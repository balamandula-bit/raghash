import json

import pytest

from raghash.duplicates import find_duplicates
from raghash.reader import read_corpus
from raghash.synth.drift import apply_drift
from raghash.synth.generate import generate
from raghash.testset import build_testset, diff_testsets, grade, grade_answer, question_id
from raghash.truth import load_manifest, load_truth
from raghash.versions import find_versions

KEY = ("topic", "subject", "scope", "region")


def build(corpus, **kw):
    docs = read_corpus(corpus)
    dup = find_duplicates(docs)
    return build_testset(docs, dup, find_versions(docs, dup), **kw)


def keyof(d):
    return tuple(d[k].casefold() for k in KEY)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w")
    generate(root, seed=42)
    return root, build(root / "corpus")


def by_type(ts, t):
    return [q for q in ts["questions"] if q["type"] == t]


def test_ids_are_stable_and_unique(world):
    _, ts = world
    ids = [q["id"] for q in ts["questions"]]
    assert len(ids) == len(set(ids))
    q = ts["questions"][0]
    assert q["id"] == question_id(keyof(q["key"]))


def test_conflict_questions_are_exactly_the_planted_contradictions(world):
    root, ts = world
    labels = json.loads((root / "ground_truth" / "labels.json").read_text("utf-8"))
    want = {keyof(f["fact"]) for f in labels["faults"] if f["type"] == "contradiction"}
    got = {keyof(q["key"]) for q in by_type(ts, "conflict")}
    assert got == want and all(q["expected"] == "FLAG_CONFLICT" for q in by_type(ts, "conflict"))


def test_stale_traps_are_exactly_the_changed_facts_of_version_pairs(world):
    root, ts = world
    labels = json.loads((root / "ground_truth" / "labels.json").read_text("utf-8"))
    changed = {keyof(c): c for f in labels["faults"] if f["type"] == "version_pair" for c in f["changed_facts"]}
    stale = {keyof(q["key"]): q for q in by_type(ts, "stale_trap")}
    assert set(stale) == set(changed)
    for k, q in stale.items():
        assert q["expected"].casefold() == changed[k]["new_value"].casefold()
        assert [x["value"].casefold() for x in q["distractors"] if x["why"] == "stale"] == [changed[k]["old_value"].casefold()]


def test_scope_traps_show_up_as_other_scope_distractors(world):
    root, ts = world
    labels = json.loads((root / "ground_truth" / "labels.json").read_text("utf-8"))
    q_by_key = {keyof(q["key"]): q for q in ts["questions"]}
    traps = [f for f in labels["faults"] if f["type"] == "scope_trap"]
    assert traps
    for f in traps:
        q = q_by_key[keyof(f["base_fact"])]
        assert f["faq_fact"]["value_text"] in [x["value"] for x in q["distractors"] if x["why"] == "other_scope"]


def test_expected_answers_match_the_answer_key(world):
    """Every non-conflict question expects a value the manifest gives for that fact in a non-superseded document."""
    root, ts = world
    truth = load_truth(root / "ground_truth")
    old_docs = {old for old, _ in truth.versions}
    allowed = {}
    for row in load_manifest(root / "ground_truth"):
        if row["path"] in old_docs:
            continue
        for f in row["facts"]:
            allowed.setdefault(keyof(f), set()).add(f["value_text"].casefold())
    for q in ts["questions"]:
        if q["type"] != "conflict":
            assert q["expected"].casefold() in allowed[keyof(q["key"])], q["id"]


def test_limiting_plain_questions_keeps_every_hard_one(world):
    root, ts = world
    small = build(root / "corpus", max_fact_questions=10)
    assert small["counts"] == {"fact": 10, "stale_trap": 30, "conflict": 20}


# ----------------------------------------------------------------------------- grading
def answers_for(ts, fn):
    return {q["id"]: fn(q) for q in ts["questions"]}


def test_a_perfect_system_scores_100(world):
    _, ts = world
    r = grade(ts, answers_for(ts, lambda q: "There is a conflict between the documents." if q["type"] == "conflict"
                              else f"It is {q['expected']}."))
    assert r["accuracy"] == 1.0, r["outcomes"]


def test_a_stale_system_is_caught(world):
    _, ts = world

    def stale(q):
        for d in q["distractors"]:
            if d["why"] == "stale":
                return f"It is {d['value']}."
        return f"It is {q['expected']}." if q["type"] != "conflict" else "Conflict."
    r = grade(ts, answers_for(ts, stale))
    assert r["outcomes"]["stale_answer"] == 30 and r["by_type"]["stale_trap"] == {"stale_answer": 30}


def test_scope_confusion_and_missed_conflicts_are_named(world):
    _, ts = world
    scope_q = next(q for q in ts["questions"] if any(d["why"] == "other_scope" for d in q["distractors"]))
    wrong = next(d for d in scope_q["distractors"] if d["why"] == "other_scope")["value"]
    assert grade_answer(scope_q, f"{wrong}") == "other_scope_answer"
    region_q = next(q for q in ts["questions"] if q["kind"] == "num" and q["type"] == "fact"
                    and all(d["why"] == "other_region" for d in q["distractors"]) and q["distractors"])
    assert grade_answer(region_q, region_q["distractors"][0]["value"]) == "other_region_answer"
    conflict_q = by_type(ts, "conflict")[0]
    pick = next(iter(conflict_q["values"].values()))
    assert grade_answer(conflict_q, f"It is {pick}.") == "conflict_missed"
    assert grade_answer(conflict_q, "These documents contradict each other.") == "conflict_flagged"
    assert grade_answer(conflict_q, "I don't know") == "no_answer"
    assert grade_answer(scope_q, None) == "no_answer"
    assert grade_answer(scope_q, "about 12345 parsecs") == "wrong"


def test_hedging_between_the_right_and_a_stale_value_is_not_correct(world):
    _, ts = world
    q = next(x for x in by_type(ts, "stale_trap") if x["kind"] == "num")
    old = next(d["value"] for d in q["distractors"] if d["why"] == "stale")
    assert grade_answer(q, f"It was {old} but is now {q['expected']}.") == "hedged"


def test_yes_no_answers_are_graded_for_decision_questions(world):
    _, ts = world
    q = next(q for q in ts["questions"] if q["kind"] == "bool" and q["type"] == "fact")
    yes = q["expected"].startswith(q["asks"])
    assert grade_answer(q, "Yes." if yes else "No.") == "correct"
    assert grade_answer(q, "No." if yes else "Yes.") != "correct"


# ----------------------------------------------------------------------------- drift
def test_drift_flags_exactly_the_questions_whose_answer_changed(world, tmp_path):
    root, before = world
    log = apply_drift(root, tmp_path / "later", n=5, seed=1)
    assert len(log) == 5
    after = build(tmp_path / "later" / "corpus")
    d = diff_testsets(before, after)
    assert d["added"] == [] and d["removed"] == []
    assert {c["id"] for c in d["changed"]} == {question_id(keyof(x["fact"])) for x in log}
    assert all(c["why"] == "answer changed" and c["new_type"] == "stale_trap" for c in d["changed"])
    assert d["unchanged"] == len(before["questions"]) - 5 and d["needs_review"] == 5
    assert diff_testsets(before, before)["needs_review"] == 0

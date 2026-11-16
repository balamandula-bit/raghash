import json
from collections import Counter

import pytest

from raghash.baseline import BM25, run as run_baseline
from raghash.diagnose import build_context, diagnose, diagnose_question, relation, supporters, wasted_slots
from raghash.duplicates import find_duplicates
from raghash.reader import read_corpus
from raghash.synth.faulty_rag import inject
from raghash.synth.generate import generate
from raghash.testset import build_testset, grade
from raghash.versions import find_versions

ALL_CAUSES = {"stale_source_only", "stale_beat_current", "conflict_hidden", "conflict_ignored", "duplicate_crowding",
              "scope_confusion", "region_confusion", "generation_error", "retrieval_miss"}


def world(root, seed):
    generate(root, seed=seed)
    docs = read_corpus(root / "corpus")
    dup = find_duplicates(docs)
    ver = find_versions(docs, dup)
    return docs, build_testset(docs, dup, ver), build_context(docs, dup, ver)


@pytest.fixture(scope="module")
def w42(tmp_path_factory):
    return world(tmp_path_factory.mktemp("d"), 42)


def test_a_perfect_system_has_no_failures(w42):
    _, ts, ctx = w42
    answers = {q["id"]: {"answer": "These documents conflict." if q["type"] == "conflict" else q["expected"],
                         "retrieved": q["sources"]} for q in ts["questions"]}
    rep = diagnose(ts, answers, ctx)
    assert rep["failures"] == 0 and rep["accuracy"] == 1.0


def test_documents_are_labelled_against_the_question(w42):
    _, ts, ctx = w42
    stale_q = next(q for q in ts["questions"] if q["type"] == "stale_trap")
    old = next(d["source"] for d in stale_q["distractors"] if d["why"] == "stale")
    assert relation(ctx, old, stale_q) == "stale" and relation(ctx, stale_q["sources"][0], stale_q) == "current"
    conflict_q = next(q for q in ts["questions"] if q["type"] == "conflict")
    assert {relation(ctx, p, conflict_q) for p in conflict_q["values"]} == {"conflict_side"}
    scope_q = next(q for q in ts["questions"] if any(d["why"] == "other_scope" for d in q["distractors"]))
    other = next(d for d in scope_q["distractors"] if d["why"] == "other_scope")
    assert relation(ctx, other["source"], scope_q) in ("other_scope", "current")  # one policy can hold both scopes
    assert supporters(ctx, scope_q, other["value"], [(other["source"], "x")]) == [(other["source"], "other_scope")]


def test_copies_in_the_results_waste_slots(w42):
    _, _, ctx = w42
    group = next(sorted(p for p, g in ctx.group_of.items() if g == gid) for gid in sorted(set(ctx.group_of.values())))
    assert wasted_slots(ctx, group[:2]) == 1 and wasted_slots(ctx, group[:1]) == 0


def test_hand_made_failures(w42):
    _, ts, ctx = w42
    stale_q = next(q for q in ts["questions"] if q["type"] == "stale_trap" and q["kind"] == "num")
    old = next(d for d in stale_q["distractors"] if d["why"] == "stale")
    only = diagnose_question(stale_q, {"answer": old["value"], "retrieved": [old["source"]]}, ctx)
    assert (only["cause"], only["blame"], only["files"]) == ("stale_source_only", "corpus", [old["source"]])
    both = diagnose_question(stale_q, {"answer": old["value"], "retrieved": [old["source"], stale_q["sources"][0]]}, ctx)
    assert both["cause"] == "stale_beat_current"
    hedge = diagnose_question(stale_q, {"answer": f"It was {old['value']}, now {stale_q['expected']}",
                                        "retrieved": [old["source"], stale_q["sources"][0]]}, ctx)
    assert hedge["outcome"] == "hedged" and hedge["cause"] == "stale_beat_current"
    gen = diagnose_question(stale_q, {"answer": "999 days", "retrieved": [stale_q["sources"][0]]}, ctx)
    assert (gen["cause"], gen["blame"]) == ("generation_error", "generator")
    assert diagnose_question(stale_q, {"answer": stale_q["expected"], "retrieved": []}, ctx) is None
    miss = diagnose_question(stale_q, {"answer": "I don't know", "retrieved": []}, ctx)
    assert (miss["cause"], miss["blame"]) == ("retrieval_miss", "retriever")
    assert diagnose_question(stale_q, None, ctx)["cause"] == "retrieval_miss"


@pytest.mark.parametrize("seed", [42, 3, 11])
def test_every_injected_failure_is_diagnosed_exactly(tmp_path, seed):
    """The harness breaks answers in known ways; the diagnoser must name each cause and blame correctly."""
    _, ts, ctx = world(tmp_path, seed)
    answers, labels = inject(ts, ctx, seed=seed)
    got = {d["id"]: d for d in diagnose(ts, answers, ctx)["diagnoses"]}
    assert set(got) == set(labels)  # failed exactly where we broke it, and nowhere else
    wrong = {i: (labels[i], (got[i]["cause"], got[i]["blame"])) for i in labels
             if (got[i]["cause"], got[i]["blame"]) != (labels[i]["cause"], labels[i]["blame"])}
    assert wrong == {}
    assert {v["cause"] for v in labels.values()} == ALL_CAUSES  # every rule was exercised


def test_baseline_failures_all_get_a_cause_and_a_fix(w42):
    docs, ts, ctx = w42
    answers = run_baseline(ts, docs)
    rep = diagnose(ts, answers, ctx)
    graded = grade(ts, answers)
    assert rep["failures"] == graded["total"] - graded["outcomes"]["correct"] - graded["outcomes"].get("conflict_flagged", 0)
    assert all(d["cause"] in ALL_CAUSES and d["fix"] for d in rep["diagnoses"])
    assert rep["by_cause"].get("conflict_ignored", 0) + rep["by_cause"].get("conflict_hidden", 0) == 20  # never flags conflicts
    assert sum(rep["by_blame"].values()) == rep["failures"] and 0 < rep["corpus_share"] < 1
    counts = [f["questions_affected"] for f in rep["files_to_fix"]]
    assert counts == sorted(counts, reverse=True)


def test_grade_accepts_answers_with_retrieval_info(w42):
    _, ts, _ = w42
    q = ts["questions"][0]
    assert grade(ts, {q["id"]: {"answer": q["expected"], "retrieved": []}})["rows"][0]["outcome"] == grade(ts, {q["id"]: q["expected"]})["rows"][0]["outcome"]


def test_bm25_puts_the_named_document_first():
    class D:  # tiny stand-in with the two fields BM25 reads
        def __init__(self, path, text):
            self.path, self.text = path, text
    docs = [D("a.md", "leave policy india interns twelve days"), D("b.md", "expense policy germany receipts"),
            D("c.md", "leave policy germany managers")]
    assert BM25(docs).search("leave policy for interns in india", 2)[0] == "a.md"
    assert BM25(docs).search("zzzz", 2) == []

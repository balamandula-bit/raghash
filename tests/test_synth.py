import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from itertools import combinations

import pytest

from raghash.evaluate import score
from raghash.synth import catalog as C
from raghash.synth.generate import DEFAULT_COUNTS, generate
from raghash.truth import Truth


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    generate(out, seed=42)
    gt = out / "ground_truth"
    rows = [json.loads(line) for line in (gt / "manifest.jsonl").read_text("utf-8").splitlines()]
    labels = json.loads((gt / "labels.json").read_text("utf-8"))
    return {"corpus": out / "corpus", "manifest": {r["path"]: r for r in rows},
            "labels": labels, "truth": Truth(labels)}


def faults(gen, kind):
    return [f for f in gen["labels"]["faults"] if f["type"] == kind]


def text(gen, path):
    return (gen["corpus"] / path).read_text("utf-8")


def key(f):
    return (f["topic"], f["subject"], f["scope"], f["region"])


def facts_by_key(gen, path):
    return {key(f): f for f in gen["manifest"][path]["facts"]}


# ----------------------------------------------------------------------------- catalogue
def test_catalogue_is_well_formed():
    assert len(C.TOPICS) * len(C.REGIONS) == 120
    for t in C.TEMPLATES:
        if t.kind == "int":
            assert len(range(t.lo, t.hi + 1, t.step)) >= 3, t.subject
    for _, topic in C.TOPICS:
        assert any(t.topic == topic and t.kind == "int" and len(t.scopes) >= 3 for t in C.TEMPLATES), \
            f"{topic} cannot host a scope trap"


# ----------------------------------------------------------------------------- generator
def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, seed=7)
    generate(b, seed=7)

    def snapshot(root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}

    assert snapshot(a) == snapshot(b)
    generate(b, seed=8)
    assert snapshot(a) != snapshot(b)


def test_counts_and_files(gen):
    assert len(gen["manifest"]) == 120 + sum(DEFAULT_COUNTS.values())
    on_disk = {p.relative_to(gen["corpus"]).as_posix() for p in gen["corpus"].rglob("*") if p.is_file()}
    assert on_disk == set(gen["manifest"])
    assert len({p.lower() for p in on_disk}) == len(on_disk), "paths collide on case-insensitive systems"
    for kind, n in DEFAULT_COUNTS.items():
        assert len(faults(gen, kind)) == n


def test_untouched_documents_exist(gen):
    labelled = {p for f in gen["labels"]["faults"] for p in f["docs"]}
    assert len(gen["manifest"]) - len(labelled) == 120 - sum(DEFAULT_COUNTS.values())


def test_exact_duplicates_are_byte_identical(gen):
    for f in faults(gen, "exact_duplicate"):
        a, b = f["docs"]
        assert (gen["corpus"] / a).read_bytes() == (gen["corpus"] / b).read_bytes()


def test_near_duplicates_differ_but_keep_every_fact(gen):
    for f in faults(gen, "near_duplicate"):
        a, b = f["docs"]
        assert text(gen, a) != text(gen, b)
        assert gen["manifest"][a]["facts"] == gen["manifest"][b]["facts"]
        assert f["edits"]
        assert f["word_3gram_jaccard"] >= 0.7


def test_version_pairs(gen):
    for f in faults(gen, "version_pair"):
        old, new = f["docs"]
        assert gen["manifest"][new]["effective_date"] > gen["manifest"][old]["effective_date"]
        fo, fn = facts_by_key(gen, old), facts_by_key(gen, new)
        assert fo.keys() == fn.keys()
        changed = {k for k in fo if fo[k]["value"] != fn[k]["value"]}
        assert changed == {key(c) for c in f["changed_facts"]}
        assert 1 <= len(changed) <= 2


def test_contradictions_really_disagree(gen):
    for f in faults(gen, "contradiction"):
        base, faq = f["docs"]
        k = key(f["fact"])
        vb, vf = facts_by_key(gen, base)[k], facts_by_key(gen, faq)[k]
        assert vb["value"] != vf["value"]
        assert (vb["value_text"], vf["value_text"]) == (f["values"][base], f["values"][faq])
        assert vb["value_text"] in text(gen, base) and vf["value_text"] in text(gen, faq)


def test_scope_traps_are_not_conflicts(gen):
    for f in faults(gen, "scope_trap"):
        base, faq = f["docs"]
        bf, ff = f["base_fact"], f["faq_fact"]
        assert bf["subject"] == ff["subject"] and bf["scope"] != ff["scope"]
        same_subject = [x for x in gen["manifest"][base]["facts"] if x["subject"] == ff["subject"]]
        assert all(x["scope"] != ff["scope"] for x in same_subject)  # scope absent from base
        assert all(x["value"] != ff["value"] for x in same_subject)  # but values differ: a real trap
        assert key(ff) in facts_by_key(gen, faq)


def test_consistent_restatements_match_base(gen):
    for f in faults(gen, "consistent_restatement"):
        base, faq = f["docs"]
        base_facts, faq_facts = facts_by_key(gen, base), facts_by_key(gen, faq)
        assert len(faq_facts) == 3
        assert all(base_facts[k]["value"] == v["value"] for k, v in faq_facts.items())


def test_no_unlabelled_conflicts(gen):
    """Any two documents stating different values for the same fact must be a labelled conflict."""
    values = defaultdict(dict)
    for path, row in gen["manifest"].items():
        for fact in row["facts"]:
            values[key(fact)][path] = fact["value"]
    for k, by_doc in values.items():
        for a, b in combinations(sorted(by_doc), 2):
            if by_doc[a] != by_doc[b]:
                assert gen["truth"].relation(a, b) in ("contradiction", "version_pair"), (k, a, b)


def test_date_signals_and_mtimes(gen):
    for path, row in gen["manifest"].items():
        d = date.fromisoformat(row["effective_date"])
        body = text(gen, path)
        if row["date_signal"] == "in_text":
            assert f"{d.day} {d.strftime('%B %Y')}" in body
        else:
            assert "Effective:" not in body and "Last updated:" not in body
        if row["date_signal"] == "filename":
            assert d.isoformat() in path
        mtime = datetime.fromtimestamp((gen["corpus"] / path).stat().st_mtime, tz=timezone.utc).date()
        delta = (mtime - d).days
        if row["role"] in ("exact_duplicate_copy", "near_duplicate_copy"):
            assert delta >= 1
        else:
            assert 0 <= delta <= 3


# ----------------------------------------------------------------------------- scorer
def oracle(truth):
    return {
        "duplicates": [sorted(p) for p in truth.pairs["exact_duplicate"] | truth.pairs["near_duplicate"]],
        "versions": [list(v) for v in truth.versions],
        "contradictions": [sorted(p) for p in truth.pairs["contradiction"]],
    }


def test_scorer_perfect_prediction(gen):
    result = score(oracle(gen["truth"]), gen["truth"])
    for name in ("duplicates", "versions", "contradictions"):
        assert result[name]["precision"] == 1.0 and result[name]["recall"] == 1.0


def test_scorer_penalises_scope_traps_and_wrong_order(gen):
    pred = oracle(gen["truth"])
    pred["contradictions"].append(sorted(next(iter(gen["truth"].pairs["scope_trap"]))))
    old, new = next(iter(gen["truth"].versions))
    pred["versions"] = [v for v in pred["versions"] if v != [old, new]] + [[new, old]]
    result = score(pred, gen["truth"])
    assert result["contradictions"]["fp_breakdown"] == {"scope_trap": 1}
    assert result["contradictions"]["precision"] < 1.0
    assert result["versions"]["wrong_order"] == 1
    assert result["versions"]["recall"] < 1.0


def test_boilerplate_carries_no_facts():
    """Filler text must hold no numbers or decision words, so editing it can never change a fact."""
    import re
    pools = [*C.POOLS.values(), C.REVISION_NOTES, [f"{q} {a}" for q, a in C.FAQ_FILLER]]
    banned = re.compile(r"\d|\b(required|mandatory|optional|allowed|permitted|prohibited|forbidden)\b", re.I)
    for pool in pools:
        for sentence in pool:
            assert not banned.search(sentence), sentence

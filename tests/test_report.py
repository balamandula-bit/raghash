import json
import re

import pytest

from raghash.baseline import run as run_baseline
from raghash.duplicates import find_duplicates
from raghash.reader import read_corpus
from raghash.report import build_report_data, render_html
from raghash.synth.drift import apply_drift
from raghash.synth.generate import generate
from raghash.testset import build_testset
from raghash.versions import find_versions


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("rep")
    generate(root, seed=42)
    docs = read_corpus(root / "corpus")
    dup = find_duplicates(docs)
    ts = build_testset(docs, dup, find_versions(docs, dup))
    return root, docs, ts, run_baseline(ts, docs)


def test_counts_match_what_was_planted(world):
    _, docs, _, _ = world
    c = build_report_data(docs, generated="2026-01-01")["counts"]
    assert (c["documents"], c["copies"], c["superseded"], c["contradictions"]) == (215, 35, 20, 20)


def test_every_planted_problem_becomes_exactly_one_action(world):
    _, docs, _, _ = world
    acts = build_report_data(docs)["actions"]
    kinds = {k: sum(a["action"] == k for a in acts) for k in ("Resolve conflict", "Retire old version", "Remove copy")}
    assert kinds == {"Resolve conflict": 20, "Retire old version": 20, "Remove copy": 35}
    assert len({a["path"] for a in acts}) == len(acts)


def test_actions_are_ranked_by_questions_repaired(world):
    _, docs, _, answers = world
    d = build_report_data(docs, answers)
    counts = [a["questions_affected"] for a in d["actions"]]
    assert counts == sorted(counts, reverse=True) and counts[0] > 0
    assert d["diagnosis"]["failures"] == 132 and len(d["diagnosis"]["examples"]) >= 4
    assert all(a["questions_affected"] == 0 for a in build_report_data(docs)["actions"])  # no answers, no ranking


def test_html_is_self_contained_and_complete(world):
    _, docs, _, answers = world
    page = render_html(build_report_data(docs, answers, title="T", generated="2026-01-01"))
    for heading in ("What to fix first", "Why answers went wrong", "Corpus health", "Test set", "Contradictions (20)"):
        assert heading in page
    assert not re.search(r"(src|href)=[\"']https?:", page) and "<script" not in page
    assert "Why answers went wrong" not in render_html(build_report_data(docs, generated="2026-01-01"))


def test_text_from_documents_cannot_inject_html():
    evil = "<script>alert(1)</script>"
    data = {"title": evil, "generated": "x", "counts": {"documents": 1, "current": 1, "copies": 1, "superseded": 0,
            "contradictions": 0, "scope_differences": 0, "claims": 0}, "duplicates": [{"keep": evil, "copies": [evil], "exact": True}],
            "versions": [], "contradictions": [], "testset": {"total": 0, "counts": {}}, "drift": None, "diagnosis": None,
            "actions": [{"action": "Remove copy", "path": evil, "files": [evil], "detail": evil, "questions_affected": 0}]}
    page = render_html(data)
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_drift_section_lists_the_changed_questions(world, tmp_path):
    root, docs, ts, _ = world
    log = apply_drift(root, tmp_path / "later", n=5, seed=1)
    d = build_report_data(read_corpus(tmp_path / "later" / "corpus"), previous=ts)
    assert d["drift"]["needs_review"] == 5 and len(log) == 5
    page = render_html(d)
    assert "Since the previous test set" in page and "5</b> questions need a new expected answer" in page
    assert d["counts"]["superseded"] == 25


def test_report_data_is_json_serialisable(world):
    _, docs, _, answers = world
    json.dumps(build_report_data(docs, answers))


def test_a_clean_corpus_says_so(tmp_path):
    (tmp_path / "a.md").write_text("# Leave Policy — India\n\nApplies to: India\n\n## 1. D\n\nThe sick leave for interns is 8 days.\n", encoding="utf-8")
    page = render_html(build_report_data(read_corpus(tmp_path), generated="2026-01-01"))
    assert "Nothing to clean up" in page and "Contradictions (0)" in page

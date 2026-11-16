import random

import pytest

from raghash.contradictions import extract_claims, find_contradictions, parse_value, predictions
from raghash.duplicates import find_duplicates
from raghash.evaluate import score
from raghash.reader import read_corpus
from raghash.synth.generate import generate
from raghash.truth import load_manifest, load_truth
from raghash.versions import find_versions


WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber basil cedar dune ember").split()
_RNG = random.Random(1)
FILLER = " ".join(_RNG.choice(WORDS) for _ in range(160))


def policy(rows, region="India", title="Leave Policy", extra=""):
    table = "\n".join(f"| {s} | {sc} | {v} |" for s, sc, v in rows)
    return (f"# {title} — {region}\n\nPolicy code: POL-HR-001\nApplies to: {region}\nEffective: 1 January 2024\n\n"
            f"## 1. Details\n\n| Item | Applies to | Value |\n|---|---|---|\n{table}\n\n{FILLER}\n{extra}\n")


def faq(qas, region="India", title="Leave Policy FAQ"):
    body = "\n\n".join(f"**Q: What is the {q}?**\nA: It is {a}." for q, a in qas)
    return f"# {title} — {region}\n\nApplies to: {region}\nLast updated: 1 March 2024\n\n## 1. FAQ\n\n{body}\n"


def run(folder, files):
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    docs = read_corpus(folder)
    dup = find_duplicates(docs)
    return find_contradictions(docs, dup, find_versions(docs, dup))


def pairs(report):
    return {(c.a, c.b) for c in report.contradictions}


def test_values_parse_and_compare():
    assert parse_value("5 years").same_as(parse_value("60 months")) is True
    assert parse_value("2 weeks").same_as(parse_value("14 days")) is True
    assert parse_value("4 hours").same_as(parse_value("240 minutes")) is True
    assert parse_value("20 days").same_as(parse_value("25 days")) is False
    assert parse_value("1,000 USD").same_as(parse_value("1000 USD")) is True
    assert parse_value("20 days").same_as(parse_value("20 USD")) is None
    assert parse_value("allowed").same_as(parse_value("not allowed")) is False
    assert parse_value("required").same_as(parse_value("allowed")) is None
    assert parse_value("the limit") is None


def test_every_sentence_shape_is_read(tmp_path):
    text = ("# Leave Policy — India\n\nApplies to: India\n\n## 1. D\n\n"
            "The annual leave entitlement for interns is 12 days per year. "
            "For managers, the notice required for planned leave is set at 5 days. "
            "The policy sets the sick leave for all employees at 8 days.\n\n"
            "Remote work is allowed for all employees. For interns, night work is not allowed. "
            "The policy states that exit interview is required for managers.\n\n"
            "- Directors: bonus cap — 30 percent\n")
    (tmp_path / "a.md").write_text(text, encoding="utf-8")
    got = {(c.key[1], c.key[2], c.value.raw) for c in extract_claims(read_corpus(tmp_path)[0])}
    assert got == {("annual leave entitlement", "interns", "12 days per year"),
                   ("notice required for planned leave", "managers", "5 days"),
                   ("sick leave", "all employees", "8 days"),
                   ("remote work", "all employees", "allowed"),
                   ("night work", "interns", "not allowed"),
                   ("exit interview", "managers", "required"),
                   ("bonus cap", "directors", "30 percent")}


def test_a_numeric_conflict_between_policy_and_faq(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Sick leave", "All employees", "8 days")]),
                       "f.md": faq([("sick leave for all employees", "12 days")])})
    assert pairs(r) == {("f.md", "p.md")} and r.contradictions[0].severity == "numeric"


def test_a_flipped_permission_is_a_polarity_conflict(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Remote work", "Interns", "allowed")]),
                       "f.md": faq([("position on remote work for interns", "not allowed")])})
    assert pairs(r) == {("f.md", "p.md")} and r.contradictions[0].severity == "polarity"


def test_same_value_in_different_words_is_not_a_conflict(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Probation", "Interns", "5 years")]),
                       "f.md": faq([("probation for interns", "60 months")])})
    assert r.contradictions == []


def test_a_different_scope_is_not_a_conflict(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Sick leave", "Interns", "8 days")]),
                       "f.md": faq([("sick leave for managers", "12 days")])})
    assert r.contradictions == [] and len(r.scope_differences) == 1


def test_a_different_region_is_not_a_conflict(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Sick leave", "Interns", "8 days")], region="India"),
                       "f.md": faq([("sick leave for interns", "12 days")], region="Germany")})
    assert r.contradictions == []


def test_units_that_cannot_be_converted_are_reported_not_flagged(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Sick leave", "Interns", "8 days")]),
                       "f.md": faq([("sick leave for interns", "8 USD")])})
    assert r.contradictions == [] and len(r.incomparable) == 1


def test_a_superseded_version_is_not_a_live_conflict(tmp_path):
    old = policy([("Sick leave", "Interns", "8 days")]).replace("1 January 2024", "1 January 2020")
    new = policy([("Sick leave", "Interns", "12 days")], extra="Reviewed again this year.")
    r = run(tmp_path, {"old.md": old, "new.md": new, "f.md": faq([("sick leave for interns", "12 days")])})
    assert r.contradictions == [] and r.stats["skipped_superseded"] == 1


def test_one_file_contradicting_itself_is_reported(tmp_path):
    r = run(tmp_path, {"p.md": policy([("Sick leave", "Interns", "8 days"), ("Sick leave", "Interns", "9 days")])})
    assert len(r.internal) == 1 and r.contradictions == []


@pytest.mark.parametrize("seed", [42, 3, 11])
def test_reads_every_planted_fact_exactly(tmp_path, seed):
    """The extractor must reproduce the answer key's facts for every document."""
    generate(tmp_path, seed=seed)
    manifest = {r["path"]: r for r in load_manifest(tmp_path / "ground_truth")}
    for doc in read_corpus(tmp_path / "corpus"):
        want = {(f["topic"].casefold(), f["subject"].casefold(), f["scope"].casefold(), f["region"].casefold(),
                 f["value_text"].casefold()) for f in manifest[doc.path]["facts"]}
        got = {(*c.key, c.value.raw.casefold()) for c in extract_claims(doc)}
        assert got == want, doc.path


@pytest.mark.parametrize("seed", [42, 3, 11])
def test_finds_every_planted_contradiction_and_no_trap(tmp_path, seed):
    generate(tmp_path, seed=seed)
    truth = load_truth(tmp_path / "ground_truth")
    docs = read_corpus(tmp_path / "corpus")
    dup = find_duplicates(docs)
    report = find_contradictions(docs, dup, find_versions(docs, dup))
    res = score(predictions(report), truth)["contradictions"]
    assert res["fp"] == 0 and res["fn"] == 0, res
    assert report.incomparable == [] and report.internal == []

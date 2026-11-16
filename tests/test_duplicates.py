import random

import pytest

from raghash.duplicates import find_duplicates, meaning_tokens, numbers_of, predictions
from raghash.evaluate import score
from raghash.reader import read_corpus
from raghash.synth.generate import generate
from raghash.truth import load_truth

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber basil cedar dune ember "
         "fjord glade harbor iris jade kelp lotus maple nectar olive pearl quartz reed sage thyme umber "
         "vale willow yarrow zinc acorn birch clover daisy elm fern ginger hazel ivy juniper laurel").split()


def fake_text(seed, n=160):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(n))
    return (f"# Title {seed}\n\n## 1. Body\n\n{body}\n\nThis policy is reviewed every year by the board.\n\n"
            "Remote access is allowed for interns.\n\nThe limit is 20 days.\n")


def test_numbers_ignore_order_and_keep_decimals():
    assert numbers_of("a 20 days, 1.5 and 1,000") == numbers_of("1,000 then 1.5 then 20")
    assert numbers_of("20 days") != numbers_of("25 days")


def test_decision_words_count_negation_and_ignore_case():
    assert meaning_tokens("Access is allowed.") == meaning_tokens("access IS Allowed")
    assert meaning_tokens("Access is allowed.") != meaning_tokens("Access is not allowed.")
    assert meaning_tokens("It is required.") != meaning_tokens("It is not required.")
    assert meaning_tokens("It is unallowed and requested.") == meaning_tokens("")  # whole words only


def test_every_kind_on_a_tiny_corpus(tmp_path):
    base = fake_text(1)
    files = {
        "a.md": base,
        "b_copy.md": base,                                                       # exact
        "c_spaces.md": base.replace(" ", "  ", 5),                               # same text once cleaned
        "d_typo.md": base.replace("This policy", "This polcy"),                  # near
        "e_number.md": base.replace("20 days", "25 days"),                       # similar, number changed
        "g_flipped.md": base.replace("is allowed", "is not allowed"),            # similar, permission flipped
        "f_other.md": fake_text(2),                                              # unrelated
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    report = find_duplicates(read_corpus(tmp_path))
    kind = {frozenset((p.a, p.b)): p.kind for p in report.pairs}
    assert kind[frozenset(("a.md", "b_copy.md"))] == "exact"
    assert kind[frozenset(("a.md", "c_spaces.md"))] == "same_text"
    assert kind[frozenset(("b_copy.md", "c_spaces.md"))] == "same_text"
    assert kind[frozenset(("a.md", "d_typo.md"))] == "near"
    assert not any({"e_number.md", "g_flipped.md", "f_other.md"} & set(k) for k in kind)
    assert report.clusters == [["a.md", "b_copy.md", "c_spaces.md", "d_typo.md"]]
    copies = ("a.md", "b_copy.md", "c_spaces.md", "d_typo.md")
    expected = {frozenset((x, y)) for x in copies for y in ("e_number.md", "g_flipped.md")}
    expected.add(frozenset(("e_number.md", "g_flipped.md")))
    assert {frozenset((p.a, p.b)) for p in report.revision_candidates} == expected


@pytest.mark.parametrize("seed", [42, 3, 11])
def test_finds_every_planted_duplicate_and_nothing_else(tmp_path, seed):
    generate(tmp_path, seed=seed)
    truth = load_truth(tmp_path / "ground_truth")
    report = find_duplicates(read_corpus(tmp_path / "corpus"))
    result = score(predictions(report), truth)
    dup = result["duplicates"]
    assert dup["fp"] == 0 and dup["fn"] == 0, dup
    assert dup["recall_exact"] == 1.0 and dup["recall_near"] == 1.0
    rev = result["revision_candidates"]
    assert rev["fp"] == 0 and rev["fn"] == 0, rev  # every version pair is handed to Part 3
    stats = report.stats
    assert stats["lsh_candidates"] < 0.1 * stats["possible_pairs"]

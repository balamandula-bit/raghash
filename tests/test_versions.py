import os
import random
from datetime import datetime, timezone

import pytest

from raghash.duplicates import find_duplicates
from raghash.evaluate import score
from raghash.reader import read_corpus
from raghash.synth.generate import generate
from raghash.truth import load_truth
from raghash.versions import find_versions, predictions, version_tuple

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber basil cedar dune ember "
         "fjord glade harbor iris jade kelp lotus maple nectar olive pearl quartz reed sage thyme umber "
         "vale willow yarrow zinc acorn birch clover daisy elm fern ginger hazel ivy juniper laurel").split()


def write(folder, name, limit, title="Leave Policy — India", version=None, effective=None, code=None,
          mtime=None, seed=1):
    header = [f"# {title}", ""]
    for label, value in (("Policy code", code), ("Version", version), ("Effective", effective)):
        if value:
            header.append(f"{label}: {value}")
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(160))
    path = folder / name
    path.write_text("\n".join(header) + f"\n\n## 1. Body\n\n{body}\n\nThe limit is {limit} days.\n",
                    encoding="utf-8")
    if mtime:
        ts = datetime.strptime(mtime, "%Y-%m-%d").replace(hour=9, tzinfo=timezone.utc).timestamp()
        os.utime(path, (ts, ts))
    return path


def run(folder):
    docs = read_corpus(folder)
    return find_versions(docs, find_duplicates(docs))


def test_version_labels_parse():
    assert version_tuple("2.10") == (2, 10) and version_tuple("v3") == (3,) and version_tuple("draft") is None
    assert version_tuple("1.9") < version_tuple("1.10")


def test_orders_by_written_date(tmp_path):
    write(tmp_path, "new.md", 25, version="2.0", effective="1 March 2023")
    write(tmp_path, "old.md", 20, version="1.0", effective="1 January 2021")
    report = run(tmp_path)
    (chain,) = report.chains
    assert chain.members == ["old.md", "new.md"] and chain.current == ["new.md"] and chain.superseded == ["old.md"]
    link = chain.links[0]
    assert (link.confidence, link.conflict, link.margin_days) == ("high", False, 789)


def test_file_name_dates_are_used(tmp_path):
    write(tmp_path, "leave_2024-05-01.md", 25)
    write(tmp_path, "leave_2022-02-01.md", 20)
    (chain,) = run(tmp_path).chains
    assert chain.members == ["leave_2022-02-01.md", "leave_2024-05-01.md"]
    assert chain.links[0].confidence == "high"


def test_modified_time_is_the_last_resort_and_only_medium(tmp_path):
    write(tmp_path, "a.md", 25, mtime="2024-06-01")
    write(tmp_path, "b.md", 20, mtime="2021-06-01")
    (chain,) = run(tmp_path).chains
    assert chain.members == ["b.md", "a.md"] and chain.links[0].confidence == "medium"


def test_a_re_saved_copy_cannot_pose_as_the_newer_version(tmp_path):
    write(tmp_path, "v1.md", 20, mtime="2021-03-01")
    write(tmp_path, "v1_copy.md", 20, mtime="2025-09-01")   # same content, copied much later
    write(tmp_path, "v2.md", 25, mtime="2023-03-01")
    report = run(tmp_path)
    (chain,) = report.chains
    assert chain.members == ["v1.md", "v2.md"]
    assert chain.current == ["v2.md"] and chain.superseded == ["v1.md", "v1_copy.md"]
    assert predictions(report)["versions"] == [["v1.md", "v2.md"]]


def test_version_label_breaks_a_date_tie_with_low_confidence(tmp_path):
    write(tmp_path, "a.md", 20, version="1.0", effective="1 January 2022")
    write(tmp_path, "b.md", 25, version="2.0", effective="1 January 2022")
    (chain,) = run(tmp_path).chains
    assert chain.members == ["a.md", "b.md"] and chain.links[0].confidence == "low"


def test_date_and_version_disagreement_is_flagged(tmp_path):
    write(tmp_path, "a.md", 20, version="3.0", effective="1 January 2021")
    write(tmp_path, "b.md", 25, version="1.0", effective="1 March 2023")
    (chain,) = run(tmp_path).chains
    assert chain.members == ["a.md", "b.md"]  # the date wins
    assert chain.links[0].conflict and chain.links[0].confidence == "low"


def test_similar_files_with_different_titles_are_not_versions(tmp_path):
    write(tmp_path, "leave.md", 20, title="Leave Policy — India", code="POL-HR-011", effective="1 January 2021")
    write(tmp_path, "travel.md", 25, title="Travel Policy — India", code="POL-FIN-052", effective="1 March 2023")
    report = run(tmp_path)
    assert report.chains == [] and len(report.rejected) == 1
    assert report.rejected[0]["paths"] == ["leave.md", "travel.md"]


def test_three_versions_form_one_ordered_chain(tmp_path):
    write(tmp_path, "c.md", 30, effective="1 January 2024")
    write(tmp_path, "a.md", 10, effective="1 January 2020")
    write(tmp_path, "b.md", 20, effective="1 January 2022")
    report = run(tmp_path)
    (chain,) = report.chains
    assert chain.members == ["a.md", "b.md", "c.md"]
    assert chain.current == ["c.md"] and chain.superseded == ["a.md", "b.md"]
    assert predictions(report)["versions"] == [["a.md", "b.md"], ["b.md", "c.md"]]


def test_an_unorderable_pair_is_reported_not_guessed(tmp_path):
    write(tmp_path, "a.md", 20, effective="1 January 2022")
    write(tmp_path, "b.md", 25, effective="1 January 2022")
    report = run(tmp_path)
    assert report.chains == []
    assert report.undetermined[0]["paths"] == ["a.md", "b.md"]
    assert predictions(report)["versions"] == []


@pytest.mark.parametrize("seed", [42, 3, 11])
def test_orders_every_planted_version_pair(tmp_path, seed):
    generate(tmp_path, seed=seed)
    truth = load_truth(tmp_path / "ground_truth")
    report = run(tmp_path / "corpus")
    result = score(predictions(report), truth)["versions"]
    assert result["fp"] == 0 and result["fn"] == 0 and result["wrong_order"] == 0, result
    assert report.undetermined == [] and report.rejected == []
    links = [link for chain in report.chains for link in chain.links]
    assert len(links) == 20 and not any(link.conflict for link in links)
    assert {link.confidence for link in links} <= {"high", "medium"}

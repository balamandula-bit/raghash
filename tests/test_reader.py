from datetime import date

import pytest

from raghash.reader import (list_files, normalize, parse_date, parse_header, read_corpus,
                                read_document, split_sections)
from raghash.reader_check import check
from raghash.synth.generate import generate
from raghash.truth import load_manifest


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("reader")
    generate(out, seed=42)
    return {"corpus": out / "corpus", "rows": load_manifest(out / "ground_truth")}


# ----------------------------------------------------------------------------- whole corpus
def test_reader_matches_the_answer_key(built):
    docs = read_corpus(built["corpus"])
    assert len(docs) == len(built["rows"])
    result = check(docs, built["rows"])
    assert result["failures"] == [], result["failures"][:5]
    for metric, (ok, total) in result["tally"].items():
        assert ok == total > 0, metric


def test_sections_and_text(built):
    docs = {d.path: d for d in read_corpus(built["corpus"])}
    for row in built["rows"]:
        d = docs[row["path"]]
        assert d.text.strip() and d.n_words > 50
        assert len(d.sections) == (3 if row["role"].startswith("faq") else 5), row["path"]
        assert "  " not in d.text and "\r" not in d.text


def test_exact_copies_share_a_hash_and_near_copies_do_not(built):
    docs = {d.path: d for d in read_corpus(built["corpus"])}
    by_role = {}
    for row in built["rows"]:
        by_role.setdefault(row["role"], []).append(row)
    hashes = [d.sha256 for d in docs.values()]
    assert len(hashes) - len(set(hashes)) == len(by_role["exact_duplicate_copy"])


# ----------------------------------------------------------------------------- helpers
@pytest.mark.parametrize("text, expected", [
    ("1 April 2024", date(2024, 4, 1)),
    ("April 1, 2024", date(2024, 4, 1)),
    ("1 Apr 2024", date(2024, 4, 1)),
    ("2024-04-01", date(2024, 4, 1)),
    (" 1 April 2024. ", date(2024, 4, 1)),
    ("sometime in spring", None),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_normalize_handles_bom_crlf_and_spaces():
    raw = "﻿Hello   world  \r\n\r\n\r\n\r\nNext\tline\r\n"
    assert normalize(raw) == "Hello world\n\nNext line\n"


def test_header_parsing_tolerates_other_labels():
    pre = ["# T", "", "Policy ID: ABC-9", "Region: India", "Effective date: April 1, 2024", "Version: 3.1"]
    h = parse_header(pre)
    assert h["policy_code"] == "ABC-9" and h["applies_to"] == "India" and h["version"] == "3.1"
    assert h["date"] == date(2024, 4, 1)


def test_sections_drop_numbering():
    pre, secs = split_sections("# T\nOwner: X\n\n## 1. Purpose\nA\n\n## 2) Scope\nB\n")
    assert [s.heading for s in secs] == ["Purpose", "Scope"] and secs[1].text == "B"
    assert pre[0] == "# T"


def test_single_file_variants(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "my_policy_2023-05-06.txt"
    f.write_bytes("﻿# My Policy\r\nVersion: 2.0\r\n\r\n## 1. Intro\r\nHello.\r\n".encode("utf-8"))
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = read_corpus(tmp_path)
    assert [d.path for d in docs] == ["sub/my_policy_2023-05-06.txt"]
    d = docs[0]
    assert d.title == "My Policy" and d.version == "2.0"
    assert d.date_in_text is None and d.date_in_filename == date(2023, 5, 6)
    assert d.date_source == "filename" and d.best_date == date(2023, 5, 6)
    assert [p.name for p in list_files(tmp_path)[1]] == ["image.png"]


def test_title_falls_back_to_filename(tmp_path):
    f = tmp_path / "empty_notes.md"
    f.write_text("", encoding="utf-8")
    assert read_document(tmp_path, f).title == "empty notes"

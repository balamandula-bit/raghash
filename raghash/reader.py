"""Part 1, the Reader: turn every file in a corpus folder into one clean, uniform record.

For each .md / .txt file it extracts:
  - normalised text (stray spaces, CRLF and BOM removed) and its hash
  - title, and the header fields: policy code, owner, "applies to", version
  - every date clue: written in the text, written in the file name, and the file's modified time
  - the sections (split on "## " headings)

    python -m raghash.reader data/synth_v1/corpus --out data/synth_v1/parsed.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

SUPPORTED = {".md", ".txt"}
DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d")
HEADER_LABELS = {
    "policy code": "policy_code",
    "policy id": "policy_code",
    "document id": "policy_code",
    "owner": "owner",
    "applies to": "applies_to",
    "region": "applies_to",
    "version": "version",
    "effective": "date",
    "effective date": "date",
    "last updated": "date",
    "date": "date",
}
_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z ]{1,30}):\s*(.+?)\s*$")
_SECTION_RE = re.compile(r"^##\s+(?:\d+[.)]\s*)?(.+?)\s*$")
_ISO_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Document:
    path: str  # relative to the corpus folder, forward slashes
    title: str
    text: str  # normalised
    sections: list
    sha256: str  # of the raw bytes (exact duplicates)
    text_sha256: str  # of the normalised text (duplicates that differ only in whitespace)
    n_words: int
    policy_code: Optional[str] = None
    owner: Optional[str] = None
    applies_to: Optional[str] = None
    version: Optional[str] = None
    date_in_text: Optional[date] = None
    date_in_filename: Optional[date] = None
    mtime: Optional[date] = None

    @property
    def best_date(self) -> Optional[date]:
        """Most trustworthy date clue: written in the text, else in the file name, else file mtime."""
        return self.date_in_text or self.date_in_filename or self.mtime

    @property
    def date_source(self) -> Optional[str]:
        if self.date_in_text:
            return "text"
        if self.date_in_filename:
            return "filename"
        return "mtime" if self.mtime else None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("date_in_text", "date_in_filename", "mtime"):
            d[key] = d[key].isoformat() if d[key] else None
        best = self.best_date
        d["best_date"] = best.isoformat() if best else None
        d["date_source"] = self.date_source
        return d


# ----------------------------------------------------------------------------- helpers
def parse_date(value: str) -> Optional[date]:
    value = value.strip().rstrip(".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def normalize(text: str) -> str:
    """Drop BOM, unify line endings, collapse repeated spaces, strip trailing spaces, squeeze blank lines."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"(?<=\S)[ \t]+(?=\S)", " ", ln).rstrip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def split_sections(text: str):
    """Return (preamble_lines, sections). Sections start at '## ' headings; numbering is dropped."""
    preamble, sections, current = [], [], None
    for line in text.split("\n"):
        m = _SECTION_RE.match(line)
        if m:
            if current:
                sections.append(current)
            current = [m.group(1), []]
        elif current:
            current[1].append(line)
        else:
            preamble.append(line)
    if current:
        sections.append(current)
    return preamble, [Section(h, "\n".join(body).strip()) for h, body in sections]


def parse_header(preamble: list) -> dict:
    fields = {}
    for line in preamble:
        m = _HEADER_RE.match(line)
        if not m:
            continue
        key = HEADER_LABELS.get(m.group(1).strip().lower())
        if key and key not in fields:
            fields[key] = m.group(2)
    if "date" in fields:
        fields["date"] = parse_date(fields["date"])
    return fields


def find_title(text: str, path: Path) -> str:
    lines = text.split("\n")
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    for line in lines:
        if line.strip():
            return line.strip()
    return path.stem.replace("_", " ").strip()


# ----------------------------------------------------------------------------- public API
def read_document(root: Path, path: Path) -> Document:
    raw = path.read_bytes()
    text = normalize(raw.decode("utf-8-sig", errors="replace"))
    preamble, sections = split_sections(text)
    header = parse_header(preamble)
    name_match = _ISO_IN_NAME.search(path.stem)
    return Document(
        path=path.relative_to(root).as_posix(),
        title=find_title(text, path),
        text=text,
        sections=sections,
        sha256=hashlib.sha256(raw).hexdigest(),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        n_words=len(text.split()),
        policy_code=header.get("policy_code"),
        owner=header.get("owner"),
        applies_to=header.get("applies_to"),
        version=header.get("version"),
        date_in_text=header.get("date"),
        date_in_filename=parse_date(name_match.group(1)) if name_match else None,
        mtime=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date(),
    )


def list_files(root: Path):
    """(supported files, skipped files), both sorted by relative path."""
    everything = sorted((p for p in Path(root).rglob("*") if p.is_file()),
                        key=lambda p: p.relative_to(root).as_posix().lower())
    return ([p for p in everything if p.suffix.lower() in SUPPORTED],
            [p for p in everything if p.suffix.lower() not in SUPPORTED])


def read_corpus(root) -> list:
    root = Path(root)
    return [read_document(root, p) for p in list_files(root)[0]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="corpus folder")
    ap.add_argument("--out", help="write one JSON record per document to this .jsonl file")
    args = ap.parse_args()
    root = Path(args.corpus)
    docs = read_corpus(root)
    skipped = list_files(root)[1]
    print(f"Read {len(docs)} documents ({len(skipped)} unsupported files skipped)")
    print("  date source:", dict(Counter(d.date_source for d in docs)))
    print("  with version:", sum(1 for d in docs if d.version), "| with policy code:",
          sum(1 for d in docs if d.policy_code))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            for d in docs:
                fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        print("Wrote", out)


if __name__ == "__main__":
    main()

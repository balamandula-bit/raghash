"""Part 3, the Version finder: for each set of revisions of one document, which file is newest?

Input: the Reader's documents and the Duplicate finder's report (its `revision_candidates`).
Steps:
  1. Keep only candidates that are really the same document (same title, or same policy code).
  2. Collapse duplicate copies into one group, so a re-saved copy cannot pose as a newer version.
  3. Give each group a date: written in the text, else in the file name, else the file modified time
     (for a group of copies, the EARLIEST modified time, which is the original's).
  4. Order the groups oldest to newest. A tie on date is broken by the version label ("1.0" < "2.0").
  5. Report every chain: the current file(s), the superseded ones, and how confident we are.

Confidence:  high    ordered by two real dates (text or file name)
             medium  ordered by dates, but one came from a file modified time
             low     ordered only by version label, or the date and the version label disagree
A set whose order cannot be decided is listed under `undetermined` rather than guessed.

    python -m raghash.versions data/synth_v1/corpus --out data/synth_v1/pred_versions.json
    python -m raghash.evaluate --truth data/synth_v1/ground_truth --pred data/synth_v1/pred_versions.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from functools import cmp_to_key
from pathlib import Path
from typing import Optional

from .duplicates import DuplicateReport, find_duplicates
from .reader import Document, read_corpus


@dataclass(frozen=True)
class Stamp:
    when: Optional[date]
    source: str  # text | filename | mtime | none


@dataclass(frozen=True)
class Item:
    """One duplicate group: every file that is a copy of the same content."""
    paths: tuple
    rep: str  # the original: earliest modified time
    stamp: Stamp
    version: Optional[tuple]
    label: Optional[str]


@dataclass
class VersionLink:
    old: str
    new: str
    confidence: str
    margin_days: Optional[int]
    conflict: bool
    evidence: dict

    def to_dict(self) -> dict:
        return {"old": self.old, "new": self.new, "confidence": self.confidence,
                "margin_days": self.margin_days, "conflict": self.conflict, "evidence": self.evidence}


@dataclass
class VersionChain:
    members: list  # oldest to newest, one representative per group
    current: list  # every file in the newest group
    superseded: list  # every file in an older group
    links: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"members": self.members, "current": self.current, "superseded": self.superseded,
                "links": [link.to_dict() for link in self.links]}


@dataclass
class VersionReport:
    chains: list
    undetermined: list  # sets of files that are revisions of each other but cannot be ordered
    rejected: list  # similar files that are NOT the same document


def version_tuple(label: Optional[str]) -> Optional[tuple]:
    if not label:
        return None
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", label.strip(), re.IGNORECASE)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def same_document(a: Document, b: Document) -> Optional[str]:
    """Why two files are revisions of one document, or None if nothing says so."""
    if a.policy_code and a.policy_code == b.policy_code:
        return "policy code"
    if _norm(a.title) and _norm(a.title) == _norm(b.title):
        return "title"
    return None


def effective_stamp(doc: Document, group: list) -> Stamp:
    if doc.date_in_text:
        return Stamp(doc.date_in_text, "text")
    if doc.date_in_filename:
        return Stamp(doc.date_in_filename, "filename")
    mtimes = [d.mtime for d in group if d.mtime]
    return Stamp(min(mtimes), "mtime") if mtimes else Stamp(None, "none")


def _compare(x: Item, y: Item):
    """(-1 if x is older, 1 if newer, 0 unknown), plus the date verdict, version verdict and conflict flag."""
    by_date = 0
    if x.stamp.when and y.stamp.when and x.stamp.when != y.stamp.when:
        by_date = -1 if x.stamp.when < y.stamp.when else 1
    by_version = 0
    if x.version and y.version and x.version != y.version:
        by_version = -1 if x.version < y.version else 1
    conflict = bool(by_date and by_version and by_date != by_version)
    return (by_date or by_version), by_date, by_version, conflict


def _confidence(old: Item, new: Item, by_date: int, conflict: bool) -> str:
    if conflict or by_date == 0:
        return "low"
    if "mtime" in (old.stamp.source, new.stamp.source):
        return "medium"
    return "high"


def find_versions(docs: list, dup: DuplicateReport) -> VersionReport:
    by_path = {d.path: d for d in docs}
    cluster_of = {p: tuple(c) for c in dup.clusters for p in c}

    def group_of(path: str) -> tuple:
        return cluster_of.get(path, (path,))

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rejected = []
    for pair in dup.revision_candidates:
        a, b = by_path[pair.a], by_path[pair.b]
        reason = same_document(a, b)
        if reason is None:
            rejected.append({"paths": sorted([pair.a, pair.b]), "similarity": pair.similarity,
                             "reason": "similar text, but different title and policy code"})
            continue
        parent[find(group_of(pair.a))] = find(group_of(pair.b))

    components = defaultdict(set)
    for group in list(parent):
        components[find(group)].add(group)

    chains, undetermined = [], []
    for groups in components.values():
        items = []
        for group in groups:
            rep = min(group, key=lambda p: (by_path[p].mtime or date.max, p))
            doc = by_path[rep]
            items.append(Item(tuple(sorted(group)), rep, effective_stamp(doc, [by_path[p] for p in group]),
                              version_tuple(doc.version), doc.version))
        items.sort(key=cmp_to_key(lambda x, y: _compare(x, y)[0]))
        links = []
        for old, new in zip(items, items[1:]):
            decision, by_date, _, conflict = _compare(old, new)
            if decision != -1:
                links = None
                break
            margin = (new.stamp.when - old.stamp.when).days if by_date else None
            links.append(VersionLink(
                old=old.rep, new=new.rep, confidence=_confidence(old, new, by_date, conflict),
                margin_days=margin, conflict=conflict,
                evidence={
                    "old_date": old.stamp.when.isoformat() if old.stamp.when else None,
                    "old_date_source": old.stamp.source, "old_version": old.label,
                    "new_date": new.stamp.when.isoformat() if new.stamp.when else None,
                    "new_date_source": new.stamp.source, "new_version": new.label,
                    "same_document_by": same_document(by_path[old.rep], by_path[new.rep]),
                }))
        if links is None:
            undetermined.append({"paths": sorted(p for it in items for p in it.paths),
                                 "reason": "no date or version label puts these in order"})
            continue
        chains.append(VersionChain(
            members=[it.rep for it in items], current=list(items[-1].paths),
            superseded=[p for it in items[:-1] for p in it.paths], links=links))
    chains.sort(key=lambda c: c.members[0])
    undetermined.sort(key=lambda u: u["paths"])
    rejected.sort(key=lambda r: r["paths"])
    return VersionReport(chains, undetermined, rejected)


def predictions(report: VersionReport) -> dict:
    """The JSON that `raghash.evaluate` reads, plus the details a person would want to open."""
    return {
        "versions": [[link.old, link.new] for chain in report.chains for link in chain.links],
        "superseded": sorted(p for chain in report.chains for p in chain.superseded),
        "chains": [chain.to_dict() for chain in report.chains],
        "undetermined": report.undetermined,
        "rejected": report.rejected,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="corpus folder")
    ap.add_argument("--out", help="write predictions JSON here")
    args = ap.parse_args()
    docs = read_corpus(args.corpus)
    report = find_versions(docs, find_duplicates(docs))
    links = [link for chain in report.chains for link in chain.links]
    print(f"Version chains: {len(report.chains)}  ({len(links)} ordered pairs)")
    print("  confidence:", dict(Counter(link.confidence for link in links)))
    print("  date conflicts flagged:", sum(1 for link in links if link.conflict))
    print(f"Could not order: {len(report.undetermined)}   Similar but not the same document: {len(report.rejected)}")
    print("Superseded files:", sum(len(c.superseded) for c in report.chains))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(predictions(report), indent=1), encoding="utf-8")
        print("Wrote", args.out)


if __name__ == "__main__":
    main()

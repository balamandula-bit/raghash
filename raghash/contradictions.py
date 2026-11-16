"""Part 4, the Contradiction finder: two current documents that state different values for the same fact.

Steps:
  1. Extract claims. A claim is (topic, subject, scope, region) -> value, read from table rows, bullets,
     FAQ question/answer pairs and plain sentences. Topic comes from the title, region from "Applies to".
  2. Drop what cannot be a live conflict: files Part 3 marked superseded, and extra copies of a duplicate group.
  3. Group claims by fact key. Two documents with different values for one key are a contradiction.
     The key includes scope and region, so "tier-1 vs tier-2" or "India vs Germany" is NOT a contradiction.
  4. Values are compared as numbers with units ("5 years" equals "60 months") or as a decision
     (allowed / not allowed, required / not required). Values in different units that cannot be converted
     are listed as `incomparable`, never guessed.

    python -m raghash.contradictions data/synth_v1/corpus --out data/synth_v1/pred_contradictions.json
    python -m raghash.evaluate --truth data/synth_v1/ground_truth --pred data/synth_v1/pred_contradictions.json

Limit: the extractor knows the sentence shapes of the practice corpus. Real documents need a smarter extractor
(an LLM or a trained tagger); everything after step 1 stays the same.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from .duplicates import DuplicateReport, find_duplicates
from .reader import Document, read_corpus
from .versions import VersionReport, find_versions

POLARITY = {"allowed": ("allow", True), "not allowed": ("allow", False),
            "required": ("req", True), "not required": ("req", False)}
# unit -> (base unit, multiplier). Only conversions that are always exact.
UNITS = {"minute": ("min", 1), "hour": ("min", 60), "day": ("day", 1), "week": ("day", 7),
         "month": ("mo", 1), "year": ("mo", 12)}
POL = r"(?:not allowed|allowed|not required|required)"


@dataclass(frozen=True)
class Value:
    kind: str  # "num" | "bool"
    amount: float  # number, or 1.0 / 0.0 for a decision
    unit: str  # base unit for numbers; "allow" | "req" for decisions
    raw: str

    def same_as(self, other: "Value") -> Optional[bool]:
        """True / False, or None when the two cannot be compared."""
        if self.kind != other.kind or self.unit != other.unit:
            return None
        return abs(self.amount - other.amount) < 1e-9


@dataclass(frozen=True)
class Claim:
    path: str
    key: tuple  # (topic, subject, scope, region), all casefolded
    value: Value
    line: str


def parse_value(text: str) -> Optional[Value]:
    s = re.sub(r"\s+", " ", text.strip().rstrip(".")).strip()
    if s.casefold() in POLARITY:
        unit, yes = POLARITY[s.casefold()]
        return Value("bool", 1.0 if yes else 0.0, unit, s)
    m = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)(?: ([A-Za-z][A-Za-z /-]*))?", s)
    if not m:
        return None
    amount, unit = float(m.group(1).replace(",", "")), (m.group(2) or "").casefold().strip()
    stem = unit[:-1] if unit.endswith("s") and unit[:-1] in UNITS else unit
    if stem in UNITS:
        base, mult = UNITS[stem]
        return Value("num", amount * mult, base, s)
    return Value("num", amount, unit, s)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip().casefold()


def _topic(title: str) -> str:
    t = re.sub(r"\s*[—–-]\s*[^—–-]*$", "", title or "").strip()
    return _clean(re.sub(r"\s+FAQ$", "", t, flags=re.I))


def _sentences(line: str) -> list:
    return re.split(r"(?<=\.)\s+(?=[A-Z])", line.strip())


# (regex, group order). Groups are named so the frame can put them anywhere.
SENTENCE_FRAMES = [
    re.compile(r"^For (?P<scope>.+?), the (?P<subject>.+) is set at (?P<value>.+?)\.?$"),
    re.compile(r"^The policy sets the (?P<subject>.+) for (?P<scope>.+?) at (?P<value>.+?)\.?$"),
    re.compile(r"^The policy states that (?P<subject>.+) is (?P<value>" + POL + r") for (?P<scope>.+?)\.?$"),
    re.compile(r"^For (?P<scope>.+?), (?P<subject>.+) is (?P<value>" + POL + r")\.?$"),
    re.compile(r"^The (?P<subject>.+) for (?P<scope>.+?) is (?P<value>.+?)\.?$"),
    re.compile(r"^(?P<subject>.+?) is (?P<value>" + POL + r") for (?P<scope>.+?)\.?$"),
]
FAQ_Q = re.compile(r"^\*\*Q: What is the (?:position on )?(?P<subject>.+) for (?P<scope>.+?)\?\*\*$")
FAQ_A = re.compile(r"^A: (?:It is )?(?P<value>.+?)\.?$")
TABLE_ROW = re.compile(r"^\|\s*(?P<subject>[^|]+?)\s*\|\s*(?P<scope>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|$")
BULLET = re.compile(r"^- (?P<scope>[^:]+): (?P<subject>.+?) — (?P<value>.+)$")


def extract_claims(doc: Document) -> list:
    topic, region = _topic(doc.title), _clean(doc.applies_to or "")
    claims, pending = [], None

    def add(subject, scope, value, line):
        v = parse_value(value)
        if v is not None:
            claims.append(Claim(doc.path, (topic, _clean(subject), _clean(scope), region), v, line.strip()))
        return v is not None

    for line in doc.text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = FAQ_Q.match(line)
        if m:
            pending = (m["subject"], m["scope"])
            continue
        if pending:
            a = FAQ_A.match(line)
            if a:
                add(pending[0], pending[1], a["value"], line)
            pending = None
            if a:
                continue
        m = TABLE_ROW.match(line)
        if m:
            if m["subject"].casefold() != "item" and not set(m["subject"]) <= {"-", " "}:
                add(m["subject"], m["scope"], m["value"], line)
            continue
        m = BULLET.match(line)
        if m:
            add(m["subject"], m["scope"], m["value"], line)
            continue
        for sentence in _sentences(line):
            for frame in SENTENCE_FRAMES:
                m = frame.match(sentence)
                if m and add(m["subject"], m["scope"], m["value"], sentence):
                    break
    return claims


@dataclass
class Contradiction:
    a: str
    b: str
    severity: str  # numeric | polarity
    facts: list  # one dict per fact the two files disagree on

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "severity": self.severity, "facts": self.facts}


@dataclass
class ContradictionReport:
    contradictions: list
    incomparable: list  # same fact, values that cannot be compared (different units)
    internal: list  # one file states two different values for the same fact
    scope_differences: list  # same subject, different scope or region, different values: NOT conflicts
    stats: dict


def _fact_dict(key: tuple) -> dict:
    return dict(zip(("topic", "subject", "scope", "region"), key))


def split_documents(docs: list, dup: DuplicateReport, ver: VersionReport):
    """(live documents, superseded paths, copy path -> original path). Only live documents state current facts."""
    by_path = {d.path: d for d in docs}
    superseded = {p for chain in ver.chains for p in chain.superseded}
    copies = {}
    for cluster in dup.clusters:
        rep = min(cluster, key=lambda p: (by_path[p].mtime or date.max, p))
        for p in cluster:
            if p != rep:
                copies[p] = rep
    return [d for d in docs if d.path not in superseded and d.path not in copies], superseded, copies


def find_contradictions(docs: list, dup: DuplicateReport, ver: VersionReport) -> ContradictionReport:
    live, superseded, copies = split_documents(docs, dup, ver)

    claims = [c for d in live for c in extract_claims(d)]
    by_key = defaultdict(lambda: defaultdict(list))
    for c in claims:
        by_key[c.key][c.path].append(c)

    pairs = defaultdict(list)
    incomparable, internal = [], []
    for key, per_doc in by_key.items():
        for path, cs in per_doc.items():
            if any(cs[0].value.same_as(x.value) is False for x in cs[1:]):
                internal.append({"path": path, "fact": _fact_dict(key), "values": sorted({x.value.raw for x in cs})})
        paths = sorted(per_doc)
        for i, pa in enumerate(paths):
            for pb in paths[i + 1:]:
                ca, cb = per_doc[pa][0], per_doc[pb][0]
                same = ca.value.same_as(cb.value)
                if same is None:
                    incomparable.append({"paths": [pa, pb], "fact": _fact_dict(key),
                                         "values": {pa: ca.value.raw, pb: cb.value.raw}})
                elif not same:
                    pairs[(pa, pb)].append({"fact": _fact_dict(key), "values": {pa: ca.value.raw, pb: cb.value.raw},
                                            "lines": {pa: ca.line, pb: cb.line}, "kind": ca.value.kind})
    contradictions = [Contradiction(a, b, "polarity" if all(f["kind"] == "bool" for f in fs) else "numeric",
                                    [{k: v for k, v in f.items() if k != "kind"} for f in fs])
                      for (a, b), fs in sorted(pairs.items())]

    # Same topic + subject + region, different scope, different value: the trap we must NOT flag.
    loose = defaultdict(lambda: defaultdict(set))
    for c in claims:
        topic, subject, scope, region = c.key
        loose[(topic, subject)][(scope, region)].add((c.path, c.value.raw))
    near = []
    for (topic, subject), scopes in loose.items():
        if len({v for vals in scopes.values() for _, v in vals}) > 1 and len(scopes) > 1:
            near.append({"topic": topic, "subject": subject, "scopes": len(scopes)})
    near.sort(key=lambda n: (n["topic"], n["subject"]))
    stats = {"documents": len(docs), "live_documents": len(live), "skipped_superseded": len(superseded),
             "skipped_copies": len(copies), "claims": len(claims), "fact_keys": len(by_key),
             "keys_in_two_or_more_documents": sum(1 for d in by_key.values() if len(d) > 1)}
    return ContradictionReport(contradictions, incomparable, internal, near, stats)


def predictions(report: ContradictionReport) -> dict:
    return {"contradictions": [[c.a, c.b] for c in report.contradictions],
            "details": [c.to_dict() for c in report.contradictions],
            "incomparable": report.incomparable, "internal": report.internal}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="corpus folder")
    ap.add_argument("--out", help="write predictions JSON here")
    args = ap.parse_args()
    docs = read_corpus(args.corpus)
    dup = find_duplicates(docs)
    report = find_contradictions(docs, dup, find_versions(docs, dup))
    s = report.stats
    print(f"Claims read: {s['claims']} from {s['live_documents']} current documents "
          f"(skipped {s['skipped_superseded']} superseded, {s['skipped_copies']} copies)")
    print(f"Facts stated in 2+ documents: {s['keys_in_two_or_more_documents']} of {s['fact_keys']}")
    print(f"Contradictions: {len(report.contradictions)}  "
          f"({dict(Counter(c.severity for c in report.contradictions))})")
    print(f"Same subject, different scope/region, different value (not conflicts): {len(report.scope_differences)}")
    print(f"Incomparable units: {len(report.incomparable)}   One file contradicting itself: {len(report.internal)}")
    for c in report.contradictions[:3]:
        f = c.facts[0]
        print(f"  e.g. {c.a}  vs  {c.b}: {f['fact']['subject']} ({f['fact']['scope']}, {f['fact']['region']}) "
              f"{f['values'][c.a]} vs {f['values'][c.b]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(predictions(report), indent=1), encoding="utf-8")
        print("Wrote", args.out)


if __name__ == "__main__":
    main()

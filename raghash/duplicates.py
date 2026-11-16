"""Part 2, the Duplicate finder.

Finds three kinds of duplicate pair, in order of strength:
  exact      the two files are byte-for-byte identical
  same_text  identical after the Reader cleans spaces and line endings
  near       similar text (word 3-gram Jaccard >= threshold) AND the same numbers and decision words
             ("required", "not allowed", ...)

Pairs that are similar but disagree on a number or a decision word are NOT duplicates. They are probably
two revisions of one document, so they go on a separate `revision_candidates` list for the Version finder
(Part 3). Calling a revision a duplicate is the costly mistake: it invites deleting the newer file.

    python -m raghash.duplicates data/synth_v1/corpus --out data/synth_v1/pred_duplicates.json
    python -m raghash.evaluate --truth data/synth_v1/ground_truth --pred data/synth_v1/pred_duplicates.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .minhash import MinHasher, jaccard, lsh_candidates, shingle_set
from .reader import read_corpus

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_of(text: str) -> Counter:
    """Every number written in the text, as a multiset. Order is ignored."""
    return Counter(NUMBER_RE.findall(text))


DECISION_RE = re.compile(r"\b(not\s+)?(required|mandatory|optional|allowed|permitted|prohibited|forbidden)\b",
                         re.IGNORECASE)


def meaning_tokens(text: str) -> Counter:
    """Numbers plus decision words ("required", "not allowed", ...): the parts of a text that carry a rule.

    Two copies that differ only in typos or wording agree on these. A revision that changes a limit or
    flips a permission does not. A heuristic: Part 4 compares the actual facts.
    """
    tokens = numbers_of(text)
    for negation, word in DECISION_RE.findall(text):
        tokens[("not " if negation else "") + word.lower()] += 1
    return tokens


@dataclass(frozen=True)
class DuplicatePair:
    a: str
    b: str
    kind: str  # exact | same_text | near | revision
    similarity: float

    def to_list(self) -> list:
        return sorted([self.a, self.b])


@dataclass
class DuplicateReport:
    pairs: list  # exact, same_text and near pairs
    revision_candidates: list  # similar text, different numbers
    clusters: list  # groups of mutually duplicate files, as sorted path lists
    stats: dict


def _groups(docs: list, key) -> list:
    buckets = defaultdict(list)
    for d in docs:
        buckets[key(d)].append(d.path)
    return [sorted(paths) for paths in buckets.values() if len(paths) > 1]


def _clusters(paths: list, pairs: list) -> list:
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        parent[find(pair.a)] = find(pair.b)
    groups = defaultdict(list)
    for p in paths:
        groups[find(p)].append(p)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def find_duplicates(docs: list, threshold: float = 0.7, num_perm: int = 128, bands: int = 32,
                    seed: int = 1) -> DuplicateReport:
    by_path = {d.path: d for d in docs}
    found = {}  # frozenset(path, path) -> DuplicatePair

    for group in _groups(docs, lambda d: d.sha256):
        for a, b in combinations(group, 2):
            found[frozenset((a, b))] = DuplicatePair(a, b, "exact", 1.0)
    for group in _groups(docs, lambda d: d.text_sha256):
        for a, b in combinations(group, 2):
            found.setdefault(frozenset((a, b)), DuplicatePair(a, b, "same_text", 1.0))

    shingles = {d.path: shingle_set(d.text) for d in docs}
    hasher = MinHasher(num_perm, seed)
    signatures = {path: hasher.signature(s) for path, s in shingles.items()}
    candidates = lsh_candidates(signatures, bands)

    revisions = []
    for a, b in sorted(tuple(sorted(c)) for c in candidates):
        if frozenset((a, b)) in found:
            continue
        sim = jaccard(shingles[a], shingles[b])
        if sim < threshold:
            continue
        if meaning_tokens(by_path[a].text) == meaning_tokens(by_path[b].text):
            found[frozenset((a, b))] = DuplicatePair(a, b, "near", round(sim, 4))
        else:
            revisions.append(DuplicatePair(a, b, "revision", round(sim, 4)))

    pairs = sorted(found.values(), key=lambda p: (p.a, p.b))
    n = len(docs)
    return DuplicateReport(
        pairs=pairs,
        revision_candidates=revisions,
        clusters=_clusters([d.path for d in docs], pairs),
        stats={"documents": n, "possible_pairs": n * (n - 1) // 2, "lsh_candidates": len(candidates)},
    )


def predictions(report: DuplicateReport) -> dict:
    """The JSON that `raghash.evaluate` reads."""
    return {
        "duplicates": [p.to_list() for p in report.pairs],
        "revision_candidates": [p.to_list() for p in report.revision_candidates],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="corpus folder")
    ap.add_argument("--out", help="write predictions JSON here")
    ap.add_argument("--threshold", type=float, default=0.7, help="near-duplicate similarity (default 0.7)")
    args = ap.parse_args()
    report = find_duplicates(read_corpus(args.corpus), threshold=args.threshold)
    kinds = Counter(p.kind for p in report.pairs)
    s = report.stats
    print(f"Compared {s['lsh_candidates']} candidate pairs out of {s['possible_pairs']} possible "
          f"({s['documents']} documents)")
    print(f"Duplicate pairs: {len(report.pairs)}  {dict(kinds)}")
    print(f"Duplicate groups: {len(report.clusters)}")
    print(f"Revision candidates (similar, but numbers or decision words differ): {len(report.revision_candidates)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(predictions(report), indent=1), encoding="utf-8")
        print("Wrote", args.out)


if __name__ == "__main__":
    main()

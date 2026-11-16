"""MinHash + LSH: find similar documents without comparing every pair.

A document becomes a set of word 3-grams ("shingles"). MinHash squeezes that set into a short signature;
the share of positions where two signatures agree estimates the Jaccard similarity of the two sets.
LSH then cuts each signature into bands: documents that share a whole band become candidate pairs, and
only candidates are compared exactly. Pure Python, no dependencies, deterministic for a given seed.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from itertools import combinations

_P = (1 << 61) - 1  # Mersenne prime used for universal hashing


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def shingle_set(text: str, n: int = 3) -> set:
    """Hashed word n-grams. Case and punctuation are ignored; short texts give one shingle."""
    words = re.findall(r"\w+", text.lower())
    return {hash64(" ".join(words[i : i + n])) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self._params = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(num_perm)]

    def signature(self, shingles: set) -> tuple:
        if not shingles:
            return tuple([_P] * self.num_perm)
        return tuple(min([(a * x + b) % _P for x in shingles]) for a, b in self._params)


def estimate_jaccard(sig_a: tuple, sig_b: tuple) -> float:
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


def lsh_candidates(signatures: dict, bands: int = 32) -> set:
    """Pairs (as frozensets of keys) whose signatures agree on every row of at least one band."""
    if not signatures:
        return set()
    num_perm = len(next(iter(signatures.values())))
    if num_perm % bands:
        raise ValueError(f"{num_perm} permutations cannot be split into {bands} equal bands")
    rows = num_perm // bands
    out = set()
    for band in range(bands):
        buckets = defaultdict(list)
        for key, sig in signatures.items():
            buckets[sig[band * rows : (band + 1) * rows]].append(key)
        for members in buckets.values():
            for a, b in combinations(sorted(members), 2):
                out.add(frozenset((a, b)))
    return out

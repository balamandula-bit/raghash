import random

import pytest

from raghash.minhash import MinHasher, estimate_jaccard, jaccard, lsh_candidates, shingle_set


def make_sets(overlap, size=300, seed=0):
    rng = random.Random(seed)
    base = rng.sample(range(10**6), size)
    shared = int(size * overlap)
    a = set(base)
    b = set(base[:shared]) | set(rng.sample(range(10**6, 2 * 10**6), size - shared))
    return a, b


@pytest.mark.parametrize("overlap", [0.1, 0.5, 0.8, 0.95])
def test_estimate_is_close_to_true_jaccard(overlap):
    a, b = make_sets(overlap)
    hasher = MinHasher(128, seed=1)
    assert abs(estimate_jaccard(hasher.signature(a), hasher.signature(b)) - jaccard(a, b)) < 0.12


def test_identical_sets_estimate_one():
    a, _ = make_sets(0.5)
    sig = MinHasher(64, seed=3).signature(a)
    assert estimate_jaccard(sig, sig) == 1.0


def test_signatures_are_deterministic():
    a, _ = make_sets(0.5)
    assert MinHasher(64, seed=7).signature(a) == MinHasher(64, seed=7).signature(a)
    assert MinHasher(64, seed=7).signature(a) != MinHasher(64, seed=8).signature(a)


def test_lsh_pairs_similar_documents_and_skips_unrelated_ones():
    x, y = make_sets(0.9, seed=1)
    z = set(random.Random(5).sample(range(3 * 10**6, 4 * 10**6), 300))
    hasher = MinHasher(128, seed=1)
    sigs = {"x": hasher.signature(x), "y": hasher.signature(y), "z": hasher.signature(z)}
    found = lsh_candidates(sigs, bands=32)
    assert frozenset(("x", "y")) in found
    assert frozenset(("x", "z")) not in found and frozenset(("y", "z")) not in found


def test_lsh_rejects_uneven_bands():
    sigs = {"a": (1,) * 10, "b": (1,) * 10}
    with pytest.raises(ValueError):
        lsh_candidates(sigs, bands=3)


def test_shingles_ignore_case_and_punctuation_and_handle_short_text():
    assert shingle_set("Hello, World again!") == shingle_set("hello world AGAIN")
    assert len(shingle_set("two words")) == 1
    assert len(shingle_set("")) == 1

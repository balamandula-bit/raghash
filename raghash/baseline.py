"""A deliberately plain RAG system to test the diagnoser on: BM25 over whole documents plus a rule-based answerer.

It is NOT part of RagHash's checks. It stands in for the system you want to diagnose, and its output
file has the same shape a real system's would:   {"Q-1a2b3c4d": {"answer": "25 days", "retrieved": ["hr/a.md", ...]}}

    python -m raghash.baseline data/synth_v1/testset.json data/synth_v1/corpus --out data/synth_v1/answers.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .contradictions import extract_claims
from .reader import read_corpus

TOKEN = re.compile(r"[a-z0-9]+")


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(TOKEN.findall(d.text.lower())) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 5) -> list:
        q = TOKEN.findall(query.lower())
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in q:
                f = self.tf[i].get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scored.append((-s, d.path))
        scored.sort()
        return [p for s, p in scored[:k] if s < 0]


def answer(question: dict, retrieved: list, claims_of) -> str:
    """Take the first retrieved document that states the same subject and scope. Looks at nothing else:
    not the region, not whether the file is current, not whether another file disagrees."""
    k = question["key"]
    topic, subject, scope = k["topic"].casefold(), k["subject"].casefold(), k["scope"].casefold()
    for path in retrieved:
        for c in claims_of(path):
            if c.key[0] == topic and c.key[1] == subject and c.key[2] == scope:
                return c.value.raw
    return "I could not find this in the documents."


def run(testset: dict, docs: list, k: int = 5) -> dict:
    index, by_path, cache = BM25(docs), {d.path: d for d in docs}, {}

    def claims_of(path):
        if path not in cache:
            cache[path] = extract_claims(by_path[path])
        return cache[path]

    out = {}
    for q in testset["questions"]:
        retrieved = index.search(q["question"], k)
        out[q["id"]] = {"answer": answer(q, retrieved, claims_of), "retrieved": retrieved}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("testset")
    ap.add_argument("corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    ts = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    out = run(ts, read_corpus(args.corpus), args.k)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Answered {len(out)} questions (BM25 top-{args.k}). Wrote {args.out}")


if __name__ == "__main__":
    main()

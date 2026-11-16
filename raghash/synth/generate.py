"""Generate the synthetic corpus, its manifest and its ground-truth labels.

    python -m raghash.synth.generate --out data/synth_v1 --seed 42

Output layout (the scanner under test must only ever read corpus/):
    <out>/corpus/...                 the documents
    <out>/ground_truth/manifest.jsonl  one row per document (true dates, facts, role)
    <out>/ground_truth/labels.json     every injected fault
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .builder import build_base_docs
from .faults import (make_consistent, make_contradiction, make_exact_duplicate, make_near_duplicate,
                     make_scope_trap, make_version)
from .model import render

DEFAULT_COUNTS = {
    "exact_duplicate": 15,
    "near_duplicate": 20,
    "version_pair": 20,
    "contradiction": 20,
    "scope_trap": 10,
    "consistent_restatement": 10,
}
MAKERS = [
    ("exact_duplicate", make_exact_duplicate),
    ("near_duplicate", make_near_duplicate),
    ("version_pair", make_version),
    ("contradiction", make_contradiction),
    ("scope_trap", make_scope_trap),
    ("consistent_restatement", make_consistent),
]


def _write(corpus_dir: Path, doc, text: str) -> bytes:
    target = corpus_dir / doc.path
    target.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    target.write_bytes(data)
    m = doc.mtime
    ts = datetime(m.year, m.month, m.day, 9, 0, tzinfo=timezone.utc).timestamp()
    os.utime(target, (ts, ts))
    return data


def generate(out_dir, seed: int = 42, counts: dict = None) -> dict:
    out = Path(out_dir)
    corpus_dir, gt_dir = out / "corpus", out / "ground_truth"
    for d in (corpus_dir, gt_dir):
        if d.exists():
            shutil.rmtree(d)
    gt_dir.mkdir(parents=True)

    rng = random.Random(seed)
    counts = {**DEFAULT_COUNTS, **(counts or {})}
    bases = build_base_docs(rng)
    if sum(counts.values()) > len(bases):
        raise ValueError(f"need {sum(counts.values())} base documents, only {len(bases)} exist")

    shuffled = bases[:]
    rng.shuffle(shuffled)  # each base document is used by at most one fault
    taken = {d.path.lower() for d in bases}
    faults, new_docs, cursor = [], [], 0
    for kind, maker in MAKERS:
        for base in shuffled[cursor : cursor + counts[kind]]:
            doc, label = maker(rng, base, taken)
            faults.append(label)
            new_docs.append(doc)
        cursor += counts[kind]
    for i, label in enumerate(faults, 1):
        label["id"] = f"F{i:03d}"
        label.move_to_end("id", last=False) if hasattr(label, "move_to_end") else None

    fault_ids: dict = {}
    for label in faults:
        for p in label["docs"]:
            fault_ids.setdefault(p, []).append(label["id"])

    docs = sorted(bases + new_docs, key=lambda d: d.path.lower())
    rows = []
    for n, doc in enumerate(docs, 1):
        text = render(doc)
        data = _write(corpus_dir, doc, text)
        rows.append({
            "id": f"D{n:04d}",
            "path": doc.path,
            "role": doc.role,
            "dept": doc.dept,
            "topic": doc.topic,
            "region": doc.region,
            "title": doc.title,
            "policy_code": doc.policy_code,
            "version_label": doc.version,
            "effective_date": doc.effective.isoformat(),  # TRUE date, regardless of what the file shows
            "date_signal": doc.date_signal,
            "style": doc.style,
            "owner": doc.owner,
            "sha256": hashlib.sha256(data).hexdigest(),
            "n_words": len(text.split()),
            "facts": [f.to_dict() for f in doc.facts],
            "fault_ids": fault_ids.get(doc.path, []),
        })

    with open(gt_dir / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    labels = {
        "schema_version": 1,
        "seed": seed,
        "n_documents": len(docs),
        "counts": counts,
        "faults": faults,
    }
    (gt_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"n_documents": len(docs), "roles": dict(Counter(d.role for d in docs)), "counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synth_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    summary = generate(args.out, args.seed)
    print(f"Wrote {summary['n_documents']} documents to {args.out}/corpus")
    for role, n in sorted(summary["roles"].items()):
        print(f"  {role:<22}{n:>4}")
    print(f"Ground truth: {args.out}/ground_truth/labels.json")


if __name__ == "__main__":
    main()

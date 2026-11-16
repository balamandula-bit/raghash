"""Simulate a corpus changing over time: new versions of some documents appear, with one changed value each.

    python -m raghash.synth.drift --corpus data/synth_v1 --out data/synth_v2 --n 5

Copies the corpus (and its ground truth) to --out, then for n untouched documents writes a `_v2` file with one
number changed, a later date and a bumped version. The old file stays, as it would in a real shared drive.
`ground_truth/drift.json` lists exactly what changed, so a test set rebuild can be checked against it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from ..contradictions import extract_claims
from ..reader import read_document

NEW_DATE = "1 June 2026"


def _bump_version(text: str) -> str:
    def bump(m):
        parts = m.group(1).split(".")
        return f"Version: {int(parts[0]) + 1}.0"
    return re.sub(r"Version: (\d+(?:\.\d+)*)", bump, text, count=1)


def _set_date(text: str) -> str:
    if re.search(r"^Effective: .*$", text, re.M):
        return re.sub(r"^Effective: .*$", f"Effective: {NEW_DATE}", text, count=1, flags=re.M)
    return re.sub(r"^(Applies to: .*)$", rf"\1\nEffective: {NEW_DATE}", text, count=1, flags=re.M)


def apply_drift(src: Path, out: Path, n: int = 5, seed: int = 1) -> list:
    src, out = Path(src), Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    gt = out / "ground_truth"
    rows = [json.loads(x) for x in (gt / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    untouched = sorted(r["path"] for r in rows if r["role"] == "base" and not r["fault_ids"])
    random.Random(seed).shuffle(untouched)
    log = []
    for path in untouched:
        if len(log) == n:
            break
        raw = (out / "corpus" / path).read_text(encoding="utf-8")
        doc = read_document(out / "corpus", out / "corpus" / path)
        for c in extract_claims(doc):
            m = re.fullmatch(r"(\d+) (.+)", c.value.raw)
            if c.value.kind != "num" or not m or raw.count(c.line) != 1:
                continue
            old_n = int(m.group(1))
            new_raw = f"{old_n + max(2, old_n // 4)} {m.group(2)}"
            new_text = _set_date(_bump_version(raw.replace(c.line, c.line.replace(c.value.raw, new_raw))))
            new_path = re.sub(r"(\.\w+)$", r"_v2\1", path)
            (out / "corpus" / new_path).write_text(new_text, encoding="utf-8")
            log.append({"old_doc": path, "new_doc": new_path, "fact": dict(zip(("topic", "subject", "scope", "region"), c.key)),
                        "old_value": c.value.raw, "new_value": new_raw})
            break
    (gt / "drift.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="folder holding corpus/ and ground_truth/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    log = apply_drift(args.corpus, args.out, args.n, args.seed)
    print(f"Wrote {len(log)} new versions to {args.out}/corpus")
    for x in log:
        f = x["fact"]
        print(f"  {x['new_doc']}: {f['subject']} ({f['scope']}, {f['region']}) {x['old_value']} -> {x['new_value']}")


if __name__ == "__main__":
    main()

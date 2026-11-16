"""Check the Reader against the ground-truth manifest.

    python -m raghash.reader_check --corpus data/synth_v1/corpus --truth data/synth_v1/ground_truth

Rules:
  - title, region, policy code: must match exactly.
  - version: must match when the file shows one; must be empty when it does not.
  - date source: must be the one the file really offers (text, else file name, else mtime).
  - date: exact when read from text or file name; within 3 days when read from mtime.
  - Copies whose ONLY date clue is mtime are excluded from the date check: their mtime is the
    day the copy was made, not the document's date. Later parts must not trust it.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .reader import read_corpus
from .truth import load_manifest

METRICS = ("title", "region", "policy_code", "version", "date_source", "date")
COPY_ROLES = ("exact_duplicate_copy", "near_duplicate_copy")
SOURCE_FOR_SIGNAL = {"in_text": "text", "filename": "filename", "mtime_only": "mtime"}


def check(docs: list, rows: list) -> dict:
    by_path = {d.path: d for d in docs}
    tally = {m: [0, 0] for m in METRICS}  # [correct, total]
    failures, excluded = [], 0

    def record(metric, ok, path, got, want):
        tally[metric][1] += 1
        if ok:
            tally[metric][0] += 1
        else:
            failures.append({"path": path, "metric": metric, "got": str(got), "want": str(want)})

    for row in rows:
        path, signal, role = row["path"], row["date_signal"], row["role"]
        d = by_path.get(path)
        if d is None:
            failures.append({"path": path, "metric": "found", "got": "missing", "want": "present"})
            continue
        true_date = date.fromisoformat(row["effective_date"])
        record("title", d.title == row["title"], path, d.title, row["title"])
        record("region", d.applies_to == row["region"], path, d.applies_to, row["region"])
        record("policy_code", d.policy_code == (row["policy_code"] or None), path,
               d.policy_code, row["policy_code"] or None)
        want_version = row["version_label"] if signal == "in_text" and not role.startswith("faq") else None
        record("version", d.version == want_version, path, d.version, want_version)
        record("date_source", d.date_source == SOURCE_FOR_SIGNAL[signal], path,
               d.date_source, SOURCE_FOR_SIGNAL[signal])
        if signal == "mtime_only":
            if role in COPY_ROLES:
                excluded += 1
                continue
            ok = d.best_date is not None and 0 <= (d.best_date - true_date).days <= 3
        else:
            ok = d.best_date == true_date
        record("date", ok, path, d.best_date, true_date)
    return {"tally": tally, "failures": failures, "excluded_dates": excluded, "documents": len(rows)}


def format_report(result: dict) -> str:
    lines = [f"{'metric':<14}{'correct':>8}{'total':>7}"]
    for m in METRICS:
        ok, total = result["tally"][m]
        lines.append(f"{m:<14}{ok:>8}{total:>7}")
    if result["excluded_dates"]:
        lines.append(f"({result['excluded_dates']} copies with an mtime-only date left out of the date check: "
                     "their mtime is the copy date)")
    if result["failures"]:
        lines.append(f"\n{len(result['failures'])} failures, first 10:")
        for f in result["failures"][:10]:
            lines.append(f"  {f['path']} | {f['metric']}: got {f['got']!r}, want {f['want']!r}")
    else:
        lines.append("\nNo failures.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    print(format_report(check(read_corpus(Path(args.corpus)), load_manifest(args.truth))))


if __name__ == "__main__":
    main()

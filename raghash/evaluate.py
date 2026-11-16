"""Score a detector's output against the injected ground truth.

Prediction file (JSON). Paths are relative to the corpus folder, with forward slashes:

    {
      "duplicates":     [["a.md", "b.md"], ...],       # unordered pairs (exact or near)
      "versions":       [["old.md", "new.md"], ...],   # ordered: [superseded, current]
      "contradictions": [["a.md", "b.md"], ...]        # unordered pairs
    }

    python -m raghash.evaluate --truth data/synth_v1/ground_truth --pred predictions.json

Strict scoring: a predicted pair that is not in the truth set for that detector is a false positive.
`fp_breakdown` shows what those false positives really are (a scope trap, a version pair, ...).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .truth import Truth, load_truth


def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _pairs(items) -> set:
    return {frozenset(p) for p in items if len(set(p)) == 2}


def _recall(pred: set, true: set) -> float:
    return round(len(pred & true) / len(true), 4) if true else 1.0


def _section(pred: set, true: set, truth: Truth) -> dict:
    tp, fp, fn = pred & true, pred - true, true - pred
    precision, recall, f1 = _prf(len(tp), len(fp), len(fn))
    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_breakdown": dict(Counter(truth.relation(*tuple(x)) for x in fp)),
        "missed": sorted(sorted(x) for x in map(tuple, fn))[:50],
    }


def score(pred: dict, truth: Truth) -> dict:
    """Score only the detectors that appear in `pred`."""
    out = {}
    if "duplicates" in pred:
        dup_pred = _pairs(pred["duplicates"])
        dup_true = truth.pairs["exact_duplicate"] | truth.pairs["near_duplicate"]
        out["duplicates"] = _section(dup_pred, dup_true, truth)
        out["duplicates"]["recall_exact"] = _recall(dup_pred, truth.pairs["exact_duplicate"])
        out["duplicates"]["recall_near"] = _recall(dup_pred, truth.pairs["near_duplicate"])
    if "revision_candidates" in pred:  # unordered pairs that should cover the version pairs
        out["revision_candidates"] = _section(
            _pairs(pred["revision_candidates"]), {frozenset(v) for v in truth.versions}, truth)
    if "versions" in pred:
        ver_pred = {tuple(p) for p in pred["versions"] if len(set(p)) == 2}
        out["versions"] = _section(ver_pred, truth.versions, truth)
        out["versions"]["wrong_order"] = sum(1 for p in ver_pred if (p[1], p[0]) in truth.versions)
    if "contradictions" in pred:
        out["contradictions"] = _section(
            _pairs(pred["contradictions"]), truth.pairs["contradiction"], truth)
    return out


def format_report(scores: dict) -> str:
    lines = [f"{'detector':<16}{'precision':>10}{'recall':>8}{'f1':>7}{'tp':>5}{'fp':>5}{'fn':>5}"]
    for name, s in scores.items():
        lines.append(f"{name:<16}{s['precision']:>10.3f}{s['recall']:>8.3f}{s['f1']:>7.3f}"
                     f"{s['tp']:>5}{s['fp']:>5}{s['fn']:>5}")
        if s["fp_breakdown"]:
            lines.append(f"    false positives by true relation: {s['fp_breakdown']}")
        if s.get("wrong_order"):
            lines.append(f"    found but in the wrong order: {s['wrong_order']}")
        if "recall_exact" in s:
            lines.append(f"    recall exact={s['recall_exact']:.3f}  near={s['recall_near']:.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="ground_truth folder")
    ap.add_argument("--pred", required=True, help="predictions JSON file")
    args = ap.parse_args()
    with open(args.pred, encoding="utf-8") as fh:
        pred = json.load(fh)
    print(format_report(score(pred, load_truth(args.truth))))


if __name__ == "__main__":
    main()

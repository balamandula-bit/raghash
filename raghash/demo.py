"""One command, the whole pipeline, about twenty seconds:

    python -m raghash.demo                  # writes ./raghash_demo/report.html
    python -m raghash.demo --out my_demo --seed 7

Steps: make a practice corpus with a hidden answer key -> scan it (copies, old versions, contradictions) and score the
scan against the key -> build a test set -> run a plain baseline RAG on it -> diagnose its wrong answers -> write the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import contradictions, duplicates, versions
from .baseline import run as run_baseline
from .diagnose import build_context, diagnose
from .evaluate import format_report, score
from .reader import read_corpus
from .report import build_report_data, render_html
from .synth.generate import generate
from .testset import build_testset, grade
from .truth import load_truth


def run_demo(out, seed: int = 42, echo=print) -> dict:
    out = Path(out)
    echo(f"RagHash demo (seed {seed})")
    generate(out / "data", seed=seed)
    docs = read_corpus(out / "data" / "corpus")
    echo(f"[1/5] Practice corpus: {len(docs)} documents. The answer key is kept apart; the scanners never see it.")

    dup = duplicates.find_duplicates(docs)
    ver = versions.find_versions(docs, dup)
    con = contradictions.find_contradictions(docs, dup, ver)
    pred = {**duplicates.predictions(dup), **versions.predictions(ver), **contradictions.predictions(con)}
    scores = score(pred, load_truth(out / "data" / "ground_truth"))
    echo("[2/5] Corpus scan, scored against the answer key:")
    echo("\n".join("      " + line for line in format_report(
        {k: scores[k] for k in ("duplicates", "versions", "contradictions")}).splitlines()))

    ts = build_testset(docs, dup, ver)
    echo(f"[3/5] Test set: {len(ts['questions'])} questions {ts['counts']}")

    answers = run_baseline(ts, docs)
    graded = grade(ts, answers)
    echo(f"[4/5] Baseline RAG (BM25 + rule-based answerer) scores {graded['accuracy']:.1%}")
    diag = diagnose(ts, answers, build_context(docs, dup, ver))
    echo(f"      {diag['failures']} wrong answers, by cause: {diag['by_cause']}")
    echo(f"      by who should fix it: {diag['by_blame']}  ({diag['corpus_share']:.0%} trace to the documents)")

    data = build_report_data(docs, answers, title=f"RagHash demo (seed {seed})")
    report = out / "report.html"
    report.write_text(render_html(data), encoding="utf-8")
    (out / "testset.json").write_text(json.dumps(ts, indent=1), encoding="utf-8")
    (out / "answers.json").write_text(json.dumps(answers, indent=1), encoding="utf-8")
    echo(f"[5/5] Report with {len(data['actions'])} clean-up actions: {report}")
    return {"scores": scores, "accuracy": graded["accuracy"], "failures": diag["failures"],
            "actions": len(data["actions"]), "report": str(report)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="raghash_demo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_demo(args.out, args.seed)


if __name__ == "__main__":
    main()

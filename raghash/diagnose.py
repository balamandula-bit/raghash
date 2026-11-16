"""Part 6, the Failure diagnoser: for every wrong answer, WHY it was wrong and WHO is to blame.

Input: the Part 5 test set, and a system's answers with what it retrieved for each question:
    {"Q-1a2b3c4d": {"answer": "25 days", "retrieved": ["hr/leave.md", "wiki/leave_faq.md"]}}
Each retrieved document is labelled against the question, using what Parts 2-4 know about the corpus:
    current        states the right value (a copy of a current file counts)
    stale          a superseded version that states a different value
    conflict_side  one of the current documents that disagree
    other_scope / other_region   same subject, different scope or region
    irrelevant
Then the wrong answer gets one cause. Blame is the part to fix:
    corpus      the documents themselves make a right answer impossible or unlikely (stale file, conflict, copies)
    retriever   the needed document never reached the model
    generator   the right document was in front of the model and it still answered wrong

    python -m raghash.diagnose data/synth_v1/testset.json data/synth_v1/answers.json --corpus data/synth_v1/corpus
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .contradictions import extract_claims, parse_value, split_documents
from .duplicates import find_duplicates
from .reader import read_corpus
from .testset import _number_candidates, _polarity, grade_answer
from .versions import find_versions

KEY = ("topic", "subject", "scope", "region")
OK = {"correct", "conflict_flagged"}
FIX = {
    "stale_source_only": "Retire or archive the superseded file(s); index only current versions.",
    "stale_beat_current": "Keep superseded versions out of retrieval, or rank the newest version first.",
    "conflict_hidden": "Resolve the contradiction in the source documents (only one side was even retrieved).",
    "conflict_ignored": "Resolve the contradiction in the source documents, or make the system flag disagreement.",
    "duplicate_crowding": "Remove duplicate copies from the index; they filled the top results.",
    "scope_confusion": "Filter by scope metadata, or make the answerer check the scope before it answers.",
    "region_confusion": "Filter by region metadata, or make the answerer check the region before it answers.",
    "generation_error": "The right document was retrieved: tighten the answer prompt or use a stronger model.",
    "retrieval_miss": "The needed document was not retrieved: improve chunking, query rewriting or embeddings.",
}


@dataclass
class Context:
    superseded: set
    group_of: dict  # path -> duplicate group id
    claims: dict  # path -> {fact key: Value}


def build_context(docs: list, dup, ver) -> Context:
    _, superseded, _ = split_documents(docs, dup, ver)
    group_of = {p: min(c) for c in dup.clusters for p in c}
    claims = {}
    for d in docs:
        claims[d.path] = {}
        for c in extract_claims(d):
            claims[d.path].setdefault(c.key, c.value)
    return Context(superseded, group_of, claims)


def qkey(q: dict) -> tuple:
    return tuple(q["key"][k].casefold() for k in KEY)


def relation(ctx: Context, path: str, q: dict) -> str:
    key, held = qkey(q), ctx.claims.get(path, {})
    if key in held:
        if q["type"] == "conflict":
            return "conflict_side"
        if path in ctx.superseded and held[key].same_as(parse_value(q["expected"])) is False:
            return "stale"
        return "current"
    topic, subject, scope, region = key
    for k in held:
        if k[:2] == (topic, subject):
            if k[3] == region and k[2] != scope:
                return "other_scope"
            if k[2] == scope and k[3] != region:
                return "other_region"
    return "irrelevant"


def wasted_slots(ctx: Context, retrieved: list) -> int:
    """Result slots taken by copies of a document that is already in the results."""
    return len(retrieved) - len({ctx.group_of.get(p, p) for p in retrieved})


def supporters(ctx: Context, q: dict, answer, rels: list) -> list:
    """(path, kind) for retrieved documents holding a claim that matches the value the system gave but is NOT the
    right answer: kind is stale (an old version of this fact), other_scope or other_region. Best rank first.
    Works on claims, not whole documents, because one policy can state the right scope and the wrong one."""
    text, key = (answer or ""), qkey(q)
    expected = parse_value(q["expected"]) if q["type"] != "conflict" else None
    if q["kind"] == "bool":
        pol = _polarity(text, q.get("asks", ""))
        same = lambda v: pol is not None and v.kind == "bool" and v.amount == float(pol)
    else:
        cands = _number_candidates(text)
        same = lambda v: any(v.same_as(c) for c in cands)
    out, found = [], {}
    for path, _ in rels:
        for k, v in ctx.claims.get(path, {}).items():
            if k[:2] != key[:2] or not same(v):
                continue
            if k == key:
                kind = "stale" if path in ctx.superseded and expected and v.same_as(expected) is False else None
            elif k[3] == key[3]:
                kind = "other_scope"
            elif k[2] == key[2]:
                kind = "other_region"
            else:
                kind = None
            if kind:
                found.setdefault(path, set()).add(kind)
        if path in found:  # a document can match several ways: the stale file itself is the most specific
            out.append((path, next(k for k in ("stale", "other_scope", "other_region") if k in found[path])))
    return out


def diagnose_question(q: dict, rec: dict, ctx: Context):
    """One diagnosis dict, or None when the answer was right."""
    answer, retrieved = (rec or {}).get("answer"), list((rec or {}).get("retrieved") or [])
    outcome = grade_answer(q, answer)
    if outcome in OK:
        return None
    rels = [(p, relation(ctx, p, q)) for p in retrieved]
    n = Counter(r for _, r in rels)
    pick = lambda kind: [p for p, r in rels if r == kind]
    files = []
    if q["type"] == "conflict":
        sides = {ctx.claims[p][qkey(q)].raw for p in pick("conflict_side")}
        cause = "conflict_ignored" if len(sides) >= 2 else "conflict_hidden"
        blame, files = "corpus", sorted(q.get("values", {}))
    elif (src := supporters(ctx, q, answer, rels)):
        rel = src[0][1]  # the best-ranked document that states the value the system gave
        if rel == "stale":
            cause = "stale_beat_current" if n["current"] else "stale_source_only"
            blame, files = "corpus", [p for p, r in src if r == "stale"]
        else:
            cause = "scope_confusion" if rel == "other_scope" else "region_confusion"
            blame = "generator" if n["current"] else "retriever"
    elif n["current"]:
        cause, blame = "generation_error", "generator"
    elif wasted_slots(ctx, retrieved) >= 2:
        cause, blame = "duplicate_crowding", "corpus"
        groups = Counter(ctx.group_of.get(p, p) for p in retrieved)
        files = [p for p in retrieved if groups[ctx.group_of.get(p, p)] > 1]
    else:
        cause, blame = "retrieval_miss", "retriever"
    return {"id": q["id"], "question": q["question"], "type": q["type"], "outcome": outcome, "cause": cause,
            "blame": blame, "fix": FIX[cause], "files": files, "answer": answer, "expected": q["expected"],
            "retrieved": [{"path": p, "relation": r} for p, r in rels]}


def diagnose(testset: dict, answers: dict, ctx: Context) -> dict:
    rows = [d for q in testset["questions"] if (d := diagnose_question(q, answers.get(q["id"]), ctx))]
    total = len(testset["questions"])
    by_file = defaultdict(lambda: {"questions": 0, "causes": Counter()})
    for r in rows:
        for f in set(r["files"]):
            by_file[f]["questions"] += 1
            by_file[f]["causes"][r["cause"]] += 1
    files = sorted(({"path": p, "questions_affected": v["questions"], "causes": dict(v["causes"])}
                    for p, v in by_file.items()), key=lambda x: (-x["questions_affected"], x["path"]))
    blame = Counter(r["blame"] for r in rows)
    return {"total": total, "failures": len(rows), "accuracy": round(1 - len(rows) / max(1, total), 4),
            "by_cause": dict(Counter(r["cause"] for r in rows)), "by_blame": dict(blame),
            "corpus_share": round(blame["corpus"] / max(1, len(rows)), 4),
            "files_to_fix": files, "diagnoses": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("testset")
    ap.add_argument("answers")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    docs = read_corpus(args.corpus)
    dup = find_duplicates(docs)
    rep = diagnose(load(args.testset), load(args.answers), build_context(docs, dup, find_versions(docs, dup)))
    print(f"{rep['failures']} wrong answers out of {rep['total']} ({rep['accuracy']:.1%} correct)")
    print("Causes:", rep["by_cause"])
    print("Blame:", rep["by_blame"], f"-> {rep['corpus_share']:.0%} of failures trace to the corpus")
    print("Files to fix first:")
    for f in rep["files_to_fix"][:5]:
        print(f"  {f['path']}  ({f['questions_affected']} questions, {f['causes']})")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print("Wrote", args.out)


if __name__ == "__main__":
    main()

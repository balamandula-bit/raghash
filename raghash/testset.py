"""Part 5, the Test question maker: a test set that follows the corpus instead of going stale.

One question per fact the corpus states. Every question carries the answer a correct system must give, and the
wrong answers worth watching for, each with a reason:
    fact          current value, nothing nearby to confuse it with
    stale_trap    an older version of the document gave a different value (answering it = stale answer)
    conflict      two current documents disagree: the right behaviour is to flag it, not to pick one
    distractors   other_scope / other_region values for the same subject (answering one = scope confusion)
Question ids come from the fact key, so rebuilding after the documents change keeps the ids and `diff` can say
exactly which questions need a new answer. `grade` sorts a system's answers into outcomes.

    python -m raghash.testset build data/synth_v1/corpus --out data/synth_v1/testset.json
    python -m raghash.testset grade data/synth_v1/testset.json answers.json      # {"Q-1a2b3c4d": "25 days", ...}
    python -m raghash.testset diff old_testset.json new_testset.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .contradictions import extract_claims, parse_value, split_documents
from .duplicates import find_duplicates
from .reader import read_corpus
from .versions import find_versions

CONFLICT_WORDS = re.compile(r"conflict|contradict|inconsisten|disagree|discrepan|differ", re.I)
DONT_KNOW = re.compile(r"don'?t know|do not know|cannot (?:find|say|tell)|not sure|unknown|no information", re.I)
ASKS = {"allow": "allowed", "req": "required"}


def question_id(key: tuple) -> str:
    return "Q-" + hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:8]


def _title(s: str) -> str:
    return s.title() if s.islower() else s


def _distinct(claims: list) -> list:
    """One representative claim per distinct value."""
    out = []
    for c in claims:
        if not any(c.value.same_as(o.value) for o in out):
            out.append(c)
    return out


def build_testset(docs: list, dup, ver, max_fact_questions=None) -> dict:
    live, superseded, copies = split_documents(docs, dup, ver)
    live_by_key, old_by_key = defaultdict(list), defaultdict(list)
    for d in live:
        for c in extract_claims(d):
            live_by_key[c.key].append(c)
    for d in docs:
        if d.path in superseded:
            for c in extract_claims(d):
                old_by_key[c.key].append(c)
    by_subject = defaultdict(list)  # (topic, subject) -> keys
    for key in live_by_key:
        by_subject[key[:2]].append(key)

    questions = []
    for key, claims in live_by_key.items():
        topic, subject, scope, region = key
        values = _distinct(claims)
        v0 = values[0].value
        sources = sorted({c.path for c in claims})
        q = {"id": question_id(key), "key": dict(zip(("topic", "subject", "scope", "region"), key)),
             "sources": sources, "kind": v0.kind, "distractors": []}
        if v0.kind == "bool":
            asks = ASKS[v0.unit]
            q["question"] = f"Is {subject} {asks} for {scope} in {_title(region)}?"
            q["asks"] = asks
        else:
            q["question"] = f"What is the {subject} for {scope} in {_title(region)}?"
        if len(values) > 1:
            q["type"] = "conflict"
            q["expected"] = "FLAG_CONFLICT"
            q["values"] = {c.path: c.value.raw for c in claims}
            q["sources"] = sorted(q["values"])
            questions.append(q)
            continue
        q["expected"] = values[0].value.raw
        seen = [v0]

        def add(claim, why):
            if claim.value.same_as(v0) is not False or any(claim.value.same_as(s) for s in seen):
                return
            seen.append(claim.value)
            q["distractors"].append({"value": claim.value.raw, "why": why, "source": claim.path})

        for c in old_by_key.get(key, []):
            add(c, "stale")
        siblings = [o for o in by_subject[(topic, subject)] if o != key]
        for want, why in (("scope", "other_scope"), ("region", "other_region")):  # scope outranks region
            for other in siblings:
                if (want == "scope" and other[3] == region) or (want == "region" and other[3] != region and other[2] == scope):
                    for c in _distinct(live_by_key[other]):
                        add(c, why)
        q["type"] = "stale_trap" if any(x["why"] == "stale" for x in q["distractors"]) else "fact"
        questions.append(q)

    questions.sort(key=lambda q: q["id"])
    if max_fact_questions is not None:
        keep = [q for q in questions if q["type"] == "fact"][:max_fact_questions]
        ids = {q["id"] for q in keep}
        questions = [q for q in questions if q["type"] != "fact" or q["id"] in ids]
    return {"version": 1, "questions": questions, "counts": dict(Counter(q["type"] for q in questions)),
            "built_from": {"documents": len(docs), "live_documents": len(live), "superseded": len(superseded)}}


# ----------------------------------------------------------------------------- grading
def _number_candidates(text: str) -> list:
    out = []
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)
    toks = re.findall(r"\d+(?:\.\d+)?|[A-Za-z/-]+", cleaned)
    for i, t in enumerate(toks):
        if t[0].isdigit():
            for n in range(0, 4):
                v = parse_value(" ".join(toks[i:i + 1 + n]))
                if v:
                    out.append(v)
    return out


def _polarity(text: str, asks: str):
    t = text.casefold()
    if re.search(r"\bnot (?:allowed|required|permitted)\b|\bno\b|n't|\bnot\b", t):
        return False
    if re.search(r"\b(?:allowed|required|permitted|yes)\b", t):
        return True
    return None


def grade_answer(q: dict, answer) -> str:
    text = (answer or "").strip()
    if not text or DONT_KNOW.search(text):
        return "no_answer"
    if q["type"] == "conflict":
        if CONFLICT_WORDS.search(text):
            return "conflict_flagged"
        vals = [parse_value(v) for v in q["values"].values()]
        if q["kind"] == "bool":
            return "conflict_missed" if _polarity(text, q.get("asks", "")) is not None else "wrong"
        got = _number_candidates(text)
        return "conflict_missed" if any(g.same_as(v) for g in got for v in vals if v) else "wrong"
    expected = parse_value(q["expected"])
    if q["kind"] == "bool":
        pol = _polarity(text, q.get("asks", ""))
        if pol is None:
            return "wrong"
        if pol == bool(expected.amount):
            return "correct"
        for d in q["distractors"]:
            if parse_value(d["value"]).amount == float(pol):
                return {"stale": "stale_answer", "other_scope": "other_scope_answer",
                        "other_region": "other_region_answer"}[d["why"]]
        return "wrong"
    got = _number_candidates(text)
    has_expected = any(g.same_as(expected) for g in got)
    hit = [d for d in q["distractors"] if any(g.same_as(parse_value(d["value"])) for g in got)]
    if has_expected and not hit:
        return "correct"
    if has_expected and hit:
        return "hedged"
    if hit:
        return {"stale": "stale_answer", "other_scope": "other_scope_answer",
                "other_region": "other_region_answer"}[hit[0]["why"]]
    return "wrong"


def grade(testset: dict, answers: dict) -> dict:
    def text_of(a):  # an answer is plain text, or {"answer": text, "retrieved": [...]}
        return a.get("answer") if isinstance(a, dict) else a
    rows = [{"id": q["id"], "type": q["type"], "outcome": grade_answer(q, text_of(answers.get(q["id"])))}
            for q in testset["questions"]]
    ok = {"correct", "conflict_flagged"}
    by_type = defaultdict(Counter)
    for r in rows:
        by_type[r["type"]][r["outcome"]] += 1
    return {"total": len(rows), "accuracy": round(sum(r["outcome"] in ok for r in rows) / max(1, len(rows)), 4),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "by_type": {t: dict(c) for t, c in by_type.items()}, "rows": rows}


# ----------------------------------------------------------------------------- drift
def _same(a, b) -> bool:
    va, vb = parse_value(a) if a != "FLAG_CONFLICT" else None, parse_value(b) if b != "FLAG_CONFLICT" else None
    if va and vb and va.same_as(vb) is not None:
        return bool(va.same_as(vb))
    return a == b


def diff_testsets(old: dict, new: dict) -> dict:
    o, n = {q["id"]: q for q in old["questions"]}, {q["id"]: q for q in new["questions"]}
    changed = []
    for i in sorted(o.keys() & n.keys()):
        a, b = o[i], n[i]
        if not _same(a["expected"], b["expected"]) or a["type"] != b["type"]:
            changed.append({"id": i, "question": b["question"], "old_expected": a["expected"],
                            "new_expected": b["expected"], "old_type": a["type"], "new_type": b["type"],
                            "why": "answer changed" if not _same(a["expected"], b["expected"]) else "type changed"})
    added, removed = sorted(n.keys() - o.keys()), sorted(o.keys() - n.keys())
    return {"added": [{"id": i, "question": n[i]["question"], "type": n[i]["type"]} for i in added],
            "removed": [{"id": i, "question": o[i]["question"], "type": o[i]["type"]} for i in removed],
            "changed": changed, "unchanged": len(o.keys() & n.keys()) - len(changed),
            "needs_review": len(added) + len(changed)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("corpus")
    b.add_argument("--out", required=True)
    b.add_argument("--max-fact-questions", type=int, default=None, help="cap plain fact questions (default: all)")
    g = sub.add_parser("grade")
    g.add_argument("testset")
    g.add_argument("answers")
    d = sub.add_parser("diff")
    d.add_argument("old")
    d.add_argument("new")
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    if args.cmd == "build":
        docs = read_corpus(args.corpus)
        dup = find_duplicates(docs)
        ts = build_testset(docs, dup, find_versions(docs, dup), args.max_fact_questions)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(ts, indent=1), encoding="utf-8")
        print(f"Test set: {len(ts['questions'])} questions  {ts['counts']}")
        extra = Counter(x["why"] for q in ts["questions"] for x in q["distractors"])
        print("Wrong answers to watch for:", dict(extra))
        print("Wrote", args.out)
    elif args.cmd == "grade":
        r = grade(load(args.testset), load(args.answers))
        print(f"Accuracy {r['accuracy']:.1%} over {r['total']} questions")
        print("Outcomes:", r["outcomes"])
    else:
        r = diff_testsets(load(args.old), load(args.new))
        print(f"Unchanged {r['unchanged']}  Changed {len(r['changed'])}  Added {len(r['added'])}  "
              f"Removed {len(r['removed'])}  -> {r['needs_review']} questions need review")
        for c in r["changed"][:10]:
            print(f"  {c['id']} {c['question']}  {c['old_expected']} -> {c['new_expected']}  ({c['why']})")


if __name__ == "__main__":
    main()

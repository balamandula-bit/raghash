"""A fake RAG system that fails on purpose, so the diagnoser can be checked against known causes.

For each question it either answers correctly, or injects one failure whose cause and blame are known
by construction (e.g. "retrieved only the old file and quoted it" = stale_source_only / corpus).
`inject` returns (answers, labels); answers have the same shape as a real system's.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..contradictions import parse_value
from ..diagnose import Context, relation, supporters

FLIP = {"allowed": "not allowed", "not allowed": "allowed", "required": "not required", "not required": "required"}


def _wrong_value(q: dict) -> str:
    expected = parse_value(q["expected"])
    if q["kind"] == "bool":
        return FLIP[q["expected"]]
    amount, _, unit = q["expected"].partition(" ")
    taken = [expected] + [parse_value(d["value"]) for d in q["distractors"]]
    n = int(float(amount.replace(",", ""))) + 7
    while any(parse_value(f"{n} {unit}".strip()).same_as(t) for t in taken):
        n += 1
    return f"{n} {unit}".strip()


def inject(testset: dict, ctx: Context, seed: int = 0, p_fail: float = 0.7):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for p, g in ctx.group_of.items():
        groups[g].append(p)
    copy_groups = [sorted(v) for _, v in sorted(groups.items()) if len(v) >= 2]
    all_paths = sorted(ctx.claims)
    answers, labels = {}, {}

    for q in testset["questions"]:
        rel = lambda p: relation(ctx, p, q)
        quiet = [p for p in all_paths if rel(p) == "irrelevant"]
        current = q["sources"][0]
        by_why = defaultdict(list)
        for d in q["distractors"]:
            by_why[d["why"]].append(d)
        options = []  # (cause, blame, retrieved, answer)
        if q["type"] == "conflict":
            side = rng.choice(sorted(q["values"]))
            options.append(("conflict_hidden", "corpus", [side], q["values"][side]))
            first = sorted(q["values"])[0]
            options.append(("conflict_ignored", "corpus", sorted(q["values"]), q["values"][first]))
        else:
            for d in by_why["stale"]:
                options.append(("stale_source_only", "corpus", [d["source"]], d["value"]))
                options.append(("stale_beat_current", "corpus", [d["source"], current], d["value"]))
            for why, cause in (("other_scope", "scope_confusion"), ("other_region", "region_confusion")):
                for d in by_why[why][:1]:
                    options.append((cause, "retriever", [d["source"]], d["value"]))
                    options.append((cause, "generator", [d["source"], current], d["value"]))
            wrong = _wrong_value(q)
            # A flipped yes/no is indistinguishable from quoting another scope's value that sits in the same
            # policy, so only inject a generation error that no document in front of the model could explain.
            if not supporters(ctx, q, wrong, [(current, "x")]):
                options.append(("generation_error", "generator", [current], wrong))
            seen_groups, picks = set(), []
            for p in rng.sample(quiet, min(len(quiet), 12)):
                if ctx.group_of.get(p, p) not in seen_groups and len(picks) < 3:
                    seen_groups.add(ctx.group_of.get(p, p))
                    picks.append(p)
            if len(picks) == 3:
                options.append(("retrieval_miss", "retriever", picks, "I don't know"))
            crowd = [g for g in copy_groups if all(rel(p) == "irrelevant" for p in g)]
            if len(crowd) >= 2:
                a, b = rng.sample(crowd, 2)
                options.append(("duplicate_crowding", "corpus", a[:2] + b[:2], "I don't know"))
        if options and rng.random() < p_fail:
            cause, blame, retrieved, text = rng.choice(options)
            if cause in ("scope_confusion", "region_confusion"):  # blame follows what was in front of the model
                blame = "generator" if any(rel(p) == "current" for p in retrieved) else "retriever"
            labels[q["id"]] = {"cause": cause, "blame": blame}
            answers[q["id"]] = {"answer": text, "retrieved": retrieved}
        else:
            text = "These documents conflict." if q["type"] == "conflict" else q["expected"]
            answers[q["id"]] = {"answer": text, "retrieved": list(q["sources"])}
    return answers, labels

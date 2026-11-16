"""Builds clean base documents and FAQ documents from the fact catalogue."""
from __future__ import annotations

import random
import re
from datetime import date, timedelta

from . import catalog as C
from .model import Block, Doc, Fact, Section

DATE_LO = date(2021, 1, 1)
DATE_HI = date(2025, 6, 30)
CAP = date(2026, 9, 30)  # nothing in the corpus is dated after this
SIGNALS = ("in_text", "filename", "mtime_only")
SIGNAL_WEIGHTS = (70, 15, 15)

TEMPLATE_BY = {(t.topic, t.subject): t for t in C.TEMPLATES}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def rand_date(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randrange((hi - lo).days + 1))


def unique_path(taken: set, path: str) -> str:
    """Return a path unique under case-insensitive comparison (Windows/macOS safe)."""
    if path.lower() not in taken:
        taken.add(path.lower())
        return path
    stem, _, ext = path.rpartition(".")
    n = 2
    while f"{stem}_{n}.{ext}".lower() in taken:
        n += 1
    out = f"{stem}_{n}.{ext}"
    taken.add(out.lower())
    return out


def draw_value(rng: random.Random, t: C.FactTemplate):
    if t.kind == "int":
        return rng.randrange(t.lo, t.hi + 1, t.step)
    return rng.random() < 0.5


def other_value(rng: random.Random, t: C.FactTemplate, current):
    if t.kind != "int":
        return not current
    return rng.choice([v for v in range(t.lo, t.hi + 1, t.step) if v != current])


def facts_for(rng: random.Random, topic: str, region: str) -> list:
    facts = []
    for t in (x for x in C.TEMPLATES if x.topic == topic):
        if len(t.scopes) == 1:
            chosen = list(t.scopes)
        else:
            k = rng.randint(1, min(t.max_scopes, len(t.scopes)))
            chosen = [t.scopes[i] for i in sorted(rng.sample(range(len(t.scopes)), k))]
        values = [draw_value(rng, t) for _ in chosen]
        if t.kind == "int" and t.scopes in C.ORDERED_SCOPE_SETS:
            values.sort()
        for scope, value in zip(chosen, values):
            facts.append(Fact(topic, t.subject, scope, region, t.kind, value, t.unit))
    return facts


def make_ctx(dept: str, topic: str, region: str) -> dict:
    return {
        "dept_name": C.DEPARTMENTS[dept],
        "topic_lc": topic.lower(),
        "region": region,
        "email": f"{dept}-help@meridian-works.test",
    }


def pick_text(rng: random.Random, pool: str, ctx: dict, n: int) -> list:
    return [Block("text", s.format(**ctx), pool=pool) for s in rng.sample(C.POOLS[pool], n)]


def make_path(rng: random.Random, dept: str, topic: str, region: str, effective: date, signal: str) -> str:
    stem = f"{slugify(topic)}_{slugify(region)}"
    if signal == "filename":
        stem += f"_{effective.isoformat()}"
    if rng.random() < 0.2:
        stem += rng.choice(["_final", "_FINAL", "_final_final"])
    ext = ".md" if rng.random() < 0.85 else ".txt"
    return f"{dept}/{stem}{ext}"


def build_base_doc(rng: random.Random, ti: int, ri: int, dept: str, topic: str, region: str) -> Doc:
    ctx = make_ctx(dept, topic, region)
    facts = facts_for(rng, topic, region)
    style = rng.choice(["sentences", "bullets", "table"])
    effective = rand_date(rng, DATE_LO, DATE_HI)
    signal = rng.choices(SIGNALS, weights=SIGNAL_WEIGHTS)[0]
    version = rng.choice(["1.0", "1.0", "1.1", "1.2", "2.0"])
    details = pick_text(rng, "details", ctx, 1) + [
        Block("fact", fact=f, frame=rng.randrange(3)) for f in facts
    ]
    sections = [
        Section("Purpose", pick_text(rng, "purpose", ctx, 2)),
        Section("Scope", pick_text(rng, "scope", ctx, 2)),
        Section("Policy details", details),
        Section("Exceptions and escalation", pick_text(rng, "exceptions", ctx, 2)),
        Section("Contacts", pick_text(rng, "contacts", ctx, 2)),
    ]
    return Doc(
        path=make_path(rng, dept, topic, region, effective, signal),
        dept=dept,
        topic=topic,
        region=region,
        title=f"{topic} — {region}",
        policy_code=f"POL-{dept.upper()[:3]}-{ti + 1:02d}{ri + 1}",
        owner=C.DEPARTMENTS[dept],
        version=version,
        effective=effective,
        date_signal=signal,
        style=style,
        sections=sections,
        ctx=ctx,
        mtime=effective + timedelta(days=rng.randint(0, 3)),
    )


def build_base_docs(rng: random.Random) -> list:
    docs = []
    for ti, (dept, topic) in enumerate(C.TOPICS):
        for ri, region in enumerate(C.REGIONS):
            docs.append(build_base_doc(rng, ti, ri, dept, topic, region))
    return docs


def build_faq(rng: random.Random, base: Doc, restated: list, taken: set) -> Doc:
    """An intranet FAQ page that restates some of `base`'s facts (possibly altered)."""
    ctx = base.ctx
    filler = [
        Block("text", f"**Q: {q}**\nA: {a.format(**ctx)}")
        for q, a in rng.sample(C.FAQ_FILLER, 3)
    ]
    items = [Block("fact", fact=f) for f in restated] + filler
    rng.shuffle(items)
    sections = [
        Section("Overview", [Block(
            "text", f"This page collects answers to common questions about the {ctx['topic_lc']} in {ctx['region']}.")]),
        Section("Frequently asked questions", items),
        Section("Contacts", [Block("text", f"Questions about this page can be sent to {ctx['email']}.")]),
    ]
    effective = min(base.effective + timedelta(days=rng.randint(10, 300)), CAP)
    signal = rng.choices(["in_text", "mtime_only"], weights=[80, 20])[0]
    path = unique_path(taken, f"wiki/{slugify(base.topic)}_faq_{slugify(base.region)}.md")
    return Doc(
        path=path,
        dept="wiki",
        topic=base.topic,
        region=base.region,
        title=f"{base.topic} FAQ — {base.region}",
        policy_code="",
        owner=C.FAQ_OWNER,
        version="1.0",
        effective=effective,
        date_signal=signal,
        style="faq",
        sections=sections,
        ctx=ctx,
        mtime=effective + timedelta(days=rng.randint(0, 3)),
        is_faq=True,
    )

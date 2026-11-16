"""Fault injectors. Each takes (rng, base_doc, taken_paths) and returns (new_doc, label)."""
from __future__ import annotations

import copy
import random
import re
from dataclasses import replace
from datetime import date, timedelta

from . import catalog as C
from .builder import (CAP, SIGNAL_WEIGHTS, SIGNALS, TEMPLATE_BY, build_faq, other_value, slugify,
                      unique_path)
from .model import Block, Doc, Fact, render, shingle_jaccard


def _split(path: str):
    folder, _, name = path.rpartition("/")
    stem, _, ext = name.rpartition(".")
    return folder, name, stem, ext


# ----------------------------------------------------------------------------- duplicates
def make_exact_duplicate(rng: random.Random, doc: Doc, taken: set):
    _, name, stem, ext = _split(doc.path)
    folder = rng.choice(["archive", "shared", *[d for d in C.DEPARTMENTS if d != doc.dept]])
    new_name = rng.choice([f"Copy of {name}", f"{stem} (1).{ext}", f"{stem}_backup.{ext}"])
    path = unique_path(taken, f"{folder}/{new_name}")
    new = replace(doc, path=path, role="exact_duplicate_copy",
                  mtime=min(doc.mtime + timedelta(days=rng.randint(1, 400)), CAP))
    label = {"type": "exact_duplicate", "docs": [doc.path, path], "canonical": doc.path}
    return new, label


def _apply_typos(rng: random.Random, doc: Doc) -> bool:
    blocks = [b for s in doc.sections for b in s.blocks if b.kind == "text"]
    rng.shuffle(blocks)
    done = 0
    for b in blocks:
        words = list(re.finditer(r"\b[A-Za-z]{6,}\b", b.text))
        if not words:
            continue
        m = rng.choice(words)
        w = m.group()
        i = rng.randint(1, len(w) - 3)
        if w[i] == w[i + 1]:
            continue
        b.text = b.text[: m.start()] + w[:i] + w[i + 1] + w[i] + w[i + 2:] + b.text[m.end():]
        done += 1
        if done == 2:
            break
    return done > 0


def _apply_reword(rng: random.Random, doc: Doc) -> bool:
    blocks = [b for s in doc.sections for b in s.blocks if b.kind == "text" and b.pool in C.POOLS]
    used = {b.text for s in doc.sections for b in s.blocks if b.kind == "text"}
    rng.shuffle(blocks)
    for b in blocks:
        options = [s.format(**doc.ctx) for s in C.POOLS[b.pool]]
        options = [o for o in options if o not in used]
        if options:
            b.text = rng.choice(options)
            return True
    return False


def _whitespace_noise(rng: random.Random, text: str) -> str:
    """Trailing spaces, doubled spaces, extra blank lines. Never touches lines with digits (facts)."""
    out = []
    for line in text.split("\n"):
        if line.strip() and not re.search(r"\d", line):
            if rng.random() < 0.3:
                line += "  "
            if rng.random() < 0.15:
                line = re.sub(r"(?<=\w) (?=\w)", "  ", line, count=1)
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n## ", lambda m: "\n\n## " if rng.random() < 0.3 else m.group(0), text)


def make_near_duplicate(rng: random.Random, doc: Doc, taken: set):
    original = render(doc)
    new = copy.deepcopy(doc)
    new.role = "near_duplicate_copy"
    edits = []
    for kind in rng.choice([["typos"], ["reworded"], ["typos", "reworded"]]):
        if kind == "typos" and _apply_typos(rng, new):
            edits.append("typos")
        elif kind == "reworded" and _apply_reword(rng, new):
            edits.append("reworded")
    if not edits and _apply_typos(rng, new):
        edits.append("typos")
    text = render(new)
    if rng.random() < 0.5:
        text = _whitespace_noise(rng, text)
        edits.append("whitespace")
    assert text != original, "near-duplicate must differ from its source"
    new.text_override = text
    folder, _, stem, ext = _split(doc.path)
    folder = rng.choice([doc.dept, doc.dept, "shared"])
    path = unique_path(taken, f"{folder}/{stem}{rng.choice(['_final', '_FINAL', '_reviewed', '_copy'])}.{ext}")
    new.path = path
    new.mtime = min(doc.mtime + timedelta(days=rng.randint(1, 200)), CAP)
    label = {"type": "near_duplicate", "docs": [doc.path, path], "canonical": doc.path, "edits": edits,
             "word_3gram_jaccard": round(shingle_jaccard(original, text), 4)}
    return new, label


# ----------------------------------------------------------------------------- versions
def _bump(version: str) -> str:
    return f"{int(version.split('.')[0]) + 1}.0"


def make_version(rng: random.Random, doc: Doc, taken: set):
    """A newer revision of `doc`: same title, 1-2 values changed, later effective date."""
    new = copy.deepcopy(doc)
    new.role = "version_new"
    fact_blocks = [b for s in new.sections for b in s.blocks if b.kind == "fact"]
    changed = []
    for b in rng.sample(fact_blocks, rng.choice([1, 2])):
        old = b.fact
        b.fact = old.with_value(other_value(rng, TEMPLATE_BY[(old.topic, old.subject)], old.value))
        changed.append({**old.key_dict(), "old_value": old.value_text, "new_value": b.fact.value_text})
    new.version = _bump(doc.version)
    new.effective = min(doc.effective + timedelta(days=rng.randint(200, 700)), date(2026, 6, 30))
    shared = rng.random() < 0.6
    if not shared:
        prefix, num = doc.policy_code.rsplit("-", 1)
        new.policy_code = f"{prefix}-{int(num) + 400}"
    new.date_signal = rng.choices(SIGNALS, weights=SIGNAL_WEIGHTS)[0]
    new.mtime = new.effective + timedelta(days=rng.randint(0, 3))
    if rng.random() < 0.5:
        new.sections[0].blocks.insert(0, Block("text", rng.choice(C.REVISION_NOTES)))
    stem = f"{slugify(doc.topic)}_{slugify(doc.region)}"
    if new.date_signal == "filename":
        stem += f"_{new.effective.isoformat()}"
    stem += rng.choice(["_v2", "_updated", "_new", ""])
    ext = doc.path.rpartition(".")[2]
    new.path = unique_path(taken, f"{doc.dept}/{stem}.{ext}")
    label = {
        "type": "version_pair",
        "docs": [doc.path, new.path],  # [old, new]
        "changed_facts": changed,
        "policy_code_shared": shared,
        "date_signals": {"old": doc.date_signal, "new": new.date_signal},
        "effective": {"old": doc.effective.isoformat(), "new": new.effective.isoformat()},
    }
    return new, label


# ----------------------------------------------------------------------------- FAQ-based faults
def make_contradiction(rng: random.Random, base: Doc, taken: set):
    """An FAQ that states a different value for one of the base document's facts."""
    facts = base.facts
    f = rng.choice(facts)
    conflict = f.with_value(other_value(rng, TEMPLATE_BY[(f.topic, f.subject)], f.value))
    others = rng.sample([x for x in facts if x.key != f.key], 2)
    faq = build_faq(rng, base, [conflict] + others, taken)
    faq.role = "faq_contradiction"
    label = {
        "type": "contradiction",
        "docs": [base.path, faq.path],
        "fact": f.key_dict(),
        "values": {base.path: f.value_text, faq.path: conflict.value_text},
        "severity": "numeric" if f.kind == "int" else "categorical",
        "also_restates_consistently": [x.key_dict() for x in others],
    }
    return faq, label


def make_scope_trap(rng: random.Random, base: Doc, taken: set):
    """An FAQ that gives a different value for the same subject but a DIFFERENT scope: not a conflict."""
    facts = base.facts
    used: dict = {}
    for f in facts:
        used.setdefault(f.subject, set()).add(f.scope)
    cands = [t for t in C.TEMPLATES
             if t.topic == base.topic and t.kind == "int" and len(t.scopes) >= 3
             and len(used[t.subject]) < len(t.scopes)]
    t = rng.choice(cands)
    scope2 = rng.choice([s for s in t.scopes if s not in used[t.subject]])
    base_values = {f.value for f in facts if f.subject == t.subject}
    value = rng.choice([v for v in range(t.lo, t.hi + 1, t.step) if v not in base_values])
    trap = Fact(base.topic, t.subject, scope2, base.region, "int", value, t.unit)
    ref = next(f for f in facts if f.subject == t.subject)
    others = rng.sample([x for x in facts if x.subject != t.subject], 2)
    faq = build_faq(rng, base, [trap] + others, taken)
    faq.role = "faq_scope_trap"
    label = {
        "type": "scope_trap",
        "docs": [base.path, faq.path],
        "subject": t.subject,
        "base_fact": ref.to_dict(),
        "faq_fact": trap.to_dict(),
        "expected": "not_a_conflict",
    }
    return faq, label


def make_consistent(rng: random.Random, base: Doc, taken: set):
    """An FAQ that restates three base facts with identical values (reworded): no conflict."""
    restated = rng.sample(base.facts, 3)
    faq = build_faq(rng, base, restated, taken)
    faq.role = "faq_consistent"
    label = {
        "type": "consistent_restatement",
        "docs": [base.path, faq.path],
        "facts": [f.key_dict() for f in restated],
        "expected": "not_a_conflict",
    }
    return faq, label

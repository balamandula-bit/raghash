"""Document model and renderer for the synthetic corpus."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Optional

INT_FRAMES = (
    "The {subject} for {scope} is {value}.",
    "For {scope}, the {subject} is set at {value}.",
    "The policy sets the {subject} for {scope} at {value}.",
)
BOOL_FRAMES = (
    "{Subject} is {value} for {scope}.",
    "For {scope}, {subject} is {value}.",
    "The policy states that {subject} is {value} for {scope}.",
)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def fmt_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


@dataclass(frozen=True)
class Fact:
    topic: str
    subject: str
    scope: str
    region: str
    kind: str  # int | req | allow
    value: object  # int, or bool for req/allow
    unit: str = ""

    @property
    def key(self) -> tuple:
        return (self.topic, self.subject, self.scope, self.region)

    @property
    def value_text(self) -> str:
        if self.kind == "int":
            return f"{self.value} {self.unit}".strip()
        yes, no = ("required", "not required") if self.kind == "req" else ("allowed", "not allowed")
        return yes if self.value else no

    def with_value(self, value) -> "Fact":
        return replace(self, value=value)

    def key_dict(self) -> dict:
        return {"topic": self.topic, "subject": self.subject, "scope": self.scope, "region": self.region}

    def to_dict(self) -> dict:
        return {**self.key_dict(), "kind": self.kind, "value": self.value, "value_text": self.value_text}


@dataclass
class Block:
    kind: str  # "text" | "fact"
    text: str = ""
    fact: Optional[Fact] = None
    frame: int = 0
    pool: str = ""  # boilerplate pool a text block came from ("" = not rewordable)


@dataclass
class Section:
    heading: str
    blocks: list = field(default_factory=list)


@dataclass
class Doc:
    path: str
    dept: str
    topic: str
    region: str
    title: str
    policy_code: str
    owner: str
    version: str
    effective: date
    date_signal: str  # in_text | filename | mtime_only
    style: str  # sentences | bullets | table | faq
    sections: list
    ctx: dict
    role: str = "base"
    mtime: Optional[date] = None
    text_override: Optional[str] = None
    is_faq: bool = False

    @property
    def facts(self) -> list:
        return [b.fact for s in self.sections for b in s.blocks if b.kind == "fact"]


# ----------------------------------------------------------------------------- rendering
def _fact_line(fact: Fact, style: str, frame: int) -> str:
    subj, scope, val = fact.subject, fact.scope, fact.value_text
    if style == "bullets":
        return f"- {cap(scope)}: {subj} — {val}"
    frames = INT_FRAMES if fact.kind == "int" else BOOL_FRAMES
    return frames[frame % len(frames)].format(
        subject=subj, Subject=cap(subj), scope=scope, Scope=cap(scope), value=val
    )


def _faq_lines(fact: Fact) -> list:
    if fact.kind == "int":
        q = f"**Q: What is the {fact.subject} for {fact.scope}?**"
    else:
        q = f"**Q: What is the position on {fact.subject} for {fact.scope}?**"
    return [q, f"A: It is {fact.value_text}.", ""]


def _render_fact_group(blocks: list, style: str) -> list:
    facts = [b.fact for b in blocks]
    if style == "faq":
        out = []
        for f in facts:
            out += _faq_lines(f)
        return out
    if style == "table":
        out = ["| Item | Applies to | Value |", "|---|---|---|"]
        out += [f"| {cap(f.subject)} | {cap(f.scope)} | {f.value_text} |" for f in facts]
        return out + [""]
    if style == "bullets":
        return [_fact_line(b.fact, style, b.frame) for b in blocks] + [""]
    sentences = [_fact_line(b.fact, style, b.frame) for b in blocks]
    out = []
    for i in range(0, len(sentences), 3):
        out += [" ".join(sentences[i : i + 3]), ""]
    return out


def _render_blocks(blocks: list, style: str) -> list:
    out, i = [], 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "text":
            if style == "faq":
                out += [b.text, ""]
                i += 1
            else:
                parts, j = [], i
                while j < len(blocks) and blocks[j].kind == "text":
                    parts.append(blocks[j].text)
                    j += 1
                out += [" ".join(parts), ""]
                i = j
        else:
            group, j = [], i
            while j < len(blocks) and blocks[j].kind == "fact":
                group.append(blocks[j])
                j += 1
            out += _render_fact_group(group, style)
            i = j
    return out


def render(doc: Doc) -> str:
    if doc.text_override is not None:
        return doc.text_override
    lines = [f"# {doc.title}", ""]
    if doc.policy_code:
        lines.append(f"Policy code: {doc.policy_code}")
    lines.append(f"Owner: {doc.owner}")
    lines.append(f"Applies to: {doc.region}")
    if doc.date_signal == "in_text":
        if doc.is_faq:
            lines.append(f"Last updated: {fmt_date(doc.effective)}")
        else:
            lines.append(f"Version: {doc.version}")
            lines.append(f"Effective: {fmt_date(doc.effective)}")
    lines.append("")
    for i, sec in enumerate(doc.sections, 1):
        lines += [f"## {i}. {sec.heading}", ""]
        lines += _render_blocks(sec.blocks, doc.style)
    return "\n".join(lines).rstrip() + "\n"


def shingle_jaccard(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity of word n-gram shingles (punctuation and case ignored)."""

    def shingles(text: str) -> set:
        words = re.findall(r"\w+", text.lower())
        return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}

    sa, sb = shingles(a), shingles(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0

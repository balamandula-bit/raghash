"""Ground-truth loading: which document pairs are related, and how."""
from __future__ import annotations

import json
from pathlib import Path

RELATIONS = (
    "exact_duplicate",
    "near_duplicate",
    "version_pair",
    "contradiction",
    "scope_trap",  # different scope, different value: NOT a conflict
    "consistent_restatement",  # same facts, reworded: NOT a conflict
)


class Truth:
    def __init__(self, labels: dict):
        self.labels = labels
        self.pairs = {r: set() for r in RELATIONS}  # unordered pairs as frozensets
        self.versions = set()  # ordered pairs (old, new)
        for fault in labels["faults"]:
            a, b = fault["docs"]
            self.pairs[fault["type"]].add(frozenset((a, b)))
            if fault["type"] == "version_pair":
                self.versions.add((a, b))

    def relation(self, a: str, b: str) -> str:
        """The injected relation between two documents, or 'unrelated'."""
        pair = frozenset((a, b))
        for rel in RELATIONS:
            if pair in self.pairs[rel]:
                return rel
        return "unrelated"


def load_truth(gt_dir) -> Truth:
    return Truth(json.loads((Path(gt_dir) / "labels.json").read_text(encoding="utf-8")))


def load_manifest(gt_dir) -> list:
    text = (Path(gt_dir) / "manifest.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]

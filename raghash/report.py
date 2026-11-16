"""Part 7, the Report: everything RagHash found, as one self-contained HTML file (no internet needed).

    python -m raghash.report data/synth_v1/corpus --out report.html
    python -m raghash.report data/synth_v1/corpus --answers answers.json --previous old_testset.json --out report.html

Always: corpus health (copies, superseded versions, contradictions), a clean-up list, and the test set.
With --answers (your RAG system's output): the diagnosis, and the clean-up list is ranked by questions repaired.
With --previous (an older test set): what changed since, question by question.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .contradictions import find_contradictions, split_documents
from .diagnose import build_context, diagnose
from .duplicates import find_duplicates
from .reader import read_corpus
from .testset import build_testset, diff_testsets
from .versions import find_versions

CAUSE_WORDS = {
    "stale_source_only": "Quoted an old version (only the old one was found)",
    "stale_beat_current": "Quoted an old version although the current one was also found",
    "conflict_hidden": "Documents disagree; only one side was found",
    "conflict_ignored": "Documents disagree; both found, one picked silently",
    "duplicate_crowding": "Copies filled the search results",
    "scope_confusion": "Answered for the wrong scope (who or what it applies to)",
    "region_confusion": "Answered for the wrong region",
    "generation_error": "Right document found, answer still wrong",
    "retrieval_miss": "The needed document was never found",
}
BLAME_WORDS = {"corpus": "Fix the documents", "retriever": "Fix the search", "generator": "Fix the answering model"}
ACTION_ORDER = {"Resolve conflict": 0, "Retire old version": 1, "Remove copy": 2}


def build_report_data(docs: list, answers: dict = None, previous: dict = None, title: str = "Corpus report",
                      generated: str = None) -> dict:
    dup = find_duplicates(docs)
    ver = find_versions(docs, dup)
    con = find_contradictions(docs, dup, ver)
    live, superseded, copies = split_documents(docs, dup, ver)
    testset = build_testset(docs, dup, ver)
    diagnosis = diagnose(testset, answers, build_context(docs, dup, ver)) if answers else None
    affected = {f["path"]: f["questions_affected"] for f in diagnosis["files_to_fix"]} if diagnosis else {}

    kinds = defaultdict(set)
    for p in dup.pairs:
        kinds[copies.get(p.a, copies.get(p.b))].add(p.kind)
    groups = defaultdict(list)
    for copy, keep in copies.items():
        groups[keep].append(copy)
    dup_groups = [{"keep": k, "copies": sorted(v), "exact": kinds[k] <= {"exact", "same_text"}}
                  for k, v in sorted(groups.items())]

    chains = []
    for c in ver.chains:
        link = c.links[-1]
        chains.append({"members": c.members, "current": c.current, "superseded": c.superseded,
                       "confidence": min((l.confidence for l in c.links), key=["low", "medium", "high"].index),
                       "margin_days": link.margin_days, "source": link.evidence["new_date_source"]})

    actions = []
    for c in con.contradictions:
        f = c.facts[0]
        actions.append({"action": "Resolve conflict", "path": f"{c.a}  vs  {c.b}", "files": [c.a, c.b],
                        "detail": f"{f['fact']['subject']} ({f['fact']['scope']}, {f['fact']['region']}): "
                                  f"{f['values'][c.a]} vs {f['values'][c.b]}"})
    for c in chains:
        for old in c["superseded"]:
            note = "" if c["confidence"] == "high" else f" - check first, order is {c['confidence']} confidence"
            actions.append({"action": "Retire old version", "path": old, "files": [old],
                            "detail": f"replaced by {c['current'][0]}{note}"})
    for keep, cs in sorted(groups.items()):
        for c in sorted(cs):
            actions.append({"action": "Remove copy", "path": c, "files": [c], "detail": f"copy of {keep}"})
    for a in actions:
        a["questions_affected"] = sum(affected.get(f, 0) for f in a["files"])
    actions.sort(key=lambda a: (-a["questions_affected"], ACTION_ORDER[a["action"]], a["path"]))

    examples = []
    if diagnosis:
        seen = Counter()
        for d in diagnosis["diagnoses"]:
            if seen[d["cause"]] < 2:
                seen[d["cause"]] += 1
                examples.append(d)
    return {
        "title": title, "generated": generated or date.today().isoformat(),
        "counts": {"documents": len(docs), "current": len(live), "copies": len(copies),
                   "superseded": len(superseded), "contradictions": len(con.contradictions),
                   "scope_differences": len(con.scope_differences), "claims": con.stats["claims"]},
        "duplicates": dup_groups, "versions": chains, "contradictions": [c.to_dict() for c in con.contradictions],
        "testset": {"total": len(testset["questions"]), "counts": testset["counts"]},
        "drift": diff_testsets(previous, testset) if previous else None,
        "diagnosis": ({k: diagnosis[k] for k in ("total", "failures", "accuracy", "by_cause", "by_blame",
                                                  "corpus_share")} | {"examples": examples}) if diagnosis else None,
        "actions": actions,
    }


# ----------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#5d5d58;--line:#e2e1dc;--corpus:#b45309;--retriever:#1d4ed8;
--generator:#6d28d9;--ok:#15803d;--bad:#b91c1c;--chip:#efeee9}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141412;--card:#1d1d1a;--ink:#ecebe6;--mute:#a3a29a;
--line:#34332e;--corpus:#f59e0b;--retriever:#7aa2ff;--generator:#b794f6;--ok:#4ade80;--bad:#f87171;--chip:#2a2a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:36px 0 4px}h3{font-size:15px;margin:18px 0 6px}p{margin:6px 0}.sub{color:var(--mute)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:26px;line-height:1.1}.tile span{color:var(--mute);font-size:13px}
table{width:100%;min-width:600px;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);background:var(--chip)}
tr:last-child td{border-bottom:0}code,.path{font:12.5px ui-monospace,Consolas,monospace;word-break:break-all}
.wrap{overflow-x:auto}.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);font-size:12px;margin:1px 2px 1px 0}
.bar{height:10px;border-radius:5px;background:var(--chip);min-width:60px}.bar i{display:block;height:100%;border-radius:5px}
.corpus{color:var(--corpus)}.retriever{color:var(--retriever)}.generator{color:var(--generator)}
.i-corpus{background:var(--corpus)}.i-retriever{background:var(--retriever)}.i-generator{background:var(--generator)}
.ok{color:var(--ok)}.bad{color:var(--bad)}details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin:8px 0}
summary{cursor:pointer;font-weight:600}.vs{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.vs div{background:var(--chip);border-radius:8px;padding:8px;font-size:13px}@media(max-width:640px){.vs{grid-template-columns:1fr}}
.note{border-left:3px solid var(--line);padding:2px 12px;color:var(--mute);margin:12px 0}
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _table(head: list, rows: list) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><tr>{th}</tr>{body}</table></div>'


def _path(p) -> str:
    return f'<span class="path">{_e(p)}</span>'


def render_html(d: dict) -> str:
    c, out = d["counts"], []
    tiles = [(c["documents"], "documents scanned"), (c["copies"], "copies of other files"),
             (c["superseded"], "old versions still in the folder"), (c["contradictions"], "contradictions between current files"),
             (d["testset"]["total"], "test questions")]
    if d["diagnosis"]:
        tiles += [(f"{d['diagnosis']['accuracy']:.0%}", "of test questions answered correctly"),
                  (f"{d['diagnosis']['corpus_share']:.0%}", "of wrong answers caused by the documents")]
    out.append(f"<h1>{_e(d['title'])}</h1><p class='sub'>Generated {_e(d['generated'])} by RagHash</p>")
    out.append('<div class="tiles">' + "".join(f"<div class='tile'><b>{_e(v)}</b><span>{_e(k)}</span></div>" for v, k in tiles) + "</div>")

    # ---- what to fix first
    out.append("<h2>What to fix first</h2>")
    acts = d["actions"]
    if not acts:
        out.append("<p>Nothing to clean up. No copies, old versions or contradictions were found.</p>")
    else:
        ranked = " Ranked by how many test questions each file is behind." if d["diagnosis"] else \
            " Run with an answers file to rank these by questions repaired."
        out.append(f"<p class='sub'>{len(acts)} clean-up actions.{_e(ranked)}</p>")
        rows = [[_e(a["action"]), _path(a["path"]), _e(a["detail"]), _e(a["questions_affected"]) if d["diagnosis"] else "-"]
                for a in acts[:25]]
        out.append(_table(["Action", "File", "Why", "Questions repaired"], rows))
        if len(acts) > 25:
            out.append(f"<p class='sub'>Showing 25 of {len(acts)}. The full list is in the JSON next to this report.</p>")

    # ---- diagnosis
    dg = d["diagnosis"]
    if dg:
        out.append("<h2>Why answers went wrong</h2>")
        out.append(f"<p>{dg['failures']} of {dg['total']} answers were wrong. Each one is traced to a cause and to the part that should be fixed.</p>")
        mx = max(dg["by_blame"].values(), default=1)
        rows = [[f"<b class='{b}'>{_e(BLAME_WORDS[b])}</b>", _e(n),
                 f"<div class='bar'><i class='i-{b}' style='width:{100 * n / mx:.0f}%'></i></div>"]
                for b, n in sorted(dg["by_blame"].items(), key=lambda x: -x[1])]
        out.append(_table(["Where the fix belongs", "Wrong answers", ""], rows))
        mxc = max(dg["by_cause"].values(), default=1)
        rows = [[_e(CAUSE_WORDS[k]), _e(n), f"<div class='bar'><i class='i-corpus' style='width:{100 * n / mxc:.0f}%'></i></div>"]
                for k, n in sorted(dg["by_cause"].items(), key=lambda x: -x[1])]
        out.append("<h3>By cause</h3>" + _table(["Cause", "Wrong answers", ""], rows))
        out.append("<h3>Examples</h3>")
        for ex in dg["examples"]:
            chips = "".join(f"<span class='chip'>{_e(r['relation'])}</span> {_path(r['path'])}<br>" for r in ex["retrieved"]) or "nothing retrieved"
            out.append(f"<details><summary>{_e(ex['question'])}</summary><p><b>Cause:</b> {_e(CAUSE_WORDS[ex['cause']])} "
                       f"(<span class='{ex['blame']}'>{_e(BLAME_WORDS[ex['blame']])}</span>)</p>"
                       f"<p><b>Right answer:</b> {_e(ex['expected'])} &nbsp; <b>System said:</b> {_e(ex['answer'])}</p>"
                       f"<p><b>What it retrieved:</b><br>{chips}</p><p class='sub'>{_e(ex['fix'])}</p></details>")

    # ---- corpus health
    out.append("<h2>Corpus health</h2>")
    out.append(f"<h3>Contradictions ({c['contradictions']})</h3>")
    if d["contradictions"]:
        out.append(f"<p class='sub'>Two current files state different values for the same fact, scope and region. "
                   f"{c['scope_differences']} look similar but differ in scope or region, and are correctly not flagged.</p>")
        for x in d["contradictions"][:12]:
            f = x["facts"][0]
            out.append(f"<details><summary>{_e(f['fact']['subject'])} ({_e(f['fact']['scope'])}, {_e(f['fact']['region'])}): "
                       f"{_e(f['values'][x['a']])} vs {_e(f['values'][x['b']])}</summary><div class='vs'>"
                       f"<div>{_path(x['a'])}<br>{_e(f['lines'][x['a']])}</div><div>{_path(x['b'])}<br>{_e(f['lines'][x['b']])}</div></div></details>")
        if len(d["contradictions"]) > 12:
            out.append(f"<p class='sub'>Showing 12 of {len(d['contradictions'])}.</p>")
    else:
        out.append("<p>None found.</p>")

    out.append(f"<h3>Old versions still in the folder ({c['superseded']})</h3>")
    if d["versions"]:
        rows = [[_path(v["current"][0]), "".join(_path(s) + "<br>" for s in v["superseded"]),
                 f"<span class='chip'>{_e(v['confidence'])}</span>", _e(f"{v['margin_days']} days" if v["margin_days"] is not None else "unknown")]
                for v in d["versions"][:15]]
        out.append("<p class='sub'>High confidence means two written dates agree. Medium means a date came from the file's last-modified time, which can lie.</p>")
        out.append(_table(["Current file", "Superseded by it", "Confidence", "Newer by"], rows))
        if len(d["versions"]) > 15:
            out.append(f"<p class='sub'>Showing 15 of {len(d['versions'])} version chains.</p>")
    else:
        out.append("<p>None found.</p>")

    out.append(f"<h3>Copies ({c['copies']} files in {len(d['duplicates'])} groups)</h3>")
    if d["duplicates"]:
        pct = 100 * c["copies"] / max(1, c["documents"])
        out.append(f"<p class='sub'>{pct:.0f}% of files are copies, so they take search results away from other documents.</p>")
        rows = [[_path(g["keep"]), "".join(_path(x) + "<br>" for x in g["copies"]), "identical" if g["exact"] else "near-identical"]
                for g in d["duplicates"][:15]]
        out.append(_table(["Original", "Copies", "Kind"], rows))
        if len(d["duplicates"]) > 15:
            out.append(f"<p class='sub'>Showing 15 of {len(d['duplicates'])} groups.</p>")
    else:
        out.append("<p>None found.</p>")

    # ---- test set
    ts = d["testset"]
    out.append("<h2>Test set</h2>")
    words = {"fact": "plain facts", "stale_trap": "stale traps (an old version says something else)",
             "conflict": "conflicts (the right answer is to flag the disagreement)"}
    out.append(_table(["Kind", "Questions"], [[_e(words.get(k, k)), _e(v)] for k, v in sorted(ts["counts"].items())]))
    if d["drift"]:
        dr = d["drift"]
        out.append(f"<h3>Since the previous test set</h3><p>{dr['unchanged']} unchanged, {len(dr['changed'])} changed, "
                   f"{len(dr['added'])} new, {len(dr['removed'])} retired. <b>{dr['needs_review']}</b> questions need a new expected answer.</p>")
        if dr["changed"]:
            out.append(_table(["Question", "Was", "Now"], [[_e(x["question"]), _e(x["old_expected"]), _e(x["new_expected"])] for x in dr["changed"][:15]]))

    out.append("<div class='note'>How to read this: facts are read with patterns built for the practice corpus, so on real "
               "documents the extraction step needs a stronger reader. Version order relies on dates; where one came from a "
               "file's last-modified time the confidence is lower.</div>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(d['title'])}</title><style>{CSS}</style></head><body><main>" + "".join(out) + "</main></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--out", required=True, help="HTML file to write; a .json with the full data is written beside it")
    ap.add_argument("--answers", help="a system's answers JSON (adds the diagnosis)")
    ap.add_argument("--previous", help="an older testset.json (adds the drift section)")
    ap.add_argument("--title", default="Corpus report")
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    data = build_report_data(read_corpus(args.corpus), load(args.answers) if args.answers else None,
                             load(args.previous) if args.previous else None, args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Report: {len(data['actions'])} clean-up actions, {data['counts']['contradictions']} contradictions"
          + (f", {data['diagnosis']['failures']} wrong answers diagnosed" if data["diagnosis"] else ""))
    print("Wrote", out, "and", out.with_suffix(".json"))


if __name__ == "__main__":
    main()

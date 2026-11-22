"""Final results (session C): merge, failure cases, tables for the report.

    python -m src.C.final merge        # results/all_results.csv
    python -m src.C.final tables       # print the 'all' tables (markdown) used in the report
    python -m src.C.final failures     # ../local/final/failure-cases.md (local only: has passage text)

Failure cases: questions that every 10k setup with per-question rankings missed. Rankings come from
../local/work/runs/C/10k (session C), results/B/runs and results/A/runs ({qid, chunk_ids}).
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import pandas as pd

from src.common import load as L
from src.common.chunk import chunks_path, load_chunks
from src.common.score import OPTIONAL, SCHEMA, load_questions
from src.common.verses import load_verses

FINAL = L.LOCAL / "final"
ALL = L.ROOT / "results" / "all_results.csv"


def merge() -> pd.DataFrame:
    parts = []
    for s in "ABCD":
        p = L.ROOT / "results" / s / "results.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
            print(s, len(parts[-1]), "rows")
        else:
            print(s, "missing")
    df = pd.concat(parts, ignore_index=True)
    cols = SCHEMA + [c for c in OPTIONAL if c in df.columns] + [c for c in df.columns if c not in SCHEMA + OPTIONAL]
    df = df[cols]
    df.to_csv(ALL, index=False)
    print("wrote", ALL, df.shape)
    return df


def table_all(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.question_group == "all"]
    piv = a.pivot_table(index=["session", "stage", "run_id", "corpus_size"], columns="level",
                        values=["recall_5", "recall_10", "recall_20", "mrr_10"], aggfunc="first")
    piv.columns = [f"{lvl}_{m}" for m, lvl in piv.columns]
    lat = a.groupby(["session", "stage", "run_id", "corpus_size"])[["p50_ms", "p95_ms", "paid_usd_per_1000"]].first()
    t = piv.join(lat).reset_index()
    keep = ["session", "stage", "run_id", "corpus_size", "chunk_recall_5", "chunk_recall_10", "chunk_mrr_10",
            "verse_recall_10", "verse_mrr_10", "p50_ms", "p95_ms", "paid_usd_per_1000"]
    return t[[c for c in keep if c in t.columns]].sort_values(["corpus_size", "stage", "chunk_recall_10"], ascending=[True, True, False])


def md(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join((f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else
                                      f"{v:.0f}" if isinstance(v, float) else str(v)).replace("|", "/") for v in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- failure cases
def chunk_verse_map() -> dict:
    m = {}
    for p in [chunks_path(), chunks_path(150), chunks_path(600)]:
        if p.exists():
            c = pd.read_parquet(p, columns=["chunk_id", "verse_id"])
            m.update(dict(zip(c.chunk_id, c.verse_id)))
    return m


def all_runs() -> dict:
    runs = {}
    for p in glob.glob(str(L.WORK / "runs" / "C" / "10k" / "fused__*.json")):
        runs["C:" + Path(p).stem[7:]] = json.loads(Path(p).read_text(encoding="utf-8"))
    for s in "AB":
        for p in glob.glob(str(L.ROOT / "results" / s / "runs" / "*.jsonl")):
            rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
            runs[f"{s}:{Path(p).stem}"] = {r["qid"]: r["chunk_ids"] for r in rows}
    return runs


def best_ranks(runs: dict, q: pd.DataFrame, cv: dict, vg: dict, ch_records: dict):
    """Per question: best chunk-level rank (gold chunk, or overlapping span for other chunk sets) and
    best verse-level rank over all runs, within the top 20."""
    best = {qid: [99, 99, None] for qid in q.qid}
    qi = q.set_index("qid")
    for name, r in runs.items():
        for qid, ids in r.items():
            if qid not in best:
                continue
            g = qi.loc[qid]
            for i, c in enumerate(ids[:20], 1):
                v = cv.get(c)
                if v is not None and vg.get(v) == g.verse_group and i < best[qid][1]:
                    best[qid][1] = i
                hit = c == g.gold_chunk_id
                if not hit and c in ch_records:
                    rec, a, b = ch_records[c]
                    hit = rec == g.record_id and a < g.char_end and b > g.char_start
                if hit and i < best[qid][0]:
                    best[qid][0], best[qid][2] = i, name
    return best


def failures(n: int = 20):
    q = load_questions()
    v = load_verses()
    vg = dict(zip(v.verse_id, v.verse_group))
    cv = chunk_verse_map()
    ch_records = {}
    for t in (150, 600):
        p = chunks_path(t)
        if p.exists():
            c = pd.read_parquet(p, columns=["chunk_id", "record_id", "char_start", "char_end"])
            ch_records.update({r.chunk_id: (r.record_id, r.char_start, r.char_end) for r in c.itertuples()})
    runs = all_runs()
    best = best_ranks(runs, q, cv, vg, ch_records)
    q["best_chunk_rank"] = q.qid.map(lambda k: best[k][0])
    q["best_verse_rank"] = q.qid.map(lambda k: best[k][1])
    missed = q[q.best_chunk_rank > 10].sort_values(["best_verse_rank", "qid"], ascending=[False, True])
    print(f"{len(runs)} runs; chunk-level missed @10 by every setup: {len(missed)}; "
          f"verse-level missed @20 by every setup: {int((q.best_verse_rank > 20).sum())}")
    return q, missed.head(n), runs


def failure_report(n: int = 20) -> str:
    q, sel, runs = failures(n)
    ch = load_chunks().set_index("chunk_id")
    v = load_verses().set_index("verse_id")
    best_run = json.loads((L.WORK / "runs" / "C" / "10k" / "fused__bm25_dense__orig_deva_en.json").read_text(encoding="utf-8"))
    lines = [f"Questions that every one of {len(runs)} 10k setups with per-question rankings missed: the gold chunk is in nobody's "
             f"top 10. Sorted by how badly the verse was missed (99 = right verse in nobody's top 20). Local only: holds passage text.\n"]
    for i, r in enumerate(sel.itertuples(), 1):
        g = ch.loc[r.gold_chunk_id]
        top = best_run[r.qid][0]
        t = ch.loc[top] if top in ch.index else None
        same = t is not None and v.loc[t.verse_id, "verse_group"] == r.verse_group
        lines.append(f"### {i}. {r.qid} ({r.group}, {r.style}; question {r.q_lang}, gold chunk {r.chunk_lang}; "
                     f"best chunk rank {r.best_chunk_rank if r.best_chunk_rank < 99 else '>20'}, best verse rank "
                     f"{r.best_verse_rank if r.best_verse_rank < 99 else '>20'})")
        lines.append(f"- Question: {r.question}")
        lines.append(f"- Gold: {r.verse_id} | {r.author_id} | sections {', '.join(g.sections)} | {g.n_tokens} tokens")
        lines.append(f"- Gold text (first 300 chars): {g.text[:300].replace(chr(10), ' ')}")
        if t is not None:
            lines.append(f"- Best setup's rank 1: {t.verse_id} | {t.author_id} | {'same verse group' if same else 'other verse'}: "
                         f"{t.text[:160].replace(chr(10), ' ')}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if cmd == "merge":
        merge()
    elif cmd == "tables":
        print(md(table_all(pd.read_csv(ALL))))
    elif cmd == "failures":
        FINAL.mkdir(parents=True, exist_ok=True)
        (FINAL / "failure-cases.md").write_text(failure_report(), encoding="utf-8")
        print("wrote", FINAL / "failure-cases.md")

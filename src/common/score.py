"""Scoring and the results writer. One function scores a run.

    from src.common.score import score_run, result_rows, write_results, load_questions

    ranked = {qid: [chunk_id, chunk_id, ...], ...}          # best first, at least 20 per question
    rows = result_rows(ranked, chunks, session="A", stage="C2b", run_id="qwen-dense",
                       setup="Qwen3-Embedding-0.6B dense", corpus_size="10k",
                       latencies_ms=[...], paid_usd_per_1000=0.0, notes="")
    write_results(rows, "results/A/results.csv")

Levels:
  chunk  a hit is the gold chunk. With `overlap_gold=True` (C6 chunk-size variants) a hit is any
         chunk of the gold record whose span overlaps the gold character span.
  verse  a hit is any chunk whose verse is in the gold verse's group (shared/... verse_group;
         the same mantra in several Vedas counts).
Every metric uses the rank of the first hit: recall@k = hit in the top k; MRR@10 = 1/rank;
nDCG@10 = 1/log2(rank+1) (one relevant item). Question groups are reported separately and as
'all' (generated groups only). A group 'real', when added, is never folded into 'all'.
"""
from __future__ import annotations

import csv
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load as L

QUESTIONS = L.ROOT / "shared" / "test_questions.jsonl"
GENERATED_GROUPS = ("hi", "en", "hi_roman", "sa_quote")
SCHEMA = ["session", "stage", "run_id", "setup", "corpus_size", "question_group", "level", "recall_5", "recall_10",
          "recall_20", "mrr_10", "ndcg_10", "p50_ms", "p95_ms", "paid_usd_per_1000", "machine", "notes"]
OPTIONAL = ["recall_8", "mrr_8", "ndcg_8"]


def load_questions(path: Path | str = QUESTIONS) -> pd.DataFrame:
    q = pd.read_json(path, lines=True, dtype={"qid": str})
    base = q.q_lang.str.replace("-Latn", "", regex=False)
    q["lang_match"] = np.where(base == q.chunk_lang, "same", "cross")
    return q


def _verse_groups() -> dict:
    from src.common.verses import load_verses
    v = load_verses()
    return dict(zip(v.verse_id, v.verse_group))


def first_ranks(ranked: dict, chunks: pd.DataFrame, questions: pd.DataFrame, level: str,
                overlap_gold: bool = False, depth: int = 20) -> dict:
    """qid -> 1-based rank of the first hit within `depth`, or None."""
    ci = chunks.set_index("chunk_id")
    out = {}
    if level == "verse":
        vg = _verse_groups()
        chunk_group = {c: vg.get(v) for c, v in zip(chunks.chunk_id, chunks.verse_id)}
    for q in questions.itertuples():
        ids = list(ranked.get(q.qid, []))[:depth]
        rank = None
        for i, c in enumerate(ids, 1):
            if level == "chunk":
                if overlap_gold:
                    if c in ci.index:
                        r = ci.loc[c]
                        hit = (r.record_id == q.record_id and r.char_start < q.char_end and r.char_end > q.char_start)
                    else:
                        hit = False
                else:
                    hit = c == q.gold_chunk_id
            else:
                hit = chunk_group.get(c) is not None and chunk_group.get(c) == q.verse_group
            if hit:
                rank = i
                break
        out[q.qid] = rank
    return out


def metrics(ranks: list, at8: bool = False) -> dict:
    n = len(ranks)
    if n == 0:
        return {}
    m = {f"recall_{k}": sum(1 for r in ranks if r and r <= k) / n for k in (5, 10, 20)}
    m["mrr_10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    m["ndcg_10"] = sum(1 / math.log2(r + 1) for r in ranks if r and r <= 10) / n
    if at8:
        m["recall_8"] = sum(1 for r in ranks if r and r <= 8) / n
        m["mrr_8"] = sum(1 / r for r in ranks if r and r <= 8) / n
        m["ndcg_8"] = sum(1 / math.log2(r + 1) for r in ranks if r and r <= 8) / n
    m["n"] = n
    return m


def score_run(ranked: dict, chunks: pd.DataFrame, questions: pd.DataFrame | None = None, by: str = "group",
              overlap_gold: bool = False, at8: bool = False) -> pd.DataFrame:
    """Metrics per value of question column `by` (group, chunk_lang, lang_match, ...) and level.
    Adds 'all' over the generated groups."""
    q = load_questions() if questions is None else questions
    rows = []
    for level in ("chunk", "verse"):
        fr = first_ranks(ranked, chunks, q, level, overlap_gold)
        q2 = q.assign(rank=q.qid.map(fr))
        gen = q2[q2.group.isin(GENERATED_GROUPS)]
        parts = [(k, g) for k, g in q2.groupby(by)] + [("all", gen)]
        for key, g in parts:
            rows.append(dict(question_group=key, level=level, **metrics(list(g["rank"]), at8)))
    return pd.DataFrame(rows)


def machine_info() -> str:
    """'laptop1 | CPU | GPU | RAM'. laptop1 = sessions C and D, laptop2 = A and B. No host names."""
    try:
        letter = (L.ROOT / ".session").read_text(encoding="utf-8").strip().upper()
    except OSError:
        letter = "?"
    laptop = "laptop1" if letter in ("C", "D") else "laptop2" if letter in ("A", "B") else "laptop?"
    cpu = platform.processor() or platform.machine()
    ram = ""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("t", ctypes.c_ulonglong),
                        ("a", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        s = MS()
        s.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        ram = f"{s.t / 2**30:.0f}GB RAM"
    except Exception:
        pass
    gpu = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0) + f" {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GB"
    except Exception:
        pass
    return " | ".join(x for x in (laptop, cpu, gpu or "no GPU", ram) if x)


def pct(xs, p) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), p)) if len(xs) else float("nan")


def result_rows(ranked: dict, chunks: pd.DataFrame, session: str, stage: str, run_id: str, setup: str,
                corpus_size: str, latencies_ms=(), paid_usd_per_1000: float = 0.0, notes: str = "",
                questions: pd.DataFrame | None = None, overlap_gold: bool = False, at8: bool = False,
                machine: str | None = None) -> list[dict]:
    sc = score_run(ranked, chunks, questions, "group", overlap_gold, at8)
    mach = machine or machine_info()
    p50, p95 = pct(list(latencies_ms), 50), pct(list(latencies_ms), 95)
    out = []
    for r in sc.to_dict("records"):
        row = dict(session=session, stage=stage, run_id=run_id, setup=setup, corpus_size=corpus_size,
                   question_group=r["question_group"], level=r["level"],
                   **{k: round(r[k], 4) for k in ("recall_5", "recall_10", "recall_20", "mrr_10", "ndcg_10")},
                   p50_ms=round(p50, 1), p95_ms=round(p95, 1), paid_usd_per_1000=round(paid_usd_per_1000, 4),
                   machine=mach, notes=f"n={r['n']}" + (f"; {notes}" if notes else ""))
        if at8:
            row.update({k: round(r[k], 4) for k in OPTIONAL})
        out.append(row)
    return out


def write_results(rows: list[dict], path: Path | str):
    """Append rows to a results CSV in the shared schema (optional @8 columns at the end if used)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = SCHEMA + [c for c in OPTIONAL if any(c in r for r in rows)]
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, encoding="utf-8") as f:
            cols = next(csv.reader(f))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)

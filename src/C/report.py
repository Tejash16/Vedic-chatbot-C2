"""Final report (session C).

    python -m src.C.report      # ../local/final/benchmark-report.md and results/SUMMARY.md

The narrative (recommendations, what to test next) lives in src/C/report_text.md, written by
session C after reading every session's findings. Tables come from results/all_results.csv.
SUMMARY.md (in Git) holds no source text; the failure cases go only into the local report.
"""
from __future__ import annotations

import pandas as pd

from src.C.final import ALL, FINAL, failure_report, md, merge
from src.common import load as L

TEXT = L.ROOT / "src" / "C" / "report_text.md"
SUMMARY = L.ROOT / "results" / "SUMMARY.md"

KEY_ROWS = [  # (session, run_id, corpus_size, label)
    ("C", "bm25", "10k", "C1a BM25 (bm25s, own tokenizer)"),
    ("B", "C1b_os_plain", "10k", "C1b OpenSearch, unstemmed"),
    ("B", "C1b_os_lang_routed", "10k", "C1b OpenSearch, language analyzer routed by script"),
    ("C", "dense", "10k", "C2a bge-m3 dense"),
    ("C", "sparse", "10k", "C2a bge-m3 sparse"),
    ("C", "dense+sparse", "10k", "C2a bge-m3 dense+sparse"),
    ("A", "qwen_dense", "10k", "C2b Qwen3-Embedding-0.6B dense (512)"),
    ("A", "qwen_dense_ml1024", "10k", "C2b Qwen3-Embedding-0.6B dense (1024)"),
    ("A", "qwen_dense+bm25", "10k", "C2b Qwen3 dense + C1a (512)"),
    ("C", "bm25+dense", "10k", "C3 BM25 + bge-m3 dense"),
    ("C", "bm25+dense+sparse", "10k", "C3 BM25 + bge-m3 dense + sparse"),
    ("C", "dense|orig+deva+en", "10k", "C4 bge-m3 dense, 3 question forms"),
    ("C", "bm25+dense|orig+deva+en", "10k", "C4 BM25 + dense, 3 forms (candidates)"),
    ("C", "bm25+dense+sparse|orig+deva+en", "10k", "C4 BM25 + dense + sparse, 3 forms"),
    ("B", "C5_no_reranker", "10k", "C5 no reranker (candidates order)"),
    ("B", "C5_bge", "10k", "C5 bge-reranker-v2-m3"),
    ("B", "C5_qwen", "10k", "C5 Qwen3-Reranker-0.6B (512)"),
    ("B", "C5_qwen_p1024", "10k", "C5 Qwen3-Reranker-0.6B (1024)"),
    ("B", "C5_haiku_low", "10k", "C5 claude-haiku-5-5 picks 8 (low effort)"),
    ("B", "C5_haiku_medium", "10k", "C5 claude-haiku-5-5 picks 8 (medium effort)"),
    ("A", "bgem3_ds_c300", "10k", "C6 bge-m3 dense+sparse, 300 tokens"),
    ("A", "bgem3_ds_c150", "10k", "C6 bge-m3 dense+sparse, 150 tokens"),
    ("A", "bgem3_ds_c600", "10k", "C6 bge-m3 dense+sparse, 600 tokens"),
    ("A", "bgem3_ds_c300_nohdr", "10k", "C6 bge-m3 dense+sparse, 300 tokens, no header"),
    ("D", "D5-01", "1.5k", "D bge-m3 dense, passages only (sub-corpus)"),
    ("D", "D5-02", "1.5k", "D + all kept sets, hi+en"),
    ("D", "D5-10", "1.5k", "D + all sets passing filters 1-2, hi+en"),
    ("C", "dense", "full", "Full: bge-m3 dense"),
    ("C", "bm25+dense|orig+deva+en", "full", "Full: BM25 + dense, 3 forms"),
    ("C", "bm25+dense+sparse|orig+deva+en", "full", "Full: BM25 + dense + sparse, 3 forms"),
]


def key_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.question_group == "all"].copy()
    a["corpus"] = a.corpus_size.astype(str).str.split(" ").str[0]
    rows = []
    for s, rid, corpus, label in KEY_ROWS:
        sub = a[(a.session == s) & (a.run_id.astype(str).str.lower() == rid.lower()) & (a.corpus == corpus)]
        if sub.empty:
            sub = a[(a.session == s) & a.run_id.astype(str).str.lower().str.contains(rid.lower(), regex=False) & (a.corpus == corpus)]
            if sub.empty:
                continue
            sub = sub[sub.run_id == sub.run_id.iloc[0]]
        c = sub[sub.level == "chunk"].iloc[0]
        v = sub[sub.level == "verse"].iloc[0]
        r8 = c.get("recall_8") if "recall_8" in c else None
        rows.append({"setup": label, "session": s, "corpus": corpus, "chunk R@5": f"{c.recall_5:.3f}",
                     "chunk R@8": f"{r8:.3f}" if r8 is not None and pd.notna(r8) else "",
                     "chunk R@10": f"{c.recall_10:.3f}", "chunk MRR@10": f"{c.mrr_10:.3f}",
                     "verse R@10": f"{v.recall_10:.3f}", "p50 ms": f"{c.p50_ms:.0f}" if c.p50_ms >= 1 else f"{c.p50_ms:.1f}",
                     "p95 ms": f"{c.p95_ms:.0f}" if c.p95_ms >= 1 else f"{c.p95_ms:.1f}",
                     "$ per 1k q": f"{c.paid_usd_per_1000:.2f}"})
    return pd.DataFrame(rows)


def machines(df: pd.DataFrame) -> str:
    m = df.groupby("machine").session.agg(lambda x: ", ".join(sorted(set(x))))
    return "\n".join(f"- {k}: sessions {v}" for k, v in m.items())


def build():
    df = merge()
    kt = key_table(df)
    text = TEXT.read_text(encoding="utf-8") if TEXT.exists() else "(narrative to come)\n"
    head = ("# C2 benchmark report\n\nAll numbers are for the 'all' group of 360 generated questions "
            "(hi, en, hi_roman, sa_quote x 90; 222 cross-language) unless stated. Full table: results/all_results.csv.\n\n")
    table = "## Results\n\n" + md(kt) + "\n"
    mach = "\n## Machines\n\n" + machines(df) + "\n"
    SUMMARY.write_text(head + table + "\n" + text + mach, encoding="utf-8")
    fails = "\n## 20 failure cases (every setup missed)\n\n" + failure_report(20) + "\n"
    (FINAL / "benchmark-report.md").write_text(head + table + "\n" + text + fails + mach, encoding="utf-8")
    print("wrote", SUMMARY, "and", FINAL / "benchmark-report.md")


if __name__ == "__main__":
    build()

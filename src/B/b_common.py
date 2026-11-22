"""Session B glue on top of src/common (C's shared loading, scoring, fusion and writer)."""
import csv
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load as L  # noqa: E402  (sets HF_HOME to ../local/hf-cache)
from src.common.chunk import embed_text, load_chunks  # noqa: E402,F401
from src.common.fusion import rrf  # noqa: E402,F401
from src.common.score import OPTIONAL, SCHEMA, load_questions, result_rows, score_run  # noqa: E402,F401
from src.common.score import write_results as _write_results  # noqa: E402

LOCAL = L.LOCAL
WORK = L.WORK
SHARED = ROOT / "shared"
RES = ROOT / "results" / "B"
RESULTS_CSV = RES / "results.csv"
CORPUS_SIZE = "10k"

# Spec C1b / C's B_READY note: hi and sa chunks -> Hindi analyzer, en chunks -> English analyzer.
ROUTE = {"hi": "hi", "sa": "hi", "en": "en"}


def questions_list():
    return load_questions().to_dict("records")


def index_text(row):
    """What the keyword index and the rerankers see: header + newline + text (src.common.chunk.embed_text)."""
    return unicodedata.normalize("NFC", embed_text(row, header=True))


def index_rows(chunks):
    for row in chunks.to_dict("records"):
        if row["lang"] not in ROUTE:
            raise ValueError("unexpected chunk lang %r" % row["lang"])
        yield {"chunk_id": row["chunk_id"], "verse_id": row["verse_id"], "lang": row["lang"],
               "route": ROUTE[row["lang"]], "text": index_text(row)}


def save_run(name, run):
    """Top-k chunk IDs per question (IDs only, no text) for fusion by other sessions."""
    d = RES / "runs"
    d.mkdir(parents=True, exist_ok=True)
    safe = name.replace("+", "_plus_").replace("/", "_")
    with open(d / ("%s.jsonl" % safe), "w", encoding="utf-8") as f:
        for qid in sorted(run):
            f.write(json.dumps({"qid": qid, "chunk_ids": run[qid]}) + "\n")


def write_stage(rows, stage):
    """Replace this stage's rows in results/B/results.csv and keep the rest. The header always
    carries the optional @8 columns, so C1b rows (empty @8) and C5 rows share one file."""
    old = []
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        with open(RESULTS_CSV, encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if r.get("stage") != stage]
        RESULTS_CSV.unlink()
    allrows = []
    for r in old + list(rows):
        r = dict(r)
        for c in OPTIONAL:
            r.setdefault(c, "")
        allrows.append(r)
    _write_results(allrows, RESULTS_CSV)


def print_summary(rows, at8=False):
    for r in rows:
        if r["question_group"] != "all":
            continue
        s = "%-28s %-6s R@5 %.3f R@10 %.3f R@20 %.3f MRR@10 %.3f nDCG@10 %.3f" % (
            r["setup"], r["level"], r["recall_5"], r["recall_10"], r["recall_20"], r["mrr_10"], r["ndcg_10"])
        if at8:
            s += " | R@8 %.3f MRR@8 %.3f nDCG@8 %.3f" % (r["recall_8"], r["mrr_8"], r["ndcg_8"])
        s += " | p50 %.1f p95 %.1f ms $/1k %.3f" % (r["p50_ms"], r["p95_ms"], r["paid_usd_per_1000"])
        print(s)

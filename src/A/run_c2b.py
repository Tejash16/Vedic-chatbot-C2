"""Step C2b (session A): Qwen/Qwen3-Embedding-0.6B dense, alone and fused (RRF k=60) with C1a BM25.

Run from the clone root:  python -m src.A.run_c2b
Mirrors C's C2a method (src/C/retrieval.py) so the numbers compare:
  passages  embed_text(row, header=True), max length 512, fp16, L2-normalised
  queries   the model's own 'query' prompt, encoded one at a time (batch 1) on the GPU
  search    numpy D @ q + top 100 on CPU (C's search_dense)
  BM25      C's C1a class (bm25s lucene k1=1.5 b=0.75, own tokenizer, header+text)
  latency   dense = encode + search; fused = BM25 + encode + search + RRF (sequential sum)
Ranked lists (chunk IDs only) go to ../local/work/A; metrics to results/A/results.csv.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.common import load as L  # sets HF_HOME
from src.common.chunk import embed_text, load_chunks
from src.common.fusion import rrf
from src.common.score import load_questions, result_rows, score_run, write_results
from src.C.retrieval import BM25, search_dense, TOP
from src.A.embed_qwen import CACHE, QwenEncoder, cached_encode
from src.A.gpu import gpu_lock

RESULTS = L.ROOT / "results" / "A" / "results.csv"


def encode_queries(enc, texts):
    enc.encode(["warm up"], is_query=True)
    out, ms = [], []
    for t in texts:
        t0 = time.perf_counter()
        out.append(enc.encode([t], is_query=True)[0])
        ms.append((time.perf_counter() - t0) * 1000)
    return np.stack(out).astype(np.float32), ms


def report(run_id, ranked, ms, chunks):
    (CACHE / f"ranked_C2b_{run_id}.json").write_text(json.dumps({"ranked": ranked, "ms": ms}), encoding="utf-8")
    sc = score_run(ranked, chunks)
    lm = score_run(ranked, chunks, by="lang_match")
    lm.to_csv(CACHE / f"langmatch_C2b_{run_id}.csv", index=False)
    print(f"\n=== {run_id}  p50 {np.percentile(ms, 50):.1f} ms  p95 {np.percentile(ms, 95):.1f} ms")
    print(sc.round(3).to_string(index=False))
    print(lm.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=512,
                    help="passage max length; 1024 = supplementary run (Qwen's tokenizer needs ~2.4x the "
                         "bge-m3 tokens on Devanagari, so 512 truncates ~40%% of chunks)")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    sfx = "" if a.max_len == 512 else f"_ml{a.max_len}"

    chunks = load_chunks()
    q = load_questions()
    qids, qtexts = list(q.qid), list(q.question)
    ids = chunks.chunk_id.tolist()
    texts = [embed_text(r, header=True) for r in chunks[["header", "text"]].to_dict("records")]
    print(f"{len(ids)} chunks, {len(q)} questions")
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---- C1a BM25 (C's class, CPU) --------------------------------------------------
    bm = BM25(chunks)
    b_lists, b_ms = bm.search(qtexts)

    # ---- Qwen dense: encode on GPU under the lock ------------------------------------------
    holder = {}

    def factory():
        if "e" not in holder:
            holder["e"] = QwenEncoder(max_len=a.max_len, batch_size=a.batch)
        return holder["e"]

    from transformers import AutoTokenizer
    from src.A.embed_qwen import MODEL_ID, local_path
    ntok = [len(x) for x in AutoTokenizer.from_pretrained(local_path(MODEL_ID))(texts)["input_ids"]]
    trunc = float(np.mean([n > a.max_len for n in ntok]))
    print(f"Qwen tokens per passage: median {np.median(ntok):.0f}, max {max(ntok)}; "
          f"truncated at {a.max_len}: {trunc:.1%}")

    with gpu_lock():
        try:
            P, secs, fresh = cached_encode(factory, texts, "c300_hdr" + sfx)
            print(f"passages encoded in {secs / 60:.1f} min" + ("" if fresh else " (cached)"))
            Qd, enc_ms = encode_queries(factory(), qtexts)
        finally:
            if "e" in holder:
                holder.pop("e").close()

    D = P.astype(np.float32)
    d_lists, s_ms = search_dense(D, Qd, ids)
    dense_ms = [e + s for e, s in zip(enc_ms, s_ms)]

    f_lists, f_ms = [], []
    for d, b, dm, bmm in zip(d_lists, b_lists, dense_ms, b_ms):
        t0 = time.perf_counter()
        f_lists.append(rrf([d, b], top=TOP))
        f_ms.append(dm + bmm + (time.perf_counter() - t0) * 1000)

    note = ("passages header+text, max_len %d (%.0f%% of passages truncated by Qwen's tokenizer), fp16, "
            "batch %d; queries: model default 'query' instruction, batch 1 on GPU; search numpy CPU; "
            "passage encoding %s; latency re-measured for all A setups back-to-back in one quiet window" %
            (a.max_len, 100 * trunc, a.batch,
             f"{secs / 60:.1f} min for {len(texts)} chunks on this GPU" if secs else "time not recorded"))
    ml = "" if a.max_len == 512 else f", passages max_length {a.max_len} (supplementary)"
    runs = {
        "qwen_dense" + sfx: ("Qwen3-Embedding-0.6B dense" + ml, d_lists, dense_ms, note),
        "qwen_dense+bm25" + sfx: ("Qwen3-Embedding-0.6B dense + C1a BM25, RRF k=60" + ml, f_lists, f_ms,
                                  note + "; BM25 = src.C.retrieval.BM25 (C1a)"),
        "bm25_ref": ("C1a BM25 on laptop 2 (reference only)", b_lists, b_ms, ""),
    }
    for run_id, (setup, lists, ms, nt) in runs.items():
        ranked = dict(zip(qids, lists))
        report(run_id, ranked, ms, chunks)
        if a.write and run_id != "bm25_ref":
            write_results(result_rows(ranked, chunks, session="A", stage="C2b", run_id=run_id, setup=setup,
                                      corpus_size="10k", latencies_ms=ms, paid_usd_per_1000=0.0, notes=nt),
                          RESULTS)
    print("\nwrote", RESULTS if a.write else "(nothing, --write 0)")


if __name__ == "__main__":
    main()

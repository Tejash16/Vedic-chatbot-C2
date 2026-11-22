"""Step C6 (session A): chunk variants with bge-m3 dense + sparse (RRF k=60).

Run from the clone root:  python -m src.A.run_c6 [--only c150,c600,...]
Variants (chunk files built by `python -m src.common.chunk --target N`):
  c300       chunks.parquet, header + text, max_length 512   (same-machine baseline = C2a dense+sparse)
  c150       chunks_150.parquet, header + text, max_length 512
  c600       chunks_600.parquet, header + text, max_length 1024
  c300_nohdr chunks.parquet, text only, max_length 512
Scored with overlap_gold=True: any chunk of the gold record overlapping the gold span is a hit.
Uses C's bge-m3 code (src/C/retrieval.py) for the model, query encoding and search so C6 compares with C2a.
Latency per question = one bge-m3 query encoding (GPU, batch 1) + dense search + sparse search + RRF.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from src.common import load as L
from src.common.chunk import embed_text, load_chunks
from src.common.fusion import rrf
from src.common.score import load_questions, result_rows, score_run, write_results
from src.C.retrieval import TOP, bge, encode_queries, free_gpu, search_dense, search_sparse, sparse_matrix
from src.A.gpu import gpu_lock

CACHE = L.WORK / "A"
RESULTS = L.ROOT / "results" / "A" / "results.csv"

VARIANTS = {
    # name: (chunk target, header, max_length, batch, setup)
    "c300": (300, True, 512, 16, "bge-m3 dense+sparse, ~300-token chunks with header (same-machine baseline)"),
    "c150": (150, True, 512, 16, "bge-m3 dense+sparse, ~150-token chunks with header"),
    "c600": (600, True, 1024, 8, "bge-m3 dense+sparse, ~600-token chunks with header, max_length 1024"),
    "c300_nohdr": (300, False, 512, 16, "bge-m3 dense+sparse, ~300-token chunks, header removed"),
}


def embed_variant(name, chunks, header, max_len, batch):
    base = CACHE / f"bgem3_{name}"
    ids = list(chunks.chunk_id)
    if base.with_suffix(".npz").exists() and json.loads(base.with_suffix(".json").read_text()) == ids:
        side = base.with_suffix(".secs")
        secs = float(side.read_text()) if side.exists() else None
        print(f"{name}: embeddings from cache")
        return np.load(base.with_suffix(".npy")).astype(np.float32), sp.load_npz(base.with_suffix(".npz")), secs
    texts = [embed_text(r, header=header) for r in chunks[["header", "text"]].to_dict("records")]
    with gpu_lock():
        try:
            t0 = time.time()
            out = bge().encode(texts, batch_size=batch, max_length=max_len, return_dense=True, return_sparse=True)
            secs = time.time() - t0
        finally:
            free_gpu()
    D = out["dense_vecs"].astype(np.float16)
    S = sparse_matrix(out["lexical_weights"])
    np.save(base.with_suffix(".npy"), D)
    sp.save_npz(base.with_suffix(".npz"), S)
    base.with_suffix(".json").write_text(json.dumps(ids))
    base.with_suffix(".secs").write_text(f"{secs:.1f}")
    print(f"{name}: {len(texts)} chunks embedded in {secs / 60:.1f} min")
    return D.astype(np.float32), S, secs


def query_side(qtexts):
    path = CACHE / "bgem3_queries.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["dense"], sp.load_npz(CACHE / "bgem3_queries_sparse.npz"), list(z["ms"])
    with gpu_lock():
        try:
            Qd, Qs, ms = encode_queries(qtexts)
        finally:
            free_gpu()
    np.savez(path, dense=Qd, ms=np.asarray(ms))
    sp.save_npz(CACHE / "bgem3_queries_sparse.npz", Qs)
    return Qd, Qs, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of variants (default all)")
    ap.add_argument("--write", type=int, default=1)
    a = ap.parse_args()
    names = [n for n in VARIANTS if not a.only or n in a.only.split(",")]

    CACHE.mkdir(parents=True, exist_ok=True)
    q = load_questions()
    qids, qtexts = list(q.qid), list(q.question)
    Qd, Qs, enc_ms = query_side(qtexts)
    print(f"query encoding p50 {np.percentile(enc_ms, 50):.1f} ms")

    for name in names:
        target, header, max_len, batch, setup = VARIANTS[name]
        chunks = load_chunks(target)
        D, S, secs = embed_variant(name, chunks, header, max_len, batch)
        ids = list(chunks.chunk_id)
        d_lists, d_ms = search_dense(D, Qd, ids)
        s_lists, s_ms = search_sparse(S, Qs, ids)
        lists, ms = [], []
        for d, s, e, dm, sm in zip(d_lists, s_lists, enc_ms, d_ms, s_ms):
            t0 = time.perf_counter()
            lists.append(rrf([d, s], top=TOP))
            ms.append(e + dm + sm + (time.perf_counter() - t0) * 1000)
        ranked = dict(zip(qids, lists))
        (CACHE / f"ranked_C6_{name}.json").write_text(json.dumps({"ranked": ranked, "ms": ms}), encoding="utf-8")
        sc = score_run(ranked, chunks, overlap_gold=True)
        lm = score_run(ranked, chunks, by="lang_match", overlap_gold=True)
        lm.to_csv(CACHE / f"langmatch_C6_{name}.csv", index=False)
        # dense-only and sparse-only, for the findings (not written to results.csv)
        for part, pl in (("dense", d_lists), ("sparse", s_lists)):
            score_run(dict(zip(qids, pl)), chunks, overlap_gold=True).to_csv(
                CACHE / f"score_C6_{name}_{part}.csv", index=False)
        print(f"\n=== {name}: {len(ids)} chunks, p50 {np.percentile(ms, 50):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms")
        print(sc.round(3).to_string(index=False))
        print(lm.round(3).to_string(index=False))
        if a.write:
            note = (f"{len(ids)} chunks from 7763 records; overlap_gold; embed max_length {max_len}, fp16, batch {batch}"
                    + (f"; passage encoding {secs / 60:.1f} min on this GPU" if secs else "")
                    + "; latency re-measured for all A setups back-to-back in one quiet window")
            write_results(result_rows(ranked, chunks, session="A", stage="C6", run_id=f"bgem3_ds_{name}",
                                      setup=setup, corpus_size="10k", latencies_ms=ms, paid_usd_per_1000=0.0,
                                      notes=note, overlap_gold=True), RESULTS)
    print("\ndone", RESULTS if a.write else "(nothing written)")


if __name__ == "__main__":
    main()

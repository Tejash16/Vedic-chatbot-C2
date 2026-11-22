"""Stage C retrieval, session C: C1a (BM25, own tokenizer), C2a (bge-m3 dense, sparse, dense+sparse),
C3 (RRF fusion, k = 60), C4 (question forms), candidates for C5, and the full-size run.

    python -m src.C.retrieval embed  [--corpus 10k|full]   # GPU: bge-m3 dense+sparse for all chunks (cached)
    python -m src.C.retrieval run    [--corpus 10k|full]   # retrieve, fuse, score, write results/C/results.csv
    python -m src.C.retrieval candidates                   # shared/candidates_top50.jsonl from the best fused setup

Runs (top 100 chunk ids per question) are cached in ../local/work/runs/C/<corpus>/<run_id>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import load as L
from src.common.chunk import embed_text, load_chunks
from src.common.fusion import rrf
from src.common.score import load_questions, result_rows, score_run, write_results
from src.common.text import words

EMB = L.WORK / "emb"
RUNS = L.WORK / "runs" / "C"
RESULTS = L.ROOT / "results" / "C" / "results.csv"
TOP = 100
BATCH = 16
MAX_LEN = 512


# ---------------------------------------------------------------- bge-m3
_model = None


def bge():
    global _model
    if _model is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        from FlagEmbedding import BGEM3FlagModel
        path = snapshot_download("BAAI/bge-m3", local_files_only=True)
        _model = BGEM3FlagModel(path, use_fp16=True, devices="cuda:0")
    return _model


def sparse_matrix(lex: list[dict], dim: int = 250_002) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i, d in enumerate(lex):
        for t, w in d.items():
            rows.append(i)
            cols.append(int(t))
            vals.append(float(w))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(lex), dim), dtype=np.float32)


def corpus_chunks(corpus: str) -> pd.DataFrame:
    return load_chunks(full=(corpus == "full"))


def embed_corpus(corpus: str, part: int | None = None, parts: int = 1, gpu_lock=None):
    """Embed chunks (dense fp16 + sparse) in `parts` parts; each part is cached on its own so the
    GPU lock can be released between parts."""
    ch = corpus_chunks(corpus)
    EMB.mkdir(parents=True, exist_ok=True)
    texts = [embed_text(r) for r in ch[["header", "text"]].to_dict("records")]
    bounds = np.linspace(0, len(texts), parts + 1).astype(int)
    todo = range(parts) if part is None else [part]
    for p in todo:
        base = EMB / f"bge_m3_{corpus}_p{p}of{parts}"
        if (base.with_suffix(".npz")).exists():
            print("cached", base.name)
            continue
        a, b = bounds[p], bounds[p + 1]
        if gpu_lock:
            gpu_lock("acquire")
        try:
            t0 = time.time()
            out = bge().encode(texts[a:b], batch_size=BATCH, max_length=MAX_LEN, return_dense=True, return_sparse=True)
            np.save(base.with_suffix(".npy"), out["dense_vecs"].astype(np.float16))
            sp.save_npz(base.with_suffix(".npz"), sparse_matrix(out["lexical_weights"]))
            print(f"part {p + 1}/{parts}: {b - a} chunks in {time.time() - t0:.0f}s")
        finally:
            if gpu_lock:
                free_gpu()
                gpu_lock("release")
    if part is None or all((EMB / f"bge_m3_{corpus}_p{p}of{parts}.npz").exists() for p in range(parts)):
        dense = np.concatenate([np.load(EMB / f"bge_m3_{corpus}_p{p}of{parts}.npy") for p in range(parts)])
        sparse = sp.vstack([sp.load_npz(EMB / f"bge_m3_{corpus}_p{p}of{parts}.npz") for p in range(parts)]).tocsr()
        np.save(EMB / f"bge_m3_{corpus}_dense.npy", dense)
        sp.save_npz(EMB / f"bge_m3_{corpus}_sparse.npz", sparse)
        (EMB / f"bge_m3_{corpus}_ids.json").write_text(json.dumps(list(ch.chunk_id)), encoding="utf-8")
        print("wrote", EMB / f"bge_m3_{corpus}_dense.npy", dense.shape)


def free_gpu():
    global _model
    _model = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def bus_gpu(action: str):
    import subprocess
    import sys
    while True:
        r = subprocess.run([sys.executable, str(L.ROOT / "bus" / "bus.py"), "gpu", action], capture_output=True, text=True)
        print(r.stdout.strip())
        if action == "release" or r.returncode == 0:
            return


# ---------------------------------------------------------------- query side
def encode_queries(texts: list[str]):
    """Per-query encoding (batch size 1) to measure latency. Returns dense, sparse, ms per query."""
    m = bge()
    m.encode(["warm up"], batch_size=1, max_length=MAX_LEN, return_dense=True, return_sparse=True)
    dense, lex, ms = [], [], []
    for t in texts:
        t0 = time.perf_counter()
        o = m.encode([t], batch_size=1, max_length=MAX_LEN, return_dense=True, return_sparse=True)
        ms.append((time.perf_counter() - t0) * 1000)
        dense.append(o["dense_vecs"][0])
        lex.append(o["lexical_weights"][0])
    return np.stack(dense).astype(np.float32), sparse_matrix(lex), ms


def topk(scores: np.ndarray, k: int = TOP) -> np.ndarray:
    idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def search_dense(D: np.ndarray, Q: np.ndarray, ids: list[str]):
    out, ms = [], []
    for q in Q:
        t0 = time.perf_counter()
        s = D @ q
        out.append([ids[i] for i in topk(s)])
        ms.append((time.perf_counter() - t0) * 1000)
    return out, ms


def search_sparse(S: sp.csr_matrix, Qs: sp.csr_matrix, ids: list[str]):
    ST = S.T.tocsr()
    out, ms = [], []
    for i in range(Qs.shape[0]):
        t0 = time.perf_counter()
        s = np.asarray((Qs[i] @ ST).todense()).ravel()
        out.append([ids[j] for j in topk(s)])
        ms.append((time.perf_counter() - t0) * 1000)
    return out, ms


class BM25:
    def __init__(self, ch: pd.DataFrame):
        import bm25s
        self.ids = list(ch.chunk_id)
        t0 = time.time()
        self.r = bm25s.BM25()
        self.r.index([words(embed_text(r)) or ["_"] for r in ch[["header", "text"]].to_dict("records")], show_progress=False)
        print(f"BM25 index: {len(self.ids)} chunks in {time.time() - t0:.0f}s")

    def search(self, texts: list[str]):
        out, ms = [], []
        for t in texts:
            t0 = time.perf_counter()
            toks = words(t) or ["_"]
            docs, _ = self.r.retrieve([toks], k=TOP, show_progress=False, n_threads=1)
            out.append([self.ids[i] for i in docs[0]])
            ms.append((time.perf_counter() - t0) * 1000)
        return out, ms


# ---------------------------------------------------------------- runs
def run_dir(corpus: str):
    d = RUNS / corpus
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(corpus: str, run_id: str, qids, lists, ms):
    (run_dir(corpus) / f"{run_id}.json").write_text(
        json.dumps({"ranked": dict(zip(qids, lists)), "ms": list(map(float, ms))}), encoding="utf-8")


def load_run(corpus: str, run_id: str):
    d = json.loads((run_dir(corpus) / f"{run_id}.json").read_text(encoding="utf-8"))
    return d["ranked"], d["ms"]


def base_runs(corpus: str, forms: dict[str, dict[str, str]]):
    """forms: form name -> {qid: query text}. Writes bm25/dense/sparse runs for each form."""
    ch = corpus_chunks(corpus)
    ids = json.loads((EMB / f"bge_m3_{corpus}_ids.json").read_text(encoding="utf-8"))
    assert ids == list(ch.chunk_id), "embeddings do not match the chunk file; re-embed"
    D = np.load(EMB / f"bge_m3_{corpus}_dense.npy").astype(np.float32)
    S = sp.load_npz(EMB / f"bge_m3_{corpus}_sparse.npz")
    bm = BM25(ch)
    for form, qmap in forms.items():
        qids = list(qmap)
        texts = [qmap[q] for q in qids]
        b, bms = bm.search(texts)
        save_run(corpus, f"bm25__{form}", qids, b, bms)
        bus_gpu("acquire")
        try:
            Qd, Qs, enc_ms = encode_queries(texts)
        finally:
            free_gpu()
            bus_gpu("release")
        d, dms = search_dense(D, Qd, ids)
        s, sms = search_sparse(S, Qs, ids)
        save_run(corpus, f"dense__{form}", qids, d, [e + x for e, x in zip(enc_ms, dms)])
        # sparse shares the query encoding pass with dense; count encoding once per setup (see fuse)
        save_run(corpus, f"sparse__{form}", qids, s, [e + x for e, x in zip(enc_ms, sms)])
        save_run(corpus, f"_enc__{form}", qids, [[] for _ in qids], enc_ms)
        print(f"form {form}: runs written")


SETUPS_C1C2C3 = {
    # run_id: (stage, components, description)
    "bm25": ("C1a", ["bm25"], "BM25 (bm25s, lucene k1=1.5 b=0.75), own tokenizer on header+text"),
    "dense": ("C2a", ["dense"], "bge-m3 dense, fp16, max_length 512"),
    "sparse": ("C2a", ["sparse"], "bge-m3 sparse (lexical weights)"),
    "dense+sparse": ("C2a", ["dense", "sparse"], "bge-m3 dense + sparse, RRF k=60"),
    "bm25+dense": ("C3", ["bm25", "dense"], "BM25 + bge-m3 dense, RRF k=60"),
    "bm25+sparse": ("C3", ["bm25", "sparse"], "BM25 + bge-m3 sparse, RRF k=60"),
    "bm25+dense+sparse": ("C3", ["bm25", "dense", "sparse"], "BM25 + bge-m3 dense + sparse, RRF k=60"),
}


def fused(corpus: str, components: list[str], forms: list[str]):
    """RRF over every (component, form) run. Latency per question: BM25 + one bge-m3 encoding + the
    searches, summed per form (sequential)."""
    lists_by_q, ms_total = {}, None
    for form in forms:
        enc_ms = load_run(corpus, f"_enc__{form}")[1]
        f_ms = None
        for c in components:
            ranked, ms = load_run(corpus, f"{c}__{form}")
            for q, l in ranked.items():
                lists_by_q.setdefault(q, []).append(l)
            ms = np.asarray(ms)
            if c in ("dense", "sparse"):
                ms = ms - np.asarray(enc_ms)          # search only
            f_ms = ms if f_ms is None else f_ms + ms
        if any(c in ("dense", "sparse") for c in components):
            f_ms = f_ms + np.asarray(enc_ms)          # one encoding pass per form
        ms_total = f_ms if ms_total is None else ms_total + f_ms
    ranked = {q: (rrf(ls) if len(ls) > 1 else ls[0][:TOP]) for q, ls in lists_by_q.items()}
    return ranked, list(ms_total)


def run_all(corpus: str, forms_extra: dict | None = None, rewrite_ms: dict | None = None,
            rewrite_usd_per_1000: float = 0.0):
    q = load_questions()
    ch = corpus_chunks(corpus)
    size = "10k" if corpus == "10k" else f"full ({len(ch)})"
    forms = {"orig": dict(zip(q.qid, q.question))}
    if forms_extra:
        forms.update(forms_extra)
    have = [f for f in forms if not (run_dir(corpus) / f"bm25__{f}.json").exists()]
    if have:
        base_runs(corpus, {f: forms[f] for f in have})
    rows, summary = [], []
    setups = []
    for rid, (stage, comps, desc) in SETUPS_C1C2C3.items():
        setups.append((rid, stage, comps, ["orig"], desc, 0.0, None))
    if forms_extra:
        for rid, (stage, comps, desc) in SETUPS_C1C2C3.items():
            for fl, label in ((["orig", "deva"], "orig+Devanagari rewrite"), (["orig", "en"], "orig+English rewrite"),
                              (["orig", "deva", "en"], "orig+Devanagari+English rewrites")):
                setups.append((f"{rid}|{'+'.join(fl)}", "C4", comps, fl, f"{desc}; question forms {label}, RRF k=60",
                               rewrite_usd_per_1000, rewrite_ms))
    if RESULTS.exists():
        done = set(pd.read_csv(RESULTS).query("corpus_size == @size").run_id)
    else:
        done = set()
    for rid, stage, comps, fl, desc, usd, rw_ms in setups:
        ranked, ms = fused(corpus, comps, fl)
        if rw_ms and len(fl) > 1:
            qids = list(q.qid)
            ms = [m + rw_ms.get(qq, 0.0) for m, qq in zip(ms, qids)]
        sc = score_run(ranked, ch, q)
        a = sc[sc.question_group == "all"].set_index("level")
        summary.append(dict(run_id=rid, stage=stage, chunk_r5=a.loc["chunk", "recall_5"], chunk_r10=a.loc["chunk", "recall_10"],
                            chunk_mrr=a.loc["chunk", "mrr_10"], verse_r5=a.loc["verse", "recall_5"],
                            verse_r10=a.loc["verse", "recall_10"], p50=np.percentile(ms, 50), p95=np.percentile(ms, 95)))
        (run_dir(corpus) / f"fused__{rid.replace('|', '__').replace('+', '_')}.json").write_text(
            json.dumps(ranked), encoding="utf-8")
        if rid in done:
            continue
        rows += result_rows(ranked, ch, "C", stage, rid, desc, size, ms, usd,
                            notes="latency = per question on laptop1 (BM25 + one bge-m3 query encoding per form + searches"
                                  + (" + Haiku rewrite)" if usd else ")"))
    if rows:
        write_results(rows, RESULTS)
    s = pd.DataFrame(summary).sort_values("chunk_r10", ascending=False)
    print(s.round(3).to_string(index=False))
    return s


def fused_path(corpus: str, run_id: str):
    return run_dir(corpus) / f"fused__{run_id.replace('|', '__').replace('+', '_')}.json"


def write_candidates(run_id: str, corpus: str = "10k", n: int = 50):
    """shared/candidates_top50.jsonl: {"qid", "chunk_ids": top n, "setup"} per question."""
    ranked = json.loads(fused_path(corpus, run_id).read_text(encoding="utf-8"))
    q = load_questions()
    out = L.ROOT / "shared" / "candidates_top50.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for qid in q.qid:
            f.write(json.dumps({"qid": qid, "chunk_ids": ranked[qid][:n], "setup": run_id}) + "\n")
    print("wrote", out, "from", run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "run", "candidates"])
    ap.add_argument("--corpus", default="10k", choices=["10k", "full"])
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--rewrites", action="store_true", help="add the C4 question forms (needs ../local/work/rewrites.jsonl)")
    ap.add_argument("--setup", default="", help="candidates: the run_id of the best fused setup")
    a = ap.parse_args()
    if a.cmd == "embed":
        embed_corpus(a.corpus, parts=a.parts, gpu_lock=bus_gpu)
    elif a.cmd == "candidates":
        write_candidates(a.setup, a.corpus)
    elif a.rewrites:
        from src.C.rewrite import load_rewrites
        forms, ms, usd = load_rewrites()
        run_all(a.corpus, forms_extra=forms, rewrite_ms=ms, rewrite_usd_per_1000=usd)
    else:
        run_all(a.corpus)


if __name__ == "__main__":
    main()

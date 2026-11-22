"""Stage D: how much of the set-indexing gain depends on near-paraphrases of the test questions.

The test questions (session C) and the sets (session D) were both written by claude-haiku-5-5
from the same chunks, so a set question can be a near copy of a test question. For each test
question this script measures the best cosine between it and the set questions of its gold
chunk, then re-scores the set indexes with those near copies removed (per question: set vectors
of the gold chunk with cosine >= tau to that question are left out).

    python src/D/leakage.py        # GPU for ~10 s on first run (encodes 165 questions, cached)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
from dutil import WORK_D, read_jsonl, utf8_stdout  # noqa: E402
from embed import PASSAGE_MAXLEN, QUERY_MAXLEN, LazyBGEM3, cached_encode  # noqa: E402
from filters import index_langs  # noqa: E402
from stage import src_lang  # noqa: E402

TAUS = (None, 0.90, 0.85)


def main():
    utf8_stdout()
    from src.common.score import score_run
    ps = read_jsonl(WORK_D / "subcorpus.jsonl")
    pid = [p["chunk_id"] for p in ps]
    pindex = {c: i for i, c in enumerate(pid)}
    sets = read_jsonl(WORK_D / "sets_haiku.jsonl")
    q = data.questions_df()
    q = q[q.gold_chunk_id.isin(pindex)].reset_index(drop=True)
    emb = LazyBGEM3()
    try:
        P = cached_encode(emb, [data.index_text(p) for p in ps], PASSAGE_MAXLEN, "passages", WORK_D / "emb")
        Qh = cached_encode(emb, [s["question_hi"] for s in sets], QUERY_MAXLEN, "q_hi_claude-haiku-5-5", WORK_D / "emb")
        Qe = cached_encode(emb, [s["question_en"] for s in sets], QUERY_MAXLEN, "q_en_claude-haiku-5-5", WORK_D / "emb")
        T = cached_encode(emb, list(q.question), QUERY_MAXLEN, "testq", WORK_D / "emb")
    finally:
        emb.close()
    owner = np.array([pindex[s["chunk_id"]] for s in sets])
    gold = np.array([pindex[c] for c in q.gold_chunk_id])

    # similarity of each test question to the set questions of its gold chunk (all generated sets)
    S_all = np.maximum(T @ Qh.T, T @ Qe.T)                  # (n_q, n_sets)
    best_gold = np.array([S_all[i, owner == gold[i]].max() if (owner == gold[i]).any() else 0.0
                          for i in range(len(q))])
    stats = {f"share_best_gold_cos_ge_{t}": float((best_gold >= t).mean()) for t in (0.95, 0.9, 0.85, 0.8)}
    stats["median_best_gold_cos"] = float(np.median(best_gold))

    variants = {
        "passages only": [],
        "passages + all sets (hi+en)": [(i, l) for i, s in enumerate(sets) if s["kept"] for l in index_langs(s, "main")],
        "passages + all sets without filter 3 (hi+en)": [(i, l) for i, s in enumerate(sets) if s.get("kept_nof3")
                                                         for l in ("hi", "en")],
        "passages + all sets (source language only)": [(i, l) for i, s in enumerate(sets) if s["kept"]
                                                       for l in index_langs(s, "main") if l == src_lang(ps[pindex[s["chunk_id"]]])],
    }
    out = {"stats": stats, "runs": []}
    for name, vecs in variants.items():
        vi = np.array([i for i, _ in vecs], dtype=int)
        V = np.concatenate([P] + ([np.stack([(Qh if l == "hi" else Qe)[i] for i, l in vecs])] if vecs else []))
        own = np.r_[np.arange(len(pid)), owner[vi]] if vecs else np.arange(len(pid))
        is_set = np.r_[np.zeros(len(pid), bool), np.ones(len(vi), bool)]
        set_idx = np.r_[-np.ones(len(pid), int), vi]
        Sq = T @ V.T                                           # (n_q, n_vec)
        for tau in TAUS:
            ranked = {}
            for i in range(len(q)):
                s = Sq[i].copy()
                if tau is not None and vecs:
                    near = is_set & (own == gold[i]) & (Sq[i] >= tau)
                    s[near] = -np.inf
                ps_ = np.full(len(pid), -np.inf)
                np.maximum.at(ps_, own, s)
                top = np.argsort(-ps_)[:20]
                ranked[q.qid[i]] = [pid[j] for j in top]
            sc = score_run(ranked, data.chunks_df()[lambda d: d.chunk_id.isin(pindex)], q)
            row = sc[(sc.question_group == "all")]
            out["runs"].append({"setup": name, "tau": tau,
                                **{f"{r.level}_recall_10": float(r.recall_10) for r in row.itertuples()},
                                **{f"{r.level}_mrr_10": float(r.mrr_10) for r in row.itertuples()}})
            if not vecs:
                break
    (WORK_D / "leakage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    for r in out["runs"]:
        print("%-48s tau=%-5s chunk R@10 %.3f  verse R@10 %.3f  chunk MRR %.3f" % (
            r["setup"], r["tau"], r["chunk_recall_10"], r["verse_recall_10"], r["chunk_mrr_10"]))


if __name__ == "__main__":
    main()

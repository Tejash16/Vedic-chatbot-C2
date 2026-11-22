"""Stage D steps 3-6: statistics, scale-up estimate, search test, review files."""
from __future__ import annotations

import csv
import math
import random
import time
from collections import defaultdict

import numpy as np

from dutil import (BATCH_DISCOUNT, HAIKU, PRICES, SEED, n_words, norm, sentence_spans, script_of)
from embed import PASSAGE_MAXLEN, QUERY_MAXLEN, cached_encode

# ------------------------------------------------------------------ step 3: statistics


def length_bins(passages):
    w = sorted(n_words(p["text"]) for p in passages)
    if not w:
        return 0, 0
    t1 = w[len(w) // 3]
    t2 = w[(2 * len(w)) // 3]
    return t1, t2


def bin_of(nw, t1, t2):
    return "short" if nw < t1 else ("medium" if nw < t2 else "long")


def coverage(p, sets_for_p):
    """Share of the passage's sentences overlapped by at least one set's answer span."""
    sents = sentence_spans(p["text"])
    if not sents:
        return None
    spans = [s["span"] for s in sets_for_p if s.get("span")]
    cov = 0
    for a, b in sents:
        need = min(10, max(1, int(0.3 * (b - a))))
        if any(min(b, y) - max(a, x) >= need for x, y in spans):
            cov += 1
    return cov / len(sents)


def stats(passages, sets, label=""):
    """Per-bin and overall numbers for one model's sets over the given passages."""
    t1, t2 = length_bins(passages)
    by_p = defaultdict(list)
    for s in sets:
        by_p[s["chunk_id"]].append(s)
    rows = defaultdict(lambda: {"n": 0, "words": 0, "gen": 0, "kept": 0, "kept_nof3": 0, "cov": [], "cov_gen": [],
                                "cov_nof3": [], "sents": 0})
    for p in passages:
        nw = n_words(p["text"])
        ss = by_p.get(p["chunk_id"], [])
        kept = [s for s in ss if s["kept"]]
        kept_nof3 = [s for s in ss if s.get("kept_nof3")]
        c = coverage(p, kept)
        c2 = coverage(p, [s for s in ss if s.get("f1_ok")])
        c3 = coverage(p, kept_nof3)
        ns = len(sentence_spans(p["text"]))
        for key in ("all", bin_of(nw, t1, t2), "lang:" + str(p.get("lang")), "author:" + str(p.get("author_id"))):
            r = rows[key]
            r["n"] += 1
            r["words"] += nw
            r["gen"] += len(ss)
            r["kept"] += len(kept)
            r["kept_nof3"] += len(kept_nof3)
            r["sents"] += ns
            for k_, v_ in (("cov", c), ("cov_gen", c2), ("cov_nof3", c3)):
                if v_ is not None:
                    r[k_].append(v_)
    out = {}
    for k, r in rows.items():
        out[k] = {
            "passages": r["n"], "words": r["words"],
            "words_per_passage": r["words"] / max(1, r["n"]),
            "sets_generated_per_passage": r["gen"] / max(1, r["n"]),
            "sets_kept_per_passage": r["kept"] / max(1, r["n"]),
            "sets_generated_per_100_words": 100 * r["gen"] / max(1, r["words"]),
            "sets_kept_per_100_words": 100 * r["kept"] / max(1, r["words"]),
            "sets_kept_nof3_per_passage": r["kept_nof3"] / max(1, r["n"]),
            "sets_kept_nof3_per_100_words": 100 * r["kept_nof3"] / max(1, r["words"]),
            "sentences_per_passage": r["sents"] / max(1, r["n"]),
            "sentence_coverage_kept": float(np.mean(r["cov"])) if r["cov"] else None,
            "sentence_coverage_verbatim": float(np.mean(r["cov_gen"])) if r["cov_gen"] else None,
            "sentence_coverage_nof3": float(np.mean(r["cov_nof3"])) if r["cov_nof3"] else None,
        }
    out["_bins"] = {"short_below_words": t1, "long_from_words": t2}
    return out


# ------------------------------------------------------------------ step 4: scale-up


# Claude tokens per character by language group (data audit, section 7, claude-haiku-5-5 counting)
CLAUDE_TPC = {"en": 0.38, "hinglish": 0.38, "hi": 0.732, "sa": 0.848}
CLAUDE_TPC_OTHER = 0.873


def passage_ktok(text, lang):
    """Estimated Claude tokens of a passage, the size measure that works across scripts."""
    return len(norm(text)) * CLAUDE_TPC.get(lang, CLAUDE_TPC_OTHER)


def fit_tokens(passages, gen):
    """Least squares tokens = a + b * x per request, x = estimated Claude tokens of the passage,
    for input and output; overall ('all') and per script for reference."""
    pts = defaultdict(list)
    for p in passages:
        g = gen.get(p["chunk_id"])
        if not g or g.get("error"):
            continue
        x = passage_ktok(p["text"], p.get("lang"))
        sc = script_of(p["text"])
        pts[sc].append((x, g["in_tokens"], g["out_tokens"]))
        pts["all"].append((x, g["in_tokens"], g["out_tokens"]))
    fits = {}
    for sc, v in pts.items():
        a = np.array(v, dtype=float)
        X = np.c_[np.ones(len(a)), a[:, 0]]
        if len(a) >= 5:
            ci, *_ = np.linalg.lstsq(X, a[:, 1], rcond=None)
            co, *_ = np.linalg.lstsq(X, a[:, 2], rcond=None)
        else:
            ci = [a[:, 1].mean(), 0.0]
            co = [a[:, 2].mean(), 0.0]
        fits[sc] = {"in_a": float(ci[0]), "in_b": float(ci[1]), "out_a": float(co[0]), "out_b": float(co[1]),
                    "n": len(a), "mean_in": float(a[:, 1].mean()), "mean_out": float(a[:, 2].mean()),
                    "mean_x": float(a[:, 0].mean())}
    return fits


def scale_up(authors, passages, sets, gen, model=HAIKU, min_passages=15, kept_key="kept"):
    """authors: list of {author_id, author, lang, script, words, records, chunks} for the whole
    dataset. Rates come from the sub-corpus: the author's own rate if it has >= min_passages
    passages, else the rate for its script, else the overall rate."""
    by_author = defaultdict(lambda: [0, 0, 0])   # passages, words, kept
    by_script = defaultdict(lambda: [0, 0, 0])
    kept_by_p = defaultdict(int)
    for s in sets:
        if s.get(kept_key):
            kept_by_p[s["chunk_id"]] += 1
    for p in passages:
        nw = n_words(p["text"])
        k = kept_by_p[p["chunk_id"]]
        for d, key in ((by_author, str(p.get("author_id"))), (by_script, script_of(p["text"])), (by_script, "all")):
            d[key][0] += 1
            d[key][1] += nw
            d[key][2] += k
    fits = fit_tokens(passages, gen)
    f = fits["all"]
    pin, pout = PRICES[model]
    rows = []
    for a in authors:
        aid = str(a["author_id"])
        sc = a.get("script") or "deva"
        if by_author[aid][0] >= min_passages:
            src, (np_, nw, nk) = "author", by_author[aid]
        elif by_script.get(sc, [0])[0] >= min_passages:
            src, (np_, nw, nk) = "script:" + sc, by_script[sc]
        else:
            src, (np_, nw, nk) = "overall", by_script["all"]
        rate = 100 * nk / max(1, nw)
        chunks = a["chunks"]
        x_total = a["chars"] * CLAUDE_TPC.get(a.get("lang"), CLAUDE_TPC_OTHER)
        in_tok = max(chunks * f["in_a"] + x_total * f["in_b"], 0)
        out_tok = max(chunks * f["out_a"] + x_total * f["out_b"], 0)
        cost = BATCH_DISCOUNT * (in_tok * pin + out_tok * pout) / 1e6
        rows.append({**a, "rate_source": src, "sets_per_100_words": rate,
                     "est_sets": a["words"] * rate / 100, "est_requests": chunks,
                     "est_in_tokens": in_tok, "est_out_tokens": out_tok, "est_batch_usd": cost})
    return rows, fits


# ------------------------------------------------------------------ step 5: search test


def score(rankings, questions, passages, level, ks=(5, 10, 20)):
    """Own metric code, same definitions as the shared scorer: recall@k (any relevant in top k),
    MRR@10, nDCG@10 with binary relevance."""
    verse_of = {p["chunk_id"]: p.get("verse_id") for p in passages}
    by_verse = defaultdict(set)
    for p in passages:
        by_verse[p.get("verse_id")].add(p["chunk_id"])
    res = defaultdict(list)
    for q in questions:
        ranked = rankings[q["qid"]]
        if level == "chunk":
            rel = {q["gold_chunk_id"]}
        else:
            rel = by_verse.get(verse_of.get(q["gold_chunk_id"]), {q["gold_chunk_id"]})
        hits = [1 if c in rel else 0 for c in ranked[:max(ks)]]
        for k in ks:
            res["recall_%d" % k].append(1.0 if any(hits[:k]) else 0.0)
        first = next((i for i, h in enumerate(hits[:10]) if h), None)
        res["mrr_10"].append(0.0 if first is None else 1.0 / (first + 1))
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
        res["ndcg_10"].append(dcg / idcg if idcg else 0.0)
    return {k: float(np.mean(v)) if v else None for k, v in res.items()}


def best_sets(sets, k):
    """Up to k kept sets per passage with the largest margin (own-passage similarity minus the
    best other passage's, averaged over both languages): the most passage-specific sets."""
    by_p = defaultdict(list)
    for s in sets:
        if s["kept"]:
            by_p[s["chunk_id"]].append(s)
    out = []
    for v in by_p.values():
        v.sort(key=lambda s: -s["margin"])
        out.extend(v[:k])
    return out


def src_lang(p):
    """Language of the source-language question for a passage: English for Latin-script
    passages, Hindi otherwise (Devanagari Hindi and Sanskrit, and other scripts)."""
    return "en" if script_of(p["text"]) == "latin" else "hi"


def search_test(emb, passages, sets, questions, set_vecs, P, log=print):
    """passages: sub-corpus list (index order = P rows). set_vecs: {set_id: (vec_hi, vec_en)}.
    Each question vector points to its passage; passage score = max over its vectors."""
    pid = [p["chunk_id"] for p in passages]
    pindex = {c: i for i, c in enumerate(pid)}
    pmap = {p["chunk_id"]: p for p in passages}
    qs = [q for q in questions if q["gold_chunk_id"] in pindex]
    qtext = [q["question"] for q in qs]
    Qv = emb.encode(qtext, QUERY_MAXLEN) if qs else np.zeros((0, P.shape[1]), dtype=np.float32)
    # per-question encode latency, one at a time like a live query (after a warm-up)
    if qs:
        emb.encode(qtext[:2], QUERY_MAXLEN)
    lat_enc = []
    for t in qtext:
        t0 = time.perf_counter()
        emb.encode([t], QUERY_MAXLEN)
        lat_enc.append(time.perf_counter() - t0)

    from filters import index_langs

    def index_for(set_list, langs, with_passages=True, variant="main"):
        """langs: 'both' (every question of the set allowed by the filter variant) or 'src'
        (only the source-language question, if allowed)."""
        vecs, owner = [], []
        if with_passages:
            vecs.append(P)
            owner.extend(range(len(pid)))
        n_q = 0
        for s in set_list:
            vh, ve = set_vecs[s["set_id"]]
            ok = index_langs(s, variant)
            ls = ok if langs == "both" else [l for l in ok if l == src_lang(pmap[s["chunk_id"]])]
            for l in ls:
                vecs.append((vh if l == "hi" else ve)[None, :])
                owner.append(pindex[s["chunk_id"]])
                n_q += 1
        V = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, P.shape[1]), dtype=np.float32)
        owner = np.array(owner)
        order = np.argsort(owner, kind="stable")
        return V[order], owner[order]

    kept = [s for s in sets if s["kept"]]
    setups = [("passages only", [], "both", True, "main")]
    setups.append(("passages + all sets (hi+en)", kept, "both", True, "main"))
    setups.append(("passages + all sets (source language only)", kept, "src", True, "main"))
    for k in (5, 10, 20, 40):
        setups.append(("passages + best %d sets/passage (hi+en)" % k, best_sets(sets, k), "both", True, "main"))
    setups.append(("passages + best 10 sets/passage (source language only)", best_sets(sets, 10), "src", True, "main"))
    setups.append(("all sets only, no passage vectors (hi+en)", kept, "both", False, "main"))
    setups.append(("passages + all sets without filter 3 (hi+en)", [s for s in sets if s.get("kept_nof3")],
                   "both", True, "nof3"))
    setups.append(("passages + all sets, strict filter 3 (hi+en)", [s for s in sets if s.get("kept_strict")],
                   "both", True, "strict"))

    out, runs = [], []
    for name, sl, langs, wp, variant in setups:
        V, owner = index_for(sl, langs, wp, variant)
        starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
        owners_present = owner[starts]
        rankings, lat_s = {}, []
        for i, q in enumerate(qs):
            t0 = time.perf_counter()
            s = V @ Qv[i]
            ps = np.full(len(pid), -np.inf, dtype=np.float32)
            ps[owners_present] = np.maximum.reduceat(s, starts)
            top = np.argpartition(-ps, min(20, len(ps) - 1))[:20]
            top = top[np.argsort(-ps[top])]
            lat_s.append(time.perf_counter() - t0)
            rankings[q["qid"]] = [pid[j] for j in top]
        lat = np.array(lat_enc) + np.array(lat_s) if qs else np.zeros(1)
        runs.append({"setup": name, "ranked": rankings, "latencies_ms": list(lat * 1000),
                     "n_vectors": int(len(V)), "n_sets": len(sl)})
        groups = sorted({q["group"] for q in qs})
        for g in ["all"] + groups:
            sub = qs if g == "all" else [q for q in qs if q["group"] == g]
            if not sub:
                continue
            for level in ("chunk", "verse"):
                m = score(rankings, sub, passages, level)
                out.append({"setup": name, "question_group": g, "level": level, "n_questions": len(sub),
                            "n_vectors": int(len(V)), "n_sets": len(sl),
                            "p50_ms": float(np.percentile(lat, 50) * 1000),
                            "p95_ms": float(np.percentile(lat, 95) * 1000), **m})
        log("search: %-55s vectors=%6d  R@10(all,chunk)=%.3f" % (
            name, len(V), next(r["recall_10"] for r in out if r["setup"] == name and r["question_group"] == "all"
                               and r["level"] == "chunk")))
    return out, qs, runs


# ------------------------------------------------------------------ step 6: review files


def review_files(passages, sets_h, sets_s, path_csv, path_key, n_each=60):
    pmap = {p["chunk_id"]: p for p in passages}
    rnd = random.Random(SEED)
    kh = sorted([s for s in sets_h if s["kept"]], key=lambda s: s["set_id"])
    ks = sorted([s for s in sets_s if s["kept"]], key=lambda s: s["set_id"])
    pick = rnd.sample(kh, min(n_each, len(kh))) + rnd.sample(ks, min(n_each, len(ks)))
    rnd.shuffle(pick)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(path_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "verse_reference", "author", "passage", "question_hindi", "question_english",
                    "answer_quote", "right_or_wrong", "comment"])
        for i, s in enumerate(pick, 1):
            p = pmap[s["chunk_id"]]
            w.writerow(["R%03d" % i, p.get("verse_ref") or p.get("verse_id"), p.get("author") or p.get("author_id"),
                        norm(p["text"]), s["question_hi"], s["question_en"], s["answer"], "", ""])
    with open(path_key, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "model", "set_id", "chunk_id"])
        for i, s in enumerate(pick, 1):
            w.writerow(["R%03d" % i, s["model"], s["set_id"], s["chunk_id"]])
    return len([s for s in pick if s["model"] == HAIKU]), len([s for s in pick if s["model"] != HAIKU])

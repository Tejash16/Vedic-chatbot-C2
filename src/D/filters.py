"""Stage D step 2: filters, all in code.

F1 answer: the quote must be found word for word in the passage after normalization.
F2 terms:  a name or Sanskrit term in either question must appear in the passage, its header
           or the verse details.
F3 search: each question must bring back its own passage in the top 10 (bge-m3 dense, index
           of the sub-corpus passages only).
Merge:     repeats inside one passage (same answer span, or near-identical questions) keep
           the first set.

Every set records the outcome of each filter on its own, so the report can give both the
share removed at each step (in order) and the share each filter would remove alone.
"""
from __future__ import annotations

import numpy as np

from dutil import find_quote, norm

TOPK = 10
DUP_COS = 0.95


def build_sets(gen: dict, passages: dict, model: str):
    """gen: {chunk_id: parsed model output}. passages: {chunk_id: passage dict}.
    One set per point; extra sets for an already-used point and sets naming no valid point
    are counted but not kept."""
    sets, stats = [], {"points": 0, "sets_raw": 0, "sets_bad_point": 0, "sets_extra_for_point": 0,
                       "points_without_set": 0, "errors": 0}
    for cid, g in gen.items():
        if g.get("error"):
            stats["errors"] += 1
        pts = g.get("points") or []
        stats["points"] += len(pts)
        used = set()
        for s in g.get("sets") or []:
            stats["sets_raw"] += 1
            try:
                k = int(s.get("point", 0))
            except Exception:
                k = 0
            if not (1 <= k <= len(pts)):
                stats["sets_bad_point"] += 1
                continue
            if k in used:
                stats["sets_extra_for_point"] += 1
                continue
            used.add(k)
            sets.append({
                "set_id": "%s:%s:%d" % (model.split("-")[1], cid, k),
                "model": model, "chunk_id": cid, "point": k,
                "answer": pts[k - 1], "question_hi": norm(s.get("question_hi", "")),
                "question_en": norm(s.get("question_en", "")),
            })
        stats["points_without_set"] += len(pts) - len(used)
    return sets, stats


def f1_answer(sets, passages):
    for s in sets:
        span = find_quote(passages[s["chunk_id"]]["text"], s["answer"])
        s["f1_ok"] = span is not None
        s["span"] = list(span) if span else None


def f2_terms(sets, passages, checker):
    cache = {}
    for s in sets:
        cid = s["chunk_id"]
        if cid not in cache:
            p = passages[cid]
            ground = " ".join(str(p.get(k) or "") for k in
                              ("text", "header", "verse_ref", "collection", "author", "mantra", "pada",
                               "devata", "rishi", "chhanda"))
            cache[cid] = checker.grounding_keys(ground)
        miss = checker.missing(s["question_hi"], cache[cid]) + checker.missing(s["question_en"], cache[cid])
        s["f2_missing"] = miss
        s["f2_ok"] = not miss


def f3_search(sets, pid_index: dict, P: np.ndarray, Qhi: np.ndarray, Qen: np.ndarray):
    """P: passage vectors (n_passages x d), rows in pid_index order. Qhi/Qen: question vectors
    aligned with sets. Records own-passage rank and margin for each language."""
    own = np.array([pid_index[s["chunk_id"]] for s in sets], dtype=int)
    block = 2000
    for lang, Q in (("hi", Qhi), ("en", Qen)):
        for b0 in range(0, len(sets), block):
            b1 = min(len(sets), b0 + block)
            S = Q[b0:b1] @ P.T
            rows = np.arange(b1 - b0)
            own_s = S[rows, own[b0:b1]]
            rank = (S > own_s[:, None]).sum(axis=1) + 1
            S[rows, own[b0:b1]] = -np.inf
            margin = own_s - S.max(axis=1)
            for i in range(b1 - b0):
                sets[b0 + i]["rank_" + lang] = int(rank[i])
                sets[b0 + i]["margin_" + lang] = float(margin[i])
    for s in sets:
        s["f3_hi_ok"] = s["rank_hi"] <= TOPK
        s["f3_en_ok"] = s["rank_en"] <= TOPK
        # Per question: a question that misses is dropped from the index; the set goes only when
        # both of its questions miss. (Strict variant, both must hit, kept for comparison.)
        s["f3_ok"] = s["f3_hi_ok"] or s["f3_en_ok"]
        s["f3_strict_ok"] = s["f3_hi_ok"] and s["f3_en_ok"]
        ok = [s["margin_" + l] for l in ("hi", "en") if s["f3_%s_ok" % l]]
        s["margin"] = sum(ok) / len(ok) if ok else (s["margin_hi"] + s["margin_en"]) / 2


def merge_repeats(sets, Qhi, Qen, order_mask, key="dup_of"):
    """Among sets with order_mask True, mark later repeats inside the same passage in s[key].
    A repeat has nearly the same answer span (overlap > 90 % of the shorter and > 50 % of the
    longer span), or both questions near-identical (cosine >= DUP_COS)."""
    by_pid = {}
    for i, s in enumerate(sets):
        s[key] = None
        if order_mask[i]:
            by_pid.setdefault(s["chunk_id"], []).append(i)
    for pid, idx in by_pid.items():
        kept = []
        for i in idx:
            si = sets[i]
            dup = None
            for j in kept:
                sj = sets[j]
                if si["span"] and sj["span"]:
                    a0, a1 = si["span"]
                    b0, b1 = sj["span"]
                    inter = max(0, min(a1, b1) - max(a0, b0))
                    if inter / max(1, min(a1 - a0, b1 - b0)) > 0.9 and inter / max(1, max(a1 - a0, b1 - b0)) > 0.5:
                        dup = sj["set_id"]
                        break
                if float(Qhi[i] @ Qhi[j]) >= DUP_COS and float(Qen[i] @ Qen[j]) >= DUP_COS:
                    dup = sj["set_id"]
                    break
            if dup:
                si[key] = dup
            else:
                kept.append(i)


def apply_all(sets, passages, checker, pid_index, P, Qhi, Qen):
    f1_answer(sets, passages)
    f2_terms(sets, passages, checker)
    f3_search(sets, pid_index, P, Qhi, Qen)
    variants = {  # name: (filter mask, merge key, kept key)
        "main": ([s["f1_ok"] and s["f2_ok"] and s["f3_ok"] for s in sets], "dup_of", "kept"),
        "nof3": ([s["f1_ok"] and s["f2_ok"] for s in sets], "dup_of_nof3", "kept_nof3"),
        "strict": ([s["f1_ok"] and s["f2_ok"] and s["f3_strict_ok"] for s in sets], "dup_of_strict", "kept_strict"),
    }
    for mask, dkey, kkey in variants.values():
        alive = np.array(mask, dtype=bool)
        merge_repeats(sets, Qhi, Qen, alive, dkey)
        for i, s in enumerate(sets):
            s[kkey] = bool(alive[i] and s[dkey] is None)
    return sets


def index_langs(s, variant="main"):
    """Question languages of a kept set that go into the index: under the main rule only the
    questions that passed filter 3; without F3 or under the strict rule, both."""
    if variant == "main":
        return [l for l in ("hi", "en") if s["f3_%s_ok" % l]]
    return ["hi", "en"]


def filter_report(sets):
    n = len(sets)
    if n == 0:
        return {}
    r = {"sets_generated": n}
    # alone
    for k in ("f1_ok", "f2_ok", "f3_ok", "f3_hi_ok", "f3_en_ok", "f3_strict_ok"):
        r["alone_fail_" + k[:-3]] = sum(not s[k] for s in sets) / n
    # filter 3 at question level, among sets that passed F1 and F2
    b12 = [s for s in sets if s["f1_ok"] and s["f2_ok"]]
    if b12:
        r["f3_questions_removed_share"] = sum((not s["f3_hi_ok"]) + (not s["f3_en_ok"]) for s in b12) / (2 * len(b12))
    r["kept_nof3"] = sum(s.get("kept_nof3", False) for s in sets)
    r["kept_strict"] = sum(s.get("kept_strict", False) for s in sets)
    # in order F1 -> F2 -> F3 -> merge
    left = n
    a = [s for s in sets if s["f1_ok"]]
    r["seq_removed_f1"] = (left - len(a)) / n
    left = len(a)
    b = [s for s in a if s["f2_ok"]]
    r["seq_removed_f2"] = (left - len(b)) / n
    left = len(b)
    c = [s for s in b if s["f3_ok"]]
    r["seq_removed_f3"] = (left - len(c)) / n
    left = len(c)
    d = [s for s in c if s["dup_of"] is None]
    r["seq_removed_merge"] = (left - len(d)) / n
    r["sets_kept"] = len(d)
    r["kept_share"] = len(d) / n
    r["kept_question_vectors"] = sum(len(index_langs(s)) for s in d)
    return r

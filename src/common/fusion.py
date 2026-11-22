"""Reciprocal rank fusion (C3): k = 60 everywhere."""
from __future__ import annotations

RRF_K = 60


def rrf(rankings: list[list[str]], k: int = RRF_K, top: int = 100, weights: list[float] | None = None) -> list[str]:
    """Fuse ranked lists of ids (best first). score(id) = sum_i w_i / (k + rank_i), rank from 1.
    Ties are broken by id so the result is deterministic."""
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for w, ids in zip(weights, rankings):
        for r, c in enumerate(ids, 1):
            score[c] = score.get(c, 0.0) + w / (k + r)
    return [c for c, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:top]

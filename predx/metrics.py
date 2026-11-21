"""Retrieval metrics, written once so every system (BM25, LSA, zero-shot, fine-tuned) is scored identically.

Similarity (reviewer): cosine on L2-normalised embeddings is the primary scoring function; dot product is a cheap
ablation once embeddings exist. `score_matrix` is the single place that choice is made.

Rank convention: rank of a gold item = 1 + (#candidates scoring strictly higher) + (#other candidates tied with it),
i.e. PESSIMISTIC ties. This matters for BM25 where many candidates score exactly 0.
"""
from __future__ import annotations

import math

import numpy as np


def score_matrix(q: np.ndarray, d: np.ndarray, kind: str = "cosine") -> np.ndarray:
    q, d = np.asarray(q, dtype=np.float32), np.asarray(d, dtype=np.float32)
    if kind == "cosine":
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    elif kind != "dot":
        raise ValueError(f"unknown similarity {kind!r}")
    return q @ d.T


def gold_ranks_from_scores(scores: np.ndarray, n_cand: int, gold_idx) -> list[int]:
    """scores: (n_corpus,) for ONE query; only the first n_cand entries (the date-filtered candidates) compete."""
    row = scores[:n_cand]
    ranks = []
    for g in gold_idx:
        gs = row[g]
        ranks.append(1 + int((row > gs).sum()) + int((row == gs).sum()) - 1)
    return ranks


def per_query_metrics(rank_lists, recall_ks, hit_ks, ndcg_k: int = 10) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    for k in recall_ks:
        out[f"recall@{k}"] = []
    for k in hit_ks:
        out[f"hit@{k}"] = []
    out["mrr"] = []
    out[f"ndcg@{ndcg_k}"] = []
    for ranks in rank_lists:
        r = np.asarray(ranks, dtype=float)
        n = len(r)
        for k in recall_ks:
            out[f"recall@{k}"].append(float((r <= k).sum()) / n if n else 0.0)
        for k in hit_ks:
            out[f"hit@{k}"].append(float((r <= k).any()) if n else 0.0)
        out["mrr"].append(float(1.0 / r.min()) if n and np.isfinite(r.min()) else 0.0)
        dcg = sum(1.0 / math.log2(x + 1) for x in r if x <= ndcg_k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n, ndcg_k) + 1))
        out[f"ndcg@{ndcg_k}"].append(dcg / idcg if idcg else 0.0)
    return {k: np.asarray(v) for k, v in out.items()}


def summarize(pq: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean and bootstrap 95% CI over queries for each metric."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(pq.values()))) if pq else 0
    res = {"n_queries": n}
    if n == 0:
        return res
    idx = rng.integers(0, n, size=(n_boot, n))
    for name, arr in pq.items():
        boots = arr[idx].mean(axis=1)
        res[name] = {"mean": float(arr.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}
    return res


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))

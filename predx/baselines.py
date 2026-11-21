"""Sanity baselines run on the BUILT dataset, with the exact evaluation protocol (date-filtered candidates, metrics).

Purpose here is not to produce the paper's baseline numbers; it is to prove the data is wired correctly:
  * BM25 far above random  -> text and labels line up.
  * BM25 ~ random          -> something is broken (wrong join, masked text, bad dates).
  * product-code popularity -> how much of the task is solved by device category alone (proposal baseline 4).
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np

from .dataset import load_dataset
from .metrics import gold_ranks_from_scores, per_query_metrics, summarize


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import CountVectorizer

        self.vec = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-]+\b", dtype=np.float32)
        X = self.vec.fit_transform(docs).tocsr()
        N = X.shape[0]
        df = np.diff(X.tocsc().indptr).astype(np.float32)
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(axis=1)).ravel()
        avg = float(dl.mean()) if N else 1.0
        coo = X.tocoo()
        denom = coo.data + k1 * (1 - b + b * dl[coo.row] / avg)
        data = coo.data * (k1 + 1) / denom * idf[coo.col]
        self.W = sp.csr_matrix((data.astype(np.float32), (coo.row, coo.col)), shape=X.shape)

    def scores(self, queries: list[str]) -> np.ndarray:
        Q = self.vec.transform(queries)
        Q.data[:] = 1.0
        return (Q @ self.W.T).toarray()


def _text(r) -> str:
    return (r["ifu"] + " " + r["description"]).strip()


def run_baselines(cfg: dict, paths, split: str = "test", systems=("bm25", "product_code_popularity")) -> dict:
    corpus, queries, pairs, _ = load_dataset(paths)
    es = cfg["eval_spec"]
    cidx = {r["k"]: i for i, r in enumerate(corpus)}
    qs = [q for q in queries[split] if q["retrievable"]]
    rng = np.random.default_rng(cfg["seed"])
    cap = cfg["baselines"]["max_queries"]
    if len(qs) > cap:
        qs = [qs[i] for i in sorted(rng.choice(len(qs), cap, replace=False))]
    out: dict = {"split": split, "n_queries": len(qs)}
    if not qs or not corpus:
        return out
    gold_idx = [[cidx[k] for k in q["gold_in_corpus"]] for q in qs]
    rk, hk, nk = es["recall_k"], es["hit_k"], es["ndcg_k"]

    if "bm25" in systems:
        bm = BM25([_text(r) for r in corpus], cfg["baselines"]["bm25"]["k1"], cfg["baselines"]["bm25"]["b"])
        ranks = []
        for s in range(0, len(qs), 256):
            S = bm.scores([_text(q) for q in qs[s: s + 256]])
            for i, q in enumerate(qs[s: s + 256]):
                ranks.append(gold_ranks_from_scores(S[i], q["n_candidates"], gold_idx[s + i]))
        out["bm25"] = summarize(per_query_metrics(ranks, rk, hk, nk), es["bootstrap"], cfg["seed"])

    if "product_code_popularity" in systems:
        cites = Counter(pr["p"] for pr in pairs)
        by_code: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(corpus):
            by_code[r["product_code"]].append(i)
        cite_arr = np.zeros(len(corpus), dtype=np.int64)
        for k, c in cites.items():
            cite_arr[cidx[k]] = c
        ranks = []
        for q, gi in zip(qs, gold_idx):
            arr = np.asarray(by_code.get(q["product_code"], []), dtype=np.int64)
            arr = arr[: bisect_left(arr.tolist(), q["n_candidates"])] if len(arr) else arr
            order = np.lexsort((-arr, -cite_arr[arr])) if len(arr) else arr
            pos = {int(a): r + 1 for r, a in enumerate(arr[order])}
            ranks.append([pos.get(g, float("inf")) for g in gi])
        out["product_code_popularity"] = summarize(per_query_metrics(ranks, rk, hk, nk), es["bootstrap"], cfg["seed"])

    # analytic random baseline (expected values), for context
    ncand = np.array([q["n_candidates"] for q in qs], dtype=float)
    g = np.array([len(q["gold_in_corpus"]) for q in qs], dtype=float)
    out["random_expected"] = {
        **{f"recall@{k}": float(np.mean(np.minimum(1.0, k / np.maximum(ncand, 1)))) for k in rk},
        **{f"hit@{k}": float(np.mean(1 - (1 - np.minimum(1.0, k / np.maximum(ncand, 1))) ** g)) for k in hk},
    }
    return out

"""CONCEPT TEST: truncation never overflows the model window, and metrics behave on hand-checkable examples."""
import math

import numpy as np

from predx.audit import _alloc_vec
from predx.metrics import gold_ranks_from_scores, per_query_metrics, score_matrix, summarize, wilson
from predx.tokens import allocate, build_input, chunk_windows, load_tokenizer, overhead_tokens

MODEL = {"name": "t", "hf_id": "does-not-exist/nope", "max_len": 64, "query_prefix": "q: ", "doc_prefix": "d: "}
TMPL = {"ifu_header": "Indications for Use: ", "desc_header": "Device Description: ", "separator": "\n"}


def tok():
    t = load_tokenizer({**MODEL, "hf_id": "does-not-exist/nope"}, allow_approx=True)
    assert t.approx          # offline: clearly flagged as approximate
    return t


def test_budget_policy_protects_the_indications_and_respects_max_len():
    t = tok()
    ifu = " ".join(["indicated"] * 30)
    desc = " ".join(["component"] * 400)
    text, info = build_input(ifu, desc, t, MODEL, TMPL, "budget", ifu_budget=20)
    assert len(t.ids(text)) + t.n_special <= MODEL["max_len"] + 2     # tiny slack: decode/re-encode boundary effects
    assert info["kept_ifu"] >= 20 and info["desc_truncated"]
    head, hinfo = build_input(ifu, desc, t, MODEL, TMPL, "head", ifu_budget=20)
    assert hinfo["kept_ifu"] == info["avail"] or hinfo["kept_ifu"] >= info["kept_ifu"]


def test_head_policy_lets_a_long_indications_starve_the_description():
    t = tok()
    ifu = " ".join(["indicated"] * 200)
    _, h = build_input(ifu, "short description here", t, MODEL, TMPL, "head", 20)
    _, b = build_input(ifu, "short description here", t, MODEL, TMPL, "budget", 20)
    assert h["kept_desc"] == 0 < b["kept_desc"] or b["kept_desc"] >= h["kept_desc"]


def test_spare_budget_flows_back_and_vectorised_matches_scalar():
    assert allocate(50, 10, 100, "budget", 20) == (50, 10)          # nothing needs cutting
    assert allocate(50, 200, 100, "budget", 20) == (20, 80)
    ifu = np.array([5, 50, 300, 0]); desc = np.array([0, 10, 500, 90])
    for pol in ("head", "budget"):
        ki, kd = _alloc_vec(ifu, desc, 100, pol, 20)
        for i in range(4):
            assert (int(ki[i]), int(kd[i])) == allocate(int(ifu[i]), int(desc[i]), 100, pol, 20)


def test_chunk_windows_cover_the_sequence():
    w = chunk_windows(1000, 256, 128)
    assert w[0] == (0, 256) and w[-1][1] == 1000
    covered = set()
    for a, b in w:
        covered |= set(range(a, b))
    assert covered == set(range(1000)) and chunk_windows(100, 256, 128) == [(0, 100)]


def test_ranks_use_pessimistic_ties():
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
    assert gold_ranks_from_scores(scores, 5, [0]) == [5]            # gold ties with 3 zeros and loses to 0.9
    assert gold_ranks_from_scores(scores, 4, [0]) == [4]            # the 0.9 doc is not yet cleared: not a candidate
    assert gold_ranks_from_scores(np.array([.9, .5, .1]), 3, [0, 2]) == [1, 3]


def test_metrics_by_hand():
    pq = per_query_metrics([[1, 20], [3], [float("inf")]], recall_ks=[1, 5, 10], hit_ks=[1, 10], ndcg_k=10)
    assert np.allclose(pq["recall@1"], [0.5, 0, 0]) and np.allclose(pq["recall@10"], [0.5, 1, 0])
    assert np.allclose(pq["hit@1"], [1, 0, 0]) and np.allclose(pq["hit@10"], [1, 1, 0])
    assert np.allclose(pq["mrr"], [1.0, 1 / 3, 0.0])
    idcg2 = 1 + 1 / math.log2(3)
    assert np.isclose(pq["ndcg@10"][0], 1.0 / idcg2) and np.isclose(pq["ndcg@10"][1], (1 / math.log2(4)) / 1.0)


def test_oracle_hits_the_ceiling_and_bootstrap_is_sane():
    ranks = [[1]] * 40 + [[1, 2]] * 40
    pq = per_query_metrics(ranks, [1, 10], [1], 10)
    s = summarize(pq, n_boot=200, seed=1)
    assert s["recall@10"]["mean"] == 1.0 and s["recall@10"]["lo"] == s["recall@10"]["hi"] == 1.0
    assert 0 < s["recall@1"]["mean"] <= 1
    lo, hi = wilson(30, 100)
    assert lo < 0.3 < hi


def test_cosine_vs_dot_similarity():
    q = np.array([[3.0, 4.0]]); d = np.array([[3.0, 4.0], [0.6, 0.8], [0.0, 1.0]])
    cos = score_matrix(q, d, "cosine")[0]
    dot = score_matrix(q, d, "dot")[0]
    assert np.allclose(cos[:2], 1.0) and cos[2] < 1
    assert dot[0] > dot[1]                       # dot product rewards the longer vector; cosine does not

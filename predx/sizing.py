"""`predx sizing`: how much gold-standard data should we expect, what shape does it have, and is the plan realistic?

This is a PLANNING MODEL, not a measurement. Its inputs fall in three groups and the report keeps them apart:
  * COUNTED   - openFDA decision counts (statement vs summary) per period. Built in (retrieved 2026-11-19) and, when
                `data/meta` exists (after `predx meta`), recomputed from the real download.
  * PUBLISHED - predicate-age distribution and predicates per clearance from the literature (see config / README).
  * ASSUMED   - extraction yield (how many PDFs parse, how many cite K-numbers). These are the numbers the pilot audit
                replaces: re-run with e.g. `--set sizing.scenarios.mid.usable_subject=0.9 --set sizing.scenarios.mid.pred_rate=0.85`.
It answers: training pairs, retrievable test queries, recall ceiling, candidate-pool size, test-set precision, smallest
detectable model difference, rough GPU time, and what truncating the corpus start year would cost.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# openFDA 510(k), count=statement_or_summary.exact for decision_date ranges, retrieved 2026-11-19.
# (summary, statement, blank).  Sum of summaries 94,291 and of all records 176,683 vs the proposal's 94,143 / 176,529 (monthly refresh).
BUILTIN_COUNTS = {
    "1976-1995": (7684, 12384, 55409),
    "1996-2000": (12643, 6398, 457),
    "2001-2005": (13934, 3245, 45),
    "2006-2010": (13629, 1421, 22),
    "2011-2014": (11619, 811, 83),
    "2015-2023": (25987, 1400, 270),
    "2024-2024": (2935, 147, 47),
    "2025-2026": (5860, 193, 60),
}
SPLIT_BUCKET = {"train": "2015-2023", "dev": "2024-2024", "test": "2025-2026"}
SPLIT_YEARS = {"train": [(y, 1 / 9) for y in range(2015, 2024)], "dev": [(2024, 1.0)], "test": [(2025, 0.53), (2026, 0.47)]}
SE_SHARE = 0.99          # De Novo grants (DENG, ~1%) sit in the same file but cannot be subjects or predicates of a 510(k)
SCENARIOS = {            # ASSUMED extraction yield; replace with the pilot's measured values
    "low":  dict(usable_subject=0.85, pred_rate=0.80, preds_per_doc=1.5, usable_candidate=0.85),
    "mid":  dict(usable_subject=0.92, pred_rate=0.88, preds_per_doc=1.8, usable_candidate=0.92),
    "high": dict(usable_subject=0.97, pred_rate=0.95, preds_per_doc=2.1, usable_candidate=0.97),
}
# PUBLISHED: 200 consecutive traditional 510(k)s -> 393 predicates, mean age 74 months, median 53 months (FDA Law Blog, 2019).
AGE_MEDIAN_Y, AGE_MEAN_Y = 53 / 12, 74 / 12
AVG_TOKENS, TOK_PER_S = 220, {"A100": 50_000, "L4": 15_000}      # ASSUMED (average text length; training throughput of a ~300M encoder)


def _bucket_years(b: str) -> tuple[int, int]:
    a, z = b.split("-")
    return int(a), int(z)


def counts_from_meta(paths) -> dict | None:
    """Recompute the bucket table from the real openFDA download (summary / statement / blank, SE decisions only not applied)."""
    if not Path(paths.meta).exists():
        return None
    from .io import read_jsonl

    out = {b: [0, 0, 0] for b in BUILTIN_COUNTS}
    for r in read_jsonl(paths.meta):
        d = (r.get("decision_date") or "")[:4]
        if not d.isdigit():
            continue
        y = int(d)
        for b in out:
            a, z = _bucket_years(b)
            if a <= y <= z:
                s = (r.get("statement_or_summary") or "").upper()
                out[b][0 if s == "SUMMARY" else 1 if s == "STATEMENT" else 2] += 1
    return {b: tuple(v) for b, v in out.items()}


def summary_rate_by_year(counts: dict) -> dict[int, float]:
    rate = {}
    for b, (s, st, bl) in counts.items():
        a, z = _bucket_years(b)
        for y in range(a, z + 1):
            rate[y] = s / max(s + st + bl, 1)
    return rate


def age_weights(step: float = 0.05, max_age: float = 50.0):
    sigma = math.sqrt(2 * math.log(AGE_MEAN_Y / AGE_MEDIAN_Y))
    mu = math.log(AGE_MEDIAN_Y)
    ages = np.arange(step / 2, max_age, step)
    pdf = np.exp(-((np.log(ages) - mu) ** 2) / (2 * sigma**2)) / (ages * sigma * math.sqrt(2 * math.pi))
    return ages, pdf / pdf.sum(), sigma


def coverage(split: str, rate: dict[int, float], start_year: int = 1976) -> dict:
    """P(a cited predicate has a published summary decided in or after `start_year`)  and  P(its decision year >= 2024)."""
    ages, w, _ = age_weights()
    cov = unseen = 0.0
    for y, wy in SPLIT_YEARS[split]:
        t = y + 0.5
        py = t - ages
        r = np.array([rate.get(int(math.floor(v)), 0.0) if v >= start_year else 0.0 for v in py])
        cov += wy * float((w * r).sum())
        unseen += wy * float((w * (py >= 2024.0)).sum())
    return {"summary_coverage": cov, "share_pred_2024_plus": unseen}


def corpus_size_at(year: float, counts: dict, usable: float) -> float:
    n = 0.0
    for b, (s, _st, _bl) in counts.items():
        a, z = _bucket_years(b)
        frac = min(max((year - a) / (z - a + 1), 0.0), 1.0)
        n += s * frac
    return n * SE_SHARE * usable


def summaries_from(year: int, counts: dict) -> float:
    tot = 0.0
    for b, (s, _st, _bl) in counts.items():
        a, z = _bucket_years(b)
        tot += s * min(1.0, max(0.0, (z + 1 - year) / (z - a + 1)))
    return tot


def wilson_halfwidth(p: float, n: float) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / max(n, 1))


def funnel(counts: dict, params: dict, start_year: int = 1976) -> dict:
    rate = summary_rate_by_year(counts)
    out = {}
    for sp, b in SPLIT_BUCKET.items():
        s = counts[b][0] * SE_SHARE
        usable = s * params["usable_subject"]
        with_pred = usable * params["pred_rate"]
        pairs = with_pred * params["preds_per_doc"]
        cv = coverage(sp, rate, start_year)
        cov = cv["summary_coverage"] * params["usable_candidate"]
        m = params["preds_per_doc"]
        out[sp] = {
            "subjects_with_summary": s, "usable_subjects": usable, "subjects_with_predicate": with_pred,
            "gold_pairs": pairs, "pair_coverage": cov, "gold_pairs_in_corpus": pairs * cov,
            "retrievable_queries": with_pred * (1 - (1 - cov) ** m), "share_gold_decided_2024_plus": cv["share_pred_2024_plus"],
        }
    return out


def run_sizing(cfg: dict, paths) -> dict:
    counts = counts_from_meta(paths)
    src = "recomputed from data/meta (your openFDA download)"
    if counts is None or sum(v[0] for v in counts.values()) < 50_000:      # demo / partial metadata
        counts, src = dict(BUILTIN_COUNTS), "built-in openFDA counts retrieved 2026-11-19"
    scen = {k: dict(v) for k, v in SCENARIOS.items()}
    for name, over in ((cfg.get("sizing") or {}).get("scenarios") or {}).items():      # e.g. --set sizing.scenarios.mid.pred_rate=0.9
        scen.setdefault(name, dict(SCENARIOS["mid"])).update({k: float(v) for k, v in over.items()})
    rep: dict = {"counts_source": src, "counts": counts, "scenarios": {}, "notes": []}
    for name, p in scen.items():
        f = funnel(counts, p)
        te = f["test"]["retrievable_queries"]
        pool = {"2015": corpus_size_at(2015.5, counts, p["usable_candidate"]), "2020": corpus_size_at(2020.5, counts, p["usable_candidate"]),
                "2025": corpus_size_at(2025.5, counts, p["usable_candidate"]), "2026-end": corpus_size_at(2027.0, counts, p["usable_candidate"])}
        tr_pairs = f["train"]["gold_pairs_in_corpus"]
        rep["scenarios"][name] = {
            "params": p, "funnel": f, "candidate_pool": pool,
            "ci_halfwidth_recall10_test": {str(pp): wilson_halfwidth(pp, te) for pp in (0.2, 0.4, 0.6)},
            "paired_mde_pts": {str(d): 100 * (1.96 + 0.84) * math.sqrt(d / max(te, 1)) for d in (0.1, 0.2, 0.3)},
            "compute": {
                "train_tokens_per_epoch_inbatch_M": tr_pairs * 2 * AVG_TOKENS / 1e6,
                "train_tokens_per_epoch_with_1_hard_neg_M": tr_pairs * 3 * AVG_TOKENS / 1e6,
                "minutes_per_epoch_with_1_hard_neg": {g: tr_pairs * 3 * AVG_TOKENS / v / 60 for g, v in TOK_PER_S.items()},
                "corpus_embed_minutes": {g: pool["2026-end"] * AVG_TOKENS / (3 * v) / 60 for g, v in TOK_PER_S.items()},
            },
        }
    # what a later corpus start year would cost (mid scenario)
    mid = SCENARIOS["mid"]
    rate = summary_rate_by_year(counts)
    full_cov = {sp: coverage(sp, rate, 1976)["summary_coverage"] * mid["usable_candidate"] for sp in SPLIT_BUCKET}
    rows = []
    for y in (1976, 1996, 2001, 2006, 2011):
        c = {sp: coverage(sp, rate, y)["summary_coverage"] * mid["usable_candidate"] for sp in SPLIT_BUCKET}
        docs = summaries_from(y, counts)
        rows.append({"start_year": y, "docs_to_process": docs, "pair_coverage_train": c["train"], "pair_coverage_test": c["test"],
                     "relative_to_full_test": c["test"] / full_cov["test"]})
    rep["corpus_start"] = rows
    return rep


def _fmt(x, d=0):
    return f"{x:,.{d}f}"


def render_markdown(rep: dict) -> str:
    L = ["# Data sizing: how much gold-standard data, what shape, is the plan realistic?", "",
         f"_Planning model. Counts: **{rep['counts_source']}**. Extraction yields are ASSUMED until you replace them with the pilot's measured rates (`--set sizing.scenarios.mid.pred_rate=0.9` ...)._", "",
         "## 1. Counted (openFDA): decisions by period", "", "| period | summary | statement | blank | summary share |", "|---|---|---|---|---|"]
    for b, (s, st, bl) in rep["counts"].items():
        L.append(f"| {b} | {_fmt(s)} | {_fmt(st)} | {_fmt(bl)} | {s / max(s + st + bl, 1):.1%} |")
    tot_s = sum(v[0] for v in rep["counts"].values())
    L += ["", f"Total summaries {_fmt(tot_s)}. Splits: train 2015-2023, dev 2024, test 2025-2026 (by the subject's decision date).", ""]
    L += ["## 2. Expected gold-standard data (subject -> cited predicate pairs)", "",
          "Scenario assumptions: usable_subject = IFU+description extracted; pred_rate = at least one K-number found; preds_per_doc = primary+additional per document "
          "that cites any; usable_candidate = a cited predicate's own PDF parses. `pair_coverage` also folds in how many *older* predicates have a published summary at all.", ""]
    for name, sc in rep["scenarios"].items():
        p = sc["params"]
        L.append(f"**{name}** ({', '.join(f'{k}={v}' for k, v in p.items())})")
        L += ["", "| split | subjects w/ summary | usable | cite a predicate | gold pairs | pair coverage | pairs in corpus | **retrievable queries** | gold decided 2024+ |",
              "|---|---|---|---|---|---|---|---|---|"]
        for sp, f in sc["funnel"].items():
            L.append(f"| {sp} | {_fmt(f['subjects_with_summary'])} | {_fmt(f['usable_subjects'])} | {_fmt(f['subjects_with_predicate'])} | {_fmt(f['gold_pairs'])} | "
                     f"{f['pair_coverage']:.0%} | {_fmt(f['gold_pairs_in_corpus'])} | **{_fmt(f['retrievable_queries'])}** | {f['share_gold_decided_2024_plus']:.0%} |")
        L.append("")
    L += ["## 3. Search space and test-set precision", ""]
    L += ["| scenario | candidate pool at 2015 / 2020 / 2025 / end-2026 | Recall@10 CI half-width at 0.2 / 0.4 / 0.6 (test) | smallest detectable A-vs-B gap, pts (discordant 10% / 20% / 30%) |", "|---|---|---|---|"]
    for name, sc in rep["scenarios"].items():
        pl = sc["candidate_pool"]
        L.append(f"| {name} | {_fmt(pl['2015'])} / {_fmt(pl['2020'])} / {_fmt(pl['2025'])} / {_fmt(pl['2026-end'])} | "
                 + " / ".join(f"±{100 * v:.1f} pts" for v in sc['ci_halfwidth_recall10_test'].values()) + " | "
                 + " / ".join(f"{v:.1f}" for v in sc['paired_mde_pts'].values()) + " |")
    L += ["", "Random Recall@10 over a pool of ~80,000 is about 0.0001, so every non-trivial number is a large multiple of chance; the *comparisons that matter* "
          "(fine-tuned vs zero-shot vs BM25) are between methods that both beat chance by a wide margin.", ""]
    L += ["## 4. Rough compute (ASSUMED throughput: A100 50k tokens/s, L4 15k tokens/s, 220 tokens per text)", "",
          "| scenario | M tokens / epoch (in-batch) | minutes / epoch with 1 hard negative per pair (A100 / L4) | minutes to embed the corpus (A100 / L4) |", "|---|---|---|---|"]
    for name, sc in rep["scenarios"].items():
        c = sc["compute"]
        L.append(f"| {name} | {c['train_tokens_per_epoch_inbatch_M']:.1f} | {c['minutes_per_epoch_with_1_hard_neg']['A100']:.0f} / {c['minutes_per_epoch_with_1_hard_neg']['L4']:.0f} | "
                 f"{c['corpus_embed_minutes']['A100']:.0f} / {c['corpus_embed_minutes']['L4']:.0f} |")
    L += ["", "## 5. What starting the corpus later would cost (mid scenario)", "",
          "| corpus starts | summaries to download+parse | train pair coverage | test pair coverage | test ceiling vs full corpus |", "|---|---|---|---|---|"]
    for r in rep["corpus_start"]:
        L.append(f"| {r['start_year'] if r['start_year'] > 1976 else 'all years'} | {_fmt(r['docs_to_process'])} | {r['pair_coverage_train']:.0%} | {r['pair_coverage_test']:.0%} | {r['relative_to_full_test']:.0%} |")
    L += ["", "Older documents are the ones most likely to be scanned (OCR is the expensive and least accurate step), so the rows above are the trade-off between recall ceiling and extraction risk.", ""]
    return "\n".join(L)


def write_report(rep: dict, paths) -> Path:
    Path(paths.reports).mkdir(parents=True, exist_ok=True)
    (Path(paths.reports) / "sizing.json").write_text(json.dumps(rep, indent=1, default=str))
    out = Path(paths.reports) / "sizing.md"
    out.write_text(render_markdown(rep), encoding="utf-8")
    return out

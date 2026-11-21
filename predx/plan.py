"""Queues: which K-numbers to fetch/extract in each stage.

  pilot    : stratified-by-year sample of subjects from one specialty panel (config pilot.*)
  subjects : every SE + summary device decided inside a split window (2015-2026 by default)
  closure  : predicates cited by already-parsed subjects (1 hop) that are not yet extracted
  full     : every remaining SE + summary device (needed for an UNBIASED candidate corpus, see README)
"""
from __future__ import annotations

import logging
import math
import random
from collections import defaultdict

from .io import read_jsonl
from .knumbers import is_510k
from .metadata import has_summary, is_se, load_meta

log = logging.getLogger("predx.plan")


def split_windows(cfg: dict):
    return {k: tuple(v) for k, v in cfg["splits"].items()}


def in_any_window(date: str | None, cfg: dict) -> str | None:
    if not date:
        return None
    for name, (a, b) in split_windows(cfg).items():
        if a <= date <= b:
            return name
    return None


def _write_queue(paths, name: str, ks) -> list[str]:
    ks = sorted(set(ks))
    p = paths.queue(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(ks) + ("\n" if ks else ""))
    log.info("queue %s: %d K-numbers -> %s", name, len(ks), p)
    return ks


def read_queue(paths, name: str) -> list[str]:
    p = paths.queue(name)
    return [x.strip() for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def eligible(meta: dict, cfg: dict):
    return [m for m in meta.values() if is_se(m, cfg) and has_summary(m) and m.get("decision_date")]


def plan_pilot(cfg, paths, panel=None, n=None):
    meta = load_meta(paths)
    panel = (panel or cfg["pilot"]["panel"]).upper()
    n = n or cfg["pilot"]["n_subjects"]
    pool = [m for m in eligible(meta, cfg) if m["panel"] == panel and in_any_window(m["decision_date"], cfg)]
    by_year = defaultdict(list)
    for m in pool:
        by_year[m["year"]].append(m["k"])
    rng = random.Random(cfg["seed"])
    quota = math.ceil(n / max(1, len(by_year)))
    pick = []
    for y in sorted(by_year):
        ks = sorted(by_year[y])
        rng.shuffle(ks)
        pick += ks[:quota]
    rng.shuffle(pick)
    pick = pick[:n]
    log.info("pilot: panel=%s pool=%d years=%d -> %d subjects", panel, len(pool), len(by_year), len(pick))
    return _write_queue(paths, "pilot_subjects", pick)


def plan_subjects(cfg, paths):
    meta = load_meta(paths)
    ks = [m["k"] for m in eligible(meta, cfg) if in_any_window(m["decision_date"], cfg)]
    return _write_queue(paths, "subjects", ks)


def plan_closure(cfg, paths, of_queue: str | None = None, name: str = "closure"):
    meta = load_meta(paths)
    parsed = {r["k"]: r for r in read_jsonl(paths.parsed)}
    base = set(read_queue(paths, of_queue)) if of_queue else None
    need = set()
    for k, p in parsed.items():
        if base is not None and k not in base:
            continue
        for x in p.get("predicates", []):
            if x["role"] in ("primary", "additional") and is_510k(x["k"]):
                m = meta.get(x["k"])
                if m and is_se(m, cfg) and has_summary(m) and not paths.text(x["k"]).exists():
                    need.add(x["k"])
    return _write_queue(paths, name, need)


def plan_full(cfg, paths):
    meta = load_meta(paths)
    mind = cfg["corpus"].get("min_decision_date")
    ks = [m["k"] for m in eligible(meta, cfg) if (not mind or m["decision_date"] >= mind) and not paths.text(m["k"]).exists()]
    return _write_queue(paths, "full", ks)

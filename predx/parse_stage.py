"""Stage `parse`: cached text -> structured record (sections + predicates). Cheap and re-runnable: iterate on the
rules in sections.py / predicates.py, re-run this stage, re-run `label-eval`. No OCR is repeated."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os

from . import PARSE_VERSION
from .io import load_pages, write_jsonl
from .metadata import load_meta
from .predicates import extract_predicates
from .sections import parse_sections

log = logging.getLogger("predx.parse")
_G: dict = {}


def parse_document(k: str, pages: list[str], cfg: dict, known=None) -> dict:
    sec = parse_sections(pages, cfg)
    pred = extract_predicates(pages, k, known, fallback=cfg["parse"].get("first_other_fallback", True))
    return {"k": k, **sec, **pred, "parse_version": PARSE_VERSION}


def _init(cfg, known):
    _G["cfg"], _G["known"] = cfg, known


def _work(args):
    k, path = args
    try:
        return parse_document(k, load_pages(path), _G["cfg"], _G["known"])
    except Exception as e:
        return {"k": k, "parse_error": repr(e)[:300], "predicates": []}


def run_parse(cfg: dict, paths) -> int:
    known = set(load_meta(paths).keys()) if paths.meta.exists() else None
    files = sorted(paths.text_dir.glob("*/*.txt.gz")) if paths.text_dir.exists() else []
    tasks = [(f.name[: -len(".txt.gz")], str(f)) for f in files]
    log.info("parse: %d cached documents", len(tasks))
    workers = cfg["parse"].get("workers") or os.cpu_count() or 1
    if workers == 1 or len(tasks) < 64:
        _init(cfg, known)
        recs = [_work(t) for t in tasks]
    else:
        with mp.Pool(workers, initializer=_init, initargs=(cfg, known)) as pool:
            recs = list(pool.imap_unordered(_work, tasks, chunksize=32))
    recs.sort(key=lambda r: r["k"])
    n = write_jsonl(paths.parsed, recs)
    err = sum(1 for r in recs if "parse_error" in r)
    log.info("parse: wrote %d records (%d parse errors) -> %s", n, err, paths.parsed)
    return n

"""Stage `fetch`: download 510(k) summary PDFs from accessdata.fda.gov. Polite, resumable, Spot-VM safe.

URL layout (confirmed for 2021 and 2024 in public examples; older years are tried in several folder spellings and
the working one is recorded per document in fetch_status.jsonl so the pilot shows which pattern each year uses):
    {base}/pdf{YY}/{K}.pdf   e.g. .../cdrh_docs/pdf21/K210758.pdf
    {base}/pdf{Y}/{K}.pdf    single-digit years (2001-2009)
    {base}/pdf/{K}.pdf       pre-2001
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .config import shard
from .io import append_jsonl, last_by_key

log = logging.getLogger("predx.fetch")


def folder_candidates(k: str) -> list[str]:
    yy = int(shard(k)) if shard(k).isdigit() else 0
    if yy >= 76 or yy == 0:
        order = ["pdf", f"pdf{yy}"]
    elif yy < 10:
        order = [f"pdf{yy}", f"pdf{yy:02d}", "pdf"]
    else:
        order = [f"pdf{yy}", "pdf"]
    seen, out = set(), []
    for o in order:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


class RateLimiter:
    def __init__(self, rps: float):
        self.interval = 1.0 / rps if rps and rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(self.next, now)
            self.next = t + self.interval
        if t > now:
            time.sleep(t - now)


def make_session(cfg: dict) -> requests.Session:
    s = requests.Session()
    s.trust_env = bool(cfg["fetch"].get("trust_env", True))
    email = cfg.get("contact_email") or os.environ.get("CONTACT_EMAIL", "")
    s.headers["User-Agent"] = f"predx-course-project/0.1 (academic research; contact: {email or 'unset'})"
    return s


def fetch_one(session, k: str, cfg: dict, limiter: RateLimiter, paths) -> dict:
    f = cfg["fetch"]
    base = f["base_url"].rstrip("/")
    last_err = None
    for folder in folder_candidates(k):
        url = f"{base}/{folder}/{k}.pdf"
        for attempt in range(f["max_retries"]):
            limiter.wait()
            try:
                r = session.get(url, timeout=f["timeout_s"])
            except requests.RequestException as e:
                last_err = repr(e)[:200]
                time.sleep(min(30, 2 ** attempt + random.random()))
                continue
            if r.status_code == 404:
                break  # try next folder spelling
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"http {r.status_code}"
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            if r.status_code != 200:
                last_err = f"http {r.status_code}"
                break
            body = r.content
            if not body.startswith(b"%PDF"):
                last_err = "200 but not a PDF"
                break
            dest = paths.pdf(k)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name("tmp." + dest.name)
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, dest)
            return {"k": k, "status": "ok", "url": url, "folder": folder, "bytes": len(body), "error": None}
        else:
            return {"k": k, "status": "error", "url": url, "folder": folder, "bytes": 0, "error": last_err or "retries exhausted"}
    status = "not_found" if last_err in (None, "200 but not a PDF") else "error"
    return {"k": k, "status": status, "url": None, "folder": None, "bytes": 0, "error": last_err}


def run_fetch(cfg: dict, paths, ks, retry_not_found: bool = False, limit: int | None = None) -> dict:
    prev = last_by_key(paths.fetch_status)
    todo = []
    for k in ks:
        if paths.pdf(k).exists() or paths.text(k).exists():
            continue
        st = prev.get(k)
        if st and st["status"] == "not_found" and not retry_not_found:
            continue
        todo.append(k)
    if limit:
        todo = todo[:limit]
    log.info("fetch: %d to download (of %d requested)", len(todo), len(ks))
    if not todo:
        return {"ok": 0, "not_found": 0, "error": 0}
    session = make_session(cfg)
    limiter = RateLimiter(cfg["fetch"]["requests_per_second"])
    lock = threading.Lock()
    counts = {"ok": 0, "not_found": 0, "error": 0}
    t0 = time.time()
    done = 0

    def work(k):
        nonlocal done
        res = fetch_one(session, k, cfg, limiter, paths)
        with lock:
            append_jsonl(paths.fetch_status, res)
            counts[res["status"]] += 1
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                log.info("fetch: %d/%d  %.1f docs/s  %s", done, len(todo), done / el, counts)
        return res

    with ThreadPoolExecutor(max_workers=cfg["fetch"]["threads"]) as ex:
        list(ex.map(work, todo))
    return counts

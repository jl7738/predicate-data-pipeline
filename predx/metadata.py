"""Stage `meta`: openFDA 510(k) bulk file -> normalised metadata (data/meta/510k.jsonl.gz).

The bulk download is used instead of paging the API (the API caps `skip`). The partition list is read from
api.fda.gov/download.json so no file name is hard-coded; a fallback URL is used if that index cannot be read.
"""
from __future__ import annotations

import io
import json
import logging
import re
import zipfile

import requests

from .fetch import make_session
from .io import read_jsonl, write_jsonl

log = logging.getLogger("predx.meta")


def _iso_date(s) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    return m.group(0) if m else None


def normalise_record(r: dict) -> dict | None:
    k = (r.get("k_number") or "").strip().upper()
    if not k:
        return None
    date = _iso_date(r.get("decision_date"))
    sos = (r.get("statement_or_summary") or "").strip().upper() or None
    return {
        "k": k,
        "device_name": (r.get("device_name") or "").strip(),
        "applicant": (r.get("applicant") or "").strip(),
        "product_code": (r.get("product_code") or "").strip().upper(),
        "panel": (r.get("review_advisory_committee") or r.get("advisory_committee") or "").strip().upper(),
        "decision_date": date,
        "date_received": _iso_date(r.get("date_received")),
        "year": int(date[:4]) if date else None,
        "decision_code": (r.get("decision_code") or "").strip().upper(),
        "decision_description": (r.get("decision_description") or "").strip(),
        "clearance_type": (r.get("clearance_type") or "").strip(),
        "statement_or_summary": sos,
    }


def _partition_urls(cfg: dict, session) -> list[str]:
    o = cfg["openfda"]
    try:
        idx = session.get(o["download_index"], timeout=60).json()
        parts = idx["results"]["device"]["510k"]["partitions"]
        urls = [p["file"] for p in parts]
        if urls:
            return urls
    except Exception as e:
        log.warning("could not read download index (%s); using fallback URL", e)
    return [o["fallback_url"]]


def download_meta(cfg: dict, paths, force: bool = False) -> int:
    if paths.meta.exists() and not force:
        log.info("metadata already present: %s", paths.meta)
        return sum(1 for _ in read_jsonl(paths.meta))
    session = make_session(cfg)
    rows, seen = [], set()
    paths.raw.mkdir(parents=True, exist_ok=True)
    for url in _partition_urls(cfg, session):
        log.info("downloading %s", url)
        r = session.get(url, timeout=cfg["openfda"]["timeout_s"])
        r.raise_for_status()
        (paths.raw / url.rsplit("/", 1)[-1]).write_bytes(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(z.read(name))
                for rec in data.get("results", []):
                    n = normalise_record(rec)
                    if n and n["k"] not in seen:
                        seen.add(n["k"])
                        rows.append(n)
    rows.sort(key=lambda x: (x["decision_date"] or "", x["k"]))
    write_jsonl(paths.meta, rows)
    log.info("metadata: %d records", len(rows))
    return len(rows)


def load_meta(paths) -> dict[str, dict]:
    return {r["k"]: r for r in read_jsonl(paths.meta)}


def is_se(m: dict, cfg: dict) -> bool:
    return (m.get("decision_code") or "").upper().startswith(cfg["corpus"]["decision_code_prefix"])


def has_summary(m: dict) -> bool:
    """Summary available (or unknown). Only an explicit STATEMENT excludes a device."""
    return (m.get("statement_or_summary") or "").upper() != "STATEMENT"

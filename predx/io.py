from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Iterator


def open_any(path, mode: str = "rt"):
    path = str(path)
    binary = "b" in mode
    if path.endswith(".gz"):
        return gzip.open(path, mode) if binary else gzip.open(path, mode, encoding="utf-8")
    return open(path, mode) if binary else open(path, mode, encoding="utf-8")


def read_jsonl(path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open_any(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> int:
    """Atomic write (tmp + rename) so a Spot-VM preemption never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp." + path.name)
    n = 0
    with open_any(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_by_key(path, key: str = "k") -> dict:
    """Read an append-only status log; the last row per key wins."""
    out: dict = {}
    for r in read_jsonl(path):
        out[r[key]] = r
    return out


PAGE_SEP = "\f"


def save_pages(path, pages: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp." + path.name)
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(PAGE_SEP.join(pages))
    os.replace(tmp, path)


def load_pages(path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read().split(PAGE_SEP)


def write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp." + path.name)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

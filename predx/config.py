from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(paths=(), overrides=()) -> dict:
    """default.yaml, then each file in `paths`, then `a.b.c=value` overrides (value parsed as YAML)."""
    cfg: dict = {}
    for p in [DEFAULT_CONFIG, *[Path(x) for x in paths if x]]:
        with open(p, encoding="utf-8") as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def cget(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def shard(k: str) -> str:
    digits = re.sub(r"\D", "", k)
    return digits[:2] or "xx"


class Paths:
    """All on-disk locations, derived from cfg['paths']['data_dir']."""

    def __init__(self, cfg: dict, root: str | Path | None = None):
        self.root = Path(root or cfg["paths"]["data_dir"])
        self.raw = self.root / "raw"
        self.meta = self.root / "meta" / "510k.jsonl.gz"
        self.queues = self.root / "queues"
        self.pdf_dir = self.root / "pdf"
        self.fetch_status = self.root / "fetch_status.jsonl"
        self.text_dir = self.root / "text"
        self.text_stats = self.root / "text_stats.jsonl"
        self.parsed = self.root / "parsed.jsonl.gz"
        self.dataset = self.root / "dataset"
        self.reports = self.root / "reports"
        self.labeling = self.root / "labeling"

    def pdf(self, k: str) -> Path:
        return self.pdf_dir / shard(k) / f"{k}.pdf"

    def text(self, k: str) -> Path:
        return self.text_dir / shard(k) / f"{k}.txt.gz"

    def queue(self, name: str) -> Path:
        return self.queues / f"{name}.txt"

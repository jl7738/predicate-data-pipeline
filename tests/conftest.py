"""Shared fixtures. `world` builds a synthetic 510(k) universe once per test session and runs parse + build on it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predx.config import Paths, load_config  # noqa: E402

DEMO_CFG = Path(__file__).resolve().parents[1] / "config" / "demo.yaml"


def make_cfg(tmp, extra=()):
    cfg = load_config([DEMO_CFG], [f"paths.data_dir={tmp}", "fetch.trust_env=false", *extra])
    return cfg, Paths(cfg)


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    from predx.dataset import build_dataset
    from predx.parse_stage import run_parse
    from predx.synth import make_synthetic

    root = tmp_path_factory.mktemp("world")
    cfg, paths = make_cfg(root)
    make_synthetic(root, n=1500, seed=7)
    run_parse(cfg, paths)
    card = build_dataset(cfg, paths)
    truth = json.loads((root / "truth.json").read_text())
    return {"cfg": cfg, "paths": paths, "truth": truth, "card": card, "root": root}

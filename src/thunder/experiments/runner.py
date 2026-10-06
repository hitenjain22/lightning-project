"""Run a pipeline from a config and write a reproducible results folder.

Layout: results/<experiment>/<run_id>/
    config.resolved.yaml   the validated config with all defaults filled in
    meta.json              seed, config hash, git commit, versions, timestamp
    metrics.json           metrics produced by the pipeline stages
    arrays.npz             numeric outputs (may be empty)
    figures/               figures produced by the run
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from thunder.config import RunConfig


def git_commit(cwd: Path | None = None) -> dict[str, Any]:
    """Current commit hash and dirty flag, or 'unknown' outside a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
        return {"commit": sha, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}


def make_run_dir(cfg: RunConfig, now: dt.datetime | None = None) -> Path:
    now = now or dt.datetime.now(dt.UTC)
    run_id = f"{now:%Y%m%dT%H%M%S}_{cfg.config_hash()}_s{cfg.seed}"
    run_dir = Path(cfg.results_root) / cfg.experiment / run_id
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    return run_dir


def run_pipeline(cfg: RunConfig) -> Path:
    """Run every configured stage and write outputs. Returns the run directory.

    Stages are added milestone by milestone (generate, synthesize, corrupt,
    reconstruct, evaluate). For M0 the pipeline has no stages.
    """
    run_dir = make_run_dir(cfg)
    rng = np.random.default_rng(cfg.seed)

    stages: list[str] = []
    metrics: dict[str, Any] = {"stages": stages}
    arrays: dict[str, np.ndarray] = {}
    _ = rng  # passed to stages once they exist

    with open(run_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    meta = {
        "experiment": cfg.experiment,
        "seed": cfg.seed,
        "config_hash": cfg.config_hash(),
        "git": git_commit(),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(run_dir / "arrays.npz", **arrays)  # type: ignore[arg-type]
    return run_dir

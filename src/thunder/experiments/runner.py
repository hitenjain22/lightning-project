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

from thunder.channel.generator import generate_channel
from thunder.channel.stats import channel_stats
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


def write_run_files(
    cfg: RunConfig, run_dir: Path, metrics: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    """Write the resolved config, metadata, metrics and arrays into run_dir."""
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


def run_pipeline(cfg: RunConfig) -> Path:
    """Run every configured stage and write outputs. Returns the run directory.

    Stages are added milestone by milestone (generate, synthesize, corrupt,
    reconstruct, evaluate). A stage runs only if its config section is present.
    """
    run_dir = make_run_dir(cfg)
    rng = np.random.default_rng(cfg.seed)

    stages: list[str] = []
    metrics: dict[str, Any] = {"stages": stages}
    arrays: dict[str, np.ndarray] = {}

    if cfg.channel is not None:
        ch = generate_channel(cfg.channel, rng, seed=cfg.seed)
        stages.append("generate")
        metrics["channel"] = channel_stats(ch)
        arrays.update(
            channel_nodes=ch.nodes,
            channel_segments=ch.segments,
            channel_energy_per_length=ch.energy_per_length,
            channel_branch_id=ch.branch_id,
            channel_is_main=ch.is_main,
            channel_is_incloud=ch.is_incloud,
            channel_stroke_times=ch.stroke_times,
        )

    write_run_files(cfg, run_dir, metrics, arrays)
    return run_dir

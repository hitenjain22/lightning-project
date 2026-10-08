"""Run a pipeline from a config and write a reproducible results folder.

Layout: results/<experiment>/<run_id>/
    config.resolved.yaml   the validated config with all defaults filled in
    meta.json              seed, config hash, git commit, versions, timestamp
    metrics.json           metrics produced by the pipeline stages (Monte Carlo: summary with CIs)
    arrays.npz             numeric outputs (may be empty)
    figures/               figures produced by the run
  Monte Carlo runs add: bolts.csv (one row per bolt), points.csv.gz (pooled per-point errors),
  examples.pkl (full geometry of one example bolt per preset, for figures).
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from thunder.acoustics.audio import write_wavs
from thunder.channel.stats import channel_stats
from thunder.config import RunConfig
from thunder.experiments.montecarlo import run_monte_carlo
from thunder.experiments.pipeline import run_bolt


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


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return v


def run_pipeline(cfg: RunConfig) -> Path:
    """Run every configured stage for one bolt (seed = cfg.seed) and write its outputs.

    A stage runs only if its config section is present: generate (channel), synthesize
    (synthesis), sensors (sensors), reconstruct + evaluate (reconstruction).
    """
    run_dir = make_run_dir(cfg)
    res = run_bolt(cfg, cfg.seed)
    ch = res.channel
    metrics: dict[str, Any] = {"stages": res.stages, "timings_s": res.timings_s, "channel": channel_stats(ch)}
    arrays: dict[str, np.ndarray] = {
        "channel_nodes": ch.nodes,
        "channel_segments": ch.segments,
        "channel_energy_per_length": ch.energy_per_length,
        "channel_branch_id": ch.branch_id,
        "channel_is_main": ch.is_main,
        "channel_is_incloud": ch.is_incloud,
        "channel_stroke_times": ch.stroke_times,
    }
    rec, array = res.recording, res.array
    if rec is not None and array is not None:
        assert rec.truth is not None
        metrics["recording"] = {
            "duration_s": rec.duration,
            "peak_pa": float(np.max(np.abs(rec.signals))),
            "rms_pa": float(np.sqrt(np.mean(rec.signals**2))),
        }
        arrays.update(
            signals=rec.signals,
            mic_positions=rec.nominal_mic_positions,
            true_mic_positions=array.true_positions,
            truth_segment_arrival_times=rec.truth.segment_arrival_times,
        )
        if "sensors" in res.stages:
            ex = rec.truth.extra
            metrics["sensors"] = {
                "t0_error_s": ex["t0_error"],
                "clock_offset_s": ex["clock_offset"],
                "clock_drift_ppm": ex["clock_drift_ppm"],
                "position_error_m": np.linalg.norm(array.true_positions - array.nominal_positions, axis=1),
                **ex["corruption_info"],
            }
        assert cfg.synthesis is not None
        if cfg.synthesis.write_wav:
            metrics["recording"]["wav_gain_per_pa"] = write_wavs(rec, run_dir / "audio")
    if res.reconstruction is not None:
        r = res.reconstruction
        metrics["evaluation"] = res.metrics
        arrays.update(
            recon_points=r.points,
            recon_covariances=r.covariances,
            recon_window_times=r.window_times,
            recon_quality=r.quality,
            recon_gated_points=r.extra["gated_points"],
            recon_skeleton_edges=r.extra["skeleton_edges"],
            recon_point_error_m=res.per_point.get("error_m", np.zeros(0)),
        )
        demo_figures(res, run_dir / "figures")
    write_run_files(cfg, run_dir, _jsonable(metrics), arrays)
    return run_dir


def demo_figures(res: Any, fig_dir: Path) -> None:
    """Figures of a single reconstructed bolt: a 3D view of the true channel and the
    reconstructed points (colored by error) with the waveform at the first mic
    (reconstruction.png), and the same 3D view as interactive HTML (reconstruction.html)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from thunder.experiments.montecarlo import _example
    from thunder.experiments.reports import plot_example_3d
    from thunder.viz.plot3d import reconstruction_figure, write_html

    ex = _example(res)
    if ex is None:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = res.metrics
    title = (
        f"{len(ex['points'])} points, median error {m.get('point_error_median_m', float('nan')):.1f} m, "
        f"main-channel coverage (50 m) {m.get('coverage_main_50m', 0.0):.0%}"
    )
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plot_example_3d(ax, ex, title)
    ax2 = fig.add_subplot(1, 2, 2)
    sig = np.asarray(ex["signals"][0])
    ax2.plot(np.arange(len(sig)) / ex["sample_rate"], sig, lw=0.5, color="0.2")
    ax2.set_xlabel("time since the flash (s)")
    ax2.set_ylabel("pressure at mic 0 (Pa)")
    ax2.set_title("recorded thunder (first mic)")
    fig.tight_layout()
    fig.savefig(fig_dir / "reconstruction.png", dpi=120)
    plt.close(fig)
    write_html(reconstruction_figure(ex, title), fig_dir / "reconstruction.html")


def run_experiment(cfg: RunConfig) -> Path:
    """Single bolt, or a Monte Carlo if the config has a monte_carlo section."""
    if cfg.monte_carlo is None:
        return run_pipeline(cfg)
    run_dir = make_run_dir(cfg)
    mc = run_monte_carlo(cfg)
    mc.table.to_csv(run_dir / "bolts.csv", index=False)
    if len(mc.points):
        mc.points.to_csv(run_dir / "points.csv.gz", index=False)
    with open(run_dir / "examples.pkl", "wb") as f:
        pickle.dump(mc.examples, f)
    write_run_files(cfg, run_dir, _jsonable(mc.summary), {})
    return run_dir

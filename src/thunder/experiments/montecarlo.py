"""Monte Carlo over many random bolts (SPEC.md Phase 6 aggregation), in parallel.

Bolt k uses SeedSequence(cfg.seed).spawn(n_bolts)[k] and preset presets[k % len(presets)], so
results are identical for any number of workers. Per-bolt metrics go to a table; per-point
errors are pooled for error-vs-range/altitude analysis; summaries carry bootstrap 95% CIs.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from thunder.channel.stats import branch_count, segment_lengths, total_length
from thunder.config import CHANNEL_PRESETS, ChannelConfig, RunConfig
from thunder.eval.aggregate import bootstrap_ci, cluster_bootstrap_ci
from thunder.experiments.pipeline import run_bolt
from thunder.types import FloatArray


@dataclass
class MonteCarloResult:
    table: pd.DataFrame  # one row per bolt
    points: pd.DataFrame  # one row per reconstructed point (pooled)
    examples: dict[str, dict[str, Any]]  # preset -> arrays for plotting one example bolt
    summary: dict[str, Any]


def _one(args: tuple[dict, dict, int, Any, bool]) -> tuple[dict, dict[str, FloatArray], dict | None]:
    cfg_dict, channel_dict, k, seed_seq, keep = args
    cfg = RunConfig.model_validate(cfg_dict)
    res = run_bolt(cfg, seed_seq, ChannelConfig.model_validate(channel_dict))
    ch = res.channel
    strike = np.asarray(ch.metadata["strike_point"])
    row = {
        "bolt": k,
        "preset": channel_dict["preset"],
        "strike_distance_m": float(np.hypot(strike[0], strike[1])),
        "strike_azimuth_deg": float(np.degrees(np.arctan2(strike[0], strike[1])) % 360),
        "start_height_m": float(ch.nodes[0, 2]),
        "channel_length_m": total_length(ch),
        "incloud_length_m": float(segment_lengths(ch)[ch.is_incloud].sum()),
        "branch_count": branch_count(ch),
        **res.metrics,
        **{f"time_{k_}_s": v for k_, v in res.timings_s.items()},
    }
    pp = dict(res.per_point)
    pp["bolt"] = np.full(len(next(iter(pp.values()), [])), k)
    example = None
    if keep and res.reconstruction is not None and res.recording is not None:
        r = res.reconstruction
        example = {
            "nodes": ch.nodes,
            "segments": ch.segments,
            "is_main": ch.is_main,
            "is_incloud": ch.is_incloud,
            "points": r.points,
            "covariances": r.covariances,
            "error_m": res.per_point.get("error_m"),
            "gated_points": r.extra.get("gated_points"),
            "skeleton_edges": r.extra.get("skeleton_edges"),
            "strike_point": r.extra.get("strike_point"),
            "mics": res.recording.nominal_mic_positions,
            "signals": res.recording.signals[:1],
            "sample_rate": res.recording.sample_rate,
            "metrics": res.metrics,
        }
    return row, pp, example


def channel_overrides(cfg: RunConfig) -> dict[str, Any]:
    """Channel fields the user set explicitly, which every Monte Carlo preset must keep.

    Must run on the user's config (not a re-validated dump, where every field counts as set).
    Fields that merely came from the config's own preset are not overrides.
    """
    assert cfg.channel is not None
    explicit = cfg.channel.model_dump(mode="json", exclude_unset=True)
    explicit.pop("preset", None)
    base = ChannelConfig.model_validate({"preset": cfg.channel.preset}).model_dump(mode="json")
    for key in CHANNEL_PRESETS[cfg.channel.preset]:
        if key in explicit and explicit[key] == base[key]:
            explicit.pop(key)
    return explicit


def run_monte_carlo(cfg: RunConfig) -> MonteCarloResult:
    if cfg.monte_carlo is None or cfg.channel is None:
        raise ValueError("config needs monte_carlo and channel sections")
    mc = cfg.monte_carlo
    seeds = np.random.SeedSequence(cfg.seed).spawn(mc.n_bolts)
    cfg_dict = cfg.model_dump(mode="json")
    overrides = channel_overrides(cfg)
    seen: set[str] = set()
    tasks = []
    for k, s in enumerate(seeds):
        preset = mc.presets[k % len(mc.presets)]
        keep = preset not in seen  # keep full geometry of the first bolt of each preset for figures
        seen.add(preset)
        tasks.append((cfg_dict, {**overrides, "preset": preset}, k, s, keep))

    workers = mc.workers or os.cpu_count() or 1
    if workers == 1:
        outputs = [_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_one, tasks, chunksize=1))

    table = pd.DataFrame([o[0] for o in outputs])
    points = (
        pd.concat([pd.DataFrame(o[1]) for o in outputs if len(o[1].get("bolt", []))], ignore_index=True)
        if any(len(o[1].get("bolt", [])) for o in outputs)
        else pd.DataFrame()
    )
    examples = {o[0]["preset"]: o[2] for o in outputs if o[2] is not None}
    return MonteCarloResult(table, points, examples, summarize(table, points, cfg))


def summarize(table: pd.DataFrame, points: pd.DataFrame, cfg: RunConfig) -> dict[str, Any]:
    """Headline numbers with bootstrap 95% CIs, overall and per preset."""
    assert cfg.monte_carlo is not None
    rng = np.random.default_rng(cfg.seed + 1)
    n = cfg.monte_carlo.bootstrap_samples

    def block(t: pd.DataFrame, p: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"n_bolts": int(len(t)), "n_points": int(len(p))}
        if len(p):
            err, bolt = p["error_m"].to_numpy(), p["bolt"].to_numpy()
            out["pooled_point_error_median_m"] = cluster_bootstrap_ci(err, bolt, np.median, rng, n)
            out["pooled_point_error_p90_m"] = cluster_bootstrap_ci(
                err, bolt, lambda v: float(np.percentile(v, 90)), rng, n
            )
        for col in (
            "point_error_median_m",
            "point_error_p90_m",
            "chamfer_m",
            "coverage_50m",
            "coverage_main_50m",
            "coverage_100m",
            "strike_error_m",
            "radial_error_median_m",
            "transverse_error_median_m",
            "time_reconstruct_s",
        ):
            if col in t:
                stat = (
                    np.median
                    if col.startswith(("point_error", "strike", "radial", "transverse"))
                    else np.mean
                )
                out[f"{'median' if stat is np.median else 'mean'}_{col}"] = bootstrap_ci(
                    t[col].to_numpy(), stat, rng, n
                )
        if "point_error_median_m" in t:
            out["bolts_with_no_points"] = int(t["n_points"].eq(0).sum())
        return out

    summary: dict[str, Any] = {"overall": block(table, points)}
    for preset in table["preset"].unique():
        bolts = table.loc[table["preset"] == preset, "bolt"]
        summary[str(preset)] = block(
            table[table["preset"] == preset], points[points["bolt"].isin(bolts)] if len(points) else points
        )
    return summary

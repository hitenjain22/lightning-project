"""Monte Carlo over many random bolts (SPEC.md Phase 6 aggregation), in parallel.

Bolt k uses SeedSequence(cfg.seed).spawn(n_bolts)[k] and preset presets[k % len(presets)], so
results are identical for any number of workers. Per-bolt metrics go to a table; per-point
errors are pooled for error-vs-range/altitude analysis; summaries carry bootstrap 95% CIs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from thunder.channel.stats import branch_count, segment_lengths, total_length
from thunder.config import CHANNEL_PRESETS, ChannelConfig, RunConfig
from thunder.eval.aggregate import bootstrap_ci, cluster_bootstrap_ci
from thunder.experiments.pipeline import BoltResult, iter_bolt_variants, run_bolt
from thunder.types import FloatArray


@dataclass
class MonteCarloResult:
    table: pd.DataFrame  # one row per bolt
    points: pd.DataFrame  # one row per reconstructed point (pooled)
    examples: dict[str, dict[str, Any]]  # preset -> arrays for plotting one example bolt
    summary: dict[str, Any]


def _row(k: int, preset: str, res: BoltResult) -> dict[str, Any]:
    ch = res.channel
    strike = np.asarray(ch.metadata["strike_point"])
    return {
        "bolt": k,
        "preset": preset,
        "strike_distance_m": float(np.hypot(strike[0], strike[1])),
        "strike_azimuth_deg": float(np.degrees(np.arctan2(strike[0], strike[1])) % 360),
        "start_height_m": float(ch.nodes[0, 2]),
        "channel_length_m": total_length(ch),
        "incloud_length_m": float(segment_lengths(ch)[ch.is_incloud].sum()),
        "branch_count": branch_count(ch),
        **res.metrics,
        **{f"time_{k_}_s": v for k_, v in res.timings_s.items()},
    }


def _example(res: BoltResult) -> dict | None:
    if res.reconstruction is None or res.recording is None:
        return None
    ch, r = res.channel, res.reconstruction
    return {
        "nodes": ch.nodes,
        "segments": ch.segments,
        "is_main": ch.is_main,
        "is_incloud": ch.is_incloud,
        "points": r.points,
        "covariances": r.covariances,
        "window_times": r.window_times,
        "error_m": res.per_point.get("error_m"),
        "gated_points": r.extra.get("gated_points"),
        "skeleton_edges": r.extra.get("skeleton_edges"),
        "strike_point": r.extra.get("strike_point"),
        "mics": res.recording.nominal_mic_positions,
        "signals": res.recording.signals[:1],
        "sample_rate": res.recording.sample_rate,
        "metrics": res.metrics,
    }


def _one(
    args: tuple[dict, dict, int, Any, bool],
) -> tuple[list[dict], list[dict[str, FloatArray]], dict | None]:
    """One bolt: every variant (or the plain config), as table rows, per-point arrays, example."""
    cfg_dict, channel_dict, k, seed_seq, keep = args
    variants = cfg_dict["monte_carlo"].get("variants") or {}
    keep_points = cfg_dict["monte_carlo"].get("keep_points", True)
    if variants:
        results: Iterable[tuple[str, BoltResult]] = iter_bolt_variants(
            cfg_dict, variants, seed_seq, channel_dict
        )
    else:
        cfg = RunConfig.model_validate(cfg_dict)
        results = [("", run_bolt(cfg, seed_seq, ChannelConfig.model_validate(channel_dict)))]
    rows, points = [], []
    example = None
    for i, (name, res) in enumerate(results):
        if keep and i == 0:
            example = _example(res)
        row = _row(k, channel_dict["preset"], res)
        if variants:
            row["variant"] = name
        rows.append(row)
        if keep_points:
            pp = dict(res.per_point)
            npt = len(next(iter(pp.values()), []))
            pp["bolt"] = np.full(npt, k)
            if variants:
                pp["variant"] = np.full(npt, name, dtype=object)
            points.append(pp)
    return rows, points, example


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
        keep = mc.keep_examples and preset not in seen  # full geometry of the first bolt of each preset
        seen.add(preset)
        tasks.append((cfg_dict, {**overrides, "preset": preset}, k, s, keep))

    workers = mc.workers or os.cpu_count() or 1
    if workers == 1:
        outputs = [_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_one, tasks, chunksize=1))

    table = pd.DataFrame([row for o in outputs for row in o[0]])
    frames = [pd.DataFrame(pp) for o in outputs for pp in o[1] if len(pp.get("bolt", []))]
    points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    examples = {o[0][0]["preset"]: o[2] for o in outputs if o[2] is not None}
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
            "angular_error_median_deg",
            "coverage_branch_50m",
            "time_reconstruct_s",
        ):
            if col in t:
                stat = (
                    np.median
                    if col.startswith(("point_error", "strike", "radial", "transverse", "angular"))
                    else np.mean
                )
                out[f"{'median' if stat is np.median else 'mean'}_{col}"] = bootstrap_ci(
                    t[col].to_numpy(), stat, rng, n
                )
        if "point_error_median_m" in t:
            out["bolts_with_no_points"] = int(t["n_points"].eq(0).sum())
        return out

    def by_preset(t: pd.DataFrame, p: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"overall": block(t, p)}
        for preset in t["preset"].unique():
            bolts = t.loc[t["preset"] == preset, "bolt"]
            out[str(preset)] = block(t[t["preset"] == preset], p[p["bolt"].isin(bolts)] if len(p) else p)
        return out

    if "variant" not in table:
        return by_preset(table, points)
    # Variants: {variant: {"overall": ..., preset: ...}}
    return {
        str(v): by_preset(
            table[table["variant"] == v], points[points["variant"] == v] if len(points) else points
        )
        for v in table["variant"].unique()
    }

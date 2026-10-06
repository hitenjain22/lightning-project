"""Figures and markdown summaries for Monte Carlo experiment runs.

`make_report(run_dir)` reads bolts.csv / points.csv.gz / examples.pkl / metrics.json from a
Monte Carlo run folder and writes figures into run_dir/figures plus run_dir/summary.md.
Everything is regenerated from the run folder alone (SPEC.md reproducibility rule).
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from thunder.eval.aggregate import binned_quantiles  # noqa: E402
from thunder.viz.plot3d import reconstruction_figure, write_html  # noqa: E402
from thunder.viz.plots import BRANCH_COLOR, INCLOUD_COLOR, MAIN_COLOR  # noqa: E402

PRESET_COLORS = {
    "tortuous": "#1f3a93",
    "branched": "#16a085",
    "with_incloud": "#c0392b",
    "multi_stroke": "#8e44ad",
    "straight": "#7f8c8d",
}


def describe_atmosphere(a: dict | None, oracle: bool = False) -> str:
    """Short human-readable description of an atmosphere config (as stored in config.resolved.yaml)."""
    if a is None:
        return "ORACLE (the synthesis atmosphere)" if oracle else "default"
    if a.get("model", "uniform") == "uniform":
        text = f"uniform, {a.get('temperature_c', 25.0):g} C (straight rays)"
    else:
        t_c, lapse = a.get("temperature_c", 25.0), a.get("lapse_rate_k_per_km", 6.5)
        text = f"stratified, {t_c:g} C, lapse {lapse:g} K/km"
        if a.get("inversion"):
            inv = a["inversion"]
            text += f", inversion +{inv['delta_k']:g} K over {inv['base_m']:g}-{inv['top_m']:g} m"
        w = a.get("wind")
        text += f", wind {w['speed_mps']:g} m/s from {w['direction_from_deg']:g} deg" if w else ", no wind"
    extras = [name for name in ("absorption", "ground_reflection") if a.get(name)]
    return text + (f" (+{', '.join(extras)})" if extras else "")


def _ci(v: Any, unit: str = "m", digits: int = 1) -> str:
    est, lo, hi = v
    return f"{est:.{digits}f} {unit} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _pct(v: Any) -> str:
    est, lo, hi = v
    return f"{100 * est:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def plot_example_3d(ax, ex: dict[str, Any], title: str) -> None:
    nodes, segs = ex["nodes"], ex["segments"]
    side = ~(ex["is_main"] | ex["is_incloud"])
    for mask, color, lw in (
        (side, BRANCH_COLOR, 0.8),
        (ex["is_incloud"], INCLOUD_COLOR, 1.2),
        (ex["is_main"], MAIN_COLOR, 1.5),
    ):
        for a, b in segs[mask]:
            ax.plot(*np.c_[nodes[a], nodes[b]], color=color, lw=lw)
    p, err = ex["points"], ex["error_m"]
    if len(p):
        sc = ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            c=err,
            cmap="viridis",
            s=6,
            vmin=0,
            vmax=max(float(np.percentile(err, 95)), 1.0),
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="point error (m)")
    m = ex["mics"]
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], color="k", marker="^", s=30)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_zlabel("up (m)")


def make_report(run_dir: Path, docs_figures: Path | None = None, prefix: str = "") -> Path:
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    table = pd.read_csv(run_dir / "bolts.csv")
    points = pd.read_csv(run_dir / "points.csv.gz")
    summary = json.loads((run_dir / "metrics.json").read_text())
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    with open(run_dir / "examples.pkl", "rb") as f:
        examples = pickle.load(f)
    presets = list(dict.fromkeys(table["preset"]))
    saved: list[str] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=120)
        plt.close(fig)
        saved.append(name)

    # 1. Example reconstructions (one per preset), static + interactive.
    fig = plt.figure(figsize=(6 * len(examples), 6))
    for k, (preset, ex) in enumerate(examples.items()):
        ax = fig.add_subplot(1, len(examples), k + 1, projection="3d")
        med = ex["metrics"].get("point_error_median_m", np.nan)
        plot_example_3d(ax, ex, f"{preset}: median error {med:.1f} m")
        write_html(
            reconstruction_figure(ex, f"{preset} example (median error {med:.1f} m)"),
            fig_dir / f"example_{preset}.html",
        )
    save(fig, "examples_3d")

    # 2. Pooled point error vs range and vs true altitude.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, label in (
        (axes[0], "range_m", "range from array centroid (km)"),
        (axes[1], "true_altitude_m", "altitude of nearest true channel point (km)"),
    ):
        edges = np.linspace(0, points[col].quantile(0.995), 16)
        for preset in presets:
            sel = points["bolt"].isin(table.loc[table["preset"] == preset, "bolt"])
            q = binned_quantiles(
                points.loc[sel, col].to_numpy(), points.loc[sel, "error_m"].to_numpy(), edges
            )
            c = PRESET_COLORS.get(preset, "k")
            ax.plot(q["center"] / 1e3, q["q50"], "-o", ms=3, color=c, label=f"{preset} (median)")
            ax.fill_between(q["center"] / 1e3, q["q25"], q["q75"], color=c, alpha=0.15)
        ax.set_xlabel(label)
        ax.set_ylabel("point error (m)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Point error vs range (median, shaded IQR)")
    axes[1].set_title("Point error vs altitude")
    save(fig, "error_vs_range_altitude")

    # 3. Error distribution and radial / transverse split.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.logspace(-1, np.log10(max(points["error_m"].max(), 10)), 60)
    axes[0].hist(points["error_m"], bins=bins, color=MAIN_COLOR, alpha=0.8)
    med = points["error_m"].median()
    axes[0].axvline(med, color="k", ls="--", label=f"median {med:.1f} m")
    axes[0].axvline(50, color="#c0392b", ls=":", label="provisional target 50 m")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("point error (m)")
    axes[0].set_ylabel("points")
    axes[0].legend()
    axes[0].set_title("All reconstructed points")
    axes[1].hist(np.abs(points["radial_m"]), bins=bins, alpha=0.6, label="radial (along line of sight)")
    axes[1].hist(points["transverse_m"], bins=bins, alpha=0.6, label="transverse")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("error component (m)")
    axes[1].legend()
    axes[1].set_title("Radial error: t0 / sound speed. Transverse: TDOA / direction")
    save(fig, "error_distribution")

    # 4. Coverage curves.
    # plain coverage columns only: coverage_<d>m (not coverage_main_* / coverage_branch_*)
    dists = sorted(float(m.group(1)) for c in table.columns if (m := re.fullmatch(r"coverage_([0-9.]+)m", c)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for preset in presets:
        t = table[table["preset"] == preset]
        c = PRESET_COLORS.get(preset, "k")
        ax.plot(dists, [t[f"coverage_{d:g}m"].mean() for d in dists], "-o", color=c, label=f"{preset}: all")
        ax.plot(
            dists, [t[f"coverage_main_{d:g}m"].mean() for d in dists], "--", color=c, label=f"{preset}: main"
        )
    ax.set_xscale("log")
    ax.set_xlabel("distance d (m)")
    ax.set_ylabel("fraction of true channel length within d")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Coverage (completeness), mean over bolts")
    save(fig, "coverage")

    # 5. Per-bolt median error vs strike distance.
    fig, ax = plt.subplots(figsize=(7, 5))
    for preset in presets:
        t = table[table["preset"] == preset]
        ax.scatter(
            t["strike_distance_m"] / 1e3,
            t["point_error_median_m"],
            s=14,
            color=PRESET_COLORS.get(preset, "k"),
            label=preset,
        )
    ax.axhline(50, color="#c0392b", ls=":", label="provisional target 50 m")
    ax.set_yscale("log")
    ax.set_xlabel("strike distance (km)")
    ax.set_ylabel("per-bolt median point error (m)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "per_bolt_error")

    if docs_figures is not None:
        docs_figures.mkdir(parents=True, exist_ok=True)
        for name in saved:
            (docs_figures / f"{prefix}{name}.png").write_bytes((fig_dir / f"{name}.png").read_bytes())

    lines = _summary_markdown(summary, table, cfg, meta, presets)
    out = run_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _summary_markdown(
    summary: dict, table: pd.DataFrame, cfg: dict, meta: dict, presets: list[str]
) -> list[str]:
    o = summary["overall"]
    rc = cfg.get("reconstruction") or {}
    atmosphere = describe_atmosphere(rc.get("atmosphere"), oracle=True)
    truth_atm = describe_atmosphere(cfg.get("atmosphere"))
    lines = [
        f"Run `{meta['experiment']}`, seed {meta['seed']}, config hash `{meta['config_hash']}`, "
        f"commit `{meta['git']['commit'][:10]}`{' (dirty)' if meta['git'].get('dirty') else ''}.",
        "",
        f"Bolts: {o['n_bolts']} ({', '.join(presets)}), reconstructed points: {o['n_points']}, "
        f"bolts with no points: {o.get('bolts_with_no_points', 0)}. "
        f"Synthesis atmosphere: {truth_atm}. Atmosphere assumed by reconstruction: {atmosphere}.",
        "",
        "| Metric (95% bootstrap CI) | " + " | ".join(["all"] + presets) + " |",
        "|" + " --- |" * (len(presets) + 2),
    ]
    rows = [
        ("Pooled point error, median", "pooled_point_error_median_m", _ci),
        ("Pooled point error, 90th percentile", "pooled_point_error_p90_m", _ci),
        ("Per-bolt median error, median over bolts", "median_point_error_median_m", _ci),
        ("Radial error, median", "median_radial_error_median_m", _ci),
        ("Transverse error, median", "median_transverse_error_median_m", _ci),
        ("Coverage within 50 m (all channel)", "mean_coverage_50m", _pct),
        ("Coverage within 50 m (main channel)", "mean_coverage_main_50m", _pct),
        ("Coverage within 100 m (all channel)", "mean_coverage_100m", _pct),
        ("Strike-point error, median", "median_strike_error_m", lambda v: _ci(v, digits=0)),
        ("Reconstruction time per bolt, mean", "mean_time_reconstruct_s", lambda v: _ci(v, "s", 2)),
    ]
    for label, key, fmt in rows:
        cells = [fmt(summary[g][key]) if key in summary[g] else "n/a" for g in ["overall"] + presets]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines

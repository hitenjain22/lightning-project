"""E7: self-calibration (SPEC.md Phase 7). Run, or re-analyze a finished run folder.

Usage:
  python scripts/e7_self_calibration.py                       # run + analyze
  python scripts/e7_self_calibration.py --analyze RUN_DIR     # from the run folder only
  python scripts/e7_self_calibration.py --storms 2 --bolts 3 --layouts square_center   # smoke test

Storms of bolts at random azimuths share one true atmosphere, drawn per storm (wind speed,
direction and veer; lapse rate). Every method assumes the standard atmosphere (right surface
temperature, 6.5 K/km, no wind). On the same recordings:
  B_standard    Method B (best practical method in E4) with the standard atmosphere
  D_standard    Method D, standard atmosphere fixed (Bayesian error bars, no calibration)
  D_storm_n     Method D self-calibrating from the first n bolts of the storm jointly
                (n = 1: each bolt alone)
  D_oracle      Method D with the true atmosphere fixed (upper bound)
Atmosphere recovery is scored against the truth's path-averaged parameters (the same linear
fit in source height the prior uses), and in particular the path-averaged wind for a source
at 3 km height.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from thunder.atmosphere.profiles import build_atmosphere
from thunder.config import ChannelConfig, EvaluationConfig, RunConfig, load_config
from thunder.eval.aggregate import cluster_bootstrap_ci
from thunder.eval.metrics import evaluate
from thunder.experiments.pipeline import merge, run_bolt
from thunder.experiments.runner import make_run_dir, write_run_files
from thunder.experiments.sweeps import md_table, write_markdown
from thunder.recon import reconstruct, reconstruct_storm
from thunder.recon.bayes import prior_from_atmosphere

CONFIG = "configs/experiments/e7_self_calibration.yaml"
LAYOUTS = {
    "square_center": {"layout": "square_center", "aperture_m": 50.0},
    "mast": {"layout": "mast", "aperture_m": 50.0},
}
PRESETS = ["tortuous", "branched", "with_incloud"]
STORM_SIZES = [1, 2, 3, 6]
H_REF_M = 3000.0  # height at which the path-averaged wind is compared
DESIGN = {
    "wind_speed_mps": [0.0, 10.0],  # at 10 m, uniform
    "wind_direction_from_deg": [0.0, 360.0],
    "wind_shear_deg_per_km": [-20.0, 20.0],
    "lapse_rate_k_per_km": [4.5, 8.5],
}


def storm_atmosphere(base: dict, rng: np.random.Generator) -> dict:
    d = DESIGN
    return merge(
        base,
        {
            "lapse_rate_k_per_km": float(rng.uniform(*d["lapse_rate_k_per_km"])),
            "wind": {
                "speed_mps": float(rng.uniform(*d["wind_speed_mps"])),
                "direction_from_deg": float(rng.uniform(*d["wind_direction_from_deg"])),
                "shear_deg_per_km": float(rng.uniform(*d["wind_shear_deg_per_km"])),
            },
        },
    )


def _wind_at(theta: np.ndarray, h: float) -> np.ndarray:
    return np.array([theta[2] + theta[4] * h, theta[3] + theta[5] * h])


def _storm(args: tuple[dict, str, int, Any, int]) -> tuple[list[dict], list[dict]]:
    """One storm on one array (one linear-algebra thread per worker: reproducible, no
    oversubscription)."""
    with threadpool_limits(limits=1):
        return _storm_run(args)


def _storm_run(args: tuple[dict, str, int, Any, int]) -> tuple[list[dict], list[dict]]:
    """One storm on one array: synthesize its bolts, run every method, score."""
    cfg_dict, layout, s, seed, n_bolts = args
    base = RunConfig.model_validate(cfg_dict)
    rng = np.random.default_rng(seed)
    atm_true_dict = storm_atmosphere(cfg_dict["atmosphere"], rng)
    truth_cfg = RunConfig.model_validate(
        merge(cfg_dict, {"atmosphere": atm_true_dict, "array": LAYOUTS[layout], "reconstruction": None})
    )
    truth = build_atmosphere(truth_cfg.atmosphere)
    assert base.reconstruction is not None and base.reconstruction.atmosphere is not None
    standard = build_atmosphere(base.reconstruction.atmosphere)
    rcfg_d = base.reconstruction
    theta_true, _ = prior_from_atmosphere(truth, rcfg_d)
    wind_true = _wind_at(theta_true, H_REF_M)

    bolts = []
    # Derived without mutating `seed`, so every array sees exactly the same storm and bolts.
    bolt_seeds = [
        np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, 1000 + b)) for b in range(n_bolts)
    ]
    for b in range(n_bolts):
        az = float(rng.uniform(0.0, 360.0))
        ch_cfg = ChannelConfig.model_validate(
            {
                "preset": PRESETS[b % len(PRESETS)],
                "strike_distance_m": [1000.0, 3000.0],
                "strike_azimuth_deg": [az, az],
            }
        )
        res = run_bolt(truth_cfg, bolt_seeds[b], ch_cfg)
        assert res.recording is not None
        bolts.append((res, az))
    mics = bolts[0][0].recording.nominal_mic_positions  # type: ignore[union-attr]
    centroid = mics.mean(axis=0)

    rows: list[dict] = []
    theta_rows: list[dict] = []

    def score(variant: str, b: int, r, cpu: float) -> None:
        res, az = bolts[b]
        m, _ = evaluate(
            r.points, r.covariances, res.channel, centroid, EvaluationConfig(), r.extra.get("strike_point")
        )
        rows.append(
            {
                "storm": s,
                "layout": layout,
                "bolt": b,
                "azimuth_deg": az,
                "variant": variant,
                "wind_true_mps": atm_true_dict["wind"]["speed_mps"],
                "cpu_s": cpu,
                **m,
            }
        )

    def record_theta(variant: str, n: int, r) -> None:
        th, cov = r.extra.get("theta"), r.extra.get("theta_cov")
        if th is None or not r.extra.get("self_calibrated"):
            return
        est = _wind_at(th, H_REF_M)
        az = np.sort([bolts[b][1] for b in range(n)])
        gaps = np.diff(np.r_[az, az[0] + 360.0])
        span = 0.0 if n == 1 else float(360.0 - gaps.max())  # azimuth range the calibration sees
        theta_rows.append(
            {
                "storm": s,
                "layout": layout,
                "variant": variant,
                "n_bolts": n,
                **{
                    f"est_{k}": v
                    for k, v in zip(("c0", "c1", "w0x", "w0y", "w1x", "w1y"), th[:6], strict=True)
                },
                **{
                    f"sd_{k}": float(np.sqrt(cov[i, i]))
                    for i, k in enumerate(("c0", "c1", "w0x", "w0y", "w1x", "w1y"))
                },
                **{
                    f"true_{k}": v
                    for k, v in zip(("c0", "c1", "w0x", "w0y", "w1x", "w1y"), theta_true, strict=True)
                },
                "wind3km_true_e": wind_true[0],
                "wind3km_true_n": wind_true[1],
                "wind3km_est_e": est[0],
                "wind3km_est_n": est[1],
                "wind3km_error_mps": float(np.linalg.norm(est - wind_true)),
                "azimuth_span_deg": span,
            }
        )

    recs = [res.recording for res, _ in bolts]
    for b, rec in enumerate(recs):
        for variant, atm, rc in (
            ("B_standard", standard, rcfg_d.model_copy(update={"method": "B"})),
            ("D_standard", standard, rcfg_d.model_copy(update={"d_self_calibrate": False})),
            ("D_oracle", truth, rcfg_d.model_copy(update={"d_self_calibrate": False})),
            ("D_storm_1", standard, rcfg_d),
        ):
            t = time.process_time()
            r = reconstruct(rec, mics, atm, rc)
            score(variant, b, r, time.process_time() - t)
            if variant == "D_storm_1":
                record_theta(variant, 1, r)
    for n in STORM_SIZES[1:]:
        if n > n_bolts:
            continue
        t = time.process_time()
        out = reconstruct_storm(recs[:n], mics, standard, rcfg_d)
        cpu = (time.process_time() - t) / n
        for b, r in enumerate(out):
            score(f"D_storm_{n}", b, r, cpu)
        record_theta(f"D_storm_{n}", n, out[0])
    return rows, theta_rows


# --- analysis -----------------------------------------------------------------------

ORDER = ["B_standard", "D_standard", "D_storm_1", "D_storm_2", "D_storm_3", "D_storm_6", "D_oracle"]


def _cluster_ci(t: pd.DataFrame, col: str, stat, rng: np.random.Generator) -> tuple[float, float, float]:
    """Statistic of a per-bolt column with a bootstrap CI that resamples whole storms (bolts in a
    storm share the atmosphere, so they are not independent)."""
    v = t[col].to_numpy(dtype=float)
    ok = np.isfinite(v)
    return cluster_bootstrap_ci(v[ok], t["storm"].to_numpy()[ok], stat, rng, 1000)


def _method_row(layout: str, variant: str, t: pd.DataFrame, rng: np.random.Generator) -> dict:
    err = _cluster_ci(t, "point_error_median_m", np.median, rng)
    cov = _cluster_ci(t, "coverage_main_50m", np.mean, rng)
    return {
        "layout": layout,
        "variant": variant,
        "bolts": len(t),
        "median error (m)": f"{err[0]:.1f} [{err[1]:.1f}, {err[2]:.1f}]",
        "main cov. 50 m": f"{cov[0]:.0%} [{cov[1]:.0%}, {cov[2]:.0%}]",
        "calib 1/2/3 sigma": "/".join(f"{np.nanmean(t[f'calib_{k}sigma']):.2f}" for k in (1, 2, 3)),
        "angular (deg)": f"{np.nanmedian(t['angular_error_median_deg']):.3f}",
        "CPU per bolt (s)": f"{t['cpu_s'].median():.1f}",
        "_err": err[0],
        "_order": ORDER.index(variant) if variant in ORDER else 99,
    }


def _theta_row(layout: str, n: int, t: pd.DataFrame) -> dict:
    z = np.r_[(t["est_w0x"] - t["true_w0x"]) / t["sd_w0x"], (t["est_w0y"] - t["true_w0y"]) / t["sd_w0y"]]
    true_speed = np.hypot(t["wind3km_true_e"], t["wind3km_true_n"])
    return {
        "layout": layout,
        "bolts in storm": n,
        "storms": len(t),
        "wind error at 3 km (m/s, median)": f"{t['wind3km_error_mps'].median():.1f}",
        "true wind at 3 km (m/s, median)": f"{true_speed.median():.1f}",
        "c0 error (m/s, median abs)": f"{(t['est_c0'] - t['true_c0']).abs().median():.1f}",
        "c1 error (m/s per km, median abs)": f"{1000 * (t['est_c1'] - t['true_c1']).abs().median():.2f}",
        "wind z-score rms": f"{np.sqrt(np.mean(z**2)):.2f}",
    }


def analyze(run_dir: Path, docs_prefix: str | None) -> None:
    table = pd.read_csv(run_dir / "bolts.csv")
    theta = pd.read_csv(run_dir / "theta.csv")
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = [
        _method_row(layout, variant, t, rng) for (layout, variant), t in table.groupby(["layout", "variant"])
    ]
    summary = pd.DataFrame(rows).sort_values(["layout", "_order"])
    th_rows = [_theta_row(layout, int(n), t) for (layout, n), t in theta.groupby(["layout", "n_bolts"])]
    theta_summary = pd.DataFrame(th_rows)
    summary.drop(columns=["_err", "_order"]).to_csv(run_dir / "e7_summary_table.csv", index=False)
    theta_summary.to_csv(run_dir / "e7_theta_table.csv", index=False)

    _plot_methods(table, figs / "methods.png")
    _plot_storm_size(table, theta, figs / "storm_size.png")
    _plot_wind(theta, figs / "wind_recovery.png")
    _plot_vs_wind(table, figs / "error_vs_wind.png")
    _plot_span(theta, figs / "azimuth_span.png")
    design = json.loads((run_dir / "e7_design.json").read_text())
    lines = [
        f"# E7 self-calibration: {run_dir.name}",
        "",
        f"{design['storms']} storms x {design['bolts_per_storm']} bolts per array; "
        f"true atmosphere per storm: {design['design']}. Brackets: 95% bootstrap CIs, resampling storms.",
        "",
        md_table(summary.drop(columns=["_err", "_order"]).reset_index(drop=True)),
        "",
        "## Atmosphere recovery (path-averaged parameters; wind for a source at 3 km)",
        "",
        md_table(theta_summary),
    ]
    write_markdown(run_dir / "e7_summary.md", "\n".join(lines))
    if docs_prefix is not None:
        for p in figs.glob("*.png"):
            shutil.copy(p, Path("docs/figures") / f"{docs_prefix or 'e7_'}{p.name}")
    print((run_dir / "e7_summary.md").read_text())


def _variant_stat(table: pd.DataFrame, layout: str, col: str, stat) -> pd.Series:
    t = table[table["layout"] == layout]
    return t.groupby("variant")[col].agg(stat).reindex([v for v in ORDER if v in set(t["variant"])])


def _plot_methods(table: pd.DataFrame, path: Path) -> None:
    layouts = sorted(table["layout"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    w = 0.8 / len(layouts)
    for j, layout in enumerate(layouts):
        err = _variant_stat(table, layout, "point_error_median_m", "median")
        cov = _variant_stat(table, layout, "coverage_main_50m", "mean")
        cal = _variant_stat(table, layout, "calib_2sigma", "mean")
        x = np.arange(len(err))
        for ax, v in zip(axes, (err, cov, cal), strict=True):
            ax.bar(x + (j - (len(layouts) - 1) / 2) * w, v.to_numpy(dtype=float), w, label=layout)
            ax.set_xticks(x, [s.replace("_", "\n") for s in v.index], fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median point error (m)")
    axes[1].set_ylabel("main coverage within 50 m")
    axes[1].set_ylim(0, 1)
    axes[2].set_ylabel("fraction inside 2-sigma ellipsoid")
    axes[2].axhline(0.739, color="k", ls=":", label="nominal 0.739")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("E7: assumed standard atmosphere vs self-calibration vs truth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_storm_size(table: pd.DataFrame, theta: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for layout in sorted(table["layout"].unique()):
        sizes, err = [], []
        for n in STORM_SIZES:
            t = table[(table["layout"] == layout) & (table["variant"] == f"D_storm_{n}")]
            if len(t):
                sizes.append(n)
                err.append(float(t["point_error_median_m"].median()))
        axes[0].plot(sizes, err, "o-", label=f"{layout} (self-calibrated)")
        for ref, ls in (("D_standard", ":"), ("D_oracle", "--")):
            v = table[(table["layout"] == layout) & (table["variant"] == ref)][
                "point_error_median_m"
            ].median()
            axes[0].axhline(v, ls=ls, color=axes[0].lines[-1].get_color(), alpha=0.7, label=f"{layout} {ref}")
        th = theta[theta["layout"] == layout].groupby("n_bolts")["wind3km_error_mps"].median()
        axes[1].plot(th.index, th.to_numpy(), "o-", label=layout)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bolts in the storm (calibrated jointly)")
    axes[0].set_ylabel("median point error (m)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("bolts in the storm")
    axes[1].set_ylabel("wind error at 3 km (m/s, median)")
    axes[1].legend(fontsize=8)
    fig.suptitle("E7: more bolts at more azimuths calibrate the wind")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_wind(theta: pd.DataFrame, path: Path) -> None:
    sizes = sorted(theta["n_bolts"].unique())
    pick = [sizes[0], sizes[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, n in zip(axes, pick, strict=True):
        t = theta[theta["n_bolts"] == n]
        for comp, marker in (("e", "o"), ("n", "s")):
            ax.scatter(
                t[f"wind3km_true_{comp}"],
                t[f"wind3km_est_{comp}"],
                marker=marker,
                label=f"{'east' if comp == 'e' else 'north'} component",
            )
        lim = [-15, 15]
        ax.plot(lim, lim, "k:", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("true path-averaged wind at 3 km (m/s)")
        ax.set_ylabel("self-calibrated estimate (m/s)")
        ax.set_title(f"{n} bolt{'s' if n > 1 else ''} per calibration")
        ax.legend(fontsize=8)
    fig.suptitle("E7: wind recovered by self-calibration (both arrays)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_span(theta: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for n, t in theta.groupby("n_bolts"):
        rel = t["wind3km_error_mps"] / np.maximum(np.hypot(t["wind3km_true_e"], t["wind3km_true_n"]), 1.0)
        ax.scatter(t["azimuth_span_deg"], rel, label=f"{n} bolt{'s' if n > 1 else ''}")
    ax.set_xlabel("azimuth range covered by the calibrating bolts (deg)")
    ax.set_ylabel("wind error / true wind at 3 km")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.legend(fontsize=8)
    ax.set_title("E7: the cross-wind needs bolts at different azimuths")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_vs_wind(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for variant, color in (
        ("B_standard", "tab:gray"),
        ("D_storm_1", "tab:orange"),
        ("D_storm_6", "tab:green"),
        ("D_oracle", "tab:blue"),
    ):
        t = table[(table["variant"] == variant) & (table["layout"] == "square_center")]
        if len(t):
            g = t.groupby("storm").agg(w=("wind_true_mps", "first"), e=("point_error_median_m", "median"))
            ax.scatter(g["w"], g["e"], color=color, label=variant)
    ax.set_yscale("log")
    ax.set_xlabel("true wind speed at 10 m (m/s)")
    ax.set_ylabel("median point error per storm (m)")
    ax.legend(fontsize=8)
    ax.set_title("E7: error vs wind (square + center)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--storms", type=int, default=12)
    ap.add_argument("--bolts", type=int, default=6)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--docs-prefix", default=None)
    ap.add_argument("--results-root")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze, args.docs_prefix)
        return
    cfg = load_config(CONFIG)
    if args.results_root:
        cfg = cfg.model_copy(update={"results_root": args.results_root})
    run_dir = make_run_dir(cfg)
    cfg_dict = cfg.model_dump(mode="json")
    seeds = np.random.SeedSequence(cfg.seed).spawn(args.storms)  # same storms for every array
    tasks = [
        (cfg_dict, layout, s, seeds[s], args.bolts) for s in range(args.storms) for layout in args.layouts
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(_storm, tasks))
    pd.DataFrame([r for o in outputs for r in o[0]]).to_csv(run_dir / "bolts.csv", index=False)
    pd.DataFrame([r for o in outputs for r in o[1]]).to_csv(run_dir / "theta.csv", index=False)
    design = {
        "storms": args.storms,
        "bolts_per_storm": args.bolts,
        "layouts": args.layouts,
        "design": DESIGN,
        "storm_sizes": STORM_SIZES,
        "wind_reference_height_m": H_REF_M,
    }
    (run_dir / "e7_design.json").write_text(json.dumps(design, indent=2))
    write_run_files(cfg, run_dir, {"design": design}, {})
    print(f"Wrote {run_dir}", flush=True)
    analyze(run_dir, args.docs_prefix)


if __name__ == "__main__":
    main()

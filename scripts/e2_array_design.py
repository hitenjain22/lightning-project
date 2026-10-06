"""E2: array design (SPEC.md Phase 7). Run, or re-analyze a finished run folder.

Usage:
  python scripts/e2_array_design.py                       # optimize, run, analyze
  python scripts/e2_array_design.py --analyze RUN_DIR     # figures + tables from a run folder
  python scripts/e2_array_design.py --n-bolts 3 --maxiter 20 --docs-prefix ""   # smoke test

Steps (SPEC): 1 parametric sweep (layout family x mic count x aperture), 2 Cramér-Rao surrogate
for every layout (thunder.experiments.design), 3 A- and D-optimal free layouts by differential
evolution, 4 full simulation of everything on the same bolts: does the surrogate's ranking hold?
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from thunder.config import ArrayConfig, ChannelConfig, load_config
from thunder.eval.metrics import sample_channel
from thunder.experiments.design import angular_variance, evaluate_layout, optimize_layout, plane_wave_bias
from thunder.experiments.montecarlo import channel_overrides
from thunder.experiments.pipeline import run_bolt
from thunder.experiments.sweeps import (
    ci_text,
    load_run,
    md_table,
    paired_ratio,
    run_variants,
    variant_table,
    write_markdown,
)
from thunder.sensors.arrays import nominal_positions

CONFIG = "configs/experiments/e2_array_design.yaml"
SIGMA_T = 1e-4  # timing noise for the surrogate (s); rankings do not depend on it
APERTURES = [5.0, 15.0, 50.0, 150.0, 500.0]
DESIGN_APERTURE = 50.0
BASELINE = "square_center_50_A"

# Parametric layouts at the design aperture (step 1): name -> array config.
PARAMETRIC: dict[str, dict] = {
    "triangle": {"layout": "triangle"},
    "square": {"layout": "square"},
    "square_center": {"layout": "square_center"},
    "circle6": {"layout": "circle", "n_mics": 6},
    "circle8": {"layout": "circle", "n_mics": 8},
    "circle12": {"layout": "circle", "n_mics": 12},
    "l_shape5": {"layout": "l_shape", "n_mics": 5},
    "l_shape9": {"layout": "l_shape", "n_mics": 9},
    "cross5": {"layout": "cross", "n_mics": 5},
    "cross9": {"layout": "cross", "n_mics": 9},
    "mast": {"layout": "mast"},
    "random_disk6": {"layout": "random_disk", "n_mics": 6},
    "random_disk12": {"layout": "random_disk", "n_mics": 12},
}
APERTURE_SWEEP = ["square_center", "circle8", "mast"]  # every aperture, Methods A and C
# Optimized layouts (step 3): (n_mics, criterion, mast).
OPTIMIZED = [
    (5, "A", False),
    (5, "D", False),
    (8, "A", False),
    (8, "D", False),
    (12, "A", False),
    (12, "D", False),
    (5, "A", True),
]
# Distributed: three 20 m triangles (9 mics) at two separations.
DISTRIBUTED = {
    f"distributed{int(s)}": {"layout": "distributed", "aperture_m": 20.0, "subarray_separation_m": s}
    for s in (300.0, 1000.0)
}


def _optimize(args: tuple[int, str, bool, int, float]) -> tuple[str, list, dict]:
    n, crit, mast, maxiter, min_sep = args
    o = optimize_layout(
        n,
        DESIGN_APERTURE,
        crit,
        mast=mast,
        sigma_t=SIGMA_T,
        maxiter=maxiter,
        seed=n,
        min_separation_frac=min_sep,
    )
    name = f"opt{crit}{n}" + ("_mast" if mast else "")
    info = {
        "rms_deg": o.surrogate.rms_angular_error_deg,
        "logdet": o.surrogate.mean_log_det,
        "aperture_m": o.aperture_m,
        "min_separation_frac": min_sep,
    }
    return name, np.round(o.positions, 4).tolist(), info


def surrogate(array: dict, draws: int = 200) -> dict[str, float]:
    """CRB summary of a layout; random layouts are averaged over `draws` placements."""
    cfg = ArrayConfig.model_validate(array)
    rng = np.random.default_rng(0)
    reps = draws if cfg.layout == "random_disk" else 1
    s = [evaluate_layout(nominal_positions(cfg, rng), SIGMA_T) for _ in range(reps)]
    rms2 = np.mean([x.rms_angular_error_deg**2 for x in s])  # mean variance, then root
    return {
        "crb_rms_deg": float(np.sqrt(rms2)),
        "crb_worst_deg": float(np.median([x.worst_angular_error_deg for x in s])),
        "crb_logdet": float(np.mean([x.mean_log_det for x in s])),
    }


def build(
    maxiter: int, workers: int, min_sep: float = 0.2, only: str | None = None
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Variants (array/method overrides, grouped so shared stages are adjacent) and their labels."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        opt = list(pool.map(_optimize, [(*o, maxiter, min_sep) for o in OPTIMIZED]))
    arrays: list[tuple[str, dict, dict]] = []  # (label, array config, extra meta)
    for name, a in PARAMETRIC.items():
        for ap in APERTURES if name in APERTURE_SWEEP else [DESIGN_APERTURE]:
            arrays.append((name, {**a, "aperture_m": ap}, {"family": name, "optimized": False}))
    for name, pos, info in opt:
        arrays.append(
            (
                name,
                {"layout": "free_form", "positions_m": pos, "aperture_m": DESIGN_APERTURE},
                {"family": name, "optimized": True, "optimizer": info},
            )
        )
    for name, a in DISTRIBUTED.items():
        arrays.append((name, a, {"family": "distributed", "optimized": False}))

    variants: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for label, a, extra in arrays:
        cfg = ArrayConfig.model_validate(a)
        n = len(nominal_positions(cfg, np.random.default_rng(0)))
        ap = cfg.aperture_m if cfg.layout != "distributed" else cfg.subarray_separation_m
        large = label in APERTURE_SWEEP or cfg.layout == "distributed"
        for method in ("A", "C") if large else ("A",):
            name = f"{label}_{ap:g}_{method}"
            variants[name] = {"array": a, "reconstruction": {"method": method}}
            meta[name] = {
                "layout": label,
                "n_mics": n,
                "aperture_m": ap,
                "method": method,
                **extra,
                **surrogate(a),
            }
    if only is not None:
        keep = [v for v in variants if re.fullmatch(only, v)]
        variants = {v: variants[v] for v in keep}
        meta = {v: meta[v] for v in keep}
    return variants, meta


# --- analysis -----------------------------------------------------------------------


def channel_points(n_bolts: int, spacing: float = 40.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """(points (K, 3), length weights (K,)) along each simulated bolt's true channel.

    Regenerates exactly the run's channels (same seeds, presets and overrides; channels only).
    """
    cfg = load_config(CONFIG)
    assert cfg.monte_carlo is not None
    cfg = cfg.model_copy(update={"synthesis": None})
    overrides = channel_overrides(cfg)
    presets = cfg.monte_carlo.presets
    out = []
    for k, seed in enumerate(np.random.SeedSequence(cfg.seed).spawn(n_bolts)):
        ch_cfg = ChannelConfig.model_validate({**overrides, "preset": presets[k % len(presets)]})
        ch = run_bolt(cfg, seed, ch_cfg).channel
        pts, w, _ = sample_channel(ch, spacing)
        out.append((pts, w))
    return out


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(v)
    cdf = np.cumsum(w[order]) / w.sum()
    return float(v[order][np.searchsorted(cdf, 0.5)])


def channel_surrogate(
    array: dict, chans: list, draws: int = 5
) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Per draw (random layouts) and bolt: (bound sd, plane-wave bias, weight) at each channel point.

    sd: Cramér-Rao spread (deg) at SIGMA_T for the point's direction from the array centroid;
    bias: far-field (plane-wave) direction error from wavefront curvature (deg), exact.
    """
    cfg = ArrayConfig.model_validate(array)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(draws if cfg.layout == "random_disk" else 1):
        p = nominal_positions(cfg, rng)
        cen = p.mean(axis=0)
        per_bolt = []
        for pts, w in chans:
            v = pts - cen
            az = np.arctan2(v[:, 0], v[:, 1])
            el = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))
            sd = np.degrees(np.sqrt(angular_variance(p, az, el, SIGMA_T, 343.0)))
            per_bolt.append((sd, plane_wave_bias(p, pts), w))
        out.append(per_bolt)
    return out


def predicted_error(surr: list, k: float, method: str) -> float:
    """Median over bolts of the length-weighted median of sqrt((k sd)^2 + bias^2) (deg); Method C
    models curvature, so it has no bias term. Random layouts: mean over draws."""
    vals = []
    for per_bolt in surr:
        med = []
        for sd, bias, w in per_bolt:
            e = np.sqrt((k * sd) ** 2 + (0.0 if method == "C" else bias**2))
            med.append(_weighted_median(e, w))
        vals.append(float(np.median(med)))
    return float(np.mean(vals))


def bound_only(surr: list) -> float:
    return predicted_error(surr, 1.0, "C")


def median_bias(surr: list) -> float:
    return float(np.mean([np.median([_weighted_median(b, w) for _, b, w in pb]) for pb in surr]))


def analyze(run_dir: Path, docs_prefix: str | None) -> None:
    table, _, cfg, meta = load_run(run_dir)
    vt = variant_table(table)
    m = pd.DataFrame(meta).T.infer_objects()
    vt = vt.join(m)
    ratio = paired_ratio(table, "angular_error_median_deg", BASELINE)
    vt["ang_ratio"], vt["ang_ratio_lo"], vt["ang_ratio_hi"] = ratio["ratio"], ratio["lo"], ratio["hi"]
    vt.to_csv(run_dir / "e2_table.csv")
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)

    chans = channel_points(int(vt["bolts"].max()))
    variants = cfg["monte_carlo"]["variants"]
    by_array: dict[str, list] = {}  # Methods A and C share arrays: compute each once
    for v in vt.index:
        key = json.dumps(variants[v]["array"], sort_keys=True)
        if key not in by_array:
            by_array[key] = channel_surrogate(variants[v]["array"], chans)
    surr = {v: by_array[json.dumps(variants[v]["array"], sort_keys=True)] for v in vt.index}
    vt["crb_channel_deg"] = [bound_only(surr[v]) for v in vt.index]
    vt["curvature_bias_deg"] = [median_bias(surr[v]) for v in vt.index]
    design = vt[(vt["aperture_m"] == DESIGN_APERTURE) & (vt["method"] == "A")].copy()
    # Calibrate the effective timing noise on layouts without curvature bias (symmetric ones).
    sym = design[design["curvature_bias_deg"] < 0.01]
    k_ch = float(np.median(sym["angular_error_median_deg"] / sym["crb_channel_deg"]))
    vt["predicted_deg"] = [predicted_error(surr[v], k_ch, str(vt.loc[v, "method"])) for v in vt.index]
    vt.to_csv(run_dir / "e2_table.csv")
    design = vt[(vt["aperture_m"] == DESIGN_APERTURE) & (vt["method"] == "A")].copy()
    design = design.sort_values("crb_rms_deg")
    rho, p_rho = spearmanr(design["crb_rms_deg"], design["angular_error_median_deg"], nan_policy="omit")
    rho_ch, p_ch = spearmanr(design["crb_channel_deg"], design["angular_error_median_deg"], nan_policy="omit")
    rho_pr, p_pr = spearmanr(design["predicted_deg"], design["angular_error_median_deg"], nan_policy="omit")
    rho_c, p_c = spearmanr(design["crb_rms_deg"], design["coverage_main_50m"], nan_policy="omit")
    # Implied effective timing noise: for a 2-D Gaussian direction error, median = sqrt(ln 2) * RMS.
    k = float(np.median(design["angular_error_median_deg"] / design["crb_rms_deg"]))
    sigma_eff = SIGMA_T * k / np.sqrt(np.log(2))

    _plot_layouts(vt, cfg, figs / "layouts.png")
    _plot_crb_vs_sim(design, rho, rho_ch, rho_pr, figs / "crb_vs_sim.png")
    _plot_aperture(vt, k, figs / "aperture.png")
    _plot_mic_count(vt, k, figs / "mic_count.png")

    lines = [
        f"# E2 array design: {run_dir.name}",
        "",
        f"{int(vt['bolts'].max())} bolts, {len(vt)} variants. Brackets: 95% bootstrap CIs over bolts. "
        f"`ang ratio`: per-bolt angular error relative to {BASELINE}, median over bolts (paired).",
        "",
        f"Spearman rank correlation, CRB vs simulated angular error at {DESIGN_APERTURE:g} m (Method A): "
        f"rho = {rho:.2f} (p = {p_rho:.2g}); "
        f"CRB vs main-channel coverage: rho = {rho_c:.2f} (p = {p_c:.2g}).",
        f"Simulated median / CRB RMS = {k:.2f}, i.e. effective timing noise ~ {sigma_eff * 1e6:.0f} us.",
        f"Channel-weighted bound (directions of the simulated channels; median of per-point bounds): "
        f"rho = {rho_ch:.2f} (p = {p_ch:.2g}).",
        f"Bias-aware prediction sqrt((k bound)^2 + curvature bias^2), k = {k_ch:.2f} calibrated on the "
        f"{len(sym)} layouts without curvature bias "
        f"(effective timing noise ~ {SIGMA_T * k_ch * 1e6:.0f} us): "
        f"rho = {rho_pr:.2f} (p = {p_pr:.2g}).",
        "",
        "## Layouts at the design aperture (Method A)",
        "",
        md_table(_rows(design)),
        "",
        "## Aperture sweep",
        "",
        md_table(
            _rows(vt[vt["layout"].isin(APERTURE_SWEEP)].sort_values(["layout", "method", "aperture_m"]))
        ),
        "",
        "## Distributed arrays",
        "",
        md_table(_rows(vt[vt["family"] == "distributed"])),
    ]
    write_markdown(run_dir / "e2_summary.md", "\n".join(lines))
    _copy_figures(figs, docs_prefix, "e2_")
    print((run_dir / "e2_summary.md").read_text())


def _rows(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": df.index,
            "mics": df["n_mics"].astype(int),
            "CRB rms (deg)": df["crb_rms_deg"].astype(float).map(lambda v: f"{v:.3f}"),
            "CRB channel (deg)": df["crb_channel_deg"].astype(float).map(lambda v: f"{v:.3f}")
            if "crb_channel_deg" in df
            else "-",
            "curvature bias (deg)": df["curvature_bias_deg"].astype(float).map(lambda v: f"{v:.3f}")
            if "curvature_bias_deg" in df
            else "-",
            "predicted (deg)": df["predicted_deg"].astype(float).map(lambda v: f"{v:.3f}")
            if "predicted_deg" in df
            else "-",
            "angular error (deg)": [ci_text(r, "angular_error_median_deg", ".3f") for _, r in df.iterrows()],
            "ang ratio": [ci_text(r, "ang_ratio", ".2f") for _, r in df.iterrows()],
            "point error (m)": [ci_text(r, "point_error_median_m", ".1f") for _, r in df.iterrows()],
            "main cov. 50 m": df["coverage_main_50m"].map(lambda v: f"{v:.0%}"),
            "bolts w/ points": df["bolts_with_points"].astype(int).astype(str)
            + "/"
            + df["bolts"].astype(str),
            "windows passed": df["windows_passed_frac"].map(lambda v: f"{v:.0%}"),
            "recon CPU (s)": df["time_reconstruct_cpu_s"].map(lambda v: f"{v:.1f}"),
        }
    )


def _positions(array: dict) -> np.ndarray:
    return nominal_positions(ArrayConfig.model_validate(array), np.random.default_rng(0))


def _plot_layouts(vt: pd.DataFrame, cfg: dict, path: Path) -> None:
    variants = cfg["monte_carlo"]["variants"]
    names = [v for v in vt.index if vt.loc[v, "aperture_m"] == DESIGN_APERTURE and vt.loc[v, "method"] == "A"]
    names = sorted(names, key=lambda v: (bool(vt.loc[v, "optimized"]), int(vt.loc[v, "n_mics"]), v))
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow), squeeze=False)
    for ax, v in zip(axes.flat, names, strict=False):
        p = _positions(variants[v]["array"])
        raised = p[:, 2] > 2.0
        ax.scatter(p[~raised, 0], p[~raised, 1], s=28, c="tab:blue")
        ax.scatter(p[raised, 0], p[raised, 1], s=60, marker="^", c="tab:red")
        for q in p[raised]:
            ax.annotate(f"{q[2]:.0f} m", (q[0], q[1]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        circ = plt.Circle((0, 0), DESIGN_APERTURE / 2, fill=False, ls=":", color="0.6")
        ax.add_patch(circ)
        ax.set_xlim(-32, 32)
        ax.set_ylim(-32, 32)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        color = "tab:red" if vt.loc[v, "optimized"] else "black"
        ax.set_title(f"{vt.loc[v, 'layout']}  CRB {vt.loc[v, 'crb_rms_deg']:.3f}°", fontsize=8, color=color)
    for ax in list(axes.flat)[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        f"E2 layouts at {DESIGN_APERTURE:g} m aperture (red: CRB-optimized; dotted: aperture circle)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _merged_labels(names, x, y) -> dict[tuple[float, float], str]:
    """One label per point; layouts that coincide (e.g. an optimum equal to a polygon) share it."""
    out: dict[tuple[float, float], str] = {}
    for n, xi, yi in zip(names, x, y, strict=True):
        key = (float(f"{float(xi):.4g}"), float(f"{float(yi):.4g}"))
        out[key] = f"{out[key]} = {n}" if key in out else str(n)
    return out


def _plot_crb_vs_sim(design: pd.DataFrame, rho: float, rho_ch: float, rho_pr: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axes.flat
    y = design["angular_error_median_deg"]
    err = np.vstack([y - design["angular_error_median_deg_lo"], design["angular_error_median_deg_hi"] - y])
    colors = ["tab:red" if o else "tab:blue" for o in design["optimized"]]
    panels = [
        ("crb_rms_deg", f"(a) bound over a uniform direction grid, σt = {SIGMA_T * 1e6:.0f} µs (deg)", rho),
        ("crb_channel_deg", "(b) bound over the simulated channels' directions (deg)", rho_ch),
        ("predicted_deg", "(c) bias-aware: √((k·bound)² + curvature bias²) (deg)", rho_pr),
    ]
    for a, (col, label, r) in zip(list(ax)[:3], panels, strict=True):
        x = design[col].astype(float)
        a.errorbar(x, y, yerr=err, fmt="none", ecolor="0.6", lw=1)
        a.scatter(x, y, c=colors, zorder=3)
        for (xi, yi), name in _merged_labels(design["layout"], x, y).items():
            a.annotate(name, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(label)
        a.set_ylabel("simulated median angular error (deg)")
        a.set_title(f"Spearman ρ = {r:.2f}")
        if col == "predicted_deg":
            lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25]
            a.plot(lim, lim, ":", color="0.5", label="1:1")
            a.legend(fontsize=8)
    x = design["crb_channel_deg"].astype(float)
    ax[3].scatter(x, design["coverage_main_50m"], c=colors)
    for (xi, yi), name in _merged_labels(design["layout"], x, design["coverage_main_50m"]).items():
        ax[3].annotate(name, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[3].set_xscale("log")
    ax[3].set_ylim(0.5, 1.0)
    ax[3].set_xlabel("(d) bound over the simulated channels' directions (deg)")
    ax[3].set_ylabel("main-channel coverage within 50 m")
    ax[3].set_title("Coverage is outside the surrogate")
    fig.suptitle(
        f"Does the surrogate rank layouts? ({DESIGN_APERTURE:g} m aperture, Method A; red: CRB-optimized)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_aperture(vt: pd.DataFrame, k: float, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for layout, color in zip(APERTURE_SWEEP, ["tab:blue", "tab:orange", "tab:green"], strict=True):
        for method, ls in (("A", "-"), ("C", "--")):
            d = vt[(vt["layout"] == layout) & (vt["method"] == method)].sort_values("aperture_m")
            ax[0].plot(
                d["aperture_m"],
                d["angular_error_median_deg"],
                ls,
                marker="o",
                color=color,
                label=f"{layout} {method}",
            )
            ax[1].plot(
                d["aperture_m"],
                d["coverage_main_50m"],
                ls,
                marker="o",
                color=color,
                label=f"{layout} {method}",
            )
        d = vt[(vt["layout"] == layout) & (vt["method"] == "A")].sort_values("aperture_m")
        ax[0].plot(d["aperture_m"], k * d["crb_rms_deg"].astype(float), ":", color=color, alpha=0.7)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("aperture (m)")
    ax[0].set_ylabel("median angular error (deg)")
    ax[0].set_title(f"Angular error vs aperture (dotted: {k:.2f} × CRB)")
    ax[0].legend(fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("aperture (m)")
    ax[1].set_ylabel("main-channel coverage within 50 m")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Coverage vs aperture")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mic_count(vt: pd.DataFrame, k: float, path: Path) -> None:
    d = vt[(vt["aperture_m"] == DESIGN_APERTURE) & (vt["method"] == "A")]
    ring = d[d["layout"].isin(["triangle", "square", "circle6", "circle8", "circle12"])].sort_values("n_mics")
    opt_a = d[d["layout"].str.match(r"optA\d+$")].sort_values("n_mics")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(ring["n_mics"], ring["angular_error_median_deg"], "o-", label="ring (simulated)")
    ax[0].plot(ring["n_mics"], k * ring["crb_rms_deg"].astype(float), ":", label=f"ring ({k:.2f} × CRB)")
    ax[0].plot(opt_a["n_mics"], opt_a["angular_error_median_deg"], "s", color="tab:red", label="A-optimal")
    n = np.arange(3, 13)
    r0 = ring.iloc[0]
    ax[0].plot(
        n, r0["angular_error_median_deg"] * np.sqrt(r0["n_mics"] / n), "--", color="0.6", label="∝ 1/√n"
    )
    ax[0].set_xlabel("number of mics")
    ax[0].set_ylabel("median angular error (deg)")
    ax[0].legend(fontsize=8)
    ax[1].plot(ring["n_mics"], ring["coverage_main_50m"], "o-", label="ring")
    ax[1].plot(opt_a["n_mics"], opt_a["coverage_main_50m"], "s", color="tab:red", label="A-optimal")
    ax[1].set_xlabel("number of mics")
    ax[1].set_ylabel("main-channel coverage within 50 m")
    ax[1].set_ylim(0.5, 1.0)  # fixed range: autoscaling made few-point differences look large
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Mic count at {DESIGN_APERTURE:g} m aperture")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _copy_figures(figs: Path, docs_prefix: str | None, default: str) -> None:
    if docs_prefix is None:
        return
    prefix = docs_prefix or default
    out = Path("docs/figures")
    for f in figs.glob("*.png"):
        shutil.copy(f, out / f"{prefix}{f.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", type=Path, help="re-analyze this run folder (no simulation)")
    ap.add_argument("--n-bolts", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--maxiter", type=int, default=300, help="differential evolution iterations")
    ap.add_argument("--docs-prefix", default=None, help="copy figures to docs/figures/<prefix>*.png")
    ap.add_argument("--results-root")
    ap.add_argument(
        "--min-separation",
        type=float,
        default=0.2,
        help="minimum mic spacing of optimized layouts, as a fraction of the aperture",
    )
    ap.add_argument("--only", help="regex: run only matching variants (short paired report, no figures)")
    ap.add_argument("--experiment", help="override the experiment name (run folder)")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze, args.docs_prefix)
        return
    cfg = load_config(CONFIG)
    mc = {k: v for k, v in (("n_bolts", args.n_bolts), ("workers", args.workers)) if v is not None}
    upd = {"monte_carlo": cfg.monte_carlo.model_copy(update=mc)} if cfg.monte_carlo else {}
    if args.results_root:
        upd["results_root"] = args.results_root
    if args.experiment:
        upd["experiment"] = args.experiment
    cfg = cfg.model_copy(update=upd)
    variants, meta = build(args.maxiter, args.workers or 4, args.min_separation, args.only)
    print(f"{len(variants)} variants")
    run_dir = run_variants(cfg, variants, meta)
    print(f"Wrote {run_dir}")
    if args.only is None:
        analyze(run_dir, args.docs_prefix)
    else:
        short_report(run_dir)


def short_report(run_dir: Path) -> None:
    """Paired comparison of a subset run (no figures)."""
    table, _, _, meta = load_run(run_dir)
    vt = variant_table(table).join(pd.DataFrame(meta).T.infer_objects())
    base = BASELINE if BASELINE in set(table["variant"]) else str(table["variant"].iloc[0])
    r = paired_ratio(table, "angular_error_median_deg", base)
    vt["ang_ratio"], vt["ang_ratio_lo"], vt["ang_ratio_hi"] = r["ratio"], r["lo"], r["hi"]
    vt.to_csv(run_dir / "e2_table.csv")
    text = f"# E2 subset run: {run_dir.name} (ratios vs {base})\n\n" + md_table(_rows(vt))
    write_markdown(run_dir / "e2_summary.md", text)
    print(text)


if __name__ == "__main__":
    main()

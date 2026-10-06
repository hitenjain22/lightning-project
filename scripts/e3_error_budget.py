"""E3: robustness and error budget (SPEC.md Phase 7). Run, or re-analyze finished run folders.

Usage:
  python scripts/e3_error_budget.py                                   # both phases + analysis
  python scripts/e3_error_budget.py --analyze OAT_DIR [PAIR_DIR]      # from run folders only
  python scripts/e3_error_budget.py --n-bolts 3 --docs-prefix ""      # smoke test

Phase 1 varies one factor at a time from the baseline (E2's realistic field kit, 5 mics, 50 m,
still air known to the reconstruction): SNR, clock sync, mic position error, flash-time (t0)
error, unknown wind, and the assumed temperature. Phase 2 sweeps the two factors that hurt most
at their realistic "poor" levels jointly. Both phases use the same seed, hence the same bolts.
Outputs: tables, a tornado chart, sweep curves, a pairwise heatmap and the error budget
(largest tolerable level of each factor for given accuracy targets).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thunder.config import RunConfig, load_config
from thunder.experiments.sweeps import (
    crossing,
    load_run,
    md_table,
    paired_ratio,
    per_variant,
    run_variants,
    write_markdown,
)

CONFIG = "configs/experiments/e3_error_budget.yaml"
T_SURFACE = 25.0  # the base config's temperature (C)
STILL = {"model": "uniform", "temperature_c": T_SURFACE}


def _plain(v: float) -> str:
    return f"{v:g}"


@dataclass(frozen=True)
class Factor:
    name: str
    label: str  # axis label
    levels: list[float]
    override: Callable[[float], dict]
    good: float  # realistic good-hardware level
    poor: float  # realistic poor-hardware level
    sign: float = 1.0  # +1: larger level is worse; -1: smaller is worse (SNR)
    log: bool = True  # log axis / log interpolation
    fmt: Callable[[float], str] = _plain
    pair_levels: tuple[float, ...] = ()


def _ms(v: float) -> str:
    return f"{v * 1e3:g} ms" if v >= 1e-3 else f"{v * 1e6:g} us"


FACTORS = [
    Factor(
        "snr",
        "background SNR in 10-300 Hz (dB)",
        [40, 30, 20, 10, 5, 0, -5, -10],
        lambda v: {"sensors": {"noise": {"background_db_spl": None, "background_snr_db": float(v)}}},
        good=30,
        poor=0,
        sign=-1.0,
        log=False,
        fmt=lambda v: f"{v:g} dB",
        pair_levels=(30, 10, 0, -5, -10),
    ),
    Factor(
        "sync",
        "clock offset std (s)",
        [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        lambda v: {
            "sensors": {
                "timing": {
                    "preset": "perfect",
                    "offset_std_s": float(v),
                    "drift_std_ppm": 0.0,
                    "jitter_std_s": 0.0,
                    "common_clock": False,
                }
            }
        },
        good=1e-6,
        poor=3e-3,
        fmt=_ms,
        pair_levels=(1e-6, 3e-5, 1e-4, 3e-4, 1e-3),
    ),
    Factor(
        "position",
        "mic position error std per axis (m)",
        [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        lambda v: {
            "sensors": {
                "position": {"preset": "exact", "horizontal_std_m": float(v), "vertical_std_m": float(v)}
            }
        },
        good=0.01,
        poor=3.0,
        fmt=lambda v: f"{v * 100:g} cm" if v < 1 else f"{v:g} m",
        pair_levels=(0.01, 0.1, 0.3, 1.0, 3.0),
    ),
    Factor(
        "t0",
        "flash-time error std (s)",
        [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1],
        lambda v: {
            "sensors": {"flash_time": {"preset": "exact", "distribution": "normal", "scale_s": float(v)}}
        },
        good=1e-5,
        poor=1e-2,
        fmt=_ms,
        pair_levels=(1e-5, 1e-3, 1e-2, 3e-2, 1e-1),
    ),
    Factor(
        "wind",
        "unknown wind speed at 10 m (m/s)",
        [0.0, 2.0, 4.0, 8.0, 12.0],
        lambda v: {
            "atmosphere": {
                "model": "stratified",
                "temperature_c": T_SURFACE,
                "lapse_rate_k_per_km": 0.0,
                "wind": {"speed_mps": float(v), "direction_from_deg": 270.0},
            },
            "reconstruction": {"atmosphere": STILL},
        },
        good=0.0,
        poor=8.0,
        log=False,
        fmt=lambda v: f"{v:g} m/s",
        pair_levels=(0.0, 2.0, 4.0, 8.0, 12.0),
    ),
    Factor(
        "temperature",
        "assumed surface temperature error (K)",
        [0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
        lambda v: {"reconstruction": {"atmosphere": {**STILL, "temperature_c": T_SURFACE + float(v)}}},
        good=0.0,
        poor=10.0,
        log=False,
        fmt=lambda v: f"{v:g} K",
        pair_levels=(0.0, 2.0, 5.0, 10.0, 20.0),
    ),
]
BY_NAME = {f.name: f for f in FACTORS}
TARGETS = {"angular_error_median_deg": [0.1, 0.5], "point_error_median_m": [5.0, 20.0]}
COVERAGE_TARGETS = [0.5]  # main-channel coverage within 50 m must stay at or above these
METRIC_LABEL = {
    "angular_error_median_deg": "median angular error (deg)",
    "point_error_median_m": "median point error (m)",
}


def vname(f: Factor, v: float) -> str:
    return f"{f.name}={v:g}"


def oat_variants() -> tuple[dict[str, dict], dict[str, dict]]:
    variants: dict[str, dict] = {"baseline": {}}
    meta: dict[str, dict] = {"baseline": {"factor": "baseline", "level": float("nan")}}
    # Wind changes the synthesis atmosphere; keep it last so the shared synthesis stays cached.
    for f in sorted(FACTORS, key=lambda f: f.name == "wind"):
        for v in f.levels:
            variants[vname(f, v)] = f.override(v)
            meta[vname(f, v)] = {"factor": f.name, "level": float(v)}
    return variants, meta


def pair_variants(f1: Factor, f2: Factor) -> tuple[dict[str, dict], dict[str, dict]]:
    from thunder.experiments.pipeline import merge

    variants, meta = {}, {}
    outer, inner = (f2, f1) if f2.name == "wind" else (f1, f2)  # synthesis-changing factor outermost
    for a in outer.pair_levels:
        for b in inner.pair_levels:
            name = f"{vname(outer, a)},{vname(inner, b)}"
            variants[name] = merge(outer.override(a), inner.override(b))
            meta[name] = {outer.name: float(a), inner.name: float(b)}
    return variants, meta


def rank_factors(table: pd.DataFrame, metric: str = "point_error_median_m") -> pd.DataFrame:
    """Paired ratio vs the baseline at each factor's realistic good and poor levels."""
    r = paired_ratio(table, metric, "baseline")
    rows = []
    for f in FACTORS:
        rows.append(
            {
                "factor": f.name,
                "good": f.fmt(f.good),
                "poor": f.fmt(f.poor),
                "ratio_good": r.loc[vname(f, f.good), "ratio"],
                "ratio_poor": r.loc[vname(f, f.poor), "ratio"],
                "ratio_poor_lo": r.loc[vname(f, f.poor), "lo"],
                "ratio_poor_hi": r.loc[vname(f, f.poor), "hi"],
            }
        )
    out = pd.DataFrame(rows).set_index("factor")
    # No bolt kept any point at the poor level: the error is unbounded, so it ranks worst.
    for f in FACTORS:
        t = table[table["variant"] == vname(f, f.poor)]
        if not (t["n_points"] > 0).any():
            out.loc[f.name, ["ratio_poor", "ratio_poor_lo", "ratio_poor_hi"]] = np.inf
    return out.sort_values("ratio_poor", ascending=False)


# --- analysis -----------------------------------------------------------------------


def analyze(oat_dir: Path, pair_dir: Path | None, docs_prefix: str | None) -> None:
    table, _, _, meta = load_run(oat_dir)
    figs = oat_dir / "figures"
    figs.mkdir(exist_ok=True)
    curves = {}
    for metric in TARGETS:
        pv = per_variant(table, metric)
        pv = pv.join(pd.DataFrame(meta).T.infer_objects())
        curves[metric] = pv
    cov = per_variant(table, "coverage_main_50m", lambda v: float(np.mean(v)))
    base = {m: float(curves[m].loc["baseline", "value"]) for m in TARGETS}
    base_cov = float(cov.loc["baseline", "value"])

    # Error budget: largest tolerable level per factor and target (interpolated on the sweep).
    # Coverage must stay above its target: negate it so every criterion reads "value <= target".
    cov_meta = cov.join(pd.DataFrame(meta).T.infer_objects())
    cov_meta["value"] = -cov_meta["value"]
    criteria = [(metric, t, curves[metric], t, "<=") for metric, ts in TARGETS.items() for t in ts]
    criteria += [("coverage_main_50m", t, cov_meta, -t, ">=") for t in COVERAGE_TARGETS]
    budget_rows = []
    for metric, shown, data, target, rel in criteria:
        label = METRIC_LABEL.get(metric, "main coverage within 50 m").split(" (")[0]
        row: dict[str, str] = {"target": f"{label} {rel} {shown:g}"}
        for f in FACTORS:
            c = data[data["factor"] == f.name]
            lv = c["level"].to_numpy(dtype=float)
            x = crossing(f.sign * lv, c["value"].to_numpy(dtype=float), target, log_x=f.log and f.sign > 0)
            if x is None:
                row[f.name] = "never (fails at best level)"
            elif np.isinf(x):
                row[f.name] = f"any tested (to {f.fmt(max(lv) if f.sign > 0 else min(lv))})"
            else:
                row[f.name] = ("<= " if f.sign > 0 else ">= ") + f.fmt(float(f"{f.sign * x:.2g}"))
        budget_rows.append(row)
    budget = pd.DataFrame(budget_rows)
    ranking = rank_factors(table)
    ranking_ang = rank_factors(table, "angular_error_median_deg")
    budget.to_csv(oat_dir / "e3_budget.csv", index=False)
    ranking.to_csv(oat_dir / "e3_ranking.csv")
    for metric, pv in curves.items():
        pv.to_csv(oat_dir / f"e3_sweep_{metric}.csv")

    _plot_sweeps(curves, cov, base, base_cov, figs / "sweeps.png")
    _plot_tornado(ranking, ranking_ang, base, figs / "tornado.png")

    lines = [
        f"# E3 error budget: {oat_dir.name}",
        "",
        f"{int(table['bolt'].nunique())} bolts. Baseline: median angular error "
        f"{base['angular_error_median_deg']:.3f} deg, "
        f"median point error {base['point_error_median_m']:.2f} m, main coverage {base_cov:.0%}.",
        "",
        "## Error budget (largest tolerable level; linear/log interpolation between tested levels)",
        "",
        md_table(budget),
        "",
        "## Ranking at realistic poor levels (paired ratio of median point error vs baseline)",
        "",
        md_table(ranking.reset_index(), ".3g"),
        "",
        "## Ranking by angular error",
        "",
        md_table(ranking_ang.reset_index(), ".3g"),
        "",
        "## One-at-a-time sweeps",
        "",
    ]
    for f in FACTORS:
        rows = []
        for v in f.levels:
            n = vname(f, v)
            a, p = curves["angular_error_median_deg"].loc[n], curves["point_error_median_m"].loc[n]
            rows.append(
                {
                    "level": f.fmt(v),
                    "angular (deg)": f"{a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]",
                    "point (m)": f"{p['value']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}]",
                    "main cov. 50 m": f"{cov.loc[n, 'value']:.0%}",
                    "bolts w/ points": int(p["n"]),
                }
            )
        lines += [f"### {f.name}: {f.label}", "", md_table(pd.DataFrame(rows)), ""]

    if pair_dir is not None:
        lines += _pairwise(pair_dir, figs)
    write_markdown(oat_dir / "e3_summary.md", "\n".join(lines))
    if docs_prefix is not None:
        for p in figs.glob("*.png"):
            shutil.copy(p, Path("docs/figures") / f"{docs_prefix or 'e3_'}{p.name}")
    print((oat_dir / "e3_summary.md").read_text())


def _pairwise(pair_dir: Path, figs: Path) -> list[str]:
    table, _, _, meta = load_run(pair_dir)
    names = list(next(iter(meta.values())).keys())
    f_out, f_in = BY_NAME[names[0]], BY_NAME[names[1]]
    lines = [f"## Pairwise: {f_out.name} x {f_in.name} ({pair_dir.name})", ""]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric in zip(axes, TARGETS, strict=True):
        pv = per_variant(table, metric)
        grid = np.array(
            [
                [pv.loc[f"{vname(f_out, a)},{vname(f_in, b)}", "value"] for b in f_in.pair_levels]
                for a in f_out.pair_levels
            ]
        )
        im = ax.imshow(np.log10(grid), origin="lower", cmap="viridis", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.3g}", ha="center", va="center", fontsize=8, color="w")
        ax.set_xticks(range(len(f_in.pair_levels)), [f_in.fmt(v) for v in f_in.pair_levels], fontsize=8)
        ax.set_yticks(range(len(f_out.pair_levels)), [f_out.fmt(v) for v in f_out.pair_levels], fontsize=8)
        ax.set_xlabel(f_in.label)
        ax.set_ylabel(f_out.label)
        ax.set_title(METRIC_LABEL[metric])
        fig.colorbar(im, ax=ax, label="log10")
        df = pd.DataFrame(
            grid,
            index=[f_out.fmt(v) for v in f_out.pair_levels],
            columns=[f_in.fmt(v) for v in f_in.pair_levels],
        )
        lines += [
            f"{METRIC_LABEL[metric]} (rows: {f_out.name}, columns: {f_in.name})",
            "",
            md_table(df.reset_index().rename(columns={"index": f_out.name})),
            "",
        ]
    fig.suptitle(f"Pairwise sweep: {f_out.name} × {f_in.name}")
    fig.tight_layout()
    fig.savefig(figs / "pairwise.png", dpi=150)
    plt.close(fig)
    return lines


def _plot_sweeps(curves: dict, cov: pd.DataFrame, base: dict, base_cov: float, path: Path) -> None:
    fig, axes = plt.subplots(3, len(FACTORS), figsize=(3.2 * len(FACTORS), 8.5), sharey="row")
    for j, f in enumerate(FACTORS):
        for i, metric in enumerate(TARGETS):
            c = curves[metric][curves[metric]["factor"] == f.name].sort_values("level")
            ax = axes[i, j]
            ax.fill_between(c["level"], c["lo"].astype(float), c["hi"].astype(float), alpha=0.25)
            ax.plot(c["level"], c["value"], "o-")
            ax.axhline(base[metric], color="0.5", ls=":", lw=1)
            for t in TARGETS[metric]:
                ax.axhline(t, color="tab:red", ls="--", lw=0.8)
            ax.set_yscale("log")
            if f.log:
                ax.set_xscale("log")
            if f.sign < 0:
                ax.invert_xaxis()
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric])
        c = cov.join(
            pd.DataFrame(
                {
                    "level": curves["point_error_median_m"]["level"],
                    "factor": curves["point_error_median_m"]["factor"],
                }
            )
        )
        c = c[c["factor"] == f.name].sort_values("level")
        ax = axes[2, j]
        ax.plot(c["level"], c["value"], "o-", color="tab:green")
        ax.axhline(base_cov, color="0.5", ls=":", lw=1)
        ax.set_ylim(0, 1)
        if f.log:
            ax.set_xscale("log")
        if f.sign < 0:
            ax.invert_xaxis()
        ax.set_xlabel(f.label, fontsize=8)
        if j == 0:
            ax.set_ylabel("main coverage within 50 m")
    fig.suptitle("E3 one-at-a-time sweeps (band: 95% CI over bolts; dotted: baseline; red: targets)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_tornado(ranking: pd.DataFrame, ranking_ang: pd.DataFrame, base: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, rk, metric in (
        (axes[0], ranking, "point_error_median_m"),
        (axes[1], ranking_ang, "angular_error_median_deg"),
    ):
        rk = rk.sort_values("ratio_poor")
        y = np.arange(len(rk))
        b = base[metric]
        good = b * rk["ratio_good"].to_numpy(dtype=float)
        poor = b * rk["ratio_poor"].to_numpy(dtype=float)
        failed = ~np.isfinite(poor)
        cap = 3 * np.nanmax(np.r_[poor[~failed], good, b])
        poor = np.where(failed, cap, poor)
        ax.barh(y, poor - b, left=b, color="tab:red", alpha=0.8, label="realistic poor level")
        for yi in y[failed]:
            ax.barh(yi, cap - b, left=b, color="none", edgecolor="k", hatch="//")
            ax.text(cap, yi, " no usable points", va="center", ha="right", fontsize=8)
        ax.barh(y, good - b, left=b, color="tab:blue", alpha=0.8, label="realistic good level")
        ax.axvline(b, color="k", lw=1)
        ax.set_yticks(
            y,
            [f"{n}\n{g} → {p}" for n, g, p in zip(rk.index, rk["good"], rk["poor"], strict=True)],
            fontsize=8,
        )
        ax.set_xscale("log")
        ax.set_xlabel(METRIC_LABEL[metric] + " (log; paired medians vs baseline)")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("E3 tornado: which error source matters most")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", type=Path, nargs="+", metavar="RUN_DIR")
    ap.add_argument("--n-bolts", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--docs-prefix", default=None)
    ap.add_argument("--results-root")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze[0], args.analyze[1] if len(args.analyze) > 1 else None, args.docs_prefix)
        return
    cfg = load_config(CONFIG)
    assert cfg.monte_carlo is not None
    mc = {k: v for k, v in (("n_bolts", args.n_bolts), ("workers", args.workers)) if v is not None}
    upd: dict = {"monte_carlo": cfg.monte_carlo.model_copy(update=mc)}
    if args.results_root:
        upd["results_root"] = args.results_root
    cfg = cfg.model_copy(update=upd)

    variants, meta = oat_variants()
    print(f"phase 1: {len(variants)} variants", flush=True)
    oat_dir = run_variants(cfg, variants, meta)
    print(f"Wrote {oat_dir}", flush=True)

    table, _, _, _ = load_run(oat_dir)
    top = list(rank_factors(table).index[:2])
    print(f"phase 2: {top[0]} x {top[1]}", flush=True)
    pair_cfg = RunConfig.model_validate({**cfg.model_dump(mode="json"), "experiment": "e3_error_budget_pair"})
    pv, pmeta = pair_variants(BY_NAME[top[0]], BY_NAME[top[1]])
    pair_dir = run_variants(pair_cfg, pv, pmeta)
    (oat_dir / "pair_run.json").write_text(json.dumps({"pair_dir": str(pair_dir), "factors": top}))
    print(f"Wrote {pair_dir}", flush=True)
    analyze(oat_dir, pair_dir, args.docs_prefix)


if __name__ == "__main__":
    main()

"""E4: method comparison (SPEC.md Phase 7). Run, or re-analyze a finished run folder.

Usage:
  python scripts/e4_methods.py                         # run + analyze
  python scripts/e4_methods.py --analyze RUN_DIR       # from the run folder only
  python scripts/e4_methods.py --n-bolts 5 --docs-prefix ""   # smoke test

Methods A (plane-wave TDOA), B (SRP-PHAT, several sources per window) and C (multilateration),
each with three assumed atmospheres: straight rays in uniform air at the right surface
temperature, a mismatched stratified profile (right surface T, 5 K/km, no wind), and the true
atmosphere (oracle). All nine variants reconstruct the same recordings (one synthesis and one
corruption per bolt). Method D joins in M8.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thunder.config import load_config
from thunder.experiments.sweeps import (
    ci_text,
    load_run,
    md_table,
    paired_ratio,
    run_variants,
    variant_table,
    write_markdown,
)

CONFIG = "configs/experiments/e4_methods.yaml"
METHODS = ["A", "B", "C"]
ATMOSPHERES = {
    "straight": {"model": "uniform", "temperature_c": 25.0},
    "mismatched": {"model": "stratified", "temperature_c": 25.0, "lapse_rate_k_per_km": 5.0},
    "oracle": None,  # the synthesis atmosphere
}
COVERAGE_D = [10.0, 25.0, 50.0, 100.0, 200.0]


def variants() -> tuple[dict[str, dict], dict[str, dict]]:
    v: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for atm, a in ATMOSPHERES.items():
        for m in METHODS:
            name = f"{m}_{atm}"
            v[name] = {"reconstruction": {"method": m, "atmosphere": a}}
            meta[name] = {"method": m, "assumed": atm}
    return v, meta


def analyze(run_dir: Path, docs_prefix: str | None) -> None:
    table, _, _, meta = load_run(run_dir)
    vt = variant_table(table).join(pd.DataFrame(meta).T)
    ratio = paired_ratio(table, "point_error_median_m", "A_oracle")
    vt["err_vs_A"], vt["err_vs_A_lo"], vt["err_vs_A_hi"] = ratio["ratio"], ratio["lo"], ratio["hi"]
    cov_ratio = paired_ratio(table, "coverage_main_50m", "A_oracle")
    vt["cov_vs_A"] = cov_ratio["ratio"]
    vt.to_csv(run_dir / "e4_table.csv")
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)

    main = pd.DataFrame(
        {
            "method": vt["method"],
            "assumed atmosphere": vt["assumed"],
            "median error (m)": [ci_text(r, "point_error_median_m", ".1f") for _, r in vt.iterrows()],
            "error vs A oracle (paired)": [ci_text(r, "err_vs_A", ".2f") for _, r in vt.iterrows()],
            "p90 (m)": vt["point_error_p90_m"].map(lambda v: f"{v:.1f}"),
            "angular (deg)": vt["angular_error_median_deg"].map(lambda v: f"{v:.3f}"),
            "coverage 50 m all / main / branch": [
                f"{r['coverage_50m']:.0%} / {r['coverage_main_50m']:.0%} / {r['coverage_branch_50m']:.0%}"
                for _, r in vt.iterrows()
            ],
            "strike (m)": vt["strike_error_m"].map(lambda v: f"{v:.0f}"),
            "points/bolt": vt["points_per_bolt"].map(lambda v: f"{v:.0f}"),
            "recon CPU time (s)": vt["time_reconstruct_cpu_s"].map(lambda v: f"{v:.1f}"),
        }
    )
    # Per preset (oracle and mismatched), median error and main coverage.
    preset_rows = []
    for (v, preset), t in table.groupby(["variant", "preset"], sort=False):
        preset_rows.append(
            {
                "variant": v,
                "preset": preset,
                "median error (m)": float(np.nanmedian(t["point_error_median_m"])),
                "main cov. 50 m": float(t["coverage_main_50m"].mean()),
                "branch cov. 50 m": float(np.nanmean(t["coverage_branch_50m"]))
                if t["coverage_branch_50m"].notna().any()
                else np.nan,
            }
        )
    per_preset = pd.DataFrame(preset_rows)
    per_preset.to_csv(run_dir / "e4_per_preset.csv", index=False)
    err_pp = per_preset.pivot(index="preset", columns="variant", values="median error (m)")
    cov_pp = per_preset.pivot(index="preset", columns="variant", values="main cov. 50 m")

    _plot_bars(vt, figs / "methods.png")
    _plot_coverage_curves(table, figs / "coverage_curves.png")
    _plot_presets(err_pp, cov_pp, figs / "presets.png")

    lines = [
        f"# E4 method comparison: {run_dir.name}",
        "",
        f"{table['bolt'].nunique()} bolts over presets {sorted(table['preset'].unique())}; brackets: 95% "
        "bootstrap CIs over bolts; paired ratios use the per-bolt values of the same recordings.",
        "",
        md_table(main.reset_index(drop=True)),
        "",
        "## Median error per preset (m)",
        "",
        md_table(err_pp.reset_index(), ".3g"),
        "",
        "## Main-channel coverage within 50 m per preset",
        "",
        md_table(cov_pp.reset_index(), ".2f"),
    ]
    write_markdown(run_dir / "e4_summary.md", "\n".join(lines))
    if docs_prefix is not None:
        for p in figs.glob("*.png"):
            shutil.copy(p, Path("docs/figures") / f"{docs_prefix or 'e4_'}{p.name}")
    print((run_dir / "e4_summary.md").read_text())


def _plot_bars(vt: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    x = np.arange(len(ATMOSPHERES))
    w = 0.26
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    panels = [
        ("point_error_median_m", "median point error (m)", True),
        ("coverage_main_50m", "main coverage within 50 m", False),
        ("coverage_branch_50m", "branch coverage within 50 m", False),
        ("time_reconstruct_cpu_s", "reconstruction CPU time per bolt (s)", True),
    ]
    for ax, (col, label, log) in zip(axes, panels, strict=True):
        for k, m in enumerate(METHODS):
            d = vt[vt["method"] == m].set_index("assumed").loc[list(ATMOSPHERES)]
            y = d[col].to_numpy(dtype=float)
            yerr = None
            if col + "_lo" in d:
                yerr = np.vstack([y - d[col + "_lo"].to_numpy(float), d[col + "_hi"].to_numpy(float) - y])
            ax.bar(x + (k - 1) * w, y, w, yerr=yerr, capsize=2, color=colors[m], label=f"Method {m}")
        ax.set_xticks(x, list(ATMOSPHERES))
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle("E4: methods × assumed atmosphere (realistic atmosphere and sensors)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coverage_curves(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(ATMOSPHERES), figsize=(14, 4.2), sharey=True)
    for ax, atm in zip(axes, ATMOSPHERES, strict=True):
        for m in METHODS:
            t = table[table["variant"] == f"{m}_{atm}"]
            cov = [t[f"coverage_main_{d:g}m"].mean() for d in COVERAGE_D]
            ax.plot(COVERAGE_D, cov, "o-", label=f"Method {m}")
        ax.set_xscale("log")
        ax.set_xlabel("distance threshold (m)")
        ax.set_title(f"assumed: {atm}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("main-channel coverage")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_presets(err: pd.DataFrame, cov: pd.DataFrame, path: Path) -> None:
    cols = [f"{m}_oracle" for m in METHODS] + [f"{m}_mismatched" for m in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    err[cols].plot.bar(ax=axes[0], logy=True, rot=0)
    axes[0].set_ylabel("median point error (m)")
    cov[cols].plot.bar(ax=axes[1], rot=0, legend=False)
    axes[1].set_ylabel("main coverage within 50 m")
    axes[1].set_ylim(0, 1)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("E4 per preset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--n-bolts", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--docs-prefix", default=None)
    ap.add_argument("--results-root")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze, args.docs_prefix)
        return
    cfg = load_config(CONFIG)
    assert cfg.monte_carlo is not None
    mc = {k: v for k, v in (("n_bolts", args.n_bolts), ("workers", args.workers)) if v is not None}
    upd: dict = {"monte_carlo": cfg.monte_carlo.model_copy(update=mc)}
    if args.results_root:
        upd["results_root"] = args.results_root
    cfg = cfg.model_copy(update=upd)
    v, meta = variants()
    run_dir = run_variants(cfg, v, meta)
    print(f"Wrote {run_dir}", flush=True)
    analyze(run_dir, args.docs_prefix)


if __name__ == "__main__":
    main()

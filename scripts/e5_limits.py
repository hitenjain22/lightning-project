"""E5: limits (SPEC.md Phase 7). Run, or re-analyze a finished run folder.

Usage:
  python scripts/e5_limits.py                          # run + analyze
  python scripts/e5_limits.py --analyze RUN_DIR        # from the run folder only
  python scripts/e5_limits.py --n-bolts 2 --docs-prefix ""    # smoke test

Error and coverage versus strike distance (1-15 km), branching depth (0-3) and in-cloud
sections, on the same bolts (branched preset), Method B. Distance runs assume the true
atmosphere (oracle: the physical limit) and the mismatched one (practical), plus a +20 dB
louder-ambient variant (equivalent to 100x less acoustic energy; assumption A12 bound).
Diagnostics that explain the losses come from the ground truth: the fraction of channel
length in the refraction shadow (no eigenray reaches the array), the in-band SNR, the peak
level, and the fraction of analysis windows that pass the method's gates.
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

from thunder.config import CHANNEL_PRESETS, load_config
from thunder.experiments.sweeps import (
    ci_text,
    crossing,
    load_run,
    md_table,
    run_variants,
    variant_table,
    write_markdown,
)

CONFIG = "configs/experiments/e5_limits.yaml"
DISTANCES_KM = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.5, 15.0]
DEPTHS = [0, 1, 2, 3]
REFERENCE_KM = 3.0  # distance of the branching and in-cloud sweeps
MISMATCHED = {"model": "stratified", "temperature_c": 25.0, "lapse_rate_k_per_km": 5.0}
LOUD_AMBIENT_DB = 65.0  # base is 45 dB SPL


def _dist(d_km: float) -> dict:
    return {"strike_distance_m": [d_km * 1000.0, d_km * 1000.0]}


def variants() -> tuple[dict[str, dict], dict[str, dict]]:
    v: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    def add(name: str, ov: dict, **m: object) -> None:
        v[name] = ov
        meta[name] = m

    for d in DISTANCES_KM:  # the three variants of a distance share one synthesis
        add(f"d{d:g}_oracle", {"channel": _dist(d)}, sweep="distance", distance_km=d, assumed="oracle")
        add(
            f"d{d:g}_mismatched",
            {"channel": _dist(d), "reconstruction": {"atmosphere": MISMATCHED}},
            sweep="distance",
            distance_km=d,
            assumed="mismatched",
        )
        add(
            f"d{d:g}_loud",
            {"channel": _dist(d), "sensors": {"noise": {"background_db_spl": LOUD_AMBIENT_DB}}},
            sweep="distance",
            distance_km=d,
            assumed="oracle, +20 dB ambient",
        )
    for k in DEPTHS:
        add(
            f"depth{k}",
            {"channel": {**_dist(REFERENCE_KM), "branch_max_depth": k}},
            sweep="depth",
            depth=k,
            distance_km=REFERENCE_KM,
            assumed="oracle",
        )
    incloud = {
        key: list(val) if isinstance(val, tuple) else val
        for key, val in CHANNEL_PRESETS["with_incloud"].items()
    }
    add(
        "incloud_off",
        {"channel": _dist(REFERENCE_KM)},
        sweep="incloud",
        incloud=False,
        distance_km=REFERENCE_KM,
        assumed="oracle",
    )
    add(
        "incloud_on",
        {"channel": {**_dist(REFERENCE_KM), **incloud}},
        sweep="incloud",
        incloud=True,
        distance_km=REFERENCE_KM,
        assumed="oracle",
    )
    return v, meta


def analyze(run_dir: Path, docs_prefix: str | None) -> None:
    table, _, _, meta = load_run(run_dir)
    m = pd.DataFrame(meta).T.infer_objects()
    vt = variant_table(table).join(m)
    g = table.groupby("variant", sort=False)
    vt["shadow_fraction"] = g["shadow_fraction"].mean()
    vt["snr_band_db"] = g["snr_band_db"].median()
    vt["arrival_ms_per_km"] = g["arrival_s_per_km"].median() * 1000
    vt["peak_db_spl"] = g["peak_pa"].median().map(lambda p: 20 * np.log10(p / 20e-6))
    vt["branch_count"] = g["branch_count"].mean()
    vt["channel_length_km"] = g["channel_length_m"].mean() / 1000
    vt["incloud_length_km"] = g["incloud_length_m"].mean() / 1000
    vt.to_csv(run_dir / "e5_table.csv")
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)

    dist = vt[vt["sweep"] == "distance"]
    oracle = dist[dist["assumed"] == "oracle"].sort_values("distance_km")
    limits = {}
    for label, d in (
        ("oracle", oracle),
        ("mismatched", dist[dist["assumed"] == "mismatched"].sort_values("distance_km")),
        ("+20 dB ambient", dist[dist["assumed"].str.contains("ambient")].sort_values("distance_km")),
    ):
        x = d["distance_km"].to_numpy(dtype=float)
        limits[label] = {
            "main coverage < 50%": crossing(x, -d["coverage_main_50m"].to_numpy(dtype=float), -0.5),
            "median error > 1% of range": crossing(
                x, d["point_error_median_m"].to_numpy(dtype=float) / (10 * x), 1.0
            ),
        }

    def km(v: float | None) -> str:
        if v is None:
            return "already at 1 km"
        return "beyond 15 km" if np.isinf(v) else f"{v:.1f} km"

    lim = pd.DataFrame({k: {c: km(x) for c, x in v.items()} for k, v in limits.items()}).T

    def rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                key: df[key].astype(str) if key in df else df.index,
                "assumed": df["assumed"],
                "median error (m)": [ci_text(r, "point_error_median_m", ".1f") for _, r in df.iterrows()],
                "angular (deg)": df["angular_error_median_deg"].map(lambda v: f"{v:.3f}"),
                "main cov. 50 m": df["coverage_main_50m"].map(lambda v: f"{v:.0%}"),
                "all cov. 50 m": df["coverage_50m"].map(lambda v: f"{v:.0%}"),
                "branch cov. 50 m": df["coverage_branch_50m"].map(
                    lambda v: f"{v:.0%}" if np.isfinite(v) else "-"
                ),
                "in shadow": df["shadow_fraction"].map(lambda v: f"{v:.0%}"),
                "SNR (dB)": df["snr_band_db"].map(lambda v: f"{v:.0f}"),
                "peak (dB SPL)": df["peak_db_spl"].map(lambda v: f"{v:.0f}"),
                "windows passed": df["windows_passed_frac"].map(lambda v: f"{v:.0%}"),
                "ms of sound per km": df["arrival_ms_per_km"].map(lambda v: f"{v:.0f}"),
                "bolts w/ points": df["bolts_with_points"].astype(int).astype(str)
                + "/"
                + df["bolts"].astype(str),
            }
        )

    _plot_distance(dist, figs / "distance.png")
    _plot_structure(vt, figs / "structure.png")
    lines = [
        f"# E5 limits: {run_dir.name}",
        "",
        f"{table['bolt'].nunique()} bolts (branched preset), Method B. "
        "Brackets: 95% bootstrap CIs over bolts.",
        "",
        "## Where it turns to mush (interpolated distance)",
        "",
        md_table(lim.reset_index().rename(columns={"index": "assumed"})),
        "",
        "## Distance",
        "",
        md_table(rows(dist.sort_values(["assumed", "distance_km"]), "distance_km")),
        "",
        f"## Branching depth (at {REFERENCE_KM:g} km, oracle)",
        "",
        md_table(rows(vt[vt["sweep"] == "depth"].sort_values("depth"), "depth")),
        "",
        "Mean branch count per depth: "
        + ", ".join(
            f"{int(r['depth'])}: {r['branch_count']:.1f}" for _, r in vt[vt["sweep"] == "depth"].iterrows()
        ),
        "",
        f"## In-cloud section (at {REFERENCE_KM:g} km, oracle)",
        "",
        md_table(rows(vt[vt["sweep"] == "incloud"], "incloud")),
        "",
        "Mean in-cloud length: "
        + ", ".join(
            f"{r['incloud']}: {r['incloud_length_km']:.1f} km of {r['channel_length_km']:.1f} km"
            for _, r in vt[vt["sweep"] == "incloud"].iterrows()
        ),
    ]
    write_markdown(run_dir / "e5_summary.md", "\n".join(lines))
    if docs_prefix is not None:
        for p in figs.glob("*.png"):
            shutil.copy(p, Path("docs/figures") / f"{docs_prefix or 'e5_'}{p.name}")
    print((run_dir / "e5_summary.md").read_text())


def _plot_distance(dist: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    styles = {
        "oracle": ("tab:blue", "-"),
        "mismatched": ("tab:red", "-"),
        "oracle, +20 dB ambient": ("tab:purple", "--"),
    }
    for assumed, (color, ls) in styles.items():
        d = dist[dist["assumed"] == assumed].sort_values("distance_km")
        x = d["distance_km"]
        kw = {"color": color, "ls": ls, "marker": "o", "label": assumed}
        axes[0, 0].plot(x, d["point_error_median_m"], **kw)
        axes[0, 0].fill_between(
            x, d["point_error_median_m_lo"], d["point_error_median_m_hi"], color=color, alpha=0.15
        )
        axes[0, 1].plot(x, d["coverage_main_50m"], **kw)
        axes[0, 2].plot(x, d["angular_error_median_deg"], **kw)
        axes[1, 1].plot(x, d["snr_band_db"], **kw)
        axes[1, 2].plot(x, d["arrival_ms_per_km"], **kw)
    o = dist[dist["assumed"] == "oracle"].sort_values("distance_km")
    axes[1, 0].plot(o["distance_km"], o["shadow_fraction"], "o-", color="k", label="channel length in shadow")
    axes[1, 0].plot(
        o["distance_km"],
        1 - o["coverage_main_50m"],
        "o--",
        color="tab:blue",
        label="main channel missed (oracle)",
    )
    x = np.array(DISTANCES_KM)
    axes[0, 0].plot(x, 10 * x, ":", color="0.5", label="1% of range")
    titles = [
        ("median point error (m)", True),
        ("main coverage within 50 m", False),
        ("median angular error (deg)", True),
        ("fraction (truth)", False),
        ("median in-band SNR (dB)", False),
        ("ms of recording per km of channel (truth)", True),
    ]
    for ax, (t, log) in zip(axes.flat, titles, strict=True):
        ax.set_xlabel("strike distance (km)")
        ax.set_ylabel(t)
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_ylim(0, 1)
    fig.suptitle("E5: error and coverage vs distance, and why (Method B, realistic atmosphere and sensors)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_structure(vt: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    d = vt[vt["sweep"] == "depth"].sort_values("depth")
    x = d["depth"].astype(int)
    axes[0].plot(x, d["coverage_main_50m"], "o-", label="main channel")
    axes[0].plot(x, d["coverage_branch_50m"], "s-", label="side branches")
    axes[0].plot(x, d["coverage_50m"], "^-", label="all")
    axes[0].set_xlabel("maximum branching depth")
    axes[0].set_ylabel("coverage within 50 m")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(DEPTHS)
    axes[0].legend(fontsize=8)
    ax2 = axes[0].twinx()
    ax2.plot(x, d["point_error_median_m"], "k:", marker="x", label="median error")
    ax2.set_ylabel("median point error (m), dotted")
    c = vt[vt["sweep"] == "incloud"].sort_values("incloud")
    labels = ["no in-cloud", "with in-cloud"]
    w = 0.25
    xx = np.arange(2)
    axes[1].bar(xx - w, c["coverage_main_50m"], w, label="main channel")
    axes[1].bar(xx, c["coverage_50m"], w, label="all (incl. in-cloud)")
    axes[1].bar(xx + w, c["coverage_branch_50m"], w, label="side branches")
    axes[1].set_xticks(xx, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("coverage within 50 m")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"E5: channel structure at {REFERENCE_KM:g} km (Method B, oracle)")
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

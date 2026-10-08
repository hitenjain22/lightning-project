"""E6: consumer-hardware feasibility (SPEC.md Phase 7). Run, or re-analyze a run folder.

Usage:
  python scripts/e6_consumer.py                        # run + analyze
  python scripts/e6_consumer.py --analyze RUN_DIR      # from the run folder only
  python scripts/e6_consumer.py --n-bolts 3 --docs-prefix ""   # smoke test

"Could someone do this with 4 phones?" Four phones on a 50 m square with phone microphones,
clocks aligned by a hand clap (~3 ms), flash time from 30 fps video and phone-GPS positions
(~3 m), in the realistic E4 atmosphere, assuming the standard atmosphere. An upgrade ladder on
the same bolts shows what each improvement buys:
  phones_50           as above, Method B
  phones_150          the phones spread over 150 m (position and sync errors matter less in angle)
  tape_50             positions measured with a tape and compass (~10 cm)
  sync_50             clocks aligned to ~0.1 ms (e.g. a GPS-time app or a shared sync chirp; VERIFY)
  tape_sync_50        both
  tape_sync_flash_50  both, plus a photodiode flash trigger
  field_kit_50        the E2-E5 field kit (measurement mics, GPS clocks, survey), for reference
  tape_sync_50_oracle the best phone setup with the true atmosphere: what remains is hardware
and Method D with noise priors matching each setup, to check that its error bars stay honest.
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

from thunder.config import FlashTimeConfig, MicConfig, PositionConfig, TimingConfig, load_config
from thunder.experiments.sweeps import (
    ci_text,
    load_run,
    md_table,
    paired_ratio,
    run_variants,
    variant_table,
    write_markdown,
)

CONFIG = "configs/experiments/e6_consumer.yaml"
BASELINE = "phones_50"
# Overrides are expanded to every field: on a resolved config a bare {"preset": ...} would keep
# the base's explicit values (presets only fill fields that are not given).
TAPE = {"position": PositionConfig(preset="tape").model_dump(mode="json")}
SYNC = {"timing": TimingConfig(offset_std_s=1e-4, drift_std_ppm=1.0).model_dump(mode="json")}  # VERIFY
FLASH = {"flash_time": FlashTimeConfig(preset="photodiode").model_dump(mode="json")}
FIELD_KIT = {
    "mic": MicConfig(preset="measurement").model_dump(mode="json"),
    "timing": TimingConfig(preset="gps_synced").model_dump(mode="json"),
    "position": PositionConfig(preset="surveyed").model_dump(mode="json"),
    "flash_time": FlashTimeConfig(preset="photodiode").model_dump(mode="json"),
}
# Method D with noise priors matching each hardware level (the user's knowledge of their kit),
# self-calibrating so that the unknown wind is part of its uncertainty (E7).
D_PHONES = {
    "method": "D",
    "d_self_calibrate": True,
    "d_array_calibration": False,  # metre/ms-level array errors: noise, not parameters (see log)
    "d_sigma_clock_s": 3e-3,
    "d_sigma_position_m": 4.0,
    "d_sigma_t0_s": 9.6e-3,
}
D_TAPE_SYNC = {
    "method": "D",
    "d_self_calibrate": True,
    "d_sigma_clock_s": 1e-4,
    "d_sigma_position_m": 0.1,
    "d_sigma_t0_s": 9.6e-3,
}


def variants() -> tuple[dict[str, dict], dict[str, dict]]:
    from thunder.experiments.pipeline import merge

    v: dict[str, dict] = {
        "phones_50": {},
        "phones_50_D": {"reconstruction": D_PHONES},
        "phones_150": {"array": {"aperture_m": 150.0}},
        "tape_50": {"sensors": TAPE},
        "sync_50": {"sensors": SYNC},
        "tape_sync_50": {"sensors": merge(TAPE, SYNC)},
        "tape_sync_50_D": {"sensors": merge(TAPE, SYNC), "reconstruction": D_TAPE_SYNC},
        "tape_sync_50_oracle": {"sensors": merge(TAPE, SYNC), "reconstruction": {"atmosphere": None}},
        "tape_sync_flash_50": {"sensors": merge(merge(TAPE, SYNC), FLASH)},
        "field_kit_50": {"sensors": FIELD_KIT},
    }
    meta = {
        "phones_50": "4 phones, 50 m (B)",
        "phones_50_D": "4 phones, 50 m (D self-calibrating, phone priors)",
        "phones_150": "4 phones, 150 m (B)",
        "tape_50": "+ tape-measured positions",
        "sync_50": "+ 0.1 ms clock sync",
        "tape_sync_50": "+ tape + 0.1 ms sync",
        "tape_sync_50_D": "+ tape + 0.1 ms sync (D self-calibrating)",
        "tape_sync_50_oracle": "+ tape + sync, true atmosphere",
        "tape_sync_flash_50": "+ tape + sync + photodiode",
        "field_kit_50": "field kit (reference)",
    }
    return v, {k: {"label": meta[k]} for k in v}


def analyze(run_dir: Path, docs_prefix: str | None) -> None:
    table, _, _, meta = load_run(run_dir)
    vt = variant_table(table).join(pd.DataFrame(meta).T)
    r = paired_ratio(table, "point_error_median_m", BASELINE)
    vt["err_vs_phones"], vt["err_vs_phones_lo"], vt["err_vs_phones_hi"] = r["ratio"], r["lo"], r["hi"]
    calib = table.groupby("variant")[["calib_1sigma", "calib_2sigma", "calib_3sigma"]].mean()
    vt = vt.join(calib)
    vt.to_csv(run_dir / "e6_table.csv")
    rows = pd.DataFrame(
        {
            "setup": vt["label"],
            "median error (m)": [ci_text(row, "point_error_median_m", ".0f") for _, row in vt.iterrows()],
            "vs phones (paired)": [ci_text(row, "err_vs_phones", ".2f") for _, row in vt.iterrows()],
            "angular (deg)": vt["angular_error_median_deg"].map(lambda v: f"{v:.2f}"),
            "main cov. 50 m": vt["coverage_main_50m"].map(lambda v: f"{v:.0%}"),
            "strike (m)": vt["strike_error_m"].map(lambda v: f"{v:.0f}"),
            "bolts w/ points": vt["bolts_with_points"].astype(int).astype(str)
            + "/"
            + vt["bolts"].astype(str),
            "calib 1/2/3 sigma": [
                f"{row['calib_1sigma']:.2f}/{row['calib_2sigma']:.2f}/{row['calib_3sigma']:.2f}"
                for _, row in vt.iterrows()
            ],
        }
    )
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    _plot(vt, figs / "ladder.png")
    lines = [
        f"# E6 consumer feasibility: {run_dir.name}",
        "",
        f"{table['bolt'].nunique()} bolts; brackets: 95% bootstrap CIs over bolts; "
        f"paired ratios vs {BASELINE}.",
        "",
        md_table(rows.reset_index(drop=True)),
    ]
    write_markdown(run_dir / "e6_summary.md", "\n".join(lines))
    if docs_prefix is not None:
        for p in figs.glob("*.png"):
            shutil.copy(p, Path("docs/figures") / f"{docs_prefix or 'e6_'}{p.name}")
    print((run_dir / "e6_summary.md").read_text())


def _plot(vt: pd.DataFrame, path: Path) -> None:
    order = [v for v in variants()[0] if v in vt.index and not v.endswith("_D")]
    d = vt.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    y = np.arange(len(d))
    err = d["point_error_median_m"].to_numpy(dtype=float)
    xerr = np.vstack(
        [
            err - d["point_error_median_m_lo"].to_numpy(float),
            d["point_error_median_m_hi"].to_numpy(float) - err,
        ]
    )
    axes[0].barh(y, err, xerr=xerr, color="tab:blue", capsize=2)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y, d["label"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("median point error (m)")
    axes[1].barh(y, d["coverage_main_50m"].to_numpy(dtype=float), color="tab:green")
    axes[1].set_yticks(y, [""] * len(d))
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("main coverage within 50 m")
    fig.suptitle("E6: 4 phones, and what each upgrade buys (Method B, standard atmosphere assumed)")
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

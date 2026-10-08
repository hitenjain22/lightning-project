"""M9 media: hero animation, interactive 3D view, signal figures and thunder audio.

Usage: python scripts/make_media.py [configs/experiments/media.yaml] [--seed N] [--frames N]

Writes into a run folder results/media/<run_id>/ (and, with --docs, copies the README/report
assets to docs/media, docs/audio and docs/figures):
  hero.gif                  points appear as their sound reaches the array (SPEC Phase 8 hero)
  reconstruction.html       interactive 3D: true channel, points by error, uncertainty, mics
  signals.png               waveform stack (all mics) and spectrograms (two mics)
  gcc_heatmap.png           GCC-PHAT between two mics over time (lag vs window)
  error_vs_range.png        point error vs range per method (from an E4 run folder, if present)
  audio/thunder_<d>km.wav   the same bolt shape at 1, 3, 8 and 15 km, one mic, each normalized
  audio/levels.json         peak level, SNR and duration of each WAV (absolute levels)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from thunder.atmosphere.profiles import build_atmosphere
from thunder.config import load_config
from thunder.experiments.montecarlo import _example
from thunder.experiments.pipeline import run_bolt
from thunder.experiments.runner import make_run_dir, write_run_files
from thunder.recon.tdoa import _phat_correlation, _segment, prepare
from thunder.viz.animate import animate_reconstruction
from thunder.viz.plot3d import reconstruction_figure, write_html
from thunder.viz.plots import plot_spectrogram, plot_waveform_stack

DISTANCES_KM = (1.0, 3.0, 8.0, 15.0)


def gcc_over_time(rec, mics, atmosphere, rcfg, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GCC-PHAT between mics i and j for every active analysis window, exactly as Method A's first
    pass computes it (band-passed signals, short window on i, long window on j, beta-PHAT).
    Returns window centre times (s), lags (ms), and the correlation image (windows x lags)."""
    st = prepare(rec, mics, atmosphere, rcfg)
    n, pad = st.corr.n_short, st.corr.n_long_pad
    max_lag = int(st.max_lag[i, j])
    rows, times = [], []
    for w in st.active:
        start = int(st.frames.starts[w])
        s = _segment(st.x[i], start, n) * st.corr.short_taper
        lg = _segment(st.x[j], start - pad, n + 2 * pad)
        cross = np.conj(np.fft.rfft(s, st.corr.nfft)) * np.fft.rfft(lg, st.corr.nfft)
        r = _phat_correlation(cross, st.corr.band_mask, rcfg.phat_beta, st.corr.nfft)
        rows.append(r[pad - max_lag : pad + max_lag + 1])
        times.append((start + n / 2) / st.fs)
    lags_ms = np.arange(-max_lag, max_lag + 1) / st.fs * 1e3
    return np.array(times), lags_ms, np.array(rows)


def error_vs_range(e4_dir: Path, path: Path) -> bool:
    pts = e4_dir / "points.csv.gz"
    if not pts.exists():
        return False
    p = pd.read_csv(pts)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    edges = np.linspace(800, 9000, 15)
    mid = 0.5 * (edges[1:] + edges[:-1]) / 1000
    styles = {"oracle": "-", "mismatched": "--", "straight": ":"}
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for variant, t in p.groupby("variant"):
        method, assumed = str(variant).split("_", 1)
        idx = np.digitize(t["range_m"], edges) - 1
        med = [t["error_m"][idx == k].median() if (idx == k).sum() >= 30 else np.nan for k in range(len(mid))]
        ax.plot(mid, med, styles[assumed], color=colors[method], label=f"{method}, {assumed}")
    ax.set_yscale("log")
    ax.set_xlabel("range from the array (km)")
    ax.set_ylabel("median point error (m)")
    ax.set_title("Point error vs range (E4: realistic atmosphere and sensors)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/experiments/media.yaml")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--docs", action="store_true", help="copy README/report assets into docs/")
    args = ap.parse_args()
    cfg = load_config(args.config, {"seed": args.seed} if args.seed is not None else None)
    assert cfg.channel is not None and cfg.reconstruction is not None and cfg.synthesis is not None
    run_dir = make_run_dir(cfg)

    # 1. The hero bolt: reconstruction, animation, interactive view.
    res = run_bolt(cfg, cfg.seed)
    ex = _example(res)
    assert ex is not None and res.recording is not None
    m = res.metrics
    title = (
        f"Method B, realistic atmosphere (known) and sensors: {len(ex['points'])} points, "
        f"median error {m['point_error_median_m']:.1f} m"
    )
    animate_reconstruction(
        ex, run_dir / "hero.gif", "Thunder → 3D lightning (Method B)", n_frames=args.frames
    )
    write_html(reconstruction_figure(ex, title), run_dir / "reconstruction.html")

    # 2. Signals: waveform stack and spectrograms.
    rec = res.recording
    fs = rec.sample_rate
    labels = [f"mic {i}" for i in range(len(rec.signals))]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], width_ratios=[2.2, 1, 0.05])
    plot_waveform_stack((fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])), rec.signals, fs, labels)
    sub = gs[1, :].subgridspec(1, 2)
    for k, i in enumerate((0, len(rec.signals) - 1)):
        plot_spectrogram(fig.add_subplot(sub[0, k]), rec.signals[i], fs, f_max=500.0, title=f"mic {i}")
    fig.tight_layout()
    fig.savefig(run_dir / "signals.png", dpi=120)
    plt.close(fig)

    # 3. GCC-PHAT over time between the centre mic and a corner mic.
    atm = build_atmosphere(cfg.atmosphere)
    i, j = len(rec.signals) - 1, 0
    t, lags, img = gcc_over_time(rec, rec.nominal_mic_positions, atm, cfg.reconstruction, i, j)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mesh = ax.pcolormesh(t, lags, img.T, shading="nearest", cmap="magma", vmin=0, vmax=1)
    ax.plot(t, lags[np.argmax(img, axis=1)], ".", color="cyan", ms=2, label="peak lag")
    fig.colorbar(mesh, ax=ax, label="normalized GCC-PHAT")
    ax.set_xlabel("time since the flash (s)")
    ax.set_ylabel(f"lag of mic {j} behind mic {i} (ms)")
    ax.set_title("GCC-PHAT over time: each column is one 100 ms window; the peak tracks the source direction")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "gcc_heatmap.png", dpi=130)
    plt.close(fig)

    # 4. Error vs range per method from the newest E4 run (if any).
    e4 = sorted(Path(cfg.results_root, "e4_methods").glob("2026*"))
    has_range = bool(e4) and error_vs_range(e4[-1], run_dir / "error_vs_range.png")

    # 5. Thunder audio: the same bolt shape at several distances (realistic sensors), one mic.
    levels = {}
    (run_dir / "audio").mkdir()
    for d in DISTANCES_KM:
        ch_cfg = cfg.channel.model_copy(update={"strike_distance_m": (d * 1000.0, d * 1000.0)})
        r = run_bolt(cfg.model_copy(update={"reconstruction": None}), cfg.seed, ch_cfg)
        assert r.recording is not None
        x = r.recording.signals[0]
        peak = float(np.max(np.abs(x)))
        sf.write(run_dir / "audio" / f"thunder_{d:g}km.wav", 0.9 * x / peak, int(round(fs)), subtype="PCM_16")
        info = r.recording.truth.extra.get("corruption_info", {}) if r.recording.truth else {}
        levels[f"{d:g} km"] = {
            "peak_pa": peak,
            "peak_db_spl": 20 * np.log10(peak / 20e-6),
            "snr_band_db": float(info.get("snr_band_db", np.nan)),
            "duration_s": float(len(x) / fs),
        }
    (run_dir / "audio" / "levels.json").write_text(json.dumps(levels, indent=2))
    write_run_files(cfg, run_dir, {"hero": m, "audio": levels}, {})

    if args.docs:
        for name, dest in (("hero.gif", "docs/media"), ("reconstruction.html", "docs/media")):
            Path(dest).mkdir(parents=True, exist_ok=True)
            shutil.copy(run_dir / name, Path(dest) / name)
        shutil.copy(run_dir / "signals.png", "docs/figures/m9_signals.png")
        shutil.copy(run_dir / "gcc_heatmap.png", "docs/figures/m9_gcc_heatmap.png")
        if has_range:
            shutil.copy(run_dir / "error_vs_range.png", "docs/figures/m9_error_vs_range.png")
        Path("docs/audio").mkdir(parents=True, exist_ok=True)
        for f in (run_dir / "audio").iterdir():
            shutil.copy(f, Path("docs/audio") / f.name)
    print(
        json.dumps(
            {
                "hero": {k: m[k] for k in ("n_points", "point_error_median_m", "coverage_main_50m")},
                "audio": levels,
            },
            indent=2,
            default=float,
        )
    )
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()

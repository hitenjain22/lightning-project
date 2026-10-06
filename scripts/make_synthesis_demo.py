"""M2 deliverable: waveform stack + spectrograms for one bolt across 4 mics, plus WAV files.

Usage: python scripts/make_synthesis_demo.py configs/experiments/synthesis_demo.yaml

Also writes WAVs (one mic) for the same bolt shape moved to 1, 3 and 8 km, to hear how the
rumble lengthens with distance.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from thunder.acoustics.audio import write_wavs  # noqa: E402
from thunder.acoustics.synth import synthesize  # noqa: E402
from thunder.atmosphere.profiles import build_atmosphere  # noqa: E402
from thunder.channel.generator import generate_channel  # noqa: E402
from thunder.config import load_config  # noqa: E402
from thunder.experiments.runner import run_pipeline  # noqa: E402
from thunder.viz.plots import plot_channel_projection, plot_spectrogram, plot_waveform_stack  # noqa: E402

DISTANCES_M = (1000.0, 3000.0, 8000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg.channel is not None and cfg.synthesis is not None

    run_dir = run_pipeline(cfg)
    with np.load(run_dir / "arrays.npz") as z:
        signals, fs = z["signals"], cfg.sample_rate_hz
    labels = [f"mic {i}" for i in range(len(signals))]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    plot_waveform_stack((axes[0], axes[1]), signals, fs, labels)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "waveform_stack.png", dpi=130)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    for ax, x, lab in zip(axes.flat, signals, labels, strict=True):
        plot_spectrogram(ax, x, fs, title=lab)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "spectrograms.png", dpi=130)

    # Same bolt shape (same seed) at several distances, one mic each, for listening.
    atm = build_atmosphere(cfg.atmosphere)
    levels = {}
    fig, axes = plt.subplots(1, len(DISTANCES_M), figsize=(15, 4))
    for ax, d in zip(axes, DISTANCES_M, strict=True):
        ch_cfg = cfg.channel.model_copy(update={"strike_distance_m": (d, d)})
        ch = generate_channel(ch_cfg, np.random.default_rng(cfg.seed), seed=cfg.seed)
        syn = cfg.synthesis
        rec = synthesize(ch, atm, np.array(cfg.array.positions_m[:1]), fs, cfg.oversample,
                         syn.emitter_spacing_m, syn.acoustic_efficiency, np.random.default_rng(cfg.seed),
                         syn.micro_turn_mean, syn.micro_scale_m)
        write_wavs(rec, run_dir / "audio_by_distance" / f"{int(d)}m")
        levels[f"{int(d)}m"] = {"peak_pa": float(np.abs(rec.signals).max()), "duration_s": rec.duration}
        plot_channel_projection(ax, ch, horizontal="range")
        ax.set_title(f"bolt at {d / 1000:g} km")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "bolt_by_distance.png", dpi=110)
    (run_dir / "levels_by_distance.json").write_text(json.dumps(levels, indent=2))
    print(json.dumps(levels, indent=2))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()

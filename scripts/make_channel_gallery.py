"""M1 deliverable: gallery of 12 channels across presets, plus a stats table per preset.

Usage: python scripts/make_channel_gallery.py configs/experiments/channel_gallery.yaml
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from thunder.channel.generator import generate_channel  # noqa: E402
from thunder.channel.stats import stats_table  # noqa: E402
from thunder.config import ChannelConfig, load_config  # noqa: E402
from thunder.experiments.runner import make_run_dir, write_run_files  # noqa: E402
from thunder.viz.plots import plot_channel_projection  # noqa: E402

GALLERY = {"straight": 2, "tortuous": 2, "branched": 3, "with_incloud": 3, "multi_stroke": 2}
N_STATS_PER_PRESET = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg)
    root = np.random.SeedSequence(cfg.seed)
    gallery_seq, stats_seq = root.spawn(2)

    # Gallery: each channel projected onto its widest horizontal direction, so lean, branches
    # and in-cloud sections are visible regardless of strike azimuth.
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    rngs = [np.random.default_rng(s) for s in gallery_seq.spawn(sum(GALLERY.values()))]
    presets = [p for p, n in GALLERY.items() for _ in range(n)]
    for ax, preset, rng in zip(axes.flat, presets, rngs, strict=True):
        ch = generate_channel(ChannelConfig(preset=preset), rng)
        plot_channel_projection(ax, ch, horizontal="principal")
        extra = f", {len(ch.stroke_times)} strokes" if len(ch.stroke_times) > 1 else ""
        ax.set_title(f"{preset}{extra}", fontsize=11)
    fig.suptitle("Synthetic lightning channels (main: dark blue, branches: light, in-cloud: red)")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "channel_gallery.png", dpi=130)

    # Stats table.
    channels = []
    for preset, seq in zip(GALLERY, stats_seq.spawn(len(GALLERY)), strict=True):
        for s in seq.spawn(N_STATS_PER_PRESET):
            channels.append(generate_channel(ChannelConfig(preset=preset), np.random.default_rng(s)))
    df = stats_table(channels)
    df.to_csv(run_dir / "channel_stats.csv", index=False)
    summary = df.groupby("preset", sort=False).agg(["mean", "std"]).round(2)
    summary.to_csv(run_dir / "channel_stats_summary.csv")

    metrics = {"n_per_preset": N_STATS_PER_PRESET, "means": df.groupby("preset").mean().round(3).to_dict()}
    write_run_files(cfg, run_dir, metrics, {})
    print(summary.xs("mean", axis=1, level=1).T.to_string())
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()

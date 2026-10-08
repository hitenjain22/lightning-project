"""Animated reconstruction (SPEC.md Phase 8 hero image).

Points appear as their sound reaches the array, over the true channel, while a cursor runs along
the recorded waveform underneath. The view turns slowly so the 3D shape reads in a GIF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from thunder.viz.plots import BRANCH_COLOR, INCLOUD_COLOR, MAIN_COLOR


def animate_reconstruction(
    example: dict[str, Any],
    path: Path,
    title: str = "",
    n_frames: int = 60,
    fps: int = 12,
    hold_frames: int = 18,
    dpi: int = 80,
    turn_deg: float = 50.0,
) -> Path:
    """Write a GIF of the reconstruction appearing in time.

    example: the dict of `experiments.montecarlo._example` (true channel, points, their window
    times on the recorder clock, point errors, mics, first-mic signal, sample rate).
    The last frame is held for `hold_frames` so the finished picture can be read.
    """
    nodes, segs = example["nodes"], example["segments"]
    pts, err = np.asarray(example["points"]), np.asarray(example["error_m"])
    times = np.asarray(example["window_times"], dtype=float)
    sig = np.asarray(example["signals"][0], dtype=float)
    fs = float(example["sample_rate"])
    t_sig = np.arange(len(sig)) / fs
    if len(pts):
        t0, t1 = float(times.min()) - 0.5, float(times.max()) + 0.5
    else:
        t0, t1 = 0.0, t_sig[-1]
    frame_t = np.linspace(t0, t1, n_frames)

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_axes((0.0, 0.3, 0.88, 0.6), projection="3d")
    assert isinstance(ax, Axes3D)
    axw = fig.add_axes((0.1, 0.06, 0.85, 0.2))
    side = ~(example["is_main"] | example["is_incloud"])
    for mask, color, lw in (
        (side, BRANCH_COLOR, 0.7),
        (example["is_incloud"], INCLOUD_COLOR, 1.0),
        (example["is_main"], MAIN_COLOR, 1.2),
    ):
        for a, b in segs[mask]:
            ax.plot(*np.c_[nodes[a], nodes[b]], color=color, lw=lw, alpha=0.35)
    mics = example["mics"]
    ax.scatter(mics[:, 0], mics[:, 1], zs=mics[:, 2], color="k", marker="^", s=30)
    vmax = max(float(np.percentile(err, 95)), 1.0) if len(err) else 1.0
    sc = ax.scatter([], [], zs=[], c=[], cmap="viridis", vmin=0, vmax=vmax, s=10)
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    lo, hi = np.minimum(lo, mics.min(axis=0)), np.maximum(hi, mics.max(axis=0))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(0, hi[2])
    ax.set_xlabel("east (m)", fontsize=8)
    ax.set_ylabel("north (m)", fontsize=8)
    ax.set_zlabel("up (m)", fontsize=8)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(sc, cax=fig.add_axes((0.9, 0.42, 0.02, 0.36)))
    cb.set_label("point error (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=11, weight="bold")
    title_text = fig.text(0.5, 0.925, "", ha="center", va="top", fontsize=9)
    axw.plot(t_sig, sig, lw=0.4, color="0.25")
    axw.set_xlim(t0, t1)
    peak = float(np.max(np.abs(sig[(t_sig >= t0) & (t_sig <= t1)]))) if len(sig) else 1.0
    axw.set_ylim(-1.05 * peak, 1.05 * peak)
    axw.set_xlabel("time since the flash (s)", fontsize=8)
    axw.set_ylabel("Pa", fontsize=8)
    axw.tick_params(labelsize=7)
    cursor = axw.axvline(t0, color="tab:red", lw=1.2)
    shade = Rectangle((t0, -2.0 * peak), 0.0, 4.0 * peak, color="tab:red", alpha=0.08)
    axw.add_patch(shade)

    def draw(i: int) -> list:
        t = frame_t[min(i, n_frames - 1)]
        seen = times <= t
        sc._offsets3d = (pts[seen, 0], pts[seen, 1], pts[seen, 2])
        sc.set_array(err[seen])
        cursor.set_xdata([t, t])
        shade.set_width(max(t - t0, 0.0))
        ax.view_init(elev=18, azim=-60 + turn_deg * min(i, n_frames - 1) / max(n_frames - 1, 1))
        title_text.set_text(f"{int(seen.sum())} points reconstructed at t = {t:.1f} s after the flash")
        return [sc, cursor, shade, title_text]

    anim = FuncAnimation(fig, draw, frames=n_frames + hold_frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path

"""Static matplotlib plots."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from thunder.types import Channel

MAIN_COLOR = "#1f3a93"
BRANCH_COLOR = "#7f9cc8"
INCLOUD_COLOR = "#c0392b"


def plot_channel_projection(ax: Axes, ch: Channel, horizontal: str = "x", km: bool = True) -> None:
    """Draw a channel projected onto a vertical plane.

    horizontal: 'x' or 'y' (ENU axes), 'range' (horizontal distance from the array origin,
    what the array sees), or 'principal' (the channel's widest horizontal direction,
    measured from the strike point; shows the most structure).
    """
    scale = 1e-3 if km else 1.0
    a, b = ch.segment_endpoints()
    if horizontal == "range":
        ha, hb = np.hypot(a[:, 0], a[:, 1]), np.hypot(b[:, 0], b[:, 1])
    elif horizontal == "principal":
        xy = ch.nodes[:, :2] - ch.nodes[:, :2].mean(axis=0)
        axis = np.linalg.svd(xy, full_matrices=False)[2][0] if np.ptp(xy) > 0 else np.array([1.0, 0.0])
        strike = ch.nodes[np.argmin(ch.nodes[:, 2]), :2]
        ha, hb = (a[:, :2] - strike) @ axis, (b[:, :2] - strike) @ axis
    else:
        k = {"x": 0, "y": 1}[horizontal]
        ha, hb = a[:, k], b[:, k]
    segs = np.stack([np.c_[ha, a[:, 2]], np.c_[hb, b[:, 2]]], axis=1) * scale
    side = ~(ch.is_main | ch.is_incloud)
    for mask, color, width in (
        (side, BRANCH_COLOR, 0.7),
        (ch.is_incloud, INCLOUD_COLOR, 1.2),
        (ch.is_main, MAIN_COLOR, 1.4),
    ):
        if mask.any():
            ax.add_collection(LineCollection(list(segs[mask]), colors=color, linewidths=width))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    unit = "km" if km else "m"
    label = "horizontal, widest direction" if horizontal == "principal" else horizontal
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel(f"z ({unit})")
    ax.axhline(0, color="0.4", lw=0.8)


def plot_waveform_stack(
    axes: tuple[Axes, Axes], signals: np.ndarray, fs: float, labels: list[str], zoom_s: float = 0.15
) -> None:
    """Left: full rumble for every mic (common scale, offset vertically). Right: zoom on the onset."""
    t = np.arange(signals.shape[1]) / fs
    scale = np.abs(signals).max() or 1.0
    onset = np.flatnonzero(np.abs(signals).max(axis=0) > 1e-3 * scale)
    t_on = t[onset[0]] if len(onset) else 0.0
    for ax, (lo, hi) in zip(axes, ((t_on - 0.5, t[-1]), (t_on - 0.02, t_on + zoom_s)), strict=True):
        sel = (t >= lo) & (t <= hi)
        for i, x in enumerate(signals):
            ax.plot(t[sel], x[sel] / scale - 2.2 * i, lw=0.5, color=MAIN_COLOR)
        ax.set_yticks([-2.2 * i for i in range(len(signals))], labels)
        ax.set_xlabel("time since flash (s)")
        ax.set_xlim(lo, hi)
    axes[0].set_title(f"Thunder at each mic (common scale, peak {scale:.1f} Pa)")
    axes[1].set_title("Onset (arrival-time differences across mics)")


def plot_spectrogram(ax: Axes, x: np.ndarray, fs: float, f_max: float = 1000.0, title: str = "") -> None:
    """Spectrogram in dB re 20 uPa^2/Hz, 128 ms Hann windows, 87.5% overlap."""
    from scipy import signal as sps

    nper = int(0.128 * fs)
    f, t, sxx = sps.spectrogram(x, fs, window="hann", nperseg=nper, noverlap=nper * 7 // 8)
    keep = f <= f_max
    db = 10 * np.log10(sxx[keep] / (20e-6) ** 2 + 1e-12)
    vmax = db.max()
    mesh = ax.pcolormesh(t, f[keep], db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
    ax.figure.colorbar(mesh, ax=ax, label="dB re 20 µPa²/Hz")
    ax.set_xlabel("time since flash (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title)

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

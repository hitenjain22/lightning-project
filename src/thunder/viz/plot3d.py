"""Interactive 3D views (plotly HTML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from thunder.viz.plots import BRANCH_COLOR, INCLOUD_COLOR, MAIN_COLOR


def _segment_trace(
    nodes: np.ndarray, segments: np.ndarray, mask: np.ndarray, color: str, name: str, width: float
):
    xs, ys, zs = [], [], []
    for a, b in segments[mask]:
        xs += [nodes[a, 0], nodes[b, 0], None]
        ys += [nodes[a, 1], nodes[b, 1], None]
        zs += [nodes[a, 2], nodes[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"color": color, "width": width}, name=name)


def reconstruction_figure(example: dict[str, Any], title: str) -> go.Figure:
    """True channel, reconstructed points colored by error, their uncertainty (toggle in the
    legend), and the microphones."""
    nodes, segs = example["nodes"], example["segments"]
    side = ~(example["is_main"] | example["is_incloud"])
    traces = [
        _segment_trace(nodes, segs, example["is_main"], MAIN_COLOR, "true main channel", 5),
        _segment_trace(nodes, segs, side, BRANCH_COLOR, "true branches", 3),
    ]
    if example["is_incloud"].any():
        traces.append(_segment_trace(nodes, segs, example["is_incloud"], INCLOUD_COLOR, "true in-cloud", 4))
    p, err = example["points"], example["error_m"]
    if len(p):
        traces.append(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="markers",
                name="reconstructed points",
                marker={
                    "size": 3,
                    "color": err,
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": float(np.percentile(err, 95)),
                    "colorbar": {"title": "error (m)"},
                },
                text=[f"error {e:.1f} m" for e in err],
            )
        )
    cov = example.get("covariances")
    if len(p) and cov is not None and len(cov) == len(p):
        traces.append(ellipsoid_axes_trace(p, np.asarray(cov)))
    mics = example["mics"]
    traces.append(
        go.Scatter3d(
            x=mics[:, 0],
            y=mics[:, 1],
            z=mics[:, 2],
            mode="markers",
            name="microphones",
            marker={"size": 5, "color": "black", "symbol": "diamond"},
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene={
            "aspectmode": "data",
            "xaxis_title": "east (m)",
            "yaxis_title": "north (m)",
            "zaxis_title": "up (m)",
        },
    )
    return fig


def ellipsoid_axes_trace(points: np.ndarray, covariances: np.ndarray, n_sigma: float = 2.0) -> go.Scatter3d:
    """Uncertainty ellipsoids drawn as their principal axes (+-n_sigma along each eigenvector):
    light enough for hundreds of points, and hidden until enabled in the legend."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for p, c in zip(points, covariances, strict=True):
        lam, vec = np.linalg.eigh(0.5 * (c + c.T))
        for k in range(3):
            half = n_sigma * np.sqrt(max(float(lam[k]), 0.0)) * vec[:, k]
            a, b = p - half, p + half
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "rgba(80,80,80,0.6)", "width": 2},
        name=f"uncertainty ({n_sigma:g}-sigma axes)",
        visible="legendonly",
    )


def write_html(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)

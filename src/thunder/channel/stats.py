"""Channel statistics: tortuosity, size, branching and fractal dimension."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from thunder.types import Channel, FloatArray


def segment_vectors(ch: Channel) -> FloatArray:
    a, b = ch.segment_endpoints()
    return b - a


def segment_lengths(ch: Channel) -> FloatArray:
    return np.linalg.norm(segment_vectors(ch), axis=1)


def incoming_segment(ch: Channel) -> np.ndarray:
    """For each node, the index of the segment ending there (-1 for the root)."""
    incoming = np.full(ch.nodes.shape[0], -1, dtype=np.int64)
    incoming[ch.segments[:, 1]] = np.arange(ch.n_segments)
    return incoming


def is_connected_tree(ch: Channel) -> bool:
    """True if every node except one root has exactly one parent and all are reachable from it."""
    n = ch.nodes.shape[0]
    children = ch.segments[:, 1]
    if ch.n_segments != n - 1 or len(np.unique(children)) != len(children):
        return False
    roots = np.setdiff1d(np.arange(n), children)
    if len(roots) != 1:
        return False
    adjacency: dict[int, list[int]] = {}
    for p, c in ch.segments:
        adjacency.setdefault(int(p), []).append(int(c))
    seen, stack = {int(roots[0])}, [int(roots[0])]
    while stack:
        for c in adjacency.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return len(seen) == n


def turn_angles(ch: Channel) -> FloatArray:
    """Angles (rad) between successive segments of the same branch.

    Branch departure joints are excluded; they follow the branch-angle distribution instead.
    Uses atan2(|a x b|, a . b), which stays accurate for small angles (arccos does not).
    """
    vec = segment_vectors(ch)
    prev = incoming_segment(ch)[ch.segments[:, 0]]
    ok = prev >= 0
    ok[ok] = ch.branch_id[prev[ok]] == ch.branch_id[ok]
    a, b = vec[prev[ok]], vec[ok]
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))


def total_length(ch: Channel) -> float:
    return float(segment_lengths(ch).sum())


def branch_count(ch: Channel) -> int:
    """Number of branches (excludes the main channel and in-cloud section)."""
    side = ~(ch.is_main | ch.is_incloud)
    return int(len(np.unique(ch.branch_id[side])))


def horizontal_extent(ch: Channel) -> float:
    """Largest horizontal distance between any two nodes, m."""
    xy = ch.nodes[:, :2]
    return float(np.max(pdist(xy))) if len(xy) > 1 else 0.0


def sample_points(ch: Channel, spacing: float) -> FloatArray:
    """Points along every segment at most `spacing` apart (includes all nodes)."""
    a, b = ch.segment_endpoints()
    n = np.maximum(1, np.ceil(segment_lengths(ch) / spacing).astype(np.int64))
    seg = np.repeat(np.arange(ch.n_segments), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = (k / n[seg])[:, None]
    return np.vstack([a[seg] + t * (b - a)[seg], ch.nodes])


def default_box_scales(ch: Channel, n: int = 8) -> FloatArray:
    """Geometric box sizes from 4 segment lengths up to 1/4 of the largest bounding-box side."""
    lo = 4.0 * float(np.median(segment_lengths(ch)))
    hi = max(float(np.max(np.ptp(ch.nodes, axis=0))) / 4.0, 8.0 * lo)
    return np.geomspace(lo, hi, n)


def box_counts(ch: Channel, scales: FloatArray) -> np.ndarray:
    pts = sample_points(ch, spacing=float(np.min(scales)) / 10.0)
    origin = pts.min(axis=0)
    return np.array([len(np.unique(np.floor((pts - origin) / s).astype(np.int64), axis=0)) for s in scales])


def fractal_dimension(ch: Channel, scales: FloatArray | None = None) -> float:
    """Box-counting dimension: slope of log N(eps) against log(1/eps)."""
    scales = default_box_scales(ch) if scales is None else np.asarray(scales, dtype=float)
    counts = box_counts(ch, scales)
    slope, _ = np.polyfit(np.log(1.0 / scales), np.log(counts), 1)
    return float(slope)


def channel_stats(ch: Channel) -> dict[str, float]:
    angles = turn_angles(ch)
    return {
        "turn_mean_deg": float(np.degrees(angles.mean())) if len(angles) else 0.0,
        "turn_median_deg": float(np.degrees(np.median(angles))) if len(angles) else 0.0,
        "total_length_m": total_length(ch),
        "main_length_m": float(segment_lengths(ch)[ch.is_main].sum()),
        "incloud_length_m": float(segment_lengths(ch)[ch.is_incloud].sum()),
        "branch_count": branch_count(ch),
        "horizontal_extent_m": horizontal_extent(ch),
        "start_height_m": float(ch.nodes[0, 2]),
        "fractal_dimension": fractal_dimension(ch),
        "n_strokes": len(ch.stroke_times),
    }


def stats_table(channels: list[Channel]) -> pd.DataFrame:
    """One row per channel, with the preset name when available."""
    rows = [{"preset": ch.metadata.get("preset", ""), **channel_stats(ch)} for ch in channels]
    return pd.DataFrame(rows)

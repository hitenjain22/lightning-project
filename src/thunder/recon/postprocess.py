"""Reconstruction post-processing (SPEC.md Phase 5).

1. Outlier removal with DBSCAN: points not density-reachable from a core point are dropped.
2. Channel skeleton: a minimum spanning tree over the cleaned points, with leaf spurs shorter
   than `spur_length` pruned repeatedly (a spur is a path from a leaf to the nearest node of
   degree >= 3).
3. Strike point: the lowest part of the reconstruction is extrapolated to the ground along its
   principal direction, with the horizontal move capped at `max_slope` x the height of the
   lowest point (channels rarely lean more than ~30 deg near the ground). On development bolts
   this beat projecting the lowest points straight down (median 39 m vs 54 m; docs/log.md).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from thunder.types import FloatArray, IntArray


def dbscan_inliers(points: FloatArray, eps: float, min_samples: int) -> np.ndarray:
    """Boolean mask of points that belong to some DBSCAN cluster."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    return labels >= 0


def mst_edges(points: FloatArray) -> IntArray:
    """(E, 2) edges of the Euclidean minimum spanning tree (E = K - 1 for K >= 1 distinct points)."""
    if len(points) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    d = squareform(pdist(points))
    d[d == 0] = 1e-9  # csgraph treats 0 as "no edge"; keep coincident points connected
    np.fill_diagonal(d, 0.0)
    tree = minimum_spanning_tree(d).tocoo()
    return np.column_stack([tree.row, tree.col]).astype(np.int64)


def prune_spurs(points: FloatArray, edges: IntArray, spur_length: float) -> IntArray:
    """Remove leaf-to-junction paths shorter than spur_length, repeating until stable."""
    work: list[tuple[int, int]] = [(int(a), int(b)) for a, b in edges]
    while True:
        adj: dict[int, set[int]] = {}
        for a, b in work:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        removed = False
        for leaf in [v for v, nb in adj.items() if len(nb) == 1]:
            if leaf not in adj or len(adj[leaf]) != 1:
                continue
            path, length, prev, cur = [leaf], 0.0, -1, leaf
            while True:
                nxt = [v for v in adj[cur] if v != prev]
                if len(nxt) != 1:
                    break
                length += float(np.linalg.norm(points[nxt[0]] - points[cur]))
                prev, cur = cur, nxt[0]
                path.append(cur)
                if len(adj[cur]) != 2:
                    break
            if len(adj[cur]) >= 3 and length < spur_length:
                drop = {frozenset(e) for e in zip(path[:-1], path[1:], strict=True)}
                work = [e for e in work if frozenset(e) not in drop]
                for a, b in zip(path[:-1], path[1:], strict=True):
                    adj[a].discard(b)
                    adj[b].discard(a)
                removed = True
        if not removed:
            return np.array(work, dtype=np.int64).reshape(-1, 2)


def skeleton(points: FloatArray, spur_length: float) -> IntArray:
    return prune_spurs(points, mst_edges(points), spur_length)


def estimate_strike_point(points: FloatArray, n_lowest: int = 8, max_slope: float = 0.5) -> FloatArray | None:
    """Ground contact point (z = 0) extrapolated from the lowest reconstructed points."""
    if len(points) == 0:
        return None
    low = points[np.argsort(points[:, 2])[:n_lowest]]
    base = low[0]
    if len(low) < 3:
        return np.array([base[0], base[1], 0.0])
    center = low.mean(axis=0)
    direction = np.linalg.svd(low - center)[2][0]
    if direction[2] > 0:
        direction = -direction  # point downward
    if abs(direction[2]) < 1e-3:
        return np.array([base[0], base[1], 0.0])
    hit = center + (-center[2] / direction[2]) * direction
    shift = hit[:2] - base[:2]
    limit = max_slope * max(base[2], 0.0)
    norm = float(np.linalg.norm(shift))
    if norm > limit:
        shift = shift * (limit / norm) if norm > 0 else shift
    return np.array([base[0] + shift[0], base[1] + shift[1], 0.0])

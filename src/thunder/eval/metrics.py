"""Phase 6 evaluation metrics (SPEC.md table).

Point error uses the exact distance to the true channel polyline (vectorized point-to-segment
distances against every segment, in chunks), so it has no sampling error. Coverage and the
truth-to-reconstruction half of the Chamfer distance use the channel densely sampled at
`truth_sample_spacing_m`, each sample weighted by the channel length it represents.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from thunder.config import EvaluationConfig
from thunder.recon.postprocess import estimate_strike_point
from thunder.types import Channel, FloatArray

# Expected fraction of 3D Gaussian errors inside the 1, 2, 3 sigma ellipsoids (chi-square, 3 dof):
# 19.9%, 73.9%, 97.1% (SPEC.md).
SIGMA_LEVELS = (1.0, 2.0, 3.0)
EXPECTED_COVERAGE = tuple(float(stats.chi2.cdf(k**2, df=3)) for k in SIGMA_LEVELS)
_CHUNK = 4096


def nearest_on_segments(
    points: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Exact distance from each point to the nearest segment a->b, the nearest point, and its segment."""
    points = np.atleast_2d(points)
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    dist = np.empty(len(points))
    nearest = np.empty_like(points)
    idx = np.empty(len(points), dtype=np.int64)
    for lo in range(0, len(points), max(1, _CHUNK * 64 // max(len(a), 1))):
        p = points[lo : lo + max(1, _CHUNK * 64 // max(len(a), 1))]
        t = np.clip(np.einsum("kij,ij->ki", p[:, None, :] - a[None], ab) / ab2, 0.0, 1.0)
        q = a[None] + t[..., None] * ab[None]
        d2 = np.sum((p[:, None, :] - q) ** 2, axis=2)
        k = np.argmin(d2, axis=1)
        rows = np.arange(len(p))
        dist[lo : lo + len(p)] = np.sqrt(d2[rows, k])
        nearest[lo : lo + len(p)] = q[rows, k]
        idx[lo : lo + len(p)] = k
    return dist, nearest, idx


def sample_channel(ch: Channel, spacing: float) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Points at the centers of sub-pieces <= spacing long, their lengths, and their segment index."""
    a, b = ch.segment_endpoints()
    length = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(length / spacing).astype(np.int64))
    seg = np.repeat(np.arange(ch.n_segments), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = ((k + 0.5) / n[seg])[:, None]
    return a[seg] + t * (b - a)[seg], length[seg] / n[seg], seg


def true_strike_point(ch: Channel) -> FloatArray:
    sp = ch.metadata.get("strike_point")
    if sp is not None:
        return np.asarray(sp, dtype=float)
    return ch.nodes[int(np.argmin(ch.nodes[:, 2]))]


def calibration(errors: FloatArray, covariances: FloatArray) -> list[float]:
    """Fraction of error vectors inside the 1, 2, 3 sigma ellipsoids of their covariances."""
    if len(errors) == 0:
        return [float("nan")] * len(SIGMA_LEVELS)
    inv = np.linalg.pinv(covariances, hermitian=True)
    m2 = np.einsum("ki,kij,kj->k", errors, inv, errors)
    return [float(np.mean(m2 <= k**2)) for k in SIGMA_LEVELS]


def evaluate(
    points: FloatArray,
    covariances: FloatArray,
    ch: Channel,
    array_centroid: FloatArray,
    cfg: EvaluationConfig,
    strike_estimate: FloatArray | None = None,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Scalar metrics for one reconstruction, plus per-point arrays for pooled analysis.

    `strike_estimate` is the method's ground-contact estimate (Reconstruction.extra
    ["strike_point"]); if omitted, the default estimator is applied to the points.
    """
    a, b = ch.segment_endpoints()
    samples, weights, sample_seg = sample_channel(ch, cfg.truth_sample_spacing_m)
    main_w = ch.is_main[sample_seg]
    centroid = np.asarray(array_centroid, dtype=float)
    metrics: dict[str, Any] = {"n_points": int(len(points))}
    per_point: dict[str, FloatArray] = {}

    if len(points) == 0:
        for d in cfg.coverage_distances_m:
            metrics[f"coverage_{d:g}m"] = 0.0
            metrics[f"coverage_main_{d:g}m"] = 0.0
        metrics.update(
            point_error_median_m=np.nan,
            point_error_p90_m=np.nan,
            point_error_mean_m=np.nan,
            chamfer_m=np.nan,
            radial_error_median_m=np.nan,
            transverse_error_median_m=np.nan,
            strike_error_m=np.nan,
            calib_1sigma=np.nan,
            calib_2sigma=np.nan,
            calib_3sigma=np.nan,
        )
        return metrics, per_point

    err, nearest, _ = nearest_on_segments(points, a, b)
    vec = points - nearest
    los = points - centroid
    los /= np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-300)
    radial = np.einsum("ij,ij->i", vec, los)
    transverse = np.linalg.norm(vec - radial[:, None] * los, axis=1)

    d_truth, _ = cKDTree(points).query(samples)
    for d in cfg.coverage_distances_m:
        covered = d_truth <= d
        metrics[f"coverage_{d:g}m"] = float(weights[covered].sum() / weights.sum())
        metrics[f"coverage_main_{d:g}m"] = float(weights[covered & main_w].sum() / weights[main_w].sum())

    strike = strike_estimate if strike_estimate is not None else estimate_strike_point(points)
    calib = calibration(vec, covariances)
    metrics.update(
        point_error_median_m=float(np.median(err)),
        point_error_p90_m=float(np.percentile(err, 90)),
        point_error_mean_m=float(np.mean(err)),
        chamfer_m=float(0.5 * (np.mean(err) + np.average(d_truth, weights=weights))),
        radial_error_median_m=float(np.median(np.abs(radial))),
        transverse_error_median_m=float(np.median(transverse)),
        strike_error_m=float(np.linalg.norm((strike - true_strike_point(ch))[:2]))
        if strike is not None
        else np.nan,
        calib_1sigma=calib[0],
        calib_2sigma=calib[1],
        calib_3sigma=calib[2],
    )
    per_point = {
        "error_m": err,
        "radial_m": radial,
        "transverse_m": transverse,
        "range_m": np.linalg.norm(points - centroid, axis=1),
        "true_altitude_m": nearest[:, 2],
    }
    return metrics, per_point

"""Monte Carlo aggregation: bootstrap confidence intervals and binned summaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from thunder.types import FloatArray


def bootstrap_ci(
    values: FloatArray,
    stat: Callable[[FloatArray], float],
    rng: np.random.Generator,
    n_resamples: int = 2000,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Statistic of the finite values with a percentile bootstrap confidence interval."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, v.size, (n_resamples, v.size))
    boots = np.array([stat(v[i]) for i in idx])
    alpha = (1 - level) / 2
    return float(stat(v)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def cluster_bootstrap_ci(
    values: FloatArray,
    groups: np.ndarray,
    stat: Callable[[FloatArray], float],
    rng: np.random.Generator,
    n_resamples: int = 2000,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI that resamples whole groups (bolts), not individual values.

    Points from one bolt share its geometry and errors, so they are not independent;
    resampling points would understate the uncertainty of pooled statistics.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size == 0:
        return (float("nan"),) * 3
    labels, inverse = np.unique(g, return_inverse=True)
    members = [v[inverse == k] for k in range(len(labels))]
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.integers(0, len(members), len(members))
        boots[b] = stat(np.concatenate([members[k] for k in pick]))
    alpha = (1 - level) / 2
    return float(stat(v)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def binned_quantiles(
    x: FloatArray, y: FloatArray, edges: FloatArray, qs: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> dict[str, np.ndarray]:
    """Quantiles of y within bins of x (NaN for bins with fewer than 5 values)."""
    x, y = np.asarray(x), np.asarray(y)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {"center": centers, "count": np.zeros(len(centers), dtype=np.int64)}
    for q in qs:
        out[f"q{int(round(q * 100))}"] = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = (x >= edges[k]) & (x < edges[k + 1]) & np.isfinite(y)
        out["count"][k] = int(sel.sum())
        if sel.sum() >= 5:
            for q in qs:
                out[f"q{int(round(q * 100))}"][k] = float(np.quantile(y[sel], q))
    return out

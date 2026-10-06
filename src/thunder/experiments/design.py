"""Phase 7 / E2: array-design surrogate (Fisher information, Cramér-Rao bound) and optimization.

Model. A plane wave from unit direction u(az, el) (toward the source) reaches mic m at
    t_m = tau0 - (m_m . u) / c + e_m,      e_m ~ N(0, sigma_t^2) independent,
with an unknown emission time tau0 (only time differences carry direction). Eliminating tau0
(Schur complement) leaves the Fisher information of theta = (az, el):
    F = 1 / (sigma_t c)^2  sum_m  g_m g_m^T,      g_m = D^T (m_m - mean(m)),   D = du/dtheta (3 x 2)
with az clockwise from north:  u = (cos el sin az, cos el cos az, sin el).
The Cramér-Rao bound on the angular (great-circle) error variance is
    var_ang >= trace(W F^-1),  W = diag(cos^2 el, 1)   (an azimuth error d_az moves u by cos(el) d_az).
For a planar array, elevation information comes only from the horizontal slowness magnitude
cos(el) and vanishes near the horizon, as it should.

Design criteria, averaged over a representative set of source directions:
  A-optimal: minimize mean trace(W F^-1)  (expected squared angular error)
  D-optimal: maximize mean log det F      (volume of the confidence region)
Free mic positions are optimized with scipy's differential evolution under the aperture
constraint (largest horizontal mic-to-mic distance <= aperture, enforced by a penalty), with
mics at a fixed height and optionally one mic on a mast of variable height.

The bound ignores what full simulation includes: several sources in one window, finite
windows, wavefront curvature on large apertures, coherence across large separations. E2 step 4
checks whether its ranking of layouts survives full simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

from thunder.types import FloatArray

DEFAULT_AZIMUTHS_DEG = np.arange(0.0, 360.0, 10.0)
DEFAULT_ELEVATIONS_DEG = np.arange(5.0, 70.1, 5.0)  # typical of bolts 1-3 km away (channel up to ~7 km)


def direction_jacobian(az: FloatArray, el: FloatArray) -> FloatArray:
    """du/d(az, el), shape (..., 3, 2)."""
    az, el = np.asarray(az, dtype=float), np.asarray(el, dtype=float)
    d_az = np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.zeros_like(az)], axis=-1)
    d_el = np.stack([-np.sin(el) * np.sin(az), -np.sin(el) * np.cos(az), np.cos(el)], axis=-1)
    return np.stack([d_az, d_el], axis=-1)


def fisher_direction(
    positions: FloatArray, az: FloatArray, el: FloatArray, sigma_t: float, c: float
) -> FloatArray:
    """Fisher information of (az, el), shape (..., 2, 2), for each direction."""
    m = np.asarray(positions, dtype=float)
    m = m - m.mean(axis=0)
    jac = direction_jacobian(az, el)  # (..., 3, 2)
    g = np.einsum("mi,...ik->...mk", m, jac)  # (..., M, 2)
    return np.einsum("...mk,...ml->...kl", g, g) / (sigma_t * c) ** 2


def angular_variance(
    positions: FloatArray, az: FloatArray, el: FloatArray, sigma_t: float, c: float
) -> FloatArray:
    """Cramér-Rao bound on the angular error variance (rad^2) per direction (inf if singular)."""
    f = fisher_direction(positions, az, el, sigma_t, c)
    det = f[..., 0, 0] * f[..., 1, 1] - f[..., 0, 1] ** 2
    el = np.asarray(el, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv00 = f[..., 1, 1] / det
        inv11 = f[..., 0, 0] / det
        var = np.cos(el) ** 2 * inv00 + inv11
    return np.where(det > 1e-30 * np.maximum(f[..., 0, 0] * f[..., 1, 1], 1e-300), var, np.inf)


def plane_wave_bias(positions: FloatArray, sources: FloatArray, c: float = 343.0) -> FloatArray:
    """Direction error (deg) of the far-field (plane-wave) fit for exact, noise-free arrivals
    from point sources at `sources` (K, 3): the wavefront-curvature bias of Methods A and B.

    The curvature term of the arrival times is quadratic in mic position, so it enters the fit
    through the layout's third moments: it vanishes (to O(1/R^2)) for layouts symmetric about
    their centroid and for regular polygons other than the triangle, and is O(aperture^2 / R)
    otherwise (a raised mast mic, L-shapes, irregular placements). The bound above, a pure
    variance, cannot see it.
    """
    from thunder.recon.tdoa import solve_direction  # local: design is analysis-side code

    m = np.asarray(positions, dtype=float)
    n = len(m)
    ii, jj = np.triu_indices(n, 1)
    diffs = m[jj] - m[ii]
    centroid = m.mean(axis=0)
    out = np.empty(len(sources))
    for k, x in enumerate(np.asarray(sources, dtype=float)):
        t = np.linalg.norm(m - x, axis=1) / c
        fit = solve_direction(diffs, t[jj] - t[ii], np.ones(len(diffs)), c, 1e-12)
        u = (x - centroid) / np.linalg.norm(x - centroid)
        out[k] = np.degrees(np.arccos(np.clip(fit.u @ u, -1.0, 1.0)))
    return out


@dataclass(frozen=True)
class Surrogate:
    rms_angular_error_deg: float  # sqrt(mean CRB angular variance) over the direction set
    mean_log_det: float  # mean log det F
    worst_angular_error_deg: float  # sqrt(max CRB angular variance)


def _grid(azimuths_deg: FloatArray, elevations_deg: FloatArray) -> tuple[FloatArray, FloatArray]:
    az, el = np.meshgrid(np.radians(azimuths_deg), np.radians(elevations_deg), indexing="ij")
    return az.ravel(), el.ravel()


def evaluate_layout(
    positions: FloatArray,
    sigma_t: float = 1e-4,
    c: float = 343.0,
    azimuths_deg: FloatArray = DEFAULT_AZIMUTHS_DEG,
    elevations_deg: FloatArray = DEFAULT_ELEVATIONS_DEG,
) -> Surrogate:
    az, el = _grid(azimuths_deg, elevations_deg)
    var = angular_variance(positions, az, el, sigma_t, c)
    f = fisher_direction(positions, az, el, sigma_t, c)
    det = np.linalg.det(f)
    logdet = np.where(det > 0, np.log(np.maximum(det, 1e-300)), -np.inf)
    return Surrogate(
        rms_angular_error_deg=float(np.degrees(np.sqrt(np.mean(var)))),
        mean_log_det=float(np.mean(logdet)),
        worst_angular_error_deg=float(np.degrees(np.sqrt(np.max(var)))),
    )


@dataclass(frozen=True)
class OptimizedLayout:
    positions: FloatArray  # (M, 3), horizontal centroid at the origin
    criterion: str
    surrogate: Surrogate
    aperture_m: float  # achieved (<= requested)


def optimize_layout(
    n_mics: int,
    aperture: float,
    criterion: str = "A",
    mast: bool = False,
    mic_height: float = 1.5,
    mast_height_max: float = 10.0,
    seed: int = 0,
    sigma_t: float = 1e-4,
    c: float = 343.0,
    maxiter: int = 400,
    min_separation_frac: float = 0.2,
) -> OptimizedLayout:
    """Free mic positions optimizing the A- or D-criterion under the placement constraints.

    Variables: x, y of every mic (and the height of mic 0 if `mast`). Constraints (penalties
    during the search, checked exactly on the result): largest horizontal distance <= aperture,
    and every pair of mics at least `min_separation_frac * aperture` apart (3-D). The bound
    assumes independent timing errors, so without a minimum spacing it rewards stacking mics
    in one spot (each copy "halves" the variance); real mics that close hear the same waveform
    and noise, so their errors are correlated and the copy adds almost nothing (E2 finding).
    """
    if criterion not in ("A", "D"):
        raise ValueError("criterion must be 'A' or 'D'")
    # Optimize on a coarser direction grid (fast); the result is scored on the full grid.
    az, el = _grid(DEFAULT_AZIMUTHS_DEG[::3], DEFAULT_ELEVATIONS_DEG[::2])

    def unpack(v: FloatArray) -> FloatArray:
        xy = v[: 2 * n_mics].reshape(n_mics, 2) * aperture
        z = np.full(n_mics, mic_height)
        if mast:
            z[0] = v[-1]
        return np.column_stack([xy, z])

    def objective(v: FloatArray) -> float:
        p = unpack(v)
        excess = max(float(np.max(pdist(p[:, :2]))) - aperture, 0.0) / aperture
        excess += max(min_sep - float(np.min(pdist(p))), 0.0) / aperture
        if criterion == "A":
            var = angular_variance(p, az, el, sigma_t, c)
            value = float(np.log(np.mean(np.minimum(var, 1e6))))  # log: scale-free, well conditioned
        else:
            det = np.linalg.det(fisher_direction(p, az, el, sigma_t, c))
            value = -float(np.mean(np.log(np.maximum(det, 1e-300))))
        return value + 1e3 * excess + 1e3 * excess**2

    min_sep = min_separation_frac * aperture
    bounds = [(-0.75, 0.75)] * (2 * n_mics) + ([(mic_height, mast_height_max)] if mast else [])
    init = _initial_population(
        n_mics,
        mast,
        bounds,
        popsize=15,
        rng=np.random.default_rng(seed),
        mic_height=mic_height,
        mast_height_max=mast_height_max,
    )
    res = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        init=init,
        tol=1e-10,
        polish=True,
        updating="deferred",
        workers=1,
    )
    p = unpack(res.x)
    # Enforce the constraint exactly (the penalty can leave a tiny excess) and center.
    ap = float(np.max(pdist(p[:, :2])))
    if ap > aperture:
        p[:, :2] *= aperture / ap
    p[:, :2] -= p[:, :2].mean(axis=0)
    # Guarantee: never worse than the parametric seeds on the full direction grid.
    candidates = [p]
    for shape in _unit_shapes(n_mics):
        q = np.column_stack([shape * aperture, np.full(n_mics, mic_height)])
        candidates.append(q)
        if mast:
            q2 = q.copy()
            q2[0, 2] = mast_height_max
            candidates.append(q2)

    def score(q: FloatArray) -> float:
        sv = evaluate_layout(q, sigma_t, c)
        return sv.rms_angular_error_deg if criterion == "A" else -sv.mean_log_det

    feasible = [q for q in candidates if float(np.min(pdist(q))) >= min_sep * (1 - 1e-9)]
    if not feasible:
        raise ValueError(f"no layout of {n_mics} mics satisfies the minimum separation {min_sep:g} m")
    best = min(feasible, key=score)
    return OptimizedLayout(
        best, criterion, evaluate_layout(best, sigma_t, c), float(np.max(pdist(best[:, :2])))
    )


def _unit_shapes(n: int) -> list[FloatArray]:
    """Known-good layouts (unit aperture) to seed the optimizer: it can then only match or beat them."""

    def scaled(xy: FloatArray) -> FloatArray:
        return (xy - xy.mean(axis=0)) / float(np.max(pdist(xy)))

    ang = 2 * np.pi * np.arange(n) / n
    shapes = [scaled(np.column_stack([np.cos(ang), np.sin(ang)]))]
    if n >= 4:
        a = 2 * np.pi * np.arange(n - 1) / (n - 1)
        shapes.append(scaled(np.vstack([np.column_stack([np.cos(a), np.sin(a)]), [[0.0, 0.0]]])))
    return shapes


def _initial_population(n, mast, bounds, popsize, rng, mic_height, mast_height_max) -> FloatArray:
    """Seed rows (parametric shapes at several rotations) plus uniform random rows."""
    dim = len(bounds)
    total = max(popsize * dim, 10)
    rows = []
    for shape in _unit_shapes(n):
        for rot in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            r = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            xy = (shape @ r.T).ravel() * 0.999
            for z0 in [mic_height, mast_height_max] if mast else [None]:
                rows.append(np.concatenate([xy, [z0]]) if z0 is not None else xy)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    rand = lo + rng.random((total - len(rows), dim)) * (hi - lo)
    return np.vstack([np.array(rows), rand])

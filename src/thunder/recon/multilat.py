"""Phase 5 Method C: absolute-time multilateration (near-field, any assumed atmosphere).

Per window, the same matched-window TDOAs as Method A give, by weighted least squares over
all pairs (tau_ij = d_j - d_i, d_ref = 0), each mic's delay d_m behind the reference mic. With
the reference arrival time t_c (energy centroid) and the reported flash time t0, every mic has
an observed absolute travel time  T_m = t_c + d_m - t0.  The source position x minimizes

    sum_m rho( (T_m(x) - T_m) / sigma ),   rho = soft-L1: 2 (sqrt(1 + z^2) - 1)

where T_m(x) is the travel time through the *assumed* atmosphere (straight rays or traced
eigenrays). Its gradient is free: dT_m/dx = -s_m(x), the ray's slowness at the source. All
windows are solved together by iteratively reweighted, Levenberg-damped Gauss-Newton (one
batched propagation per mic per iteration, warm-started from the previous iteration's rays),
starting from Method A's points. Unlike the plane-wave methods this models wavefront
curvature exactly, so it also suits large or distributed arrays.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from thunder.config import ReconstructionConfig
from thunder.recon.tdoa import Setup, assemble, method_a_fits, place_points, prepare
from thunder.types import ArrivalPath, Atmosphere, FloatArray, Reconstruction, Recording


def relative_delays(st: Setup, tau: FloatArray, weight: FloatArray) -> tuple[FloatArray, float]:
    """Per-mic delays behind the reference (s) from pair TDOAs (weighted LS), and the RMS misfit."""
    m = len(st.mics)
    rows = np.zeros((len(st.pairs), m))
    for q, (i, j) in enumerate(st.pairs):
        rows[q, i], rows[q, j] = -1.0, 1.0
    keep = [k for k in range(m) if k != st.ref]
    a = rows[:, keep] * np.sqrt(weight)[:, None]
    sol, *_ = np.linalg.lstsq(a, tau * np.sqrt(weight), rcond=None)
    d = np.zeros(m)
    d[keep] = sol
    misfit = float(np.sqrt(np.mean((rows @ d - tau) ** 2)))
    return d, misfit


def _travel_times(atmosphere: Atmosphere, x: FloatArray, mics: FloatArray, guesses: list[ArrivalPath | None]):
    paths = [atmosphere.propagate(x, m, guess=g) for m, g in zip(mics, guesses, strict=True)]
    T = np.stack([p.travel_time for p in paths], axis=1)  # (K, M)
    S = np.stack([p.source_slowness for p in paths], axis=1)  # (K, M, 3)
    return T, S, paths


def multilaterate(
    x0: FloatArray, t_obs: FloatArray, mics: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> tuple[FloatArray, FloatArray, FloatArray, np.ndarray]:
    """Robust batched Gauss-Newton. Returns positions (K, 3), covariances, RMS residual (s), validity."""
    x = np.array(x0, dtype=float)
    k = len(x)
    scale = cfg.multilat_time_scale_s
    lam = np.full(k, 1e-3)
    active = np.all(np.isfinite(x), axis=1)
    guesses: list[ArrivalPath | None] = [None] * len(mics)

    def evaluate(xx, gs):
        T, S, paths = _travel_times(atmosphere, xx, mics, gs)
        r = T - t_obs
        valid = np.all(np.isfinite(r), axis=1)
        z = np.where(np.isfinite(r), r, 0.0) / scale
        w = 1.0 / np.sqrt(1.0 + z**2)  # IRLS weights of soft-L1
        cost = np.sum(2.0 * (np.sqrt(1.0 + z**2) - 1.0), axis=1)
        return r, S, w, np.where(valid, cost, np.inf), valid, paths

    xs = np.where(active[:, None], x, 0.0)
    r, S, w, cost, valid, paths = evaluate(xs, guesses)
    guesses = list(paths)
    active &= valid
    for _ in range(cfg.multilat_max_iterations):
        if not np.any(active):
            break
        J = -S  # dr/dx = dT/dx = -s_source  (K, M, 3)
        jw = J * w[..., None]
        h = np.einsum("kmi,kmj->kij", jw, J)
        g = np.einsum("kmi,km->ki", jw, np.where(np.isfinite(r), r, 0.0))
        damp = lam[:, None, None] * (np.eye(3)[None] * np.maximum(np.einsum("kii->ki", h)[:, :, None], 1e-30))
        step = -np.linalg.solve(h + damp + 1e-30 * np.eye(3)[None], g[..., None])[..., 0]
        step = np.where(active[:, None], step, 0.0)
        trial = x + step
        r_t, S_t, w_t, cost_t, valid_t, paths_t = evaluate(np.where(active[:, None], trial, xs), guesses)
        better = active & valid_t & (cost_t <= cost)
        x = np.where(better[:, None], trial, x)
        xs = np.where(active[:, None], x, 0.0)
        r = np.where(better[:, None], r_t, r)
        S = np.where(better[:, None, None], S_t, S)
        w = np.where(better[:, None], w_t, w)
        cost = np.where(better, cost_t, cost)
        lam = np.where(better, lam * 0.3, lam * 10.0)
        guesses = paths_t
        converged = better & (np.linalg.norm(step, axis=1) < 1e-3)
        stuck = lam > 1e8
        active &= ~(converged | stuck)
    ok = np.all(np.isfinite(r), axis=1) & np.all(np.isfinite(x), axis=1)
    m = len(mics)
    rr = np.where(np.isfinite(r), r, 0.0)
    rms = np.sqrt(np.mean(rr**2, axis=1))
    sigma2 = np.maximum(np.sum(w * rr**2, axis=1) / max(m - 3, 1), (1.0 / 8000.0) ** 2 / 12.0)
    J = -S
    h = np.einsum("kmi,kmj->kij", J * w[..., None], J)
    cov = sigma2[:, None, None] * np.linalg.pinv(h)
    return x, cov, rms, ok


def reconstruct_multilateration(
    rec: Recording, mic_positions: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> Reconstruction:
    """Method C on a recording (signals, nominal positions, reported t0, assumed atmosphere only)."""
    st = prepare(rec, mic_positions, atmosphere, cfg)
    fits = method_a_fits(st, cfg)
    if not fits:
        return Reconstruction.empty("C", cfg.config_hash())
    x0, _, ok0 = place_points(st, fits, rec.reported_t0, atmosphere)
    t_obs = np.empty((len(fits), len(st.mics)))
    misfit = np.empty(len(fits))
    for i, f in enumerate(fits):
        d, misfit[i] = relative_delays(st, f.tau, np.clip(f.peak, 1e-3, None))
        t_obs[i] = f.t_c + d - rec.reported_t0
    # Seed from every coherent window, not only those passing Method A's plane-wave gates:
    # with large arrays or near sources the plane-wave residual is large by construction, and
    # removing that bias is this method's purpose. C's own residual gate applies afterwards.
    use = ok0 & np.array([f.quality >= cfg.min_peak for f in fits])
    x = np.full_like(x0, np.nan)
    cov = np.zeros((len(fits), 3, 3))
    rms = np.full(len(fits), np.inf)
    ok = np.zeros(len(fits), dtype=bool)
    if np.any(use):
        xi, ci, ri, oki = multilaterate(x0[use], t_obs[use], st.mics, atmosphere, cfg)
        x[use], cov[use], rms[use], ok[use] = xi, ci, ri, oki
    ok &= use & (rms <= cfg.max_residual_s)
    fits_c = [dataclasses.replace(f, passed=bool(o)) for f, o in zip(fits, ok, strict=True)]
    return assemble("C", st, fits_c, x, cov, ok, cfg, {"multilat_residual_s": rms, "pair_misfit_s": misfit})

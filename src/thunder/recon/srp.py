"""Phase 5 Method B: steered response power with (beta-)PHAT, several sources per window.

Steering variable. Across an array of mics at (nearly) one height, the delay of mic j behind
the reference is tau_j = s . (m_j - m_ref), with s the wave slowness at the array. In a
stratified atmosphere the horizontal slowness s_h is the ray's conserved slowness, and the
vertical component follows from the dispersion relation at mic height,
    s_z = -sqrt(|s|^2 - |s_h|^2),  |s| = (1 - v . s_h) / c   (downgoing wave).
So the steering grid is over s_h (in units of 1/c: u = c s_h, |u| <= 1 + tolerance) and does
not depend on the atmosphere above the array; the assumed atmosphere enters only when each
detection is placed (`place_points`: straight rays, or tracing s_h back with `locate`). This is
the plane-wave (far-field) form; near-field geometry is Method C's job.

Per window:
  1. GCC-PHAT curves R_j(lag) between the reference mic's short window and each other mic's
     long window (pass-1 geometry of Method A).
  2. SRP(u) = mean_j R_j(tau_j(u)) on a coarse grid; local maxima above `srp_min_power` and above
     `srp_relative_peak` x the strongest, separated by >= `srp_min_separation`, at most
     `max_sources`, each refined on a 10x finer local grid.
  3. For each peak: matched-window TDOAs for every pair searched only +-`refine_search_s`
     around the peak's predicted lags (so a weaker source is not captured by a stronger one),
     the same constrained slowness fit and gates as Method A plus a coherence gate (every
     pair's peak >= `srp_min_pair_peak`: sidelobe maxima align only some pairs, so at least
     one pair is near zero, while a real source is seen by all), and the arrival time from a
     delay-and-sum beam steered to that source (so two sources in one window get their own
     ranges).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from thunder.config import ReconstructionConfig
from thunder.recon.tdoa import (
    Setup,
    WindowFit,
    _phat_correlation,
    _segment,
    assemble,
    energy_time,
    matched_pairs,
    place_points,
    prepare,
    slowness_ok,
    solve_direction,
)
from thunder.types import Atmosphere, FloatArray, Reconstruction, Recording


def gcc_curve(st: Setup, j: int, start: int, beta: float) -> FloatArray:
    """GCC-PHAT of the reference's short window against mic j's long window, lags -pad..pad."""
    corr = st.corr
    n, pad = corr.n_short, corr.n_long_pad
    s = _segment(st.x[st.ref], start, n) * corr.short_taper
    lg = _segment(st.x[j], start - pad, n + 2 * pad)
    cross = np.conj(np.fft.rfft(s, corr.nfft)) * np.fft.rfft(lg, corr.nfft)
    return _phat_correlation(cross, corr.band_mask, beta, corr.nfft)[: 2 * pad + 1]


class Steering:
    """Predicted reference-relative delays (samples) for a set of normalized slownesses u."""

    def __init__(self, st: Setup, atmosphere: Atmosphere):
        z = np.array([float(st.mics[st.ref, 2])])
        self.c = float(atmosphere.sound_speed(z)[0])
        self.v = atmosphere.wind(z)[0, :2]
        self.d = st.mics - st.mics[st.ref]  # (M, 3)
        self.fs = st.fs

    def delays(self, u: FloatArray) -> FloatArray:
        """(K, M) delays in samples for u (K, 2)."""
        sh = u / self.c
        sabs = (1.0 - sh @ self.v) / self.c
        sz = -np.sqrt(np.clip(sabs**2 - np.sum(sh * sh, axis=1), 0.0, None))
        return (sh @ self.d[:, :2].T + sz[:, None] * self.d[None, :, 2]) * self.fs


def _power(curves: list[FloatArray], others: list[int], pad: int, lags: FloatArray) -> FloatArray:
    """Mean over pairs of the GCC curves linearly interpolated at the predicted lags."""
    total = np.zeros(len(lags))
    for q, j in enumerate(others):
        x = np.clip(lags[:, j] + pad, 0.0, 2 * pad - 1e-9)
        i = np.floor(x).astype(int)
        f = x - i
        total += (1 - f) * curves[q][i] + f * curves[q][i + 1]
    return total / len(others)


def _coarse_grid(step: float, radius: float) -> tuple[FloatArray, FloatArray, np.ndarray]:
    axis = np.arange(-radius, radius + 1e-12, step)
    ux, uy = np.meshgrid(axis, axis, indexing="ij")
    inside = ux**2 + uy**2 <= radius**2
    return axis, np.stack([ux, uy], axis=-1), inside


def srp_peaks(
    st: Setup, steer: Steering, start: int, cfg: ReconstructionConfig
) -> list[tuple[FloatArray, float]]:
    """Up to max_sources (u, power) peaks of the steered response power in one window."""
    others = [j for j in range(len(st.mics)) if j != st.ref]
    curves = [gcc_curve(st, j, start, cfg.phat_beta) for j in others]
    pad = st.corr.n_long_pad
    radius = 1.0 + cfg.slowness_tolerance
    axis, grid, inside = _coarse_grid(cfg.srp_slowness_step, radius)
    flat = grid.reshape(-1, 2)
    power = np.full(len(flat), -np.inf)
    sel = inside.ravel()
    power[sel] = _power(curves, others, pad, steer.delays(flat[sel]))
    pmap = power.reshape(inside.shape)
    is_max = (pmap == maximum_filter(pmap, size=3, mode="constant", cval=-np.inf)) & inside
    cand = np.flatnonzero(is_max.ravel())
    cand = cand[np.argsort(power[cand])[::-1]]
    peaks: list[tuple[FloatArray, float]] = []
    best = power[cand[0]] if len(cand) else -np.inf
    for idx in cand:
        p = power[idx]
        if p < cfg.srp_min_power or p < cfg.srp_relative_peak * best or len(peaks) >= cfg.max_sources:
            break
        u0 = flat[idx]
        if any(np.linalg.norm(u0 - q) < cfg.srp_min_separation for q, _ in peaks):
            continue
        # fine search on a 10x finer grid within +-1 coarse step
        f_axis = np.linspace(-cfg.srp_slowness_step, cfg.srp_slowness_step, 21)
        fx, fy = np.meshgrid(f_axis, f_axis, indexing="ij")
        fine = u0 + np.stack([fx.ravel(), fy.ravel()], axis=1)
        fine = fine[np.sum(fine**2, axis=1) <= radius**2]
        fp = _power(curves, others, pad, steer.delays(fine))
        k = int(np.argmax(fp))
        peaks.append((fine[k], float(fp[k])))
    return peaks


def method_b_fits(st: Setup, steer: Steering, cfg: ReconstructionConfig) -> tuple[list[WindowFit], list[int]]:
    """Window fits for every detected source; also returns the window index of each fit."""
    out, where = [], []
    n = st.frames.length
    refine = max(2, int(round(cfg.refine_search_s * st.fs)))
    for k in st.active:
        start = int(st.frames.starts[k])
        kept_dirs: list[FloatArray] = []
        for u, power in srp_peaks(st, steer, start, cfg):
            shift = steer.delays(u[None, :])[0]
            tau, peak = matched_pairs(st, start, shift, refine)
            fit = solve_direction(st.diffs, tau, np.clip(peak, 1e-3, None), st.c, st.sigma_floor)
            if any(float(fit.u @ d) > np.cos(np.radians(cfg.srp_duplicate_deg)) for d in kept_dirs):
                continue  # two peaks refined onto the same source
            offs = np.round(shift).astype(int)
            beam = sum(_segment(st.x[i], start + int(offs[i]), n) for i in range(len(st.mics)))
            timing = energy_time(np.asarray(beam), start, st.fs)
            if timing is None:
                continue
            passed = (
                fit.residual_rms <= min(cfg.max_residual_s, cfg.srp_max_residual_s)
                and slowness_ok(fit, cfg)
                and float(np.mean(peak)) >= cfg.min_peak
                and float(np.min(peak)) >= cfg.srp_min_pair_peak
            )
            out.append(WindowFit(timing[0], timing[1], fit, tau, peak, power, passed))
            where.append(int(k))
            kept_dirs.append(fit.u)
    return out, where


def reconstruct_srp(
    rec: Recording, mic_positions: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> Reconstruction:
    """Method B on a recording (signals, nominal positions, reported t0, assumed atmosphere only)."""
    st = prepare(rec, mic_positions, atmosphere, cfg)
    fits, where = method_b_fits(st, Steering(st, atmosphere), cfg)
    if not fits:
        return Reconstruction.empty("B", cfg.config_hash())
    pts, covs, ok = place_points(st, fits, rec.reported_t0, atmosphere)
    return assemble("B", st, fits, pts, covs, ok, cfg, {"window_index": np.array(where)})

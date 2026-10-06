"""Phase 5 Method A: plane-wave TDOA reconstruction (the baseline).

Per analysis window:
  1. GCC-PHAT time differences for every mic pair, PHAT-weighted only inside the analysis
     band (out-of-band bins carry no thunder and would be amplified to unit weight). The
     weighting is beta-PHAT, |G|^-beta with beta = 0.6 by default: partial whitening was more
     accurate than full PHAT at every SNR tested (docs/log.md). Lags are restricted to
     |tau_ij| <= d_ij / c + margin and refined to sub-sample precision with a parabola.
     A 50 m array can delay the same sound by up to ~146 ms, longer than a 100 ms window.
     Pass 1 (coarse) pairs a short Hann window on the reference mic (nearest the centroid)
     with a long window on each other mic that covers every physically possible lag.
     Mismatched windows leak extra content that PHAT amplifies (lag std ~0.4 samples), so
     pass 2 (fine) places equal Hann windows on the same sound at both mics of every pair,
     using the pass-1 delays, and measures only the residual lag (std ~0.002 samples on
     band-limited noise; see docs/log.md).
  2. Slowness p solves  min ||W^1/2 (A p + tau)||^2  subject to |p| = 1/c, where row (i, j)
     of A is m_j - m_i (far field: tau_ij = -(m_j - m_i) . p, p = u / c, u = unit vector
     toward the source). The constraint is solved exactly with the secular equation in the
     eigenbasis of H = A^T W A. For a planar array H is singular in z (the "hard case"):
     the horizontal slowness is the least-squares one and p_z = +sqrt(1/c^2 - |p_xy|^2)
     (sources are above the array). One code path covers planar, 3D and noisy cases.
  3. Range R = c (t_c - t0), where t_c is the energy centroid of the reference window (not
     the window center: removes up to +-c * window / 2 of range error). The point is
     centroid + R_c u with R_c = R + u . (m_ref - centroid) (far-field range transfer).
  4. Quality gates: mean GCC-PHAT peak, RMS least-squares residual, and |c p_free| close to 1
     (for planar arrays: |c p_xy| <= 1 + tolerance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import optimize

from thunder.config import ReconstructionConfig
from thunder.recon.preprocess import active_frames, bandpass, frame, whiten, window_energy
from thunder.types import FloatArray, Reconstruction, Recording

# --- GCC-PHAT -------------------------------------------------------------------


@dataclass(frozen=True)
class Correlator:
    """Precomputed FFT size and band mask for a given short/long window geometry."""

    n_short: int
    n_long_pad: int  # samples added on each side of the long window
    nfft: int
    band_mask: np.ndarray
    norm: float  # makes a perfect single-source match peak at exactly 1
    short_taper: FloatArray

    @classmethod
    def build(cls, n_short: int, pad: int, fs: float, band: tuple[float, float]) -> Correlator:
        n_long = n_short + 2 * pad
        nfft = int(2 ** math.ceil(math.log2(n_short + n_long)))
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        mask = (f >= band[0]) & (f <= band[1]) & (f > 0) & (f < fs / 2)
        return cls(n_short, pad, nfft, mask, nfft / (2.0 * mask.sum()), np.hanning(n_short + 2)[1:-1])


def _segment(x: FloatArray, start: int, length: int) -> FloatArray:
    """x[start:start + length] with zeros outside the recording."""
    out = np.zeros(length)
    lo, hi = max(start, 0), min(start + length, len(x))
    if hi > lo:
        out[lo - start : hi - start] = x[lo:hi]
    return out


def _phat_correlation(cross: np.ndarray, mask: np.ndarray, beta: float, nfft: int, norm: float) -> FloatArray:
    """Inverse FFT of the band-limited, (beta-)PHAT-weighted cross-spectrum."""
    mag = np.abs(cross)
    w = np.where(mask & (mag > 0), cross / np.where(mag > 0, mag, 1.0) ** beta, 0.0)
    return np.fft.irfft(w, nfft) * norm


def _parabolic(r: FloatArray, k: int, lo: int, hi: int) -> float:
    """Sub-sample offset of the peak at k from a parabola through r[k-1..k+1] (0 at bounds)."""
    if not lo < k < hi:
        return 0.0
    y0, y1, y2 = r[k - 1], r[k], r[k + 1]
    den = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / den if den < 0 else 0.0


def gcc_phat(
    x_short: FloatArray,
    x_long: FloatArray,
    short_start: int,
    max_lag: int,
    corr: Correlator,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Delay (samples, fractional) of x_long relative to x_short, and the normalized PHAT peak.

    The short window is x_short[short_start : short_start + n]; the long window extends it by
    corr.n_long_pad on both sides on x_long. Positive delay: the sound reaches x_long later.
    """
    n, pad = corr.n_short, corr.n_long_pad
    s = _segment(x_short, short_start, n) * corr.short_taper
    lg = _segment(x_long, short_start - pad, n + 2 * pad)
    cross = np.conj(np.fft.rfft(s, corr.nfft)) * np.fft.rfft(lg, corr.nfft)
    r = _phat_correlation(cross, corr.band_mask, beta, corr.nfft, corr.norm)
    lo, hi = pad - max_lag, pad + max_lag
    k = lo + int(np.argmax(r[lo : hi + 1]))
    return (k - pad) + _parabolic(r, k, lo, hi), float(r[k])


@dataclass(frozen=True)
class MatchedCorrelator:
    """Equal-length windows on both mics (pass 2)."""

    n: int
    nfft: int
    band_mask: np.ndarray
    norm: float
    taper: FloatArray
    beta: float

    @classmethod
    def build(cls, n: int, fs: float, band: tuple[float, float], beta: float) -> MatchedCorrelator:
        nfft = int(2 ** math.ceil(math.log2(2 * n)))
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        mask = (f >= band[0]) & (f <= band[1]) & (f > 0) & (f < fs / 2)
        return cls(n, nfft, mask, nfft / (2.0 * mask.sum()), np.hanning(n + 2)[1:-1], beta)


def matched_spectrum(x: FloatArray, start: int, corr: MatchedCorrelator) -> np.ndarray:
    """Spectrum of one tapered pass-2 window (computed once per mic per window, shared by pairs)."""
    return np.fft.rfft(_segment(x, start, corr.n) * corr.taper, corr.nfft)


def gcc_phat_spectra(
    a: np.ndarray, b: np.ndarray, lag_lo: int, lag_hi: int, corr: MatchedCorrelator
) -> tuple[float, float]:
    """Residual lag (samples) of window b relative to window a, searched in [lag_lo, lag_hi]."""
    r = _phat_correlation(np.conj(a) * b, corr.band_mask, corr.beta, corr.nfft, corr.norm)
    lags = np.arange(lag_lo - 1, lag_hi + 2)  # one extra on each side for the parabola
    vals = r[lags % corr.nfft]  # circular index: negative lags wrap to the end
    k = 1 + int(np.argmax(vals[1:-1]))
    # Refine only strictly inside the search range: a peak on the boundary lies beyond it.
    frac = _parabolic(vals, k, 1, len(vals) - 2)
    return float(lags[k]) + frac, float(vals[k])


def gcc_phat_matched(
    xi: FloatArray,
    xj: FloatArray,
    start_i: int,
    start_j: int,
    lag_lo: int,
    lag_hi: int,
    corr: MatchedCorrelator,
) -> tuple[float, float]:
    """Residual lag (samples) of xj's window relative to xi's, searched in [lag_lo, lag_hi]."""
    return gcc_phat_spectra(
        matched_spectrum(xi, start_i, corr), matched_spectrum(xj, start_j, corr), lag_lo, lag_hi, corr
    )


# --- Constrained slowness -----------------------------------------------------------


@dataclass(frozen=True)
class DirectionFit:
    u: FloatArray  # (3,) unit vector toward the source
    free_ratio: float  # c * |unconstrained slowness| (horizontal part for planar arrays)
    planar: bool  # True if the vertical slowness was unobservable (hard case)
    residual_rms: float  # s
    cov_u: FloatArray  # (3, 3) first-order covariance of u


def solve_direction(
    diffs: FloatArray, tau: FloatArray, weights: FloatArray, c: float, sigma_floor: float
) -> DirectionFit:
    """Minimize sum w (diffs . p + tau)^2 subject to |p| = 1/c.  diffs: (P, 3) rows m_j - m_i."""
    w = np.asarray(weights, dtype=float)
    H = diffs.T @ (diffs * w[:, None])
    g = diffs.T @ (w * tau)
    lam, vec = np.linalg.eigh(H)
    b = vec.T @ g
    r = 1.0 / c
    scale = max(float(lam[-1]), 1e-300)
    singular = lam < 1e-10 * scale
    planar = bool(singular[0] and abs(b[0]) <= 1e-8 * (np.linalg.norm(b) + 1e-300))

    def p_of(mu: float, skip: np.ndarray | None = None) -> FloatArray:
        coef = -b / (lam + mu)
        if skip is not None:
            coef = np.where(skip, 0.0, coef)
        return vec @ coef

    observable = ~singular
    p_free = vec @ np.where(observable, -b / np.where(observable, lam, 1.0), 0.0)
    free_ratio = float(np.linalg.norm(p_free) * c)

    if planar and free_ratio <= 1.0:
        # Hard case: least-squares slowness in the observable (horizontal) subspace, plus the
        # unobservable component that restores |p| = 1/c, pointing up (sources are above).
        t = math.sqrt(max(r**2 - float(p_free @ p_free), 0.0))
        null = vec[:, 0] * (1.0 if vec[2, 0] >= 0 else -1.0)
        p = p_free + t * null
    else:
        # Secular equation |p(mu)| = 1/c on mu > -lambda_min, where |p(mu)| decreases monotonically.
        skip = singular if planar else None
        lo = -lam[0] + 1e-12 * scale

        def f(mu: float) -> float:
            return float(np.linalg.norm(p_of(mu, skip))) - r

        hi = max(float(np.linalg.norm(g)) / r, 1e-12 * scale) + abs(lam[0])
        while f(hi) > 0:
            hi *= 2
        if f(lo) < 0:  # norm cannot reach 1/c from below (only for degenerate data)
            p = p_free / max(np.linalg.norm(p_free), 1e-300) * r
        else:
            mu = optimize.brentq(f, lo, hi, xtol=1e-14 * scale, rtol=1e-12, maxiter=200)
            p = p_of(mu, skip)
    u = p / np.linalg.norm(p)

    resid = diffs @ p + tau
    dof = max(len(tau) - 2, 1)  # the constrained fit has 2 free parameters (a direction)
    sigma2 = max(float(np.sum(w * resid**2) / max(np.sum(w), 1e-300) * len(tau) / dof), sigma_floor**2)
    # First-order covariance of the constrained estimate: perturbations live in the plane
    # tangent to the sphere |p| = 1/c, so restrict the normal matrix to that plane (T: 3x2
    # orthonormal tangent basis):  cov(u) = c^2 sigma^2 T (T^T H T)^-1 T^T.  This also covers
    # planar arrays (where vertical information comes only from the constraint) and correctly
    # grows without bound as the arrival becomes horizontal.
    wn = w / max(float(np.mean(w)), 1e-300)
    hn = diffs.T @ (diffs * wn[:, None])
    tb = np.linalg.svd(np.eye(3) - np.outer(u, u))[0][:, :2]
    cov_u = c**2 * sigma2 * tb @ np.linalg.pinv(tb.T @ hn @ tb) @ tb.T
    return DirectionFit(u, free_ratio, planar, float(np.sqrt(np.mean(resid**2))), cov_u)


# --- Method A -----------------------------------------------------------------------


def reconstruct_plane_wave(
    rec: Recording, mic_positions: FloatArray, sound_speed: float, cfg: ReconstructionConfig
) -> Reconstruction:
    """Method A on a recording. Uses only the signals, nominal positions, reported t0 and c."""
    fs = rec.sample_rate
    mics = np.asarray(mic_positions, dtype=float)
    m = len(mics)
    c = float(sound_speed)
    x = bandpass(rec.signals, fs, cfg.band_hz, cfg.filter_order)
    if cfg.whiten:
        x = whiten(x, fs, cfg.band_hz)

    centroid = mics.mean(axis=0)
    ref = int(np.argmin(np.linalg.norm(mics - centroid, axis=1)))
    dist = np.linalg.norm(mics[:, None, :] - mics[None, :, :], axis=2)
    max_lag = np.ceil((dist / c + cfg.lag_margin_s) * fs).astype(int)
    frames = frame(x.shape[1], fs, cfg.window_s, cfg.overlap)
    n = frames.length
    corr = Correlator.build(n, int(max_lag.max()), fs, cfg.band_hz)
    fine = MatchedCorrelator.build(n, fs, cfg.band_hz, cfg.phat_beta)
    residual = max(2, int(round(cfg.residual_search_s * fs)))
    active = active_frames(window_energy(x, frames), cfg.detection_dynamic_range_db, cfg.detection_snr_db)
    pairs = list(combinations(range(m), 2))
    diffs = np.array([mics[j] - mics[i] for i, j in pairs])
    sigma_floor = 1.0 / (fs * math.sqrt(12.0))
    t = np.arange(n) / fs

    rows = []
    for k in active:
        start = int(frames.starts[k])
        # Pass 1: delays from the reference mic.
        shift = np.zeros(m)
        for j in range(m):
            if j != ref:
                shift[j], _ = gcc_phat(x[ref], x[j], start, int(max_lag[ref, j]), corr, cfg.phat_beta)
        # Pass 2: equal windows on the same sound at both mics of every pair; residual lag only.
        offs = np.round(shift).astype(int)
        spectra = [matched_spectrum(x[i], start + int(offs[i]), fine) for i in range(m)]
        tau = np.empty(len(pairs))
        peak = np.empty(len(pairs))
        for q, (i, j) in enumerate(pairs):
            base = int(offs[j] - offs[i])
            lo = max(-residual, -int(max_lag[i, j]) - base)
            hi = min(residual, int(max_lag[i, j]) - base)
            if lo > hi:  # pass-1 delays inconsistent with geometry: unusable pair
                tau[q], peak[q] = base / fs, 0.0
                continue
            lag, peak[q] = gcc_phat_spectra(spectra[i], spectra[j], lo, hi, fine)
            tau[q] = (base + lag) / fs
        w = np.clip(peak, 1e-3, None)
        fit = solve_direction(diffs, tau, w, c, sigma_floor)

        seg = x[ref, start : start + n]
        e = seg**2
        tot = float(e.sum())
        if tot <= 0:
            continue
        t_c = start / fs + float((t * e).sum() / tot)
        spread = float(np.sqrt(max((t**2 * e).sum() / tot - (t_c - start / fs) ** 2, 0.0)))
        r_ref = c * (t_c - rec.reported_t0)
        r_cen = r_ref + float(fit.u @ (mics[ref] - centroid))
        point = centroid + r_cen * fit.u
        cov = r_cen**2 * fit.cov_u + (c * spread) ** 2 * np.outer(fit.u, fit.u)
        mean_peak = float(np.mean(peak))
        ok_ratio = (
            fit.free_ratio <= 1 + cfg.slowness_tolerance
            if fit.planar
            else (abs(fit.free_ratio - 1) <= cfg.slowness_tolerance)
        )
        passed = (
            mean_peak >= cfg.min_peak and fit.residual_rms <= cfg.max_residual_s and ok_ratio and r_cen > 0
        )
        rows.append((point, cov, t_c, mean_peak, fit.residual_rms, fit.free_ratio, passed))

    if not rows:
        return Reconstruction.empty("A", cfg.config_hash())
    pts = np.array([r[0] for r in rows])
    covs = np.array([r[1] for r in rows])
    times = np.array([r[2] for r in rows])
    peaks = np.array([r[3] for r in rows])
    resid = np.array([r[4] for r in rows])
    ratio = np.array([r[5] for r in rows])
    keep = np.array([r[6] for r in rows])
    return Reconstruction(
        points=pts[keep],
        covariances=covs[keep],
        window_times=times[keep],
        quality=peaks[keep],
        method="A",
        config_hash=cfg.config_hash(),
        extra={
            "window_points": pts,
            "window_times": times,
            "window_peak": peaks,
            "window_residual_s": resid,
            "window_slowness_ratio": ratio,
            "window_passed": keep,
            "reference_mic": ref,
            "n_windows_active": int(len(active)),
        },
    )

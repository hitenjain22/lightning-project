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
from thunder.recon.preprocess import Frames, active_frames, bandpass, frame, whiten, window_energy
from thunder.types import Atmosphere, FloatArray, Reconstruction, Recording

# --- GCC-PHAT -------------------------------------------------------------------


@dataclass(frozen=True)
class Correlator:
    """Precomputed FFT size and band mask for a given short/long window geometry."""

    n_short: int
    n_long_pad: int  # samples added on each side of the long window
    nfft: int
    band_mask: np.ndarray
    short_taper: FloatArray

    @classmethod
    def build(cls, n_short: int, pad: int, fs: float, band: tuple[float, float]) -> Correlator:
        n_long = n_short + 2 * pad
        nfft = int(2 ** math.ceil(math.log2(n_short + n_long)))
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        mask = (f >= band[0]) & (f <= band[1]) & (f > 0) & (f < fs / 2)
        return cls(n_short, pad, nfft, mask, np.hanning(n_short + 2)[1:-1])


def _segment(x: FloatArray, start: int, length: int) -> FloatArray:
    """x[start:start + length] with zeros outside the recording."""
    out = np.zeros(length)
    lo, hi = max(start, 0), min(start + length, len(x))
    if hi > lo:
        out[lo - start : hi - start] = x[lo:hi]
    return out


def _phat_correlation(cross: np.ndarray, mask: np.ndarray, beta: float, nfft: int) -> FloatArray:
    """Inverse FFT of the band-limited, (beta-)PHAT-weighted cross-spectrum, normalized so a
    perfect match (all weighted bins in phase) peaks at exactly 1 for any beta:
        r(tau) = sum_k |w_k| cos(phase_k - w_k tau) / sum_k |w_k|,   w_k = G_k / |G_k|^beta.
    (A fixed constant is only right for beta = 1, where every |w_k| = 1.)"""
    mag = np.abs(cross)
    w = np.where(mask & (mag > 0), cross / np.where(mag > 0, mag, 1.0) ** beta, 0.0)
    total = float(np.sum(np.abs(w)))
    if total <= 0:
        return np.zeros(nfft)
    return np.fft.irfft(w, nfft) * (nfft / (2.0 * total))


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
    r = _phat_correlation(cross, corr.band_mask, beta, corr.nfft)
    lo, hi = pad - max_lag, pad + max_lag
    k = lo + int(np.argmax(r[lo : hi + 1]))
    return (k - pad) + _parabolic(r, k, lo, hi), float(r[k])


@dataclass(frozen=True)
class MatchedCorrelator:
    """Equal-length windows on both mics (pass 2)."""

    n: int
    nfft: int
    band_mask: np.ndarray
    taper: FloatArray
    beta: float

    @classmethod
    def build(cls, n: int, fs: float, band: tuple[float, float], beta: float) -> MatchedCorrelator:
        nfft = int(2 ** math.ceil(math.log2(2 * n)))
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        mask = (f >= band[0]) & (f <= band[1]) & (f > 0) & (f < fs / 2)
        return cls(n, nfft, mask, np.hanning(n + 2)[1:-1], beta)


def matched_spectrum(x: FloatArray, start: int, corr: MatchedCorrelator) -> np.ndarray:
    """Spectrum of one tapered pass-2 window (computed once per mic per window, shared by pairs)."""
    return np.fft.rfft(_segment(x, start, corr.n) * corr.taper, corr.nfft)


def gcc_phat_spectra(
    a: np.ndarray, b: np.ndarray, lag_lo: int, lag_hi: int, corr: MatchedCorrelator
) -> tuple[float, float]:
    """Residual lag (samples) of window b relative to window a, searched in [lag_lo, lag_hi]."""
    r = _phat_correlation(np.conj(a) * b, corr.band_mask, corr.beta, corr.nfft)
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
    p_free: FloatArray  # (3,) unconstrained least-squares slowness (z = 0 if unobservable)


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
    return DirectionFit(u, free_ratio, planar, float(np.sqrt(np.mean(resid**2))), cov_u, p_free)


# --- Method A -----------------------------------------------------------------------


@dataclass
class Setup:
    """Everything the TDOA methods share for one recording."""

    x: FloatArray  # (M, n) band-passed (optionally whitened) signals
    fs: float
    mics: FloatArray  # (M, 3) nominal positions
    c: float  # assumed sound speed at mic height
    centroid: FloatArray
    ref: int  # reference mic (nearest the centroid)
    max_lag: np.ndarray  # (M, M) samples
    frames: Frames
    corr: Correlator  # pass 1 (short vs long windows)
    fine: MatchedCorrelator  # pass 2 (matched windows)
    residual: int  # pass-2 lag search half-width, samples
    active: np.ndarray  # active window indices
    pairs: list[tuple[int, int]]
    diffs: FloatArray  # (P, 3) m_j - m_i per pair
    sigma_floor: float


def prepare(
    rec: Recording, mic_positions: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> Setup:
    fs = rec.sample_rate
    mics = np.asarray(mic_positions, dtype=float)
    m = len(mics)
    c = float(atmosphere.sound_speed(np.array([float(np.mean(mics[:, 2]))]))[0])
    x = bandpass(rec.signals, fs, cfg.band_hz, cfg.filter_order)
    if cfg.whiten:
        x = whiten(x, fs, cfg.band_hz)
    centroid = mics.mean(axis=0)
    ref = int(np.argmin(np.linalg.norm(mics - centroid, axis=1)))
    dist = np.linalg.norm(mics[:, None, :] - mics[None, :, :], axis=2)
    max_lag = np.ceil((dist / c + cfg.lag_margin_s) * fs).astype(int)
    frames = frame(x.shape[1], fs, cfg.window_s, cfg.overlap)
    n = frames.length
    pairs = list(combinations(range(m), 2))
    return Setup(
        x=x,
        fs=fs,
        mics=mics,
        c=c,
        centroid=centroid,
        ref=ref,
        max_lag=max_lag,
        frames=frames,
        corr=Correlator.build(n, int(max_lag.max()), fs, cfg.band_hz),
        fine=MatchedCorrelator.build(n, fs, cfg.band_hz, cfg.phat_beta),
        residual=max(2, int(round(cfg.residual_search_s * fs))),
        active=active_frames(window_energy(x, frames), cfg.detection_dynamic_range_db, cfg.detection_snr_db),
        pairs=pairs,
        diffs=np.array([mics[j] - mics[i] for i, j in pairs]).reshape(-1, 3),
        sigma_floor=1.0 / (fs * math.sqrt(12.0)),
    )


def coarse_shifts(st: Setup, start: int, beta: float) -> FloatArray:
    """Pass 1: delay (samples) of each mic relative to the reference mic for this window."""
    shift = np.zeros(len(st.mics))
    for j in range(len(st.mics)):
        if j != st.ref:
            shift[j], _ = gcc_phat(st.x[st.ref], st.x[j], start, int(st.max_lag[st.ref, j]), st.corr, beta)
    return shift


def matched_pairs(st: Setup, start: int, shift: FloatArray, residual: int) -> tuple[FloatArray, FloatArray]:
    """Pass 2: TDOA tau_ij (s) and normalized peak for every pair, from equal windows placed on the
    same sound at both mics (offsets = rounded `shift`), searching +-`residual` samples."""
    offs = np.round(shift).astype(int)
    spectra = [matched_spectrum(st.x[i], start + int(offs[i]), st.fine) for i in range(len(st.mics))]
    tau = np.empty(len(st.pairs))
    peak = np.empty(len(st.pairs))
    for q, (i, j) in enumerate(st.pairs):
        base = int(offs[j] - offs[i])
        lo = max(-residual, -int(st.max_lag[i, j]) - base)
        hi = min(residual, int(st.max_lag[i, j]) - base)
        if lo > hi:  # delays inconsistent with geometry: unusable pair
            tau[q], peak[q] = base / st.fs, 0.0
            continue
        lag, peak[q] = gcc_phat_spectra(spectra[i], spectra[j], lo, hi, st.fine)
        tau[q] = (base + lag) / st.fs
    return tau, peak


def energy_time(signal_window: FloatArray, start: int, fs: float) -> tuple[float, float] | None:
    """Energy centroid time (s, recording clock) and RMS spread of a window, or None if silent."""
    e = signal_window**2
    tot = float(e.sum())
    if tot <= 0:
        return None
    t = np.arange(len(signal_window)) / fs
    t_rel = float((t * e).sum() / tot)
    spread = float(np.sqrt(max((t**2 * e).sum() / tot - t_rel**2, 0.0)))
    return start / fs + t_rel, spread


def slowness_ok(fit: DirectionFit, cfg: ReconstructionConfig) -> bool:
    if fit.planar:
        return fit.free_ratio <= 1 + cfg.slowness_tolerance
    return abs(fit.free_ratio - 1) <= cfg.slowness_tolerance


@dataclass
class WindowFit:
    """One direction measured in one window (Methods A and B), before placing the point."""

    t_c: float  # arrival time at the reference mic (recording clock)
    spread: float  # RMS time spread of that arrival within the window
    fit: DirectionFit
    tau: FloatArray  # (P,) pair TDOAs
    peak: FloatArray  # (P,) pair GCC peaks
    quality: float  # mean pair peak (A) or SRP power (B)
    passed: bool  # quality, residual and slowness gates


def place_points(
    st: Setup, fits: list[WindowFit], reported_t0: float, atmosphere: Atmosphere
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Points (K, 3), covariances (K, 3, 3) and validity for measured directions + travel times.

    Uniform assumed atmosphere: straight rays from the centroid (far-field range transfer from
    the reference mic). Otherwise: trace the measured horizontal slowness back through the
    assumed atmosphere for the travel time (`Atmosphere.locate`).
    """
    k = len(fits)
    pts = np.full((k, 3), np.nan)
    covs = np.zeros((k, 3, 3))
    ok = np.zeros(k, dtype=bool)
    m_ref = st.mics[st.ref]
    if atmosphere.is_uniform:
        for i, w in enumerate(fits):
            r_cen = st.c * (w.t_c - reported_t0) + float(w.fit.u @ (m_ref - st.centroid))
            pts[i] = st.centroid + r_cen * w.fit.u
            covs[i] = r_cen**2 * w.fit.cov_u + (st.c * w.spread) ** 2 * np.outer(w.fit.u, w.fit.u)
            ok[i] = r_cen > 0
        return pts, covs, ok
    # p points toward the source (tau_ij = -(m_j - m_i) . p); the wave slowness along the
    # direction of travel, which `locate` integrates, is -p.
    sh = -np.array([w.fit.p_free[:2] for w in fits]).reshape(-1, 2)
    times = np.array([w.t_c for w in fits]) - reported_t0
    located, valid = atmosphere.locate(m_ref, sh, times)
    for i, w in enumerate(fits):
        if valid[i]:
            u = located[i] - m_ref
            rng_i = float(np.linalg.norm(u))
            u /= rng_i
            pts[i] = located[i]
            covs[i] = rng_i**2 * w.fit.cov_u + (st.c * w.spread) ** 2 * np.outer(u, u)
            ok[i] = True
    return pts, covs, ok


def method_a_fits(st: Setup, cfg: ReconstructionConfig) -> list[WindowFit]:
    """Method A per window: one direction from the full two-pass TDOA set."""
    out = []
    n = st.frames.length
    for k in st.active:
        start = int(st.frames.starts[k])
        shift = coarse_shifts(st, start, cfg.phat_beta)
        tau, peak = matched_pairs(st, start, shift, st.residual)
        fit = solve_direction(st.diffs, tau, np.clip(peak, 1e-3, None), st.c, st.sigma_floor)
        timing = energy_time(st.x[st.ref, start : start + n], start, st.fs)
        if timing is None:
            continue
        mean_peak = float(np.mean(peak))
        passed = (
            mean_peak >= cfg.min_peak and fit.residual_rms <= cfg.max_residual_s and slowness_ok(fit, cfg)
        )
        out.append(WindowFit(timing[0], timing[1], fit, tau, peak, mean_peak, passed))
    return out


def assemble(
    method: str, st: Setup, fits: list[WindowFit], pts, covs, ok, cfg: ReconstructionConfig, extra=None
) -> Reconstruction:
    """Reconstruction from window fits and placed points (gates: fit.passed & placement ok)."""
    if not fits:
        return Reconstruction.empty(method, cfg.config_hash())
    times = np.array([w.t_c for w in fits])
    quality = np.array([w.quality for w in fits])
    keep = np.array([w.passed for w in fits]) & ok
    return Reconstruction(
        points=pts[keep],
        covariances=covs[keep],
        window_times=times[keep],
        quality=quality[keep],
        method=method,
        config_hash=cfg.config_hash(),
        extra={
            "window_points": pts,
            "window_times": times,
            "window_peak": quality,
            "window_residual_s": np.array([w.fit.residual_rms for w in fits]),
            "window_slowness_ratio": np.array([w.fit.free_ratio for w in fits]),
            "window_passed": keep,
            "reference_mic": st.ref,
            "n_windows_active": int(len(st.active)),
            **(extra or {}),
        },
    )


def reconstruct_plane_wave(
    rec: Recording, mic_positions: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> Reconstruction:
    """Method A on a recording. Uses only the signals, nominal positions, reported t0 and the
    *assumed* atmosphere (straight rays if uniform, else traced back via `locate`)."""
    st = prepare(rec, mic_positions, atmosphere, cfg)
    fits = method_a_fits(st, cfg)
    if not fits:
        return Reconstruction.empty("A", cfg.config_hash())
    pts, covs, ok = place_points(st, fits, rec.reported_t0, atmosphere)
    return assemble("A", st, fits, pts, covs, ok, cfg)

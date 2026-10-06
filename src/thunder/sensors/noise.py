"""Phase 4 noise fields at the microphones.

All fields are synthesized in the frequency domain. For one-sided PSD S(f) and coherence
matrix G(f) between mics, each frequency bin gets X(f) = A(f) W(f) sqrt(S(f) fs / 2), where
W are independent unit white-noise spectra and A A^T = G (A = V sqrt(lambda) from an
eigendecomposition, robust when G is near singular). The result has exactly PSD S at every
mic and cross-spectral coherence G between mics.

Coherence models:
  diffuse background  G_ij = sinc(2 f d_ij / c)              (isotropic 3D sound field)
  wind (turbulence)   G_ij = exp(-a f d_ij / U)              (convected eddies; incoherent
                                                               beyond a few meters)
  rain, self-noise    G = I                                   (independent per mic)
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist

from thunder.constants import (
    RAIN_HIGHPASS_HZ,
    RAIN_IMPACT_RATE_HZ,
    WIND_COHERENCE_DECAY,
    WIND_EDDY_SCALE_M,
    WIND_NOISE_COEFF,
)
from thunder.types import FloatArray

P_REF = 20e-6  # Pa, reference pressure for dB SPL (ANSI/ISO definition)
COLOR_EXPONENT = {"white": 0.0, "pink": 1.0, "brown": 2.0}
COLOR_CORNER_HZ = 0.5  # PSD flattens below this so pink/brown noise has finite power
_CHUNK = 8192  # frequency bins per eigendecomposition batch (bounds memory)

Coherence = Callable[[FloatArray], FloatArray]  # f (F,) -> (F, M, M)


def db_spl_to_pa(level_db: float) -> float:
    return P_REF * 10 ** (level_db / 20)


def pa_to_db_spl(p_rms: float) -> float:
    return 20 * math.log10(p_rms / P_REF)


def colored_psd(f: FloatArray, color: str) -> FloatArray:
    """Relative PSD shape: 1 / (f^2 + fc^2)^(alpha / 2); alpha = 0, 1, 2 for white, pink, brown."""
    alpha = COLOR_EXPONENT[color]
    return 1.0 / (np.asarray(f) ** 2 + COLOR_CORNER_HZ**2) ** (alpha / 2)


def diffuse_coherence(positions: FloatArray, sound_speed: float) -> Coherence:
    d = cdist(positions, positions)
    return lambda f: np.sinc(2.0 * f[:, None, None] * d[None] / sound_speed)


def wind_coherence(positions: FloatArray, wind_speed: float) -> Coherence:
    d = cdist(positions, positions)
    return lambda f: np.exp(-WIND_COHERENCE_DECAY * f[:, None, None] * d[None] / wind_speed)


def generate_field(
    psd: FloatArray,
    n_mics: int,
    n: int,
    fs: float,
    rng: np.random.Generator,
    coherence: Coherence | None = None,
) -> FloatArray:
    """(n_mics, n) real noise with one-sided PSD `psd` (length n // 2 + 1) and given coherence."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec = np.fft.rfft(rng.standard_normal((n_mics, n)), axis=1)  # E|W|^2 = n per bin
    if coherence is not None and n_mics > 1:
        for lo in range(0, len(f), _CHUNK):
            sl = slice(lo, min(lo + _CHUNK, len(f)))
            lam, vec = np.linalg.eigh(coherence(f[sl]))
            mix = vec * np.sqrt(np.clip(lam, 0.0, None))[:, None, :]
            spec[:, sl] = np.einsum("fij,jf->if", mix, spec[:, sl])
    spec *= np.sqrt(np.asarray(psd) * fs / 2.0)[None, :]
    return np.fft.irfft(spec, n=n, axis=1)


def wind_rms(wind_speed: float, air_density: float) -> float:
    """Wind-noise RMS pressure, Pa: WIND_NOISE_COEFF * dynamic pressure 0.5 rho U^2."""
    return WIND_NOISE_COEFF * 0.5 * air_density * wind_speed**2


def wind_psd_shape(f: FloatArray, wind_speed: float) -> FloatArray:
    """Flat below f_c = U / L, falling as f^(-5/3) above (inertial-range turbulence)."""
    fc = wind_speed / WIND_EDDY_SCALE_M
    return 1.0 / (1.0 + (np.asarray(f) / fc) ** (5.0 / 3.0))


def scaled_psd(shape: FloatArray, p_rms: float, fs: float) -> FloatArray:
    """Scale a PSD shape so its integral over 0..fs/2 equals p_rms^2."""
    df = fs / (2 * (len(shape) - 1))
    return shape * p_rms**2 / (np.sum(shape) * df)


def rain_noise(n_mics: int, n: int, fs: float, p_rms: float, rng: np.random.Generator) -> FloatArray:
    """Independent rain-impact shot noise per mic: Poisson drop impacts, high-passed, at p_rms."""
    out = np.zeros((n_mics, n))
    sos = signal.butter(1, RAIN_HIGHPASS_HZ, "highpass", fs=fs, output="sos")
    for i in range(n_mics):
        k = rng.poisson(RAIN_IMPACT_RATE_HZ * n / fs)
        idx = rng.integers(0, n, k)
        np.add.at(out[i], idx, rng.exponential(1.0, k))
        out[i] = signal.sosfilt(sos, out[i] - out[i].mean())
        out[i] *= p_rms / np.sqrt(np.mean(out[i] ** 2))
    return out


def band_power(x: FloatArray, fs: float, band: tuple[float, float]) -> FloatArray:
    """Mean power (Pa^2) of each row of x within the frequency band (ideal FFT mask, Parseval)."""
    x = np.atleast_2d(x)
    n = x.shape[1]
    spec = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    w = np.where((f == 0) | ((n % 2 == 0) & (f == fs / 2)), 1.0, 2.0)  # one-sided weights
    mask = (f >= band[0]) & (f <= band[1])
    return np.sum(w[mask] * np.abs(spec[:, mask]) ** 2, axis=1) / n**2

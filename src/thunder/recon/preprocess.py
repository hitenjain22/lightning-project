"""Shared reconstruction preprocessing (SPEC.md Phase 5).

1. Remove DC and band-pass with a zero-phase Butterworth filter (forward-backward), so the
   filter itself never shifts arrival times.
2. Optional spectral whitening.
3. Slide analysis windows over the recording and keep the span from the first to the last
   window whose band energy is both within `dynamic_range_db` of the loudest window and
   `snr_db` above the noise floor (10th percentile of window energies).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from thunder.types import FloatArray, IntArray


def bandpass(signals: FloatArray, fs: float, band: tuple[float, float], order: int) -> FloatArray:
    """Zero-phase band-pass of each row, after removing its mean."""
    x = np.atleast_2d(signals) - np.mean(signals, axis=-1, keepdims=True)
    sos = signal.butter(order, band, "bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def whiten(signals: FloatArray, fs: float, band: tuple[float, float], smooth_hz: float = 5.0) -> FloatArray:
    """Flatten each row's magnitude spectrum inside the band (moving-average smoothed)."""
    x = np.atleast_2d(signals)
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    k = max(1, int(round(smooth_hz / (f[1] - f[0]))))
    mag = signal.convolve(np.abs(spec), np.ones((1, k)) / k, mode="same")
    in_band = (f >= band[0]) & (f <= band[1])
    spec = np.where(in_band, spec / np.maximum(mag, 1e-30), 0.0)
    return np.fft.irfft(spec, n=n, axis=-1)


@dataclass(frozen=True)
class Frames:
    starts: IntArray  # (K,) first sample of each window
    length: int  # samples per window


def frame(n_samples: int, fs: float, window_s: float, overlap: float) -> Frames:
    length = max(8, int(round(window_s * fs)))
    hop = max(1, int(round(length * (1 - overlap))))
    starts = np.arange(0, max(1, n_samples - length + 1), hop, dtype=np.int64)
    return Frames(starts, length)


def window_energy(signals: FloatArray, frames: Frames) -> FloatArray:
    """Energy of each window summed over all mics."""
    power = np.sum(np.atleast_2d(signals) ** 2, axis=0)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    return csum[frames.starts + frames.length] - csum[frames.starts]


def active_frames(energy: FloatArray, dynamic_range_db: float, snr_db: float) -> IntArray:
    """Indices of windows from the first to the last one that passes both energy thresholds."""
    if energy.size == 0 or np.max(energy) <= 0:
        return np.zeros(0, dtype=np.int64)
    floor = float(np.percentile(energy, 10))
    threshold = max(np.max(energy) * 10 ** (-dynamic_range_db / 10), floor * 10 ** (snr_db / 10))
    hot = np.flatnonzero(energy >= threshold)
    if hot.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(hot[0], hot[-1] + 1, dtype=np.int64)

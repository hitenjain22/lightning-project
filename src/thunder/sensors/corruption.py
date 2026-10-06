"""Phase 4: turn a clean synthetic recording into what real hardware would record.

Clock offsets/drift and position errors are physical (they change when sound reaches each
sensor and how the recorder timestamps it), so they are applied exactly during synthesis
from the MicArray's hidden truth. This module applies everything after the sound field,
in physical order:

    1. acoustic noise at each mic: background (diffuse or incoherent, at a set SNR),
       wind (turbulent, nearly incoherent), rain (impact shot noise)
    2. microphone response: causal Butterworth high/low-pass (per-mic corner error) and
       per-mic sensitivity error
    3. electrical self-noise (white; datasheet level over 20 Hz-20 kHz converted to density)
    4. sample-clock jitter (first-order: x(t + e) ~ x(t) + e x'(t), spectral derivative)
    5. clipping at the recorder's full scale, then ADC quantization (round to nearest)
    6. flash-time error on the reported t0

Each step is skipped when disabled, so with everything off the output is bit-identical to
the input. Every realized corruption is written to the hidden ground truth.

SNR definition: 10 log10(P_signal / P_noise), both band powers in `snr_band_hz` (default
10-300 Hz, the analysis band), measured at the acoustic input (before the mic) over the
thunder's active window, averaged over mics. The noise is scaled so this holds exactly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import signal

from thunder.config import SensorsConfig
from thunder.constants import SELF_NOISE_SPEC_BAND_HZ
from thunder.sensors.noise import (
    band_power,
    colored_psd,
    db_spl_to_pa,
    diffuse_coherence,
    generate_field,
    rain_noise,
    scaled_psd,
    wind_coherence,
    wind_psd_shape,
    wind_rms,
)
from thunder.types import FloatArray, MicArray, Recording

PREROLL_S = 2.0  # noise generated before the recording so filters start in steady state


def active_window(rec: Recording) -> slice:
    """Sample range from the first to the last thunder arrival (from ground truth)."""
    assert rec.truth is not None
    ex = rec.truth.extra
    times = ex.get("segment_arrival_times_recorder", rec.truth.segment_arrival_times)
    t_lo = float(np.min(times))
    t_hi = float(np.max(times)) + float(np.max(ex["stroke_times"]))
    t_hi += float(np.max(ex["segment_pulse_duration"]))
    n = rec.signals.shape[1]
    lo = max(0, int(np.floor(t_lo * rec.sample_rate)))
    hi = min(n, int(np.ceil(t_hi * rec.sample_rate)) + 1)
    return slice(lo, hi)


def mic_filter_sos(cfg: SensorsConfig, fs: float, corner_scale: float) -> np.ndarray | None:
    """Second-order sections of one mic's band-pass response, or None if flat."""
    mic = cfg.mic
    nyq_limit = 0.45 * fs
    sections = []
    if mic.highpass_hz is not None:
        sections.append(signal.butter(mic.filter_order, mic.highpass_hz * corner_scale, "highpass", fs=fs,
                                      output="sos"))
    if mic.lowpass_hz is not None and mic.lowpass_hz * corner_scale < nyq_limit:
        sections.append(signal.butter(mic.filter_order, mic.lowpass_hz * corner_scale, "lowpass", fs=fs,
                                      output="sos"))
    return np.vstack(sections) if sections else None


def self_noise_rms(level_db_spl: float, fs: float) -> float:
    """RMS (Pa) over 0..fs/2 of white noise whose level over the datasheet band is `level_db_spl`."""
    lo, hi = SELF_NOISE_SPEC_BAND_HZ
    density = db_spl_to_pa(level_db_spl) ** 2 / (hi - lo)  # Pa^2 / Hz
    return float(np.sqrt(density * fs / 2))


def spectral_derivative(x: FloatArray, fs: float) -> FloatArray:
    """Band-limited time derivative of each row (even extension avoids edge wrap-around)."""
    n = x.shape[-1]
    ext = np.concatenate([x, x[..., ::-1]], axis=-1)
    f = np.fft.rfftfreq(2 * n, 1.0 / fs)
    d = np.fft.irfft(np.fft.rfft(ext, axis=-1) * (2j * np.pi * f), n=2 * n, axis=-1)
    return d[..., :n]


def draw_t0_error(cfg: SensorsConfig, rng: np.random.Generator) -> float:
    ft = cfg.flash_time
    if ft.scale_s == 0:
        return 0.0
    if ft.distribution == "normal":
        return float(rng.normal(0.0, ft.scale_s))
    return float(rng.uniform(-ft.scale_s, ft.scale_s))


def corrupt(
    clean: Recording,
    array: MicArray,
    cfg: SensorsConfig,
    rng: np.random.Generator,
    sound_speed: float = 343.0,
    air_density: float = 1.2,
) -> Recording:
    """Apply sensor noise, response, jitter, clipping, quantization and t0 error.

    `clean` must come from `synthesize(..., mics=array)` so clock and position errors are
    already in it. `sound_speed` and `air_density` are at mic height (diffuse-field
    coherence and wind dynamic pressure).
    """
    assert clean.truth is not None
    noise_rng, jitter_rng, t0_rng = rng.spawn(3)
    bg_rng, wind_rng, rain_rng, self_rng = noise_rng.spawn(4)
    fs = clean.sample_rate
    m, n = clean.signals.shape
    nz = cfg.noise
    mic = cfg.mic
    info: dict = {}

    filters = [mic_filter_sos(cfg, fs, float(s)) for s in np.asarray(array.corner_scale)]
    any_filter = any(f is not None for f in filters)
    any_noise = nz.background_snr_db is not None or nz.wind_speed_mps > 0 or nz.rain_db_spl is not None
    pre = int(round(PREROLL_S * fs)) if (any_filter and any_noise) else 0
    x = clean.signals
    if any_noise or pre:
        x = np.concatenate([np.zeros((m, pre)), clean.signals], axis=1)
    total = n + pre
    f = np.fft.rfftfreq(total, 1.0 / fs)
    positions = array.true_positions

    # 1. Acoustic noise.
    if nz.background_snr_db is not None:
        coh = diffuse_coherence(positions, sound_speed) if nz.background_field == "diffuse" else None
        bg = generate_field(colored_psd(f, nz.background_color), m, total, fs, bg_rng, coh)
        win = active_window(clean)
        p_sig = float(np.mean(band_power(clean.signals[:, win], fs, nz.snr_band_hz)))
        p_bg = float(np.mean(band_power(bg[:, pre:][:, win], fs, nz.snr_band_hz)))
        bg *= np.sqrt(p_sig / p_bg / 10 ** (nz.background_snr_db / 10))
        x = x + bg
        info["background_power_band_pa2"] = p_sig / 10 ** (nz.background_snr_db / 10)
        info["signal_power_band_pa2"] = p_sig
    if nz.wind_speed_mps > 0:
        p_rms = wind_rms(nz.wind_speed_mps, air_density)
        psd = scaled_psd(wind_psd_shape(f, nz.wind_speed_mps), p_rms, fs)
        x = x + generate_field(psd, m, total, fs, wind_rng, wind_coherence(positions, nz.wind_speed_mps))
        info["wind_rms_pa"] = p_rms
    if nz.rain_db_spl is not None:
        x = x + rain_noise(m, total, fs, db_spl_to_pa(nz.rain_db_spl), rain_rng)
        info["rain_rms_pa"] = db_spl_to_pa(nz.rain_db_spl)

    # 2. Microphone response and sensitivity.
    if any_filter:
        x = np.stack([signal.sosfilt(sos, row) if sos is not None else row for sos, row in zip(filters, x,
                                                                                               strict=True)])
    x = x[:, pre:] if (any_noise or pre) else x
    gain_db = np.asarray(array.gain_db)
    if np.any(gain_db != 0):
        x = x * (10 ** (gain_db / 20))[:, None]

    # 3. Electrical self-noise: datasheet level over 20 Hz-20 kHz -> white noise density.
    if mic.self_noise_db_spl is not None:
        x = x + self_rng.standard_normal((m, n)) * self_noise_rms(mic.self_noise_db_spl, fs)

    # 4. Sample-clock jitter.
    if cfg.timing.jitter_std_s > 0:
        x = x + jitter_rng.standard_normal((m, n)) * cfg.timing.jitter_std_s * spectral_derivative(x, fs)

    # 5. Clipping and quantization.
    if mic.clip_db_spl is not None:
        full_scale = db_spl_to_pa(mic.clip_db_spl)
        info["clipped_fraction"] = float(np.mean(np.abs(x) >= full_scale))
        x = np.clip(x, -full_scale, full_scale)
        if mic.adc_bits is not None:
            lsb = 2 * full_scale / 2**mic.adc_bits
            codes = np.clip(np.round(x / lsb), -(2 ** (mic.adc_bits - 1)), 2 ** (mic.adc_bits - 1) - 1)
            x = codes * lsb
            info["adc_lsb_pa"] = lsb

    # 6. Reported flash time.
    t0_err = draw_t0_error(cfg, t0_rng)

    truth = dataclasses.replace(
        clean.truth,
        extra={
            **clean.truth.extra,
            "clean_signals": clean.signals,
            "t0_error": t0_err,
            "clock_offset": np.asarray(array.clock_offset).copy(),
            "clock_drift_ppm": np.asarray(array.clock_drift_ppm).copy(),
            "mic_gain_db": gain_db.copy(),
            "mic_corner_scale": np.asarray(array.corner_scale).copy(),
            "nominal_mic_positions": array.nominal_positions.copy(),
            "corruption_info": info,
        },
    )
    return Recording(
        signals=x if x is not clean.signals else clean.signals.copy(),
        sample_rate=fs,
        nominal_mic_positions=array.nominal_positions.copy(),
        reported_t0=clean.reported_t0 + t0_err,
        truth=truth,
    )

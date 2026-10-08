import dataclasses
import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import signal
from scipy.spatial.distance import pdist

from thunder.acoustics.source import pulse_duration
from thunder.acoustics.synth import synthesize
from thunder.channel.generator import generate_channel
from thunder.config import ArrayConfig, ChannelConfig, SensorsConfig
from thunder.constants import ACOUSTIC_EFFICIENCY, P_STANDARD, ZERO_CELSIUS_K, sound_speed_dry
from thunder.sensors import arrays as A
from thunder.sensors.corruption import active_window, corrupt, mic_filter_sos, spectral_derivative
from thunder.sensors.noise import (
    band_power,
    colored_psd,
    db_spl_to_pa,
    diffuse_coherence,
    generate_field,
    pa_to_db_spl,
    rain_noise,
    wind_coherence,
    wind_psd_shape,
    wind_rms,
)
from thunder.types import Channel, MicArray, UniformAtmosphere

FS = 8000.0
OS = 8
T_AIR = ZERO_CELSIUS_K + 25.0
C0 = sound_speed_dry(T_AIR)
ATM = UniformAtmosphere(C0, T_AIR)
T_PULSE = float(pulse_duration(1e6, P_STANDARD, C0))


def polyline(points) -> Channel:
    pts = np.asarray(points, dtype=float)
    n = len(pts) - 1
    return Channel(
        nodes=pts,
        segments=np.c_[np.arange(n), np.arange(1, n + 1)],
        energy_per_length=np.full(n, 1e6),
        branch_id=np.zeros(n, dtype=np.int64),
        is_main=np.ones(n, dtype=bool),
        is_incloud=np.zeros(n, dtype=bool),
    )


def synth(ch, mics):
    return synthesize(ch, ATM, mics, FS, OS, 0.5, ACOUSTIC_EFFICIENCY)


@pytest.fixture(scope="module")
def bolt_recording():
    """A branched bolt at 3 km recorded by an ideal 5-mic array (clean)."""
    cfg = ChannelConfig(preset="branched", strike_distance_m=(3000, 3000))
    ch = generate_channel(cfg, np.random.default_rng(11))
    array = MicArray.ideal(A.square(50.0, 1.5, center=True))
    return synth(ch, array), array


# --- array geometry -----------------------------------------------------------

LAYOUTS = [
    ("triangle", {}, 3),
    ("square", {}, 4),
    ("square_center", {}, 5),
    ("circle", {"n_mics": 8}, 8),
    ("circle", {"n_mics": 7}, 7),
    ("l_shape", {"n_mics": 7}, 7),
    ("cross", {"n_mics": 9}, 9),
    ("cross", {"n_mics": 8}, 8),
    ("mast", {}, 5),
    ("random_disk", {"n_mics": 10}, 10),
]


@pytest.mark.parametrize("layout,extra,n", LAYOUTS)
def test_layout_count_aperture_centroid(layout, extra, n):
    cfg = ArrayConfig(layout=layout, aperture_m=120.0, **extra)
    p = A.nominal_positions(cfg, np.random.default_rng(0))
    assert p.shape == (n, 3)
    assert A.horizontal_aperture(p) == pytest.approx(120.0, rel=1e-12)
    np.testing.assert_allclose(p[:, :2].mean(axis=0), 0.0, atol=1e-9)
    assert np.min(pdist(p)) > 1.0  # no coincident mics
    ground = p[:, 2] == pytest.approx(1.5) if layout != "mast" else p[:4, 2] == pytest.approx(1.5)
    assert ground


def test_triangle_is_equilateral_and_circle_is_regular():
    np.testing.assert_allclose(pdist(A.triangle(30.0)), 30.0)
    p = A.circle(6, 100.0)
    r = np.linalg.norm(p[:, :2], axis=1)
    np.testing.assert_allclose(r, 50.0)
    ang = np.sort(np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi))
    np.testing.assert_allclose(np.diff(ang), 2 * np.pi / 6)


def test_mast_mic_is_raised_at_center():
    p = A.mast(40.0, 1.5, 10.0)
    np.testing.assert_allclose(p[4], [0, 0, 10.0], atol=1e-12)


def test_distributed_subarrays():
    p = A.distributed("triangle", 3, 20.0, 600.0)
    assert p.shape == (9, 3)
    centers = p.reshape(3, 3, 3).mean(axis=1)
    np.testing.assert_allclose(pdist(centers[:, :2]), 600.0, rtol=1e-12)
    for sub in p.reshape(3, 3, 3):
        assert A.horizontal_aperture(sub) == pytest.approx(20.0)
    np.testing.assert_allclose(p[:, :2].mean(axis=0), 0.0, atol=1e-9)
    two = A.distributed("square", 2, 10.0, 300.0)
    assert two.shape == (8, 3)


def test_random_disk_reproducible_and_spaced():
    a = A.random_disk(30, 100.0, np.random.default_rng(1))
    b = A.random_disk(30, 100.0, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert np.min(pdist(a[:, :2])) >= 100.0 * 0.5 / math.sqrt(30) * 0.999


def test_invalid_layouts_rejected():
    with pytest.raises(ValueError):
        A.l_shape(6, 10.0)
    with pytest.raises(ValueError):
        A.cross(6, 10.0)
    with pytest.raises(ValueError):
        A.nominal_positions(ArrayConfig(layout="circle"), np.random.default_rng(0))
    with pytest.raises(ValidationError):
        ArrayConfig(layout="free_form")
    assert ArrayConfig(positions_m=[(0, 0, 1), (5, 0, 1)]).layout == "free_form"


# --- hidden hardware errors ---------------------------------------------------


def test_no_sensors_gives_ideal_array():
    arr = A.build_mic_array(ArrayConfig(), None, np.random.default_rng(0))
    np.testing.assert_array_equal(arr.true_positions, arr.nominal_positions)
    assert not arr.has_clock_error


def test_position_error_statistics():
    pos = [(float(x), 0.0, 1.5) for x in range(4000)]
    cfg = SensorsConfig(position={"horizontal_std_m": 0.3, "vertical_std_m": 0.1})
    arr = A.build_mic_array(ArrayConfig(positions_m=pos), cfg, np.random.default_rng(2))
    err = arr.true_positions - arr.nominal_positions
    assert err[:, 0].std() == pytest.approx(0.3, rel=0.05)
    assert err[:, 1].std() == pytest.approx(0.3, rel=0.05)
    assert err[:, 2].std() == pytest.approx(0.1, rel=0.05)
    np.testing.assert_array_equal(arr.nominal_positions, np.array(pos))


def test_clock_presets():
    big = ArrayConfig(positions_m=[(float(x), 0.0, 1.5) for x in range(3000)])
    gps = A.build_mic_array(big, SensorsConfig(timing={"preset": "gps_synced"}), np.random.default_rng(3))
    assert gps.clock_offset.std() == pytest.approx(1e-6, rel=0.05)
    assert np.all(gps.clock_drift_ppm == 0)
    shared = A.build_mic_array(
        ArrayConfig(), SensorsConfig(timing={"preset": "shared_interface"}), np.random.default_rng(4)
    )
    assert np.all(shared.clock_offset == 0) and len(set(shared.clock_drift_ppm)) == 1
    hand = A.build_mic_array(big, SensorsConfig(timing={"preset": "hand_synced"}), np.random.default_rng(5))
    assert hand.clock_offset.std() == pytest.approx(3e-3, rel=0.05)
    assert hand.clock_drift_ppm.std() == pytest.approx(20.0, rel=0.05)


def test_error_streams_are_independent():
    base = SensorsConfig(timing={"preset": "hand_synced"})
    more = SensorsConfig(timing={"preset": "hand_synced"}, position={"preset": "tape"})
    a = A.build_mic_array(ArrayConfig(), base, np.random.default_rng(6))
    b = A.build_mic_array(ArrayConfig(), more, np.random.default_rng(6))
    np.testing.assert_array_equal(a.clock_offset, b.clock_offset)
    assert not np.array_equal(a.true_positions, b.true_positions)


# --- clocks and positions in synthesis ----------------------------------------


def _fft_delay(x, delay_s):
    n = x.shape[-1]
    f = np.fft.rfftfreq(2 * n, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, 2 * n) * np.exp(-2j * np.pi * f * delay_s), 2 * n)[:n]


def test_clock_offset_is_an_exact_fractional_delay():
    ch = polyline([[1500, 200, 1200], [1530, 180, 600], [1500, 200, 0]])
    pos = np.array([[0, 0, 1.5]])
    clean = synth(ch, pos).signals[0]
    offset = 1.37e-3  # 10.96 samples
    arr = MicArray(pos, pos.copy(), np.array([offset]), np.zeros(1))
    shifted = synth(ch, arr).signals[0]
    n = len(clean)
    expected = _fft_delay(clean, offset)
    assert np.abs(shifted[:n] - expected).max() < 0.01 * np.abs(clean).max()
    # Cross-correlation peak (parabolic sub-sample refinement) recovers the offset.
    xc = signal.correlate(shifted[:n], clean, mode="full")
    k = int(np.argmax(xc))
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    lag = (k - (n - 1)) + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    assert lag / FS == pytest.approx(offset, abs=0.05 / FS)


def test_clock_drift_and_offset_map_arrival_times():
    src = np.array([2000.0, 0.0, 500.0])
    ch = polyline([src, src + [0, 0, 0.05]])
    pos = np.array([[0, 0, 1.5]])
    drift, offset = 500.0, 2e-3  # exaggerated drift so it is measurable
    rec = synth(ch, MicArray(pos, pos.copy(), np.array([offset]), np.array([drift])))
    x = rec.signals[0]
    t = np.arange(len(x)) / FS
    r = np.linalg.norm(src + [0, 0, 0.025] - pos[0])
    centroid = (t * x**2).sum() / (x**2).sum()
    assert centroid - T_PULSE / 2 == pytest.approx((1 + drift * 1e-6) * r / C0 + offset, abs=1 / FS)
    assert rec.truth is not None
    np.testing.assert_allclose(
        rec.truth.extra["segment_arrival_times_recorder"],
        (1 + drift * 1e-6) * rec.truth.segment_arrival_times + offset,
    )


def test_negative_clock_time_rejected():
    pos = np.array([[0, 0, 1.5]])
    arr = MicArray(pos, pos.copy(), np.array([-1.0]), np.zeros(1))
    with pytest.raises(ValueError):
        synth(polyline([[100, 0, 50], [100, 0, 0]]), arr)


def test_synthesis_uses_true_positions_and_reports_nominal():
    nominal = np.array([[0, 0, 1.5]])
    true = np.array([[30.0, 0, 1.5]])
    ch = polyline([[1000, 0, 500], [1000, 0, 499.95]])
    rec = synth(ch, MicArray(nominal, true, np.zeros(1), np.zeros(1)))
    ref = synth(ch, true)
    np.testing.assert_array_equal(rec.signals, ref.signals)
    np.testing.assert_array_equal(rec.nominal_mic_positions, nominal)


# --- noise fields -------------------------------------------------------------


def test_field_psd_level_and_color_slopes():
    fs, n = 8000.0, 2**18
    f = np.fft.rfftfreq(n, 1 / fs)
    rng = np.random.default_rng(7)
    white = generate_field(np.ones_like(f), 1, n, fs, rng)
    assert white.var() == pytest.approx(fs / 2, rel=0.02)  # integral of PSD = 1 over 0..fs/2
    for color, slope in (("pink", -1.0), ("brown", -2.0)):
        x = generate_field(colored_psd(f, color), 1, n, fs, rng)[0]
        fw, p = signal.welch(x, fs, nperseg=8192)
        sel = (fw > 10) & (fw < 1000)
        assert np.polyfit(np.log10(fw[sel]), np.log10(p[sel]), 1)[0] == pytest.approx(slope, abs=0.05)


def test_diffuse_field_coherence_is_sinc():
    fs, n = 1000.0, 2**21
    pos = np.array([[0, 0, 0], [20.0, 0, 0]])
    f = np.fft.rfftfreq(n, 1 / fs)
    x = generate_field(np.ones_like(f), 2, n, fs, np.random.default_rng(8), diffuse_coherence(pos, 343.0))
    fc, msc = signal.coherence(x[0], x[1], fs, nperseg=16384)
    for target in (1.0, 2.0, 3.0, 5.0):
        i = np.argmin(abs(fc - target))
        assert msc[i] == pytest.approx(np.sinc(2 * fc[i] * 20 / 343.0) ** 2, abs=0.03)


def test_wind_noise_level_scaling_and_incoherence():
    fs, n = 1000.0, 2**19
    f = np.fft.rfftfreq(n, 1 / fs)
    pos = np.array([[0, 0, 1.5], [10.0, 0, 1.5]])
    rms = {}
    for u in (4.0, 8.0):
        shape = wind_psd_shape(f, u)
        df = f[1]
        psd = shape * wind_rms(u, 1.2) ** 2 / (shape.sum() * df)
        x = generate_field(psd, 2, n, fs, np.random.default_rng(9), wind_coherence(pos, u))
        rms[u] = np.sqrt(np.mean(x**2))
        assert rms[u] == pytest.approx(wind_rms(u, 1.2), rel=0.03)
    assert rms[8.0] / rms[4.0] == pytest.approx(4.0, rel=0.05)  # dynamic pressure ~ U^2
    fc, msc = signal.coherence(x[0], x[1], fs, nperseg=8192)
    # Theory: |exp(-f d / U)|^2 < 0.007 above 2 Hz at 10 m; the estimator's own bias is ~1/K.
    assert np.mean(msc[(fc > 2) & (fc < 300)]) < 0.02


def test_rain_noise_level_incoherent_and_high_frequency():
    p = db_spl_to_pa(60.0)
    x = rain_noise(2, 2**18, FS, p, np.random.default_rng(10))
    np.testing.assert_allclose(np.sqrt(np.mean(x**2, axis=1)), p, rtol=1e-12)
    fc, msc = signal.coherence(x[0], x[1], FS, nperseg=4096)
    assert np.mean(msc) < 0.01
    assert band_power(x[0], FS, (0.1, 100.0))[0] < 0.01 * band_power(x[0], FS, (500.0, 4000.0))[0]


def test_band_power_parseval():
    rng = np.random.default_rng(12)
    x = rng.standard_normal(10001)
    assert band_power(x, FS, (0.0, FS / 2))[0] == pytest.approx(np.mean(x**2), rel=1e-12)
    t = np.arange(8000) / FS
    sine = 3.0 * np.sin(2 * np.pi * 100 * t)
    assert band_power(sine, FS, (90, 110))[0] == pytest.approx(4.5, rel=1e-9)
    assert band_power(sine, FS, (200, 300))[0] < 1e-20


# --- corruption chain ---------------------------------------------------------


def test_absolute_background_level_matches_db_spl(bolt_recording):
    """background_db_spl: band power over the whole recording equals the configured level, and
    the reported SNR is the thunder-window band power over that level."""
    clean, array = bolt_recording
    cfg = SensorsConfig(noise={"background_db_spl": 45.0})
    out = corrupt(clean, array, cfg, np.random.default_rng(3), sound_speed=C0)
    noise = out.signals - clean.signals
    band = cfg.noise.snr_band_hz
    level = 10 * np.log10(np.mean(band_power(noise, FS, band)) / db_spl_to_pa(0.0) ** 2)
    assert level == pytest.approx(45.0, abs=1e-6)
    info = out.truth.extra["corruption_info"]
    p_sig = np.mean(band_power(clean.signals[:, active_window(clean)], FS, band))
    assert info["snr_band_db"] == pytest.approx(10 * np.log10(p_sig / db_spl_to_pa(45.0) ** 2), rel=1e-9)
    # The level is absolute: a weaker bolt gets the same noise and a lower SNR.
    weak = dataclasses.replace(clean, signals=0.1 * clean.signals)
    out_w = corrupt(weak, array, cfg, np.random.default_rng(3), sound_speed=C0)
    np.testing.assert_allclose(out_w.signals - weak.signals, noise, rtol=1e-9, atol=1e-15)
    assert out_w.truth.extra["corruption_info"]["snr_band_db"] == pytest.approx(info["snr_band_db"] - 20.0)


def test_db_spl_conversions_are_inverse():
    for level in (0.0, 45.0, 94.0, 137.3):
        assert pa_to_db_spl(db_spl_to_pa(level)) == pytest.approx(level, abs=1e-12)
    assert db_spl_to_pa(94.0) == pytest.approx(1.0024, rel=1e-4)  # 94 dB SPL is about 1 Pa


def test_background_level_options_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        SensorsConfig(noise={"background_snr_db": 10.0, "background_db_spl": 45.0})


def test_all_corruptions_off_is_bit_identical(bolt_recording):
    clean, array = bolt_recording
    out = corrupt(clean, array, SensorsConfig(), np.random.default_rng(0))
    assert np.array_equal(out.signals, clean.signals)
    assert out.reported_t0 == clean.reported_t0
    np.testing.assert_array_equal(out.nominal_mic_positions, array.nominal_positions)


def test_snr_matches_configuration(bolt_recording):
    clean, array = bolt_recording
    for snr in (0.0, 10.0, 25.0):
        cfg = SensorsConfig(noise={"background_snr_db": snr})
        out = corrupt(clean, array, cfg, np.random.default_rng(1), sound_speed=C0)
        noise = out.signals - clean.signals
        win = active_window(clean)
        band = cfg.noise.snr_band_hz
        exact = 10 * np.log10(
            np.mean(band_power(clean.signals[:, win], FS, band))
            / np.mean(band_power(noise[:, win], FS, band))
        )
        assert exact == pytest.approx(snr, abs=1e-6)
        # Independent estimate: Welch PSDs integrated over the band.
        f, ps = signal.welch(clean.signals[:, win], FS, nperseg=2048, axis=-1)
        _, pn = signal.welch(noise[:, win], FS, nperseg=2048, axis=-1)
        sel = (f >= band[0]) & (f <= band[1])
        welch = 10 * np.log10(ps[:, sel].sum() / pn[:, sel].sum())
        assert welch == pytest.approx(snr, abs=0.5)


def test_mic_response_corners_and_causality():
    for preset, edge, kind in (("phone", 100.0, "highpass"), ("measurement", 2000.0, "lowpass")):
        cfg = SensorsConfig(mic={"preset": preset})
        sos = mic_filter_sos(cfg, FS, 1.0)
        w, h = signal.sosfreqz(sos, worN=[edge, edge * 0.01 if kind == "lowpass" else edge * 10], fs=FS)
        assert 20 * np.log10(abs(h[0])) == pytest.approx(-3.01, abs=0.05)
        assert 20 * np.log10(abs(h[1])) == pytest.approx(0.0, abs=0.1)
    # audio low-pass at 20 kHz is above the 4 kHz Nyquist band and is skipped; its high-pass remains.
    audio = mic_filter_sos(SensorsConfig(mic={"preset": "audio"}), FS, 1.0)
    assert audio is not None and audio.shape[0] == 1
    assert mic_filter_sos(SensorsConfig(), FS, 1.0) is None
    # Corner error moves the corner.
    sos = mic_filter_sos(SensorsConfig(mic={"preset": "phone"}), FS, 1.2)
    _, h = signal.sosfreqz(sos, worN=[120.0], fs=FS)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(-3.01, abs=0.05)


def test_phone_highpass_removes_much_of_the_thunder(bolt_recording):
    clean, array = bolt_recording
    out = corrupt(
        clean,
        array,
        SensorsConfig(
            mic={
                "preset": "phone",
                "gain_tolerance_db": 0,
                "corner_tolerance": 0,
                "self_noise_db_spl": None,
                "clip_db_spl": None,
                "adc_bits": None,
            }
        ),
        np.random.default_rng(2),
    )
    removed = 1 - (out.signals**2).sum() / (clean.signals**2).sum()
    assert removed > 0.3


def test_gain_error_applied():
    pos = A.triangle(20.0)
    arr = MicArray(pos, pos.copy(), np.zeros(3), np.zeros(3), gain_db=np.array([0.0, 6.0, -3.0]))
    clean = synth(polyline([[800, 0, 400], [800, 0, 399.9]]), arr)
    out = corrupt(clean, arr, SensorsConfig(), np.random.default_rng(3))
    gains = 10 ** (np.array([0.0, 6.0, -3.0]) / 20)
    np.testing.assert_allclose(out.signals, clean.signals * gains[:, None], rtol=1e-12, atol=0)


def test_self_noise_level(bolt_recording):
    clean, array = bolt_recording
    out = corrupt(clean, array, SensorsConfig(mic={"self_noise_db_spl": 40.0}), np.random.default_rng(4))
    noise = out.signals - clean.signals
    # 40 dB over 20 Hz-20 kHz -> same density over 0-4 kHz: 40 - 10 log10(19980 / 4000) dB.
    expected = 40.0 - 10 * np.log10(19980.0 / (FS / 2))
    assert 20 * np.log10(np.sqrt(np.mean(noise**2)) / 20e-6) == pytest.approx(expected, abs=0.05)
    f, psd = signal.welch(noise[0], FS, nperseg=4096)
    density = db_spl_to_pa(40.0) ** 2 / 19980.0
    assert np.median(psd[(f > 50) & (f < 3500)]) == pytest.approx(density, rel=0.05)


def test_adc_bits_require_full_scale():
    with pytest.raises(ValidationError):
        SensorsConfig(mic={"adc_bits": 16})


def test_clipping_and_quantization(bolt_recording):
    clean, array = bolt_recording
    cfg = SensorsConfig(mic={"clip_db_spl": 110.0, "adc_bits": 12})
    out = corrupt(clean, array, cfg, np.random.default_rng(5))
    fs_pa = db_spl_to_pa(110.0)
    lsb = 2 * fs_pa / 2**12
    assert np.abs(out.signals).max() <= fs_pa + 1e-12
    codes = out.signals / lsb
    np.testing.assert_allclose(codes, np.round(codes), atol=1e-9)
    unclipped = np.abs(clean.signals) < fs_pa - lsb
    assert np.abs(out.signals - clean.signals)[unclipped].max() <= lsb / 2 + 1e-12
    assert out.truth is not None and out.truth.extra["corruption_info"]["clipped_fraction"] > 0


def test_spectral_derivative_and_jitter():
    t = np.arange(16000) / FS
    x = np.sin(2 * np.pi * 50 * t)[None]
    d = spectral_derivative(x, FS)
    interior = slice(200, -200)
    exact = 2 * np.pi * 50 * np.cos(2 * np.pi * 50 * t)
    np.testing.assert_allclose(d[0, interior], exact[interior], atol=1e-4 * np.abs(exact).max())
    # Jitter: y - x has rms sigma * rms(x').
    from thunder.types import GroundTruth, Recording

    rec = Recording(
        x.copy(),
        FS,
        np.zeros((1, 3)),
        0.0,
        GroundTruth(
            0.0,
            np.zeros((1, 3)),
            np.zeros((1, 1)),
            {"segment_pulse_duration": np.ones(1), "stroke_times": np.zeros(1)},
        ),
    )
    arr = MicArray.ideal(np.zeros((1, 3)))
    sigma = 1e-5
    out = corrupt(rec, arr, SensorsConfig(timing={"jitter_std_s": sigma}), np.random.default_rng(6))
    err = (out.signals - x)[0, interior]
    expected = sigma * 2 * np.pi * 50 / np.sqrt(2)
    assert np.sqrt(np.mean(err**2)) == pytest.approx(expected, rel=0.05)


def test_flash_time_error_presets(bolt_recording):
    clean, array = bolt_recording
    draws = {}
    for preset in ("exact", "photodiode", "lightning_network", "video_30fps"):
        cfg = SensorsConfig(flash_time={"preset": preset})
        from thunder.sensors.corruption import draw_t0_error

        rng = np.random.default_rng(7)
        draws[preset] = np.array([draw_t0_error(cfg, rng) for _ in range(4000)])
    assert np.all(draws["exact"] == 0)
    assert draws["photodiode"].std() == pytest.approx(10e-6, rel=0.05)
    assert draws["lightning_network"].std() == pytest.approx(1e-3, rel=0.05)
    v = draws["video_30fps"]
    assert v.min() >= -1 / 60 and v.max() <= 1 / 60
    assert v.std() == pytest.approx((1 / 60) / math.sqrt(3), rel=0.05)
    cfg = SensorsConfig(flash_time={"preset": "lightning_network"})
    out = corrupt(clean, array, cfg, np.random.default_rng(8))
    assert out.truth is not None
    assert out.reported_t0 == out.truth.extra["t0_error"] != 0


def test_truth_records_every_realized_corruption(bolt_recording):
    clean, array = bolt_recording
    out = corrupt(
        clean,
        array,
        SensorsConfig(noise={"background_snr_db": 10.0, "wind_speed_mps": 5.0, "rain_db_spl": 50.0}),
        np.random.default_rng(9),
    )
    assert out.truth is not None
    ex = out.truth.extra
    for key in (
        "clean_signals",
        "t0_error",
        "clock_offset",
        "clock_drift_ppm",
        "mic_gain_db",
        "nominal_mic_positions",
        "corruption_info",
    ):
        assert key in ex
    assert {"wind_rms_pa", "rain_rms_pa", "background_power_band_pa2"} <= set(ex["corruption_info"])
    np.testing.assert_array_equal(ex["clean_signals"], clean.signals)

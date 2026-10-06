import math

import numpy as np
import pytest
from scipy import integrate, signal

from thunder.acoustics.source import (
    line_amplitude,
    nwave_kernel,
    nwave_peak_x,
    peak_frequency,
    pulse_duration,
    spherical_j1,
)
from thunder.acoustics.synth import deposit, discretize, synthesize
from thunder.channel.generator import generate_channel
from thunder.config import ChannelConfig
from thunder.constants import ACOUSTIC_EFFICIENCY, P_STANDARD, ZERO_CELSIUS_K, sound_speed_dry
from thunder.types import Channel, UniformAtmosphere

FS = 8000.0
OS = 8
E_L = 1.0e6
T_AIR = ZERO_CELSIUS_K + 25.0
C0 = sound_speed_dry(T_AIR)
ATM = UniformAtmosphere(C0, T_AIR)
T_PULSE = float(pulse_duration(E_L, P_STANDARD, C0))


def polyline(points, energy=E_L, strokes=(0.0,), main=True) -> Channel:
    pts = np.asarray(points, dtype=float)
    n = len(pts) - 1
    return Channel(
        nodes=pts,
        segments=np.c_[np.arange(n), np.arange(1, n + 1)],
        energy_per_length=np.full(n, energy),
        branch_id=np.zeros(n, dtype=np.int64),
        is_main=np.full(n, main),
        is_incloud=np.zeros(n, dtype=bool),
        stroke_times=np.asarray(strokes, dtype=float),
    )


def synth(ch, mics, spacing=0.5):
    return synthesize(ch, ATM, np.atleast_2d(mics), FS, OS, spacing, ACOUSTIC_EFFICIENCY)


def support(x, rel=1e-3):
    """First and last sample times where |x| exceeds rel * max|x|."""
    idx = np.flatnonzero(np.abs(x) > rel * np.abs(x).max())
    return idx[0] / FS, idx[-1] / FS


# --- source model -----------------------------------------------------------


def test_j1_integral_is_one_quarter():
    """The closed-form amplitude calibration relies on int_0^inf j1(x)^2 / x dx = 1/4."""
    val, _ = integrate.quad(lambda x: spherical_j1(x) ** 2 / x, 1e-9, 400, limit=2000)
    assert val == pytest.approx(0.25, abs=1e-5)


def test_nwave_spectral_peak_equals_few_frequency():
    dt = 1e-6
    t = np.arange(0, T_PULSE, dt)
    n = 1 - 2 * t / T_PULSE
    nfft = 1 << 22
    spec = np.abs(np.fft.rfft(n, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    f_peak = float(peak_frequency(E_L, P_STANDARD, C0))
    assert f[np.argmax(spec)] == pytest.approx(f_peak, rel=0.005)
    assert 50 < f_peak < 300  # tens to low hundreds of Hz (spec)
    assert nwave_peak_x() == pytest.approx(2.0816, abs=1e-4)


def test_nwave_kernel_has_zero_mean():
    k = nwave_kernel(T_PULSE, 1 / (FS * OS))
    assert abs(k.sum()) < 1e-12 * len(k)
    assert k[0] == pytest.approx(0.5, abs=0.01)  # pulse starts mid-cell at t = 0
    assert k[1] > 0.9 and k.min() < -0.9


def test_deposit_conserves_mass_and_centroid():
    rng = np.random.default_rng(0)
    dt = 1e-3
    ta = rng.uniform(0.01, 0.5, 300)
    tb = ta + rng.choice([0.0, 1e-4, 5e-4, 2e-3, 0.05], 300)
    w = rng.uniform(0.1, 1.0, 300)
    m = deposit(ta, tb, w, 1000, dt)
    assert m.sum() == pytest.approx(w.sum(), rel=1e-12)
    centroid = (m * np.arange(1000) * dt).sum() / m.sum()
    assert centroid == pytest.approx((w * 0.5 * (ta + tb)).sum() / w.sum(), abs=1e-9)


# --- spec Phase 2 tests -----------------------------------------------------


def test_point_source_arrival_time_and_inverse_r_amplitude():
    src = np.array([1200.0, 300.0, 800.0])
    tiny = polyline([src, src + [0, 0, 0.05]])
    m1 = np.array([0, 0, 1.5])
    m2 = src - 2 * (src - m1)  # same line, twice the distance
    rec = synth(tiny, [m1, m2])
    for x, mic in zip(rec.signals, (m1, m2), strict=True):
        r = np.linalg.norm(src + [0, 0, 0.025] - mic)
        t = np.arange(len(x)) / FS
        centroid = (t * x**2).sum() / (x**2).sum()
        assert centroid - T_PULSE / 2 == pytest.approx(r / C0, abs=1 / FS)
    # Energy, not sampled peak: the band-limited N-wave's peak depends on sub-sample alignment.
    e1, e2 = (rec.signals**2).sum(axis=1)
    assert math.sqrt(e1 / e2) == pytest.approx(2.0, rel=0.01)


def test_vertical_line_first_and_last_arrival():
    D, H = 2000.0, 3000.0
    rec = synth(polyline([[D, 0, H], [D, 0, 0]]), [0, 0, 0])
    first, last = support(rec.signals[0])
    assert first == pytest.approx(D / C0, abs=1.5e-3)
    t_top = math.hypot(D, H) / C0
    assert t_top - 1e-3 <= last <= t_top + T_PULSE + 2e-3


def test_horizontal_segment_broadside_much_louder_than_end_on():
    seg = polyline([[-50, 0, 500], [50, 0, 500]])
    rec = synth(seg, [[0, 2000, 0], [2000, 0, 0]])
    e_broad, e_end = (rec.signals**2).sum(axis=1)
    assert e_broad / e_end > 10


def test_rumble_duration_matches_range_spread():
    cfg = ChannelConfig(preset="branched", strike_distance_m=(2500, 2500))
    ch = generate_channel(cfg, np.random.default_rng(3))
    mic = np.array([0, 0, 1.5])
    rec = synth(ch, mic)
    r = np.linalg.norm(ch.nodes - mic, axis=1)
    first, last = support(rec.signals[0])
    assert first == pytest.approx(r.min() / C0, abs=2e-3)
    assert last - first == pytest.approx((r.max() - r.min()) / C0, rel=0.05)


def test_spectrum_peak_in_thunder_band():
    cfg = ChannelConfig(preset="branched", strike_distance_m=(3000, 3000))
    ch = generate_channel(cfg, np.random.default_rng(4))
    rec = synth(ch, [0, 0, 1.5])
    f, pxx = signal.welch(rec.signals[0], FS, nperseg=4096)
    assert 20 <= f[np.argmax(pxx)] <= 300


def test_long_line_radiates_configured_acoustic_energy():
    """Energy flux through a cylinder around a long straight line equals eta * E_l per meter."""
    rho = 300.0
    line = polyline([[-6000, 0, 1000], [6000, 0, 1000]])
    rec = synth(line, [0, rho, 1000])
    flux = 2 * math.pi * rho * (rec.signals[0] ** 2).sum() / FS / (ATM.density(1000.0) * C0)
    assert flux == pytest.approx(ACOUSTIC_EFFICIENCY * E_L, rel=0.03)


def test_amplitude_scales_with_sqrt_efficiency():
    q1 = line_amplitude(E_L, 1.2, T_PULSE, 0.001)
    q4 = line_amplitude(E_L, 1.2, T_PULSE, 0.004)
    assert q4 / q1 == pytest.approx(2.0)


# --- strokes, discretization, ground truth ---------------------------------


def test_later_strokes_repeat_refiring_segments_only():
    pts = [[1500, 0, 2000], [1500, 0, 0]]
    one = synth(polyline(pts), [0, 0, 0]).signals[0]
    two = synth(polyline(pts, strokes=(0.0, 0.05)), [0, 0, 0]).signals[0]
    shift = int(round(0.05 * FS))
    expected = one.copy()
    expected[shift:] += one[: len(one) - shift]
    n = len(one)
    np.testing.assert_allclose(two[:n], expected, atol=1e-6 * np.abs(one).max())

    branch_only = synth(polyline(pts, strokes=(0.0, 0.05), main=False), [0, 0, 0]).signals[0]
    np.testing.assert_allclose(branch_only[:n], one, atol=1e-9 * np.abs(one).max())


def test_result_independent_of_emitter_spacing():
    """Continuous line deposit: refining the pieces should not change the signal."""
    ch = polyline([[1000, -40, 900], [1010, 40, 700], [990, 0, 300]])
    a = synth(ch, [0, 0, 1.5], spacing=1.0).signals[0]
    b = synth(ch, [0, 0, 1.5], spacing=0.1).signals[0]
    assert np.abs(a - b).max() < 0.01 * np.abs(b).max()


def test_discretize_preserves_geometry():
    ch = polyline([[0, 0, 10], [0, 0, 7.3], [3, 0, 7.3]])
    em = discretize(ch, 0.5)
    assert em.length.sum() == pytest.approx(5.7)
    assert np.all(em.length <= 0.5 + 1e-12)


def test_ground_truth_arrival_times():
    ch = generate_channel(ChannelConfig(preset="tortuous"), np.random.default_rng(5))
    mics = np.array([[0, 0, 1.5], [50, 0, 1.5]])
    rec = synth(ch, mics)
    a, b = ch.segment_endpoints()
    mid = 0.5 * (a + b)
    expected = np.linalg.norm(mid[:, None, :] - mics[None], axis=2) / C0
    assert rec.truth is not None
    np.testing.assert_allclose(rec.truth.segment_arrival_times, expected, rtol=1e-12)
    assert rec.signals.shape[0] == 2 and np.all(np.isfinite(rec.signals))
    assert rec.duration > expected.max()


# --- sub-segment (micro) tortuosity -------------------------------------------


def test_micro_tortuosity_matches_configured_turn_and_keeps_joints():
    ch = generate_channel(ChannelConfig(preset="branched"), np.random.default_rng(6))
    em = discretize(ch, 1.0, np.random.default_rng(7), math.radians(16.0), 1.0)
    v = em.end - em.start
    same = em.segment[1:] == em.segment[:-1]
    a, b = v[:-1][same], v[1:][same]
    ang = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
    assert np.degrees(ang.mean()) == pytest.approx(16.0, rel=0.03)
    # Pieces chain continuously and every original node is still on the path.
    first = np.r_[True, ~same]
    last = np.r_[~same, True]
    np.testing.assert_allclose(em.start[~first], em.end[:-1][same], atol=1e-9)
    np.testing.assert_allclose(em.start[first], ch.nodes[ch.segments[:, 0]], atol=1e-9)
    np.testing.assert_allclose(em.end[last], ch.nodes[ch.segments[:, 1]], atol=1e-9)


def test_micro_tortuosity_requires_rng():
    with pytest.raises(ValueError):
        discretize(polyline([[0, 0, 10], [0, 0, 0]]), 0.5, None, 0.2)


def test_micro_tortuosity_fills_end_on_gaps_but_keeps_directivity():
    """End-on, a straight segment cancels in its interior; micro-tortuosity breaks the cancellation."""
    seg = polyline([[-50, 0, 500], [50, 0, 500]])
    mics = np.array([[0, 2000, 0], [2000, 0, 0]])
    straight = synth(seg, mics)
    rough = synthesize(
        seg, ATM, mics, FS, OS, 0.5, ACOUSTIC_EFFICIENCY, np.random.default_rng(8), math.radians(16.0), 1.0
    )
    near = np.linalg.norm(np.array([50, 0, 500]) - mics[1]) / C0
    far = np.linalg.norm(np.array([-50, 0, 500]) - mics[1]) / C0
    interior = slice(int((near + 3 * T_PULSE) * FS), int((far - 3 * T_PULSE) * FS))
    rms_s, rms_r = (np.sqrt(np.mean(r.signals[1][interior] ** 2)) for r in (straight, rough))
    assert rms_r > 30 * rms_s
    e_r = (rough.signals**2).sum(axis=1)
    assert e_r[0] / e_r[1] > 100  # broadside still dominates


def test_channel_entirely_in_shadow_gives_silent_recording_not_an_error():
    """A low source 15 km away, heard against a 10 m/s headwind with a 10 K/km lapse: no ray
    reaches the array. That is a real outcome (E5), so the chain must run and report nothing."""
    from thunder.atmosphere.profiles import build_atmosphere
    from thunder.config import AtmosphereConfig, ReconstructionConfig, SensorsConfig
    from thunder.recon import reconstruct
    from thunder.sensors.corruption import corrupt
    from thunder.types import MicArray

    atm = build_atmosphere(
        AtmosphereConfig(
            model="stratified", lapse_rate_k_per_km=10.0, wind={"speed_mps": 10.0, "direction_from_deg": 90.0}
        )
    )
    ch = Channel(
        np.array([[-15000.0, 0.0, 300.0], [-15000.0, 0.0, 50.0]]),
        np.array([[0, 1]]),
        np.array([1e6]),
        np.array([0]),
        np.array([True]),
        np.array([False]),
    )
    array = MicArray.ideal(
        np.array([[0.0, 0.0, 1.5], [20.0, 0.0, 1.5], [0.0, 20.0, 1.5], [-20.0, -20.0, 1.5]])
    )
    rec = synthesize(ch, atm, array, FS, 8, 0.5, ACOUSTIC_EFFICIENCY)
    assert not np.any(rec.signals)
    assert rec.signals.shape[1] / FS >= 15000.0 / 350.0  # as long as the sound would have taken
    assert np.all(np.isnan(rec.truth.segment_arrival_times))
    noisy = corrupt(rec, array, SensorsConfig(noise={"background_db_spl": 45.0}), np.random.default_rng(0))
    assert np.any(noisy.signals)
    r = reconstruct(noisy, array.nominal_positions, atm, ReconstructionConfig(method="B"))
    assert r.n_points == 0

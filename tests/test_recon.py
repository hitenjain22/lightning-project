import math
from itertools import combinations

import numpy as np
import pytest

from thunder.acoustics.synth import synthesize
from thunder.channel.generator import generate_channel
from thunder.config import ChannelConfig, EvaluationConfig, ReconstructionConfig
from thunder.constants import ACOUSTIC_EFFICIENCY, ZERO_CELSIUS_K, sound_speed_dry
from thunder.eval.metrics import evaluate
from thunder.recon import reconstruct
from thunder.recon.postprocess import dbscan_inliers, mst_edges, prune_spurs
from thunder.recon.preprocess import active_frames, bandpass, frame, window_energy
from thunder.recon.tdoa import Correlator, MatchedCorrelator, gcc_phat, gcc_phat_matched, solve_direction
from thunder.sensors.arrays import mast, square
from thunder.types import Channel, MicArray, Recording, UniformAtmosphere

FS = 8000.0
T_AIR = ZERO_CELSIUS_K + 25.0
C0 = sound_speed_dry(T_AIR)
ATM = UniformAtmosphere(C0, T_AIR)
ARRAY = MicArray.ideal(square(50.0, 1.5, center=True))
CFG = ReconstructionConfig()


def _noise(n=40000, seed=0):
    x = np.random.default_rng(seed).standard_normal(n)
    return bandpass(x, FS, (10.0, 300.0), 4)[0]


def _delay(x, d):
    n = len(x)
    f = np.fft.rfftfreq(2 * n, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, 2 * n) * np.exp(-2j * np.pi * f * d / FS), 2 * n)[:n]


def _polyline(points) -> Channel:
    pts = np.asarray(points, dtype=float)
    n = len(pts) - 1
    return Channel(
        pts,
        np.c_[np.arange(n), np.arange(1, n + 1)],
        np.full(n, 1e6),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )


# --- GCC-PHAT -------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.0, 7.31, -42.86, 120.5])
def test_matched_gcc_recovers_fractional_delay(d):
    x = _noise()
    y = _delay(x, d)
    corr = MatchedCorrelator.build(800, FS, (10.0, 300.0), 1.0)
    errs = []
    for start in range(4000, 34000, 2500):
        coarse = int(round(d))
        lag, peak = gcc_phat_matched(x, y, start, start + coarse, -40, 40, corr)
        errs.append(coarse + lag - d)
        assert peak > 0.9
    assert np.max(np.abs(errs)) < 0.02


def test_coarse_gcc_sign_and_lag_limit():
    x = _noise(seed=1)
    y = _delay(x, 25.0)  # y hears the sound 25 samples later
    corr = Correlator.build(800, 100, FS, (10.0, 300.0))
    lag, _ = gcc_phat(x, y, 20000, 100, corr)
    assert lag == pytest.approx(25.0, abs=1.5)
    lag_limited, _ = gcc_phat(x, y, 20000, 10, corr)  # true lag outside the physical limit
    assert abs(lag_limited) <= 10.0


# --- Constrained direction solver ---------------------------------------------------


@pytest.mark.parametrize("mics", [square(50.0, 1.5, center=True), mast(50.0, 1.5, 10.0), square(20.0, 1.5)])
def test_solver_exact_for_plane_waves(mics):
    pairs = list(combinations(range(len(mics)), 2))
    diffs = np.array([mics[j] - mics[i] for i, j in pairs])
    rng = np.random.default_rng(2)
    for _ in range(200):
        el, az = rng.uniform(0.0, 1.5), rng.uniform(-np.pi, np.pi)
        u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        fit = solve_direction(diffs, -(diffs @ u) / C0, np.ones(len(pairs)), C0, 1e-6)
        assert fit.u @ u == pytest.approx(1.0, abs=1e-12)
        assert fit.free_ratio == pytest.approx(1.0 if np.ptp(mics[:, 2]) > 0 else math.cos(el), abs=1e-9)
        assert fit.planar == (np.ptp(mics[:, 2]) == 0)


def test_solver_covariance_matches_monte_carlo():
    mics = mast(50.0, 1.5, 10.0)
    pairs = list(combinations(range(len(mics)), 2))
    diffs = np.array([mics[j] - mics[i] for i, j in pairs])
    u = np.array([math.cos(0.4) * math.cos(0.7), math.cos(0.4) * math.sin(0.7), math.sin(0.4)])
    tau0 = -(diffs @ u) / C0
    sigma = 2e-4
    rng = np.random.default_rng(3)
    us, covs = [], []
    for _ in range(3000):
        # Negligible floor: test the covariance formula itself, with sigma estimated from residuals.
        fit = solve_direction(diffs, tau0 + rng.normal(0, sigma, len(tau0)), np.ones(len(tau0)), C0, 1e-12)
        us.append(fit.u)
        covs.append(fit.cov_u)
    emp = np.cov(np.array(us).T)
    mean_cov = np.mean(covs, axis=0)
    assert np.trace(mean_cov) == pytest.approx(np.trace(emp), rel=0.08)
    np.testing.assert_allclose(mean_cov, emp, atol=0.1 * np.trace(emp))


def test_planar_solver_points_up_and_clamps_noisy_slowness():
    mics = square(50.0, 1.5, center=True)
    pairs = list(combinations(range(len(mics)), 2))
    diffs = np.array([mics[j] - mics[i] for i, j in pairs])
    u = np.array([1.0, 0.0, 0.0])  # horizontal arrival
    tau = -(diffs @ u) / C0 * 1.05  # 5% too-large horizontal slowness (noise / wrong c)
    fit = solve_direction(diffs, tau, np.ones(len(pairs)), C0, 1e-6)
    assert np.linalg.norm(fit.u) == pytest.approx(1.0)
    assert fit.free_ratio == pytest.approx(1.05, rel=1e-9)
    assert fit.u[2] == pytest.approx(0.0, abs=1e-9)


# --- Method A ---------------------------------------------------------------------


def test_point_source_direction_and_range():
    """SPEC.md: single point source, ideal conditions: direction within 0.5 deg, range within 1%."""
    src = np.array([1500.0, -900.0, 700.0])
    ch = _polyline([src, src + [0, 0, 0.05]])
    rec = synthesize(ch, ATM, ARRAY, FS, 8, 0.5, ACOUSTIC_EFFICIENCY)
    r = reconstruct(rec, ARRAY.nominal_positions, ATM, CFG.model_copy(update={"dbscan_min_samples": 1}))
    assert r.n_points >= 1
    best = r.points[np.argmax(r.quality)]
    cen = ARRAY.nominal_positions.mean(axis=0)
    u_true = (src - cen) / np.linalg.norm(src - cen)
    u_est = (best - cen) / np.linalg.norm(best - cen)
    assert math.degrees(math.acos(min(1.0, u_true @ u_est))) < 0.5
    assert np.linalg.norm(best - cen) == pytest.approx(np.linalg.norm(src - cen), rel=0.01)


def test_round_trip_ideal_conditions():
    """Fast round trip (runs on every commit): generate -> synthesize -> reconstruct -> evaluate."""
    ch = generate_channel(
        ChannelConfig(preset="tortuous", strike_distance_m=(2000, 2000)), np.random.default_rng(7)
    )
    rec = synthesize(
        ch, ATM, ARRAY, FS, 8, 0.5, ACOUSTIC_EFFICIENCY, np.random.default_rng(8), math.radians(16), 1.0
    )
    r = reconstruct(rec, ARRAY.nominal_positions, ATM, CFG)
    m, _ = evaluate(r.points, r.covariances, ch, ARRAY.nominal_positions.mean(axis=0), EvaluationConfig())
    assert m["n_points"] > 50
    assert m["point_error_median_m"] < 10.0
    assert m["coverage_main_50m"] > 0.6


def test_reported_t0_shifts_range():
    ch = generate_channel(
        ChannelConfig(preset="tortuous", strike_distance_m=(1500, 1500)), np.random.default_rng(9)
    )
    rec = synthesize(ch, ATM, ARRAY, FS, 8, 0.5, ACOUSTIC_EFFICIENCY)
    late = Recording(rec.signals, FS, rec.nominal_mic_positions, 0.01)
    a = reconstruct(rec, ARRAY.nominal_positions, ATM, CFG)
    b = reconstruct(late, ARRAY.nominal_positions, ATM, CFG)
    cen = ARRAY.nominal_positions.mean(axis=0)
    ra = np.linalg.norm(a.extra["window_points"] - cen, axis=1)
    rb = np.linalg.norm(b.extra["window_points"] - cen, axis=1)
    np.testing.assert_allclose(ra - rb, C0 * 0.01, rtol=1e-9)


def test_silent_recording_gives_empty_reconstruction():
    rec = Recording(np.zeros((5, 16000)), FS, ARRAY.nominal_positions, 0.0)
    r = reconstruct(rec, ARRAY.nominal_positions, ATM, CFG)
    assert r.n_points == 0 and r.points.shape == (0, 3)


# --- preprocessing and post-processing -----------------------------------------


def test_bandpass_is_zero_phase():
    x = np.zeros(8000)
    x[4000] = 1.0
    y = bandpass(x, FS, (10.0, 300.0), 4)[0]
    t = np.arange(8000)
    assert (t * y**2).sum() / (y**2).sum() == pytest.approx(4000.0, abs=0.01)


def test_active_frames_span_first_to_last_loud_window():
    x = np.zeros((1, 8000))
    x[0, 2000:2100] = 1.0
    x[0, 6000:6100] = 0.5
    fr = frame(8000, FS, 0.05, 0.5)
    act = active_frames(window_energy(x, fr), 80.0, 10.0)
    t_lo, t_hi = fr.starts[act[0]], fr.starts[act[-1]] + fr.length
    assert t_lo <= 2000 and t_hi >= 6100
    assert len(active_frames(np.zeros(5), 40.0, 10.0)) == 0


def test_dbscan_and_skeleton():
    line = np.column_stack([np.zeros(50), np.zeros(50), np.arange(50) * 20.0])
    outlier = np.array([[5000.0, 0, 0]])
    pts = np.vstack([line, outlier])
    keep = dbscan_inliers(pts, 100.0, 3)
    assert keep[:50].all() and not keep[50]
    spur = np.array([[10.0, 0, 500.0], [20.0, 0, 500.0]])  # 20 m spur off the middle of the line
    p = np.vstack([line, spur])
    edges = mst_edges(p)
    assert len(edges) == len(p) - 1
    pruned = prune_spurs(p, edges, 50.0)
    assert len(pruned) == 49  # the line alone
    assert len(prune_spurs(p, edges, 5.0)) == len(edges)  # spur longer than 5 m survives


def test_bandpass_does_not_shift_synthetic_arrivals():
    ch = _polyline([[800.0, 0, 300.0], [800.0, 0, 299.95]])
    rec = synthesize(ch, ATM, ARRAY.nominal_positions[:1], FS, 8, 0.5, ACOUSTIC_EFFICIENCY)
    y = bandpass(rec.signals, FS, (10.0, 300.0), 4)[0]
    t = np.arange(len(y)) / FS
    c_raw = (t * rec.signals[0] ** 2).sum() / (rec.signals[0] ** 2).sum()
    c_bp = (t * y**2).sum() / (y**2).sum()
    assert c_bp == pytest.approx(c_raw, abs=2e-4)

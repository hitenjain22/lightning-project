"""Methods B (SRP-PHAT) and C (multilateration), and the beta-PHAT normalization they share with A."""

import math

import numpy as np
import pytest

from thunder.acoustics.synth import synthesize
from thunder.atmosphere.profiles import build_atmosphere
from thunder.channel.generator import generate_channel
from thunder.config import AtmosphereConfig, ChannelConfig, EvaluationConfig, ReconstructionConfig
from thunder.constants import ACOUSTIC_EFFICIENCY
from thunder.eval.metrics import evaluate
from thunder.recon import reconstruct
from thunder.recon.multilat import multilaterate, relative_delays
from thunder.recon.preprocess import bandpass
from thunder.recon.tdoa import MatchedCorrelator, gcc_phat_matched, prepare
from thunder.sensors.arrays import distributed, square
from thunder.types import Channel, MicArray, Recording, UniformAtmosphere

FS = 8000.0
ATM = UniformAtmosphere(343.0, 293.15)
ARRAY = MicArray.ideal(square(50.0, 1.5, center=True))
CEN = ARRAY.nominal_positions.mean(axis=0)


def point_sources(points) -> Channel:
    nodes = np.vstack([[p, np.asarray(p) + [0, 0, 0.05]] for p in points])
    n = len(points)
    return Channel(
        nodes,
        np.arange(2 * n).reshape(n, 2),
        np.full(n, 1e6),
        np.arange(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )


def toward(az_deg, el_deg, r, origin=CEN):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return origin + r * np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def angle_deg(u, v):
    return math.degrees(math.acos(min(1.0, float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))))


# --- beta-PHAT normalization ------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.6, 1.0])
def test_phat_peak_is_one_for_identical_signals_at_any_level(beta):
    x = bandpass(np.random.default_rng(0).standard_normal(40000), FS, (10.0, 300.0), 4)[0]
    corr = MatchedCorrelator.build(800, FS, (10.0, 300.0), beta)
    for scale in (1e-3, 1.0, 1e4):
        lag, peak = gcc_phat_matched(scale * x, scale * x, 20000, 20000, -20, 20, corr)
        assert peak == pytest.approx(1.0, abs=1e-9)
        assert lag == pytest.approx(0.0, abs=1e-9)


# --- Method B ---------------------------------------------------------------------


def test_method_b_finds_two_simultaneous_sources():
    """SPEC.md gate: two sources at different directions in the same window; B finds both."""
    sources = [toward(20, 15, 2000.0), toward(110, 30, 2010.0)]
    rec = synthesize(point_sources(sources), ATM, ARRAY, FS, 8, 0.5, 2 * ACOUSTIC_EFFICIENCY)
    cfg = ReconstructionConfig(method="B", dbscan_min_samples=1)
    r = reconstruct(rec, ARRAY.nominal_positions, ATM, cfg)
    assert r.n_points == 2
    for s in sources:
        best = min(r.points, key=lambda p, s=s: angle_deg(p - CEN, s - CEN))
        assert angle_deg(best - CEN, s - CEN) < 0.5
        assert np.linalg.norm(best - CEN) == pytest.approx(np.linalg.norm(s - CEN), rel=0.01)
    # Method A fits one direction per window and so cannot represent this window.
    a = reconstruct(rec, ARRAY.nominal_positions, ATM, ReconstructionConfig(method="A", dbscan_min_samples=1))
    assert a.n_points < 2


@pytest.mark.parametrize("method", ["B", "C"])
def test_point_source_direction_and_range(method):
    """SPEC.md (Method A criterion, applied to B and C): direction within 0.5 deg, range within 1%."""
    src = toward(250, 22, 2300.0)
    rec = synthesize(point_sources([src]), ATM, ARRAY, FS, 8, 0.5, ACOUSTIC_EFFICIENCY)
    r = reconstruct(
        rec, ARRAY.nominal_positions, ATM, ReconstructionConfig(method=method, dbscan_min_samples=1)
    )
    assert r.n_points >= 1
    best = r.points[np.argmax(r.quality)]
    assert angle_deg(best - CEN, src - CEN) < 0.5
    assert np.linalg.norm(best - CEN) == pytest.approx(np.linalg.norm(src - CEN), rel=0.01)


# --- Method C ---------------------------------------------------------------------


def test_relative_delays_from_consistent_pairs():
    rec = Recording(np.zeros((5, 8000)), FS, ARRAY.nominal_positions, 0.0)
    st = prepare(rec, ARRAY.nominal_positions, ATM, ReconstructionConfig())
    d_true = np.array([0.013, -0.02, 0.0, 0.031, -0.007])
    d_true -= d_true[st.ref]
    tau = np.array([d_true[j] - d_true[i] for i, j in st.pairs])
    d, misfit = relative_delays(st, tau, np.ones(len(tau)))
    np.testing.assert_allclose(d, d_true, atol=1e-12)
    assert misfit < 1e-12


def test_multilateration_converges_through_wind():
    atm = build_atmosphere(
        AtmosphereConfig(model="stratified", wind={"speed_mps": 8.0, "shear_deg_per_km": 15.0})
    )
    mics = ARRAY.nominal_positions
    truth = np.array([toward(30, 20, 2500.0), toward(200, 35, 1800.0), toward(300, 10, 3000.0)])
    t_obs = np.stack([atm.propagate(truth, m).travel_time for m in mics], axis=1)
    start = truth + np.random.default_rng(1).normal(0, 40.0, truth.shape)
    x, cov, rms, ok = multilaterate(start, t_obs, mics, atm, ReconstructionConfig(method="C"))
    assert ok.all()
    np.testing.assert_allclose(x, truth, atol=0.05)
    assert np.all(rms < 1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_method_c_corrects_near_field_bias_of_plane_wave():
    """A 300 m distributed array 900 m from a source: wavefront curvature biases plane-wave fits;
    absolute-time multilateration models it exactly."""
    arr = MicArray.ideal(distributed("triangle", 3, 20.0, 300.0))
    cen = arr.nominal_positions.mean(axis=0)
    src = toward(60, 25, 900.0, cen)
    rec = synthesize(point_sources([src]), ATM, arr, FS, 8, 0.5, ACOUSTIC_EFFICIENCY)
    a = reconstruct(rec, arr.nominal_positions, ATM, ReconstructionConfig(method="A", dbscan_min_samples=1))
    c = reconstruct(rec, arr.nominal_positions, ATM, ReconstructionConfig(method="C", dbscan_min_samples=1))
    assert c.n_points >= 1
    err_c = float(np.min(np.linalg.norm(c.points - src, axis=1)))
    assert err_c < 1.0
    # The plane-wave fit cannot explain the curved wavefront: its residual gate rejects the
    # window (no point), or whatever it places is far worse than C.
    if a.n_points:
        assert float(np.min(np.linalg.norm(a.points - src, axis=1))) > 3 * err_c


# --- round trips and mismatched atmospheres -------------------------------------------


@pytest.fixture(scope="module")
def ideal_bolt():
    ch = generate_channel(
        ChannelConfig(preset="branched", strike_distance_m=(2000, 2000)), np.random.default_rng(12)
    )
    rec = synthesize(
        ch, ATM, ARRAY, FS, 8, 0.5, ACOUSTIC_EFFICIENCY, np.random.default_rng(13), math.radians(16), 1.0
    )
    return ch, rec


def test_round_trip_all_methods(ideal_bolt):
    ch, rec = ideal_bolt
    metrics = {}
    for method in ("A", "B", "C"):
        r = reconstruct(rec, ARRAY.nominal_positions, ATM, ReconstructionConfig(method=method))
        m, _ = evaluate(r.points, r.covariances, ch, CEN, EvaluationConfig(), r.extra["strike_point"])
        assert m["n_points"] > 50, method
        assert m["point_error_median_m"] < 10.0, method
        metrics[method] = m
    assert metrics["B"]["n_points"] >= metrics["A"]["n_points"]  # B also keeps secondary sources


def test_mismatched_atmosphere_runs_work():
    """SPEC.md M6 gate: every method runs against a mismatched assumed atmosphere."""
    true_cfg = AtmosphereConfig(model="stratified", wind={"speed_mps": 6.0, "direction_from_deg": 250.0})
    truth = build_atmosphere(true_cfg)
    ch = generate_channel(
        ChannelConfig(preset="tortuous", strike_distance_m=(2000, 2000), start_height_m=(2500, 2500)),
        np.random.default_rng(14),
    )
    rec = synthesize(
        ch, truth, ARRAY, FS, 8, 0.5, ACOUSTIC_EFFICIENCY, np.random.default_rng(15), math.radians(16), 1.0
    )
    mismatched = build_atmosphere(AtmosphereConfig(model="stratified", lapse_rate_k_per_km=5.0))
    for method in ("A", "B", "C"):
        cfg = ReconstructionConfig(method=method)
        oracle = reconstruct(rec, ARRAY.nominal_positions, truth, cfg)
        m_o, _ = evaluate(oracle.points, oracle.covariances, ch, CEN, EvaluationConfig())
        assert m_o["point_error_median_m"] < 10.0, method  # refraction undone with the true atmosphere
        wrong = reconstruct(rec, ARRAY.nominal_positions, mismatched, cfg)
        m_w, _ = evaluate(wrong.points, wrong.covariances, ch, CEN, EvaluationConfig())
        assert m_w["n_points"] > 20 and np.isfinite(m_w["point_error_median_m"]), method
        assert m_w["point_error_median_m"] > m_o["point_error_median_m"], method  # wind unknown

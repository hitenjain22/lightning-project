"""Method D: forward model, priors, uncertainty calibration (the M8 gate), MCMC reference,
and storm self-calibration end to end."""

import math

import numpy as np
import pytest

from thunder.acoustics.synth import synthesize
from thunder.atmosphere.profiles import build_atmosphere
from thunder.config import AtmosphereConfig, EvaluationConfig, ReconstructionConfig
from thunder.constants import ACOUSTIC_EFFICIENCY
from thunder.eval.metrics import EXPECTED_COVERAGE, calibration, evaluate
from thunder.recon import reconstruct
from thunder.recon.bayes import (
    NoiseModel,
    Problem,
    conditional_cov,
    effective_times,
    infer,
    mcmc_window,
    prior_from_atmosphere,
    reconstruct_bayes_storm,
)
from thunder.sensors.arrays import mast, square
from thunder.types import Channel, MicArray

PRIOR_MEAN = np.array([347.0, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0])
PRIOR_STD = np.array([5.0, 2e-3, 8.0, 8.0, 2e-3, 2e-3, 1e-3])
CFG = ReconstructionConfig(method="D")


# --- forward model ----------------------------------------------------------------------


def test_effective_medium_solves_its_equation_and_derivatives_match():
    rng = np.random.default_rng(0)
    mics = rng.uniform(-25, 25, (5, 3))
    mics[:, 2] = 1.5
    x = np.array([[1200.0, -800, 2500], [300, 2000, 900]])
    atm = np.array([340.0, -0.004, 3.0, -2.0, 0.002, 0.001])
    T, jx, jatm = effective_times(x, mics, atm)
    h = x[:, 2:3]
    c = atm[0] + atm[1] * h
    w = np.zeros((2, 1, 3))
    w[..., 0], w[..., 1] = atm[2] + atm[4] * h, atm[3] + atm[5] * h
    d = mics[None] - x[:, None]
    np.testing.assert_allclose(np.linalg.norm(d - w * T[..., None], axis=-1), c * T, rtol=1e-12)
    e = 1e-4

    def t_at(xx):
        return effective_times(xx, mics, atm)[0]

    num = np.stack(
        [(t_at(x + e * np.eye(3)[i]) - t_at(x - e * np.eye(3)[i])) / (2 * e) for i in range(3)], axis=-1
    )
    np.testing.assert_allclose(jx, num, rtol=1e-6, atol=1e-12)
    for i, step in enumerate([1e-3, 1e-7, 1e-3, 1e-3, 1e-7, 1e-7]):
        dv = np.zeros(6)
        dv[i] = step
        num_i = (effective_times(x, mics, atm + dv)[0] - effective_times(x, mics, atm - dv)[0]) / (2 * step)
        np.testing.assert_allclose(jatm[..., i], num_i, rtol=1e-5, atol=1e-12)


def test_effective_medium_limits():
    mics = square(50.0, 1.5, center=True)
    x = np.array([[1000.0, 2000.0, 1500.0]])
    T, _, _ = effective_times(x, mics, np.array([343.0, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(T, np.linalg.norm(mics - x, axis=1)[None] / 343.0, rtol=1e-14)
    # Wind straight from source to mic: sound arrives at c + w.
    Tw, _, _ = effective_times(
        np.array([[-3000.0, 0, 1e-4]]), np.array([[0.0, 0, 1e-4]]), np.array([340.0, 0, 10, 0, 0, 0])
    )
    assert Tw[0, 0] == pytest.approx(3000.0 / 350.0, rel=1e-12)


def test_prior_from_atmosphere():
    cfg = ReconstructionConfig(method="D")
    uni = prior_from_atmosphere(build_atmosphere(AtmosphereConfig(model="uniform", temperature_c=25.0)), cfg)[
        0
    ]
    assert uni[1:] == pytest.approx(np.zeros(5), abs=1e-12)
    strat = build_atmosphere(
        AtmosphereConfig(
            model="stratified", lapse_rate_k_per_km=6.5, wind={"speed_mps": 5.0, "direction_from_deg": 270.0}
        )
    )
    m = prior_from_atmosphere(strat, cfg)[0]
    assert -3e-3 < m[1] < -1e-3  # path-averaged sound speed falls ~2 m/s per km of source height
    assert m[2] > 3.0 and abs(m[3]) < 1e-6  # west wind blows toward +x (east)
    assert m[4] > 0  # and grows with height


# --- uncertainty calibration (M8 gate) ---------------------------------------------------


def synthetic_coverage(mics, self_calibrate: bool, trials: int, k: int, seed: int):
    """Coverage of point ellipsoids and theta z-scores on observations drawn exactly from the
    likelihood, with the medium drawn from the prior (a Bayesian consistency check)."""
    rng = np.random.default_rng(seed)
    free = np.full(7, self_calibrate)

    def model(xx, atm, dm):
        return effective_times(xx, mics + dm, atm)

    cover, z = [], []
    for _ in range(trials):
        theta = PRIOR_MEAN + np.where(free, PRIOR_STD, 0.0) * rng.standard_normal(7)
        az, el, r = (
            rng.uniform(0, 2 * np.pi, k),
            np.radians(rng.uniform(5, 60, k)),
            rng.uniform(1000, 4000, k),
        )
        x = np.column_stack([r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az), r * np.sin(el)])
        a = rng.uniform(30e-6, 150e-6, k) ** 2
        b = rng.uniform(1e-3, 20e-3, k) ** 2 + (0.0 if self_calibrate else PRIOR_STD[6] ** 2)
        T, _, _ = effective_times(x, mics, theta)
        noise = (
            rng.standard_normal((k, len(mics))) * np.sqrt(a)[:, None]
            + rng.standard_normal(k)[:, None] * np.sqrt(b)[:, None]
        )
        t_obs = T + (theta[6] if self_calibrate else 0.0) + noise
        mean = PRIOR_MEAN if self_calibrate else np.r_[theta[:6], 0.0]
        pb = Problem(t_obs, NoiseModel(a, b), np.zeros(k, dtype=int), model, mean, PRIOR_STD, free)
        fit, inl = infer(pb, x + rng.normal(0, 30, x.shape), CFG)
        cover.append(calibration(fit.x[inl] - x[inl], fit.cov_x[inl]))
        if self_calibrate:
            z.append((fit.theta - theta) / np.sqrt(np.diag(fit.cov_theta)))
    return np.mean(cover, axis=0), (np.sqrt(np.mean(np.square(z), axis=0)) if z else None)


def test_calibration_fixed_atmosphere_planar_array():
    """M8 gate: 1/2/3-sigma coverage nominal (19.9 / 73.9 / 97.1%) with a known medium."""
    cov, _ = synthetic_coverage(square(50.0, 1.5, center=True), False, trials=30, k=60, seed=2)
    np.testing.assert_allclose(cov, EXPECTED_COVERAGE, atol=0.03)


def test_calibration_self_calibrating_array_with_vertical_aperture():
    """M8 gate: nominal coverage of points *and* of the medium parameters (z-scores ~1) when the
    medium is estimated; vertical aperture makes the posterior close to Gaussian."""
    cov, z = synthetic_coverage(mast(50.0, 1.5, 10.0), True, trials=25, k=60, seed=1)
    np.testing.assert_allclose(cov, EXPECTED_COVERAGE, atol=0.03)
    assert np.all((z > 0.7) & (z < 1.35))


@pytest.mark.slow
def test_calibration_self_calibrating_planar_array_known_tail():
    """Planar array + self-calibration: the core is calibrated, the 3-sigma tail is light
    (~0.91 vs 0.971) because elevation near the horizon depends nonlinearly on c (documented)."""
    cov, z = synthetic_coverage(square(50.0, 1.5, center=True), True, trials=30, k=60, seed=0)
    np.testing.assert_allclose(cov[:2], EXPECTED_COVERAGE[:2], atol=0.04)
    assert 0.85 < cov[2] < 0.95
    assert np.all((z > 0.6) & (z < 1.5))


def test_mcmc_reference_agrees_with_laplace():
    mics = square(50.0, 1.5, center=True)
    rng = np.random.default_rng(3)
    x = np.array([[800.0, 1500.0, 1200.0]])
    theta = np.r_[PRIOR_MEAN[:6], 0.0]
    a, b = np.array([(80e-6) ** 2]), np.array([(5e-3) ** 2])
    T, _, _ = effective_times(x, mics, theta)
    t_obs = T + rng.standard_normal((1, 5)) * 80e-6

    def model(xx, atm, dm):
        return effective_times(xx, mics + dm, atm)

    pb = Problem(t_obs, NoiseModel(a, b), np.zeros(1, dtype=int), model, theta, PRIOR_STD, np.zeros(7, bool))
    fit, _ = infer(pb, x, CFG)
    samples = mcmc_window(pb, 0, fit.x[0], fit.theta, rng, walkers=16, steps=1000)
    lap = conditional_cov(pb, fit.x, fit.theta)[0]
    np.testing.assert_allclose(np.sqrt(np.diag(np.cov(samples.T))), np.sqrt(np.diag(lap)), rtol=0.2)


# --- end to end ------------------------------------------------------------------------


def point_channel(points) -> Channel:
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


@pytest.mark.slow
def test_storm_self_calibration_recovers_unknown_wind():
    """Bolts at four azimuths through a windy stratified atmosphere, reconstructed assuming
    still air: storm self-calibration removes most of the wind error and recovers the wind."""
    truth = build_atmosphere(
        AtmosphereConfig(
            model="stratified", lapse_rate_k_per_km=0.0, wind={"speed_mps": 6.0, "direction_from_deg": 270.0}
        )
    )
    still = build_atmosphere(AtmosphereConfig(model="uniform", temperature_c=25.0))
    arr = MicArray.ideal(square(50.0, 1.5, center=True))
    rng = np.random.default_rng(5)
    recs, sources = [], []
    for az in (0.0, 90.0, 180.0, 270.0):
        pts = []
        for t, h in enumerate(np.linspace(300.0, 4000.0, 16)):
            a = math.radians(az + rng.uniform(-5, 5))
            r = 2000.0 + 40.0 * t
            pts.append([r * math.sin(a), r * math.cos(a), h])
        pts = np.array(pts)
        sources.append(pts)
        recs.append(synthesize(point_channel(pts), truth, arr, 8000.0, 8, 0.5, ACOUSTIC_EFFICIENCY))
    cfg = ReconstructionConfig(method="D", dbscan_min_samples=1)

    def error(rs) -> float:
        errs = [
            np.min(np.linalg.norm(r.points[:, None] - s[None], axis=2), axis=1)
            for r, s in zip(rs, sources, strict=True)
        ]
        return float(np.median(np.concatenate(errs)))

    fixed = [
        reconstruct(r, arr.nominal_positions, still, cfg.model_copy(update={"d_self_calibrate": False}))
        for r in recs
    ]
    storm = reconstruct_bayes_storm(recs, arr.nominal_positions, still, cfg)
    assert error(fixed) > 50.0  # 6 m/s of unknown wind
    assert error(storm) < error(fixed) / 2.5  # bolts at four azimuths constrain the full wind
    w0 = storm[0].extra["theta"][2:4]
    assert w0[0] > 3.0 and abs(w0[1]) < 0.2 * w0[0]  # blowing toward +x (east), as the truth


def test_method_d_round_trip_in_still_air():
    from thunder.channel.generator import generate_channel
    from thunder.config import ChannelConfig
    from thunder.types import UniformAtmosphere

    atm = UniformAtmosphere(343.0, 293.15)
    arr = MicArray.ideal(square(50.0, 1.5, center=True))
    cen = arr.nominal_positions.mean(axis=0)
    ch = generate_channel(
        ChannelConfig(preset="branched", strike_distance_m=(2000, 2000)), np.random.default_rng(12)
    )
    rec = synthesize(
        ch, atm, arr, 8000.0, 8, 0.5, ACOUSTIC_EFFICIENCY, np.random.default_rng(13), math.radians(16), 1.0
    )
    r = reconstruct(rec, arr.nominal_positions, atm, ReconstructionConfig(method="D", d_self_calibrate=False))
    m, _ = evaluate(r.points, r.covariances, ch, cen, EvaluationConfig(), r.extra["strike_point"])
    assert m["n_points"] > 50 and m["point_error_median_m"] < 10.0
    # Self-calibrating from one bolt: the cross-wind is unobservable, so its prior uncertainty
    # (+-8 m/s) stays in the answer. The estimate may drift within it; the error bars must say so.
    r = reconstruct(rec, arr.nominal_positions, atm, ReconstructionConfig(method="D"))
    m, _ = evaluate(r.points, r.covariances, ch, cen, EvaluationConfig(), r.extra["strike_point"])
    th, sd = r.extra["theta"], np.sqrt(np.diag(r.extra["theta_cov"]))
    assert np.all(np.abs(th[2:4]) < 2 * sd[2:4])  # wind consistent with the (true) still air
    assert sd[2:4].max() > 4.0  # and honestly uncertain
    assert m["calib_2sigma"] > 0.7 and m["point_error_median_m"] < 50.0


def test_array_calibration_absorbs_mic_offsets():
    """Mic position errors are the same in every window: pooled over many windows they bias the
    shared medium unless modeled. Array calibration recovers them and the wind."""
    rng = np.random.default_rng(0)
    mics = square(50.0, 1.5, center=True)
    m, n_groups = len(mics), 4
    dm_true = rng.normal(0, 0.02, (m, 3))
    atm_true = np.array([347.0, -0.002, 3.0, -2.0, 0.0005, 0.0])
    xs, groups = [], []
    for g, az0 in enumerate((0, 90, 180, 270)):
        k = 40
        az, el = np.radians(az0 + rng.uniform(-10, 10, k)), np.radians(rng.uniform(5, 60, k))
        r = rng.uniform(1000, 4000, k)
        xs.append(np.column_stack([r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az), r * np.sin(el)]))
        groups.append(np.full(k, g))
    x, group = np.vstack(xs), np.concatenate(groups)
    T, _, _ = effective_times(x, mics + dm_true, atm_true)
    a, b = np.full(len(x), (5e-6) ** 2), np.full(len(x), (1e-3) ** 2)
    t_obs = T + rng.standard_normal(T.shape) * 5e-6 + rng.standard_normal(len(x))[:, None] * 1e-3

    def model(xx, atm, dm):
        return effective_times(xx, mics + dm, atm)

    pm = np.r_[PRIOR_MEAN[:6], np.zeros(n_groups), np.zeros(m), np.zeros(3 * m)]
    ps = np.r_[PRIOR_STD[:6], np.full(n_groups, 1e-3), np.full(m, 1e-9), np.full(3 * m, 0.02)]
    pb = Problem(t_obs, NoiseModel(a, b), group, model, pm, ps, np.ones(len(pm), bool), array_cal=True)
    fit, inl = infer(pb, x + rng.normal(0, 20, x.shape), CFG)
    _, _, _, dm = pb.split(fit.theta)
    assert inl.mean() > 0.95
    np.testing.assert_allclose(fit.theta[2:4], atm_true[2:4], atol=0.6)
    assert np.median(np.abs(dm - dm_true)) < 0.01

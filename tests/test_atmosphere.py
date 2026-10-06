import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import integrate

from thunder import constants as C
from thunder.acoustics.synth import synthesize
from thunder.atmosphere import raytrace as rt
from thunder.atmosphere.absorption import absorption_db_per_m
from thunder.atmosphere.profiles import (
    StratifiedProfiles,
    build_atmosphere,
    saturation_vapor_pressure,
    sound_speed_moist,
)
from thunder.atmosphere.stratified import StratifiedAtmosphere
from thunder.channel.generator import generate_channel
from thunder.config import (
    AtmosphereConfig,
    ChannelConfig,
    EvaluationConfig,
    ReconstructionConfig,
    SensorsConfig,
)
from thunder.eval.metrics import evaluate
from thunder.recon import reconstruct
from thunder.sensors.arrays import square
from thunder.sensors.corruption import active_window, corrupt
from thunder.types import Channel, MicArray, UniformAtmosphere

FS = 8000.0
MIC = np.array([10.0, -5.0, 1.5])


def strat(**kw) -> StratifiedAtmosphere:
    return StratifiedAtmosphere(AtmosphereConfig(model="stratified", **kw))


def random_sources(n, seed=0, zmax=7000.0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(-8000, 8000, n), rng.uniform(-8000, 8000, n), rng.uniform(20, zmax, n)]
    )


def polyline(points, energy=1e6) -> Channel:
    pts = np.asarray(points, dtype=float)
    n = len(pts) - 1
    return Channel(
        pts,
        np.c_[np.arange(n), np.arange(1, n + 1)],
        np.full(n, energy),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )


# --- profiles -------------------------------------------------------------------


def test_pressure_matches_barometric_formula_and_standard_atmosphere():
    p = StratifiedProfiles(AtmosphereConfig(model="stratified", temperature_c=15.0, lapse_rate_k_per_km=6.5))
    z = np.array([0.0, 1000.0, 5000.0, 10000.0])
    t0, lapse = 288.15, 0.0065
    exact = C.P_STANDARD * ((t0 - lapse * z) / t0) ** (C.G_STANDARD / (C.R_DRY_AIR * lapse))
    np.testing.assert_allclose(p.pressure(z), exact, rtol=1e-8)
    assert p.pressure(5000.0) == pytest.approx(54020.0, rel=1e-4)  # U.S. Standard Atmosphere 1976


def test_humidity_and_sound_speed():
    assert saturation_vapor_pressure(293.15) == pytest.approx(2338.0, rel=0.003)  # e_s(20 C)
    assert sound_speed_moist(293.15, 0.0, 101325.0) == pytest.approx(C.sound_speed_dry(293.15), rel=1e-12)
    gain = sound_speed_moist(293.15, 0.5, 101325.0) / C.sound_speed_dry(293.15) - 1
    assert 0.001 < gain < 0.003  # humidity adds ~0.1-0.3% at 20 C


def test_inversion_and_tropopause():
    p = StratifiedProfiles(
        AtmosphereConfig(model="stratified", inversion={"base_m": 200, "top_m": 500, "delta_k": 4.0})
    )
    t = p.temperature(np.array([200.0, 500.0, 11000.0, 12000.0]))
    assert t[1] - t[0] == pytest.approx(4.0 - 0.0065 * 300)  # rises across the inversion
    assert t[3] == pytest.approx(t[2])  # isothermal above 11 km


def test_wind_profile():
    w = StratifiedProfiles(
        AtmosphereConfig(model="stratified", wind={"speed_mps": 5.0, "direction_from_deg": 270.0})
    )
    np.testing.assert_allclose(
        w.wind(np.array([10.0]))[0], [5.0, 0.0, 0.0], atol=1e-12
    )  # from west: toward +x
    sheared = StratifiedProfiles(
        AtmosphereConfig(
            model="stratified", wind={"speed_mps": 5.0, "direction_from_deg": 270.0, "shear_deg_per_km": 90.0}
        )
    )
    v = sheared.wind(np.array([1010.0]))[0]
    from_deg = math.degrees(math.atan2(-v[0], -v[1]))
    assert abs((from_deg + 180.0) % 360.0 - 180.0) < 1e-9  # veered by 90 deg: now "from north"


def test_config_validation():
    with pytest.raises(ValidationError):
        AtmosphereConfig(wind={"speed_mps": 3.0})  # wind needs a stratified model
    with pytest.raises(ValidationError):
        AtmosphereConfig(model="stratified", inversion={"base_m": 500, "top_m": 200, "delta_k": 2})


# --- absorption -----------------------------------------------------------------


@pytest.mark.parametrize(
    "t_c,rh,table",
    [
        (20, 0.7, [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]),
        (10, 0.7, [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0]),
        (15, 0.5, [0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0]),
        (15, 0.8, [0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8]),
    ],
)
def test_iso9613_matches_published_table(t_c, rh, table):
    """ISO 9613-2 Table 2 (dB/km at 63 Hz-8 kHz octave centres; values rounded to 0.1 / 3 digits)."""
    f = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000.0])
    got = absorption_db_per_m(f, 273.15 + t_c, rh, 101325.0) * 1000
    np.testing.assert_allclose(got, table, rtol=0.03, atol=0.05)


def test_path_absorption():
    u = UniformAtmosphere(343.0, 293.15, 0.7, 101325.0, absorption=True)
    f = np.array([125.0, 1000.0, 4000.0])
    exact = absorption_db_per_m(f, 293.15, 0.7, 101325.0)
    np.testing.assert_allclose(u.attenuation_db(f, [500.0], 1.5, [3000.0])[0], 3000 * exact, rtol=2e-3)
    assert not UniformAtmosphere(343.0, 293.15).attenuation_db(f, [500.0], 1.5, [3000.0]).any()


# --- ray integrals --------------------------------------------------------------


def test_layer_weights_are_exact_for_linear_numerator_and_S():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s0, s1 = rng.uniform(1e-4, 1e-2, 2)
        n0, n1 = rng.uniform(-2, 2, 2)
        r0, r1 = math.sqrt(s0), math.sqrt(s1)
        p0, p1, q0, q1 = rt.layer_weights(r0, r1, 1.0)

        def n(u, n0=n0, n1=n1):
            return n0 * (1 - u) + n1 * u

        def S(u, s0=s0, s1=s1):
            return s0 * (1 - u) + s1 * u

        assert n0 * p0 + n1 * p1 == pytest.approx(
            integrate.quad(lambda u: n(u) / S(u) ** 0.5, 0, 1)[0], rel=1e-9
        )
        assert n0 * q0 + n1 * q1 == pytest.approx(
            integrate.quad(lambda u: n(u) / S(u) ** 1.5, 0, 1)[0], rel=1e-9
        )


def test_height_grid():
    z = rt.height_nodes(12000.0, 20.0, (11000.0, 333.3))
    assert z[0] == 0.0 and z[1] == pytest.approx(0.05) and z[-1] == 12000.0
    assert 11000.0 in z and 333.3 in z
    assert np.all(np.diff(z) > 0) and np.max(np.diff(z)) <= 20.0 + 1e-9


def test_uniform_limit_is_exact():
    """SPEC.md: in uniform still air, ray-traced travel times match straight-line r/c (here to 1e-9)."""
    atm = strat(lapse_rate_k_per_km=0.0, relative_humidity=0.0)
    c = float(atm.sound_speed(np.array([0.0]))[0])
    src = random_sources(300)
    imp = np.sqrt(atm.density(np.array([MIC[2]])) / atm.density(src[:, 2]))  # rho c impedance (c constant)
    for reflected in (False, True):
        p = (atm.propagate_reflected if reflected else atm.propagate)(src, MIC)
        img = src * [1, 1, -1] if reflected else src
        R = np.linalg.norm(img - MIC, axis=1)
        assert p.valid.all()
        np.testing.assert_allclose(p.travel_time, R / c, rtol=1e-6)
        np.testing.assert_allclose(p.amplitude_factor * R / imp, 1.0, rtol=1e-5)
        np.testing.assert_allclose(p.arrival_direction, (MIC - img) / R[:, None], atol=1e-5)


def test_lapse_rate_shadow_zone_distance():
    """SPEC.md: under a standard lapse rate rays curve upward and a shadow zone appears.

    For c(z) ~ linear with gradient g < 0, the limiting ray grazes the receiver height and is a
    circle of radius R_c = c_r / |g|, so a source H above the receiver is shadowed beyond
    r_s = sqrt(2 R_c H - H^2).
    """
    atm = strat(relative_humidity=0.0)
    zr = 1.5
    c_r = float(atm.sound_speed(np.array([zr]))[0])
    g = float((atm.sound_speed(np.array([zr + 1.0])) - atm.sound_speed(np.array([zr - 1.0])))[0] / 2.0)
    H = 100.0
    r_s = math.sqrt(2 * (c_r / abs(g)) * H - H**2)
    r = np.linspace(0.9 * r_s, 1.1 * r_s, 41)
    src = np.column_stack([-r, np.zeros_like(r), np.full_like(r, zr + H)])
    p = atm.propagate(src, np.array([0.0, 0.0, zr]))
    last_valid = r[p.valid].max()
    assert last_valid == pytest.approx(r_s, rel=0.02)
    assert not p.valid[r > 1.02 * r_s].any() and p.valid[r < 0.98 * r_s].all()
    # rays bend upward: a ray launched horizontally from 1 km climbs (reference tracer)
    _, pos, _ = rt.trace_ray(atm, np.array([0.0, 0.0, 1000.0]), np.array([1.0, 0.0, 0.0]), 20.0)
    assert pos[-1, 2] > 1000.0 + 10.0


def test_downwind_arrives_before_upwind():
    """SPEC.md: downwind arrivals are earlier than upwind at equal distance."""
    atm = strat(wind={"speed_mps": 8.0, "direction_from_deg": 270.0})  # wind blows toward +x (east)
    src = np.array([[-3000.0, 0.0, 1000.0], [3000.0, 0.0, 1000.0]])  # west (upwind) and east (downwind)
    p = atm.propagate(src, np.array([0.0, 0.0, 1.5]))
    t_from_west, t_from_east = p.travel_time
    assert t_from_west < t_from_east  # sound from the west travels with the wind


def test_fast_solver_matches_reference_ray_tracer():
    """SPEC.md: the fast travel times match direct (solve_ivp) ray traces at random points."""
    atm = strat(
        wind={"speed_mps": 8.0, "direction_from_deg": 230.0, "shear_deg_per_km": 15.0},
        inversion={"base_m": 200.0, "top_m": 500.0, "delta_k": 3.0},
        rays={"integration_step_m": 5.0},
    )
    src = random_sources(12, seed=3, zmax=4000.0)
    p = atm.propagate(src, MIC)
    checked = 0
    for k in np.flatnonzero(p.valid)[:8]:
        s = p.source_slowness[k]
        t, pos, slow = rt.trace_ray(
            atm, src[k], s / np.linalg.norm(s), 1.2 * p.travel_time[k], z_stop=MIC[2], max_step=0.02
        )
        assert np.linalg.norm(pos[-1, :2] - MIC[:2]) < 0.5  # lands on the mic
        assert t[-1] == pytest.approx(p.travel_time[k], abs=1e-4)
        c = atm.sound_speed(pos[:, 2])
        v = atm.wind(pos[:, 2])
        hamiltonian = c * np.linalg.norm(slow, axis=1) + np.sum(v * slow, axis=1)
        # dispersion relation conserved (to the accuracy of the reference's finite differences)
        np.testing.assert_allclose(hamiltonian, 1.0, atol=5e-6)
        checked += 1
    assert checked >= 6


def test_locate_inverts_propagate():
    for wind in (None, {"speed_mps": 8.0, "shear_deg_per_km": 15.0}):
        atm = strat(wind=wind)
        src = random_sources(200, seed=4, zmax=6000.0)
        p = atm.propagate(src, MIC)
        ok = p.valid
        pos, valid = atm.locate(MIC, p.source_slowness[ok, :2], p.travel_time[ok])
        assert valid.mean() > 0.98
        np.testing.assert_allclose(pos[valid], src[ok][valid], atol=0.5)


def test_warm_start_matches_fresh_solve():
    atm = strat(wind={"speed_mps": 8.0, "shear_deg_per_km": 15.0})
    src = random_sources(200, seed=5)
    p0 = atm.propagate(src, MIC)
    m1 = MIC + [30.0, 40.0, 0.0]
    fresh = atm.propagate(src, m1)
    warm = atm.propagate(src, m1, guess=p0)
    np.testing.assert_array_equal(fresh.valid, warm.valid)
    np.testing.assert_allclose(fresh.travel_time[fresh.valid], warm.travel_time[warm.valid], atol=2e-6)


def test_build_atmosphere_models():
    u = build_atmosphere(AtmosphereConfig(temperature_c=20.0, relative_humidity=0.5))
    assert u.is_uniform and float(u.sound_speed(np.array([0.0]))[0]) == pytest.approx(
        float(sound_speed_moist(293.15, 0.5, 101325.0))
    )
    s = build_atmosphere(AtmosphereConfig(model="stratified"))
    assert not s.is_uniform and s is build_atmosphere(AtmosphereConfig(model="stratified"))  # cached


# --- synthesis through the atmosphere ----------------------------------------------


def test_node_interpolation_matches_exact_emitter_rays():
    """Synthesis solves rays at channel nodes only; Hermite interpolation (plus the gradient
    correction for sub-segment tortuosity) must reproduce exact per-emitter ray times."""
    from thunder.acoustics.synth import _emitter_times, _node_paths, discretize

    atm = strat(wind={"speed_mps": 8.0, "shear_deg_per_km": 15.0})
    ch = generate_channel(
        ChannelConfig(preset="branched", strike_distance_m=(2000, 2000), start_height_m=(2500, 2500)),
        np.random.default_rng(2),
    )
    em = discretize(ch, 0.5, np.random.default_rng(3), math.radians(16.0), 1.0)
    pick = np.random.default_rng(4).choice(len(em.length), 300, replace=False)
    node = _node_paths(atm, ch, MIC, None)[0]
    t_a, _, _, _ = _emitter_times(atm, ch, em, MIC, node, reflected=False)
    exact = atm.propagate(em.start[pick], MIC)
    interp_valid = np.isfinite(t_a[pick])
    np.testing.assert_array_equal(interp_valid, exact.valid)  # same shadow (low upwind points)
    ok = exact.valid
    assert ok.sum() > 250
    assert np.max(np.abs(t_a[pick][ok] - exact.travel_time[ok])) < 2e-6  # < 2 us (sample = 125 us)


def test_ground_reflection_adds_image_pulse():
    src = np.array([1500.0, 0.0, 600.0])
    ch = polyline([src, src + [0, 0, 0.05]])
    mic = np.array([0.0, 0.0, 1.5])
    c = 343.0
    direct = synthesize(ch, UniformAtmosphere(c, 293.15), mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY).signals[0]
    both = synthesize(
        ch, UniformAtmosphere(c, 293.15, ground_reflection=True), mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY
    ).signals[0]
    n = len(direct)
    echo = both[:n] - direct
    R = np.linalg.norm(src + [0, 0, 0.025] - mic)
    Ri = np.linalg.norm((src + [0, 0, 0.025]) * [1, 1, -1] - mic)
    t = np.arange(n) / FS
    cd = (t * direct**2).sum() / (direct**2).sum()
    ce = (t * echo**2).sum() / (echo**2).sum()
    assert ce - cd == pytest.approx((Ri - R) / c, abs=1 / FS)
    assert math.sqrt((echo**2).sum() / (direct**2).sum()) == pytest.approx(R / Ri, rel=0.02)


def test_absorption_filters_the_spectrum():
    src = np.array([6000.0, 0.0, 800.0])
    ch = polyline([src, src + [0, 0, 0.05]])
    mic = np.array([0.0, 0.0, 1.5])
    plain = synthesize(ch, UniformAtmosphere(343.0, 293.15, 0.5), mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY)
    absorbed = synthesize(
        ch, UniformAtmosphere(343.0, 293.15, 0.5, absorption=True), mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY
    )
    n = 1 << int(np.ceil(np.log2(max(plain.signals.shape[1], absorbed.signals.shape[1]))))  # whole recordings
    f = np.fft.rfftfreq(n, 1 / FS)
    ratio_db = 20 * np.log10(
        np.abs(np.fft.rfft(absorbed.signals[0], n)) / np.abs(np.fft.rfft(plain.signals[0], n))
    )
    L = np.linalg.norm(src - mic)
    for fq in (150.0, 360.0, 740.0):  # between the N-wave's spectral zeros (~266, 457, 645, 833 Hz)
        i = np.argmin(abs(f - fq))
        expected = -absorption_db_per_m(fq, 293.15, 0.5, 101325.0) * L
        assert ratio_db[i] == pytest.approx(expected, abs=0.5)


def test_shadowed_source_is_silent_and_truth_is_nan():
    atm = strat(relative_humidity=0.0)  # standard lapse: shadow beyond ~4.3 km for a 100 m source
    near = polyline([[-2000.0, 0, 100.0], [-2000.0, 0, 101.0]])
    far = polyline([[-8000.0, 0, 100.0], [-8000.0, 0, 101.0]])
    mic = np.array([[0.0, 0.0, 1.5]])
    assert np.abs(synthesize(near, atm, mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY).signals).max() > 0
    silent = synthesize(far, atm, mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY)  # all in shadow: silence
    assert not np.any(silent.signals) and silent.signals.shape[1] / FS > 8000.0 / 350.0
    both = polyline([[-2000.0, 0, 100.0], [-2000.0, 0, 101.0], [-8000.0, 0, 101.0]])
    rec = synthesize(both, atm, mic, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY)
    assert rec.truth is not None
    times = rec.truth.segment_arrival_times[:, 0]
    assert np.isfinite(times[0]) and np.isnan(times[1])
    win = active_window(rec)
    assert win.stop > win.start


def test_wind_noise_can_follow_the_atmosphere():
    atm = strat(wind={"speed_mps": 6.0})
    ch = polyline([[1200.0, 0, 500.0], [1200.0, 0, 499.9]])
    arr = MicArray.ideal(square(30.0, 1.5))
    rec = synthesize(ch, atm, arr, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY)
    u = float(np.linalg.norm(atm.wind(np.array([1.5]))[0]))
    out = corrupt(
        rec,
        arr,
        SensorsConfig(noise={"wind_from_atmosphere": True}),
        np.random.default_rng(0),
        wind_at_mics_mps=u,
    )
    assert out.truth is not None
    assert out.truth.extra["corruption_info"]["wind_speed_mps"] == pytest.approx(u)


# --- reconstruction through the atmosphere -------------------------------------------


def test_refraction_aware_method_a_oracle_round_trip():
    cfg = AtmosphereConfig(model="stratified", wind={"speed_mps": 6.0, "direction_from_deg": 250.0})
    atm = build_atmosphere(cfg)
    arr = MicArray.ideal(square(50.0, 1.5, center=True))
    ch = generate_channel(
        ChannelConfig(preset="tortuous", strike_distance_m=(2500, 2500), start_height_m=(3000, 3000)),
        np.random.default_rng(6),
    )
    rec = synthesize(
        ch, atm, arr, FS, 8, 0.5, C.ACOUSTIC_EFFICIENCY, np.random.default_rng(7), math.radians(16), 1.0
    )
    cen = arr.nominal_positions.mean(axis=0)
    ray = reconstruct(rec, arr.nominal_positions, atm, ReconstructionConfig())
    m_ray, _ = evaluate(ray.points, ray.covariances, ch, cen, EvaluationConfig(), ray.extra["strike_point"])
    straight = reconstruct(
        rec, arr.nominal_positions, build_atmosphere(AtmosphereConfig()), ReconstructionConfig()
    )
    m_str, _ = evaluate(straight.points, straight.covariances, ch, cen, EvaluationConfig())
    assert m_ray["n_points"] > 50
    assert m_ray["point_error_median_m"] < 10.0  # oracle atmosphere: refraction fully undone
    assert (
        m_str["point_error_median_m"] > 5 * m_ray["point_error_median_m"]
    )  # straight rays: refraction error

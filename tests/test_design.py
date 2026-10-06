"""E2 array-design surrogate: Fisher information / Cramér-Rao bound, and the layout optimizer."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from thunder.experiments.design import (
    _unit_shapes,
    angular_variance,
    evaluate_layout,
    fisher_direction,
    optimize_layout,
    plane_wave_bias,
)
from thunder.sensors.arrays import circle, square

SIGMA, C = 1e-4, 343.0


def unit_vector(az, el):
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def numeric_fisher(m, az, el, h=1e-6):
    """F = J^T P J / sigma^2 from finite-difference arrival times, P removing the unknown emission time."""

    def times(a, e):
        return -(m @ unit_vector(a, e)) / C

    jac = np.column_stack(
        [
            (times(az + h, el) - times(az - h, el)) / (2 * h),
            (times(az, el + h) - times(az, el - h)) / (2 * h),
        ]
    )
    n = len(m)
    proj = np.eye(n) - np.ones((n, n)) / n
    return jac.T @ proj @ jac / SIGMA**2


def test_fisher_matches_finite_differences():
    m = np.random.default_rng(0).uniform(-40, 40, (6, 3))
    for az, el in [(0.3, 0.2), (2.0, 0.9), (4.5, 0.05)]:
        f = fisher_direction(m, np.array(az), np.array(el), SIGMA, C)
        np.testing.assert_allclose(f, numeric_fisher(m, az, el), rtol=1e-6)


def test_isotropic_planar_array_closed_form():
    """Horizontal second moment s I:  F = s/(sigma c)^2 diag(cos^2 el, sin^2 el), so the angular
    variance bound is (sigma c)^2 / s * (1 + 1/sin^2 el), independent of azimuth."""
    m = circle(8, 50.0, 1.5)
    s = np.mean(np.sum((m[:, :2] - m[:, :2].mean(0)) ** 2, axis=1)) / 2 * len(m)  # sum over mics
    az = np.radians(np.arange(0, 360, 15.0))
    for el_deg in (5.0, 30.0, 60.0):
        el = np.full_like(az, np.radians(el_deg))
        var = angular_variance(m, az, el, SIGMA, C)
        np.testing.assert_allclose(var, (SIGMA * C) ** 2 / s * (1 + 1 / np.sin(el) ** 2), rtol=1e-9)


def test_bound_scales_with_aperture_and_timing_error():
    m = square(50.0, 1.5, center=True)
    base = evaluate_layout(m).rms_angular_error_deg
    assert evaluate_layout(2 * m).rms_angular_error_deg == pytest.approx(base / 2, rel=1e-9)
    assert evaluate_layout(m, sigma_t=3 * SIGMA).rms_angular_error_deg == pytest.approx(3 * base, rel=1e-9)


def test_collinear_array_has_no_bound():
    m = np.column_stack([np.linspace(-25, 25, 5), np.zeros(5), np.full(5, 1.5)])
    assert np.isinf(evaluate_layout(m).worst_angular_error_deg)


def test_mast_adds_elevation_information():
    flat = square(50.0, 1.5, center=True)
    mast = flat.copy()
    mast[-1, 2] = 10.0  # center mic raised
    assert evaluate_layout(mast).rms_angular_error_deg < evaluate_layout(flat).rms_angular_error_deg


@pytest.mark.parametrize("n,criterion,mast", [(5, "A", False), (6, "D", False), (4, "A", True)])
def test_optimizer_respects_aperture_and_beats_parametric(n, criterion, mast):
    aperture = 30.0
    o = optimize_layout(n, aperture, criterion, mast=mast, maxiter=15)
    assert o.positions.shape == (n, 3)
    assert float(np.max(pdist(o.positions[:, :2]))) <= aperture * (1 + 1e-9)
    np.testing.assert_allclose(o.positions[:, :2].mean(axis=0), 0.0, atol=1e-9)
    z = o.positions[:, 2]
    assert np.all((z >= 1.5 - 1e-9) & (z <= 10.0 + 1e-9))
    if not mast:
        np.testing.assert_allclose(z, 1.5)
    assert float(np.min(pdist(o.positions))) >= 0.2 * aperture * (1 - 1e-9)  # no stacked mics
    for shape in _unit_shapes(n):
        p = evaluate_layout(np.column_stack([shape * aperture, np.full(n, 1.5)]))
        if criterion == "A":
            assert o.surrogate.rms_angular_error_deg <= p.rms_angular_error_deg * (1 + 1e-12)
        else:
            assert o.surrogate.mean_log_det >= p.mean_log_det - 1e-12


def test_plane_wave_bias_vanishes_for_symmetric_layouts_and_scales_as_one_over_range():
    rng = np.random.default_rng(3)
    az, el = rng.uniform(0, 2 * np.pi, 30), np.radians(rng.uniform(10, 60, 30))
    u = np.column_stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    sym = square(50.0, 1.5, center=True)
    mast = sym.copy()
    mast[-1, 2] = 10.0  # one raised mic: asymmetric in z
    for r in (2000.0, 20000.0):
        src_sym = sym.mean(axis=0) + r * u
        src_mast = mast.mean(axis=0) + r * u
        assert np.median(plane_wave_bias(sym, src_sym)) < 0.004 * (2000.0 / r) ** 2  # O(1/R^2), tiny
        b = np.median(plane_wave_bias(mast, src_mast))
        if r == 2000.0:
            b2k = b
    # asymmetric: O(1/R) (10x range -> 10x smaller), and far above the symmetric O(1/R^2) level
    assert b2k > 0.05
    assert b == pytest.approx(b2k / 10.0, rel=0.05)

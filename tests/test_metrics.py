import numpy as np
import pytest
from scipy.spatial import cKDTree

from thunder.config import EvaluationConfig
from thunder.eval.aggregate import binned_quantiles, bootstrap_ci, cluster_bootstrap_ci
from thunder.eval.metrics import (
    EXPECTED_COVERAGE,
    calibration,
    evaluate,
    nearest_on_segments,
    sample_channel,
)
from thunder.recon.postprocess import estimate_strike_point
from thunder.types import Channel

CFG = EvaluationConfig()
CENTROID = np.zeros(3)


def line_channel(top=(1000.0, 0.0, 3000.0), bottom=(1000.0, 0.0, 0.0), n=300) -> Channel:
    pts = np.linspace(top, bottom, n + 1)
    return Channel(
        pts,
        np.c_[np.arange(n), np.arange(1, n + 1)],
        np.ones(n),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
        metadata={"strike_point": list(bottom)},
    )


def test_identical_reconstruction_scores_zero_error_and_full_coverage():
    ch = line_channel()
    pts, _, _ = sample_channel(ch, 1.0)
    m, pp = evaluate(pts, np.repeat(np.eye(3)[None], len(pts), 0), ch, CENTROID, CFG)
    assert m["point_error_median_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["point_error_p90_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["coverage_10m"] == pytest.approx(1.0)
    assert m["coverage_main_10m"] == pytest.approx(1.0)
    assert m["strike_error_m"] < 1.0
    assert m["chamfer_m"] < 0.3


def test_ten_meter_perpendicular_shift_gives_exactly_ten_meters():
    ch = line_channel()
    pts, _, _ = sample_channel(ch, 5.0)
    pts = pts[(pts[:, 2] > 50) & (pts[:, 2] < 2950)] + [0.0, 10.0, 0.0]  # avoid end caps
    m, pp = evaluate(pts, np.repeat(np.eye(3)[None], len(pts), 0), ch, CENTROID, CFG)
    np.testing.assert_allclose(pp["error_m"], 10.0, rtol=1e-12)
    assert m["coverage_25m"] > 0.95  # all but the trimmed end caps
    assert m["coverage_10m"] < 1.0


def test_exact_segment_distance_matches_dense_sampling():
    rng = np.random.default_rng(0)
    a = rng.uniform(-100, 100, (40, 3))
    b = a + rng.uniform(-30, 30, (40, 3))
    p = rng.uniform(-150, 150, (200, 3))
    d, q, k = nearest_on_segments(p, a, b)
    t = np.linspace(0, 1, 20001)[:, None, None]
    dense = (a[None] + t * (b - a)[None]).reshape(-1, 3)
    brute, _ = cKDTree(dense).query(p)  # dense samples are <= 0.003 m apart
    np.testing.assert_allclose(d, brute, atol=0.01)
    np.testing.assert_allclose(np.linalg.norm(p - q, axis=1), d, rtol=1e-12)


def test_radial_and_transverse_split():
    ch = line_channel(top=(1000.0, 0, 3000.0), bottom=(1000.0, 0, 1.0))
    q = np.array([[1000.0, 0.0, 1500.0]])
    los = q[0] / np.linalg.norm(q[0])
    _, pp = evaluate(q + 7.0 * los, np.eye(3)[None], ch, CENTROID, CFG)
    # Moving along the line of sight from a point on a vertical line: the error splits into a
    # radial and a transverse part relative to the *reconstructed* point's line of sight.
    e = pp["error_m"][0]
    assert pp["radial_m"][0] ** 2 + pp["transverse_m"][0] ** 2 == pytest.approx(e**2, rel=1e-9)


def test_calibration_returns_nominal_fractions_for_gaussian_errors():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + 0.5 * np.eye(3)
    e = rng.multivariate_normal(np.zeros(3), cov, 200000)
    frac = calibration(e, np.repeat(cov[None], len(e), 0))
    np.testing.assert_allclose(frac, EXPECTED_COVERAGE, atol=0.005)
    assert EXPECTED_COVERAGE == pytest.approx((0.199, 0.739, 0.971), abs=0.001)


def test_strike_point_extrapolates_lowest_section_with_slope_cap():
    # A straight channel leaning 0.2 m per m, reconstructed only above 300 m: extrapolation
    # lands on the true strike point (within the 0.5 slope cap).
    z = np.linspace(300.0, 1000.0, 30)
    pts = np.column_stack([100.0 + 0.2 * z, np.zeros_like(z), z])
    np.testing.assert_allclose(estimate_strike_point(pts), [100.0, 0.0, 0.0], atol=1e-9)
    # Leaning 2 m per m (implausibly flat): the horizontal move is capped at 0.5 x 300 m.
    flat = np.column_stack([2.0 * z, np.zeros_like(z), z])
    est = estimate_strike_point(flat)
    assert est is not None
    assert np.linalg.norm(est[:2] - [600.0, 0.0]) == pytest.approx(150.0)
    assert estimate_strike_point(np.zeros((0, 3))) is None


def test_empty_reconstruction_metrics():
    m, pp = evaluate(np.zeros((0, 3)), np.zeros((0, 3, 3)), line_channel(), CENTROID, CFG)
    assert m["n_points"] == 0 and m["coverage_50m"] == 0.0 and np.isnan(m["point_error_median_m"])
    assert pp == {}


def test_bootstrap_ci_covers_true_median():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(100):
        v = rng.normal(5.0, 1.0, 200)
        est, lo, hi = bootstrap_ci(v, np.median, rng, 500)
        hits += lo <= 5.0 <= hi
    assert 88 <= hits <= 100  # nominal 95%


def test_cluster_bootstrap_is_wider_for_correlated_groups():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(30), 50)
    values = rng.normal(0, 1, 30)[groups] + 0.1 * rng.normal(size=1500)  # strong within-group correlation
    _, lo_n, hi_n = bootstrap_ci(values, np.median, rng, 500)
    _, lo_c, hi_c = cluster_bootstrap_ci(values, groups, np.median, rng, 500)
    assert (hi_c - lo_c) > 3 * (hi_n - lo_n)


def test_binned_quantiles():
    x = np.repeat([0.5, 1.5], 100)
    y = np.r_[np.arange(100), np.arange(100) + 1000]
    out = binned_quantiles(x, y, np.array([0.0, 1.0, 2.0, 3.0]))
    assert out["q50"][0] == pytest.approx(49.5) and out["q50"][1] == pytest.approx(1049.5)
    assert np.isnan(out["q50"][2]) and list(out["count"]) == [100, 100, 0]

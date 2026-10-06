import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as sstats

from thunder.channel import stats
from thunder.channel.generator import generate_channel
from thunder.channel.walk import perpendicular_basis, rotate, turn, turn_angle_distribution
from thunder.config import CHANNEL_PRESETS, ChannelConfig
from thunder.types import Channel

PRESETS = list(CHANNEL_PRESETS)


def _channels(preset: str, n: int, seed: int = 0, **overrides) -> list[Channel]:
    cfg = ChannelConfig(preset=preset, **overrides)
    return [generate_channel(cfg, np.random.default_rng(s)) for s in np.random.SeedSequence(seed).spawn(n)]


# --- walk primitives ------------------------------------------------------


def test_perpendicular_basis_is_orthonormal():
    rng = np.random.default_rng(0)
    for d in [np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), *rng.normal(size=(20, 3))]:
        d = d / np.linalg.norm(d)
        e1, e2 = perpendicular_basis(d)
        m = np.stack([e1, e2, d])
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(m) == pytest.approx(1.0)


def test_turn_angle_is_exact_regardless_of_goal():
    rng = np.random.default_rng(1)
    for _ in range(200):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        goal = rng.normal(size=3)
        goal /= np.linalg.norm(goal)
        angle = rng.uniform(0, 1.5)
        new = turn(d, angle, goal, offset=0.0, fallback=0.0)
        assert math.acos(np.clip(new @ d, -1, 1)) == pytest.approx(angle, abs=1e-9)


def test_turn_with_zero_offset_moves_toward_goal():
    d = np.array([0.0, 0.0, -1.0])
    goal = np.array([1.0, 0.0, -1.0]) / math.sqrt(2)
    new = turn(d, 0.1, goal, offset=0.0, fallback=0.0)
    assert new @ goal > d @ goal


def test_rotate_tilts_by_angle():
    d = np.array([0.0, 0.0, -1.0])
    assert math.acos(rotate(d, 0.3, 1.0) @ d) == pytest.approx(0.3)


@pytest.mark.parametrize("name", ["halfnormal", "exponential"])
def test_turn_distribution_has_configured_mean(name):
    assert turn_angle_distribution(name, 0.28).mean() == pytest.approx(0.28)


# --- channel structure (spec Phase 1 tests) ----------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_channel_is_connected_tree_and_lands_on_target(preset):
    for ch in _channels(preset, 10, seed=2):
        assert stats.is_connected_tree(ch)
        main_nodes = ch.nodes[ch.segments[ch.is_main][:, 1]]
        landing = main_nodes[np.argmin(main_nodes[:, 2])]
        np.testing.assert_allclose(landing, ch.metadata["strike_point"], atol=1e-9)
        assert landing[2] == 0.0


@pytest.mark.parametrize("preset", PRESETS)
def test_nothing_below_ground_and_branches_never_reach_ground(preset):
    for ch in _channels(preset, 10, seed=3):
        assert ch.nodes[:, 2].min() >= 0.0
        side = ~(ch.is_main | ch.is_incloud)
        a, b = ch.segment_endpoints()
        if side.any():
            assert b[side, 2].min() >= ch.metadata["params"]["segment_length_m"]


def test_same_seed_identical_different_seed_differs():
    cfg = ChannelConfig(preset="with_incloud")
    a = generate_channel(cfg, np.random.default_rng(42))
    b = generate_channel(cfg, np.random.default_rng(42))
    c = generate_channel(cfg, np.random.default_rng(43))
    np.testing.assert_array_equal(a.nodes, b.nodes)
    np.testing.assert_array_equal(a.segments, b.segments)
    assert a.nodes.shape != c.nodes.shape or not np.array_equal(a.nodes, c.nodes)


def _check_turn_statistics(n: int) -> None:
    cfg = ChannelConfig(preset="tortuous")
    angles = np.concatenate([stats.turn_angles(ch) for ch in _channels("tortuous", n, seed=5)])
    dist = turn_angle_distribution(cfg.turn_distribution, cfg.turn_mean)
    assert angles.mean() == pytest.approx(dist.mean(), rel=0.02)
    assert np.median(angles) == pytest.approx(dist.median(), rel=0.02)
    assert sstats.kstest(angles, dist.cdf).pvalue > 0.01


def test_turn_angle_statistics_match_configuration():
    _check_turn_statistics(100)


@pytest.mark.slow
def test_turn_angle_statistics_match_configuration_500():
    _check_turn_statistics(500)


def test_branch_departure_angles_within_configured_range():
    cfg = ChannelConfig(preset="branched", branch_probability=0.05)
    ch = generate_channel(cfg, np.random.default_rng(6))
    vec = stats.segment_vectors(ch)
    prev = stats.incoming_segment(ch)[ch.segments[:, 0]]
    joints = (prev >= 0) & (ch.branch_id != ch.branch_id[np.maximum(prev, 0)]) & ~ch.is_incloud
    a, b = vec[prev[joints]], vec[joints]
    cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    ang = np.arccos(np.clip(cos, -1, 1))
    lo, hi = cfg.branch_angle
    assert len(ang) > 5
    assert ang.min() >= lo - 1e-9 and ang.max() <= hi + 1e-9


# --- presets --------------------------------------------------------------


def test_straight_preset():
    for ch in _channels("straight", 5, seed=7):
        assert stats.branch_count(ch) == 0
        assert np.degrees(stats.turn_angles(ch).mean()) < 3.0
        assert stats.horizontal_extent(ch) < 50.0


def test_tortuous_has_no_branches_branched_has_some():
    assert all(stats.branch_count(ch) == 0 for ch in _channels("tortuous", 5))
    assert np.mean([stats.branch_count(ch) for ch in _channels("branched", 20)]) > 2


def test_branch_energy_is_fraction_of_parent():
    cfg = ChannelConfig(preset="branched", branch_probability=0.05)
    ch = generate_channel(cfg, np.random.default_rng(8))
    levels = {round(math.log(e / cfg.energy_per_length_main, cfg.branch_energy_fraction), 9)
              for e in ch.energy_per_length}
    assert levels <= set(range(cfg.branch_max_depth + 1)) and {0, 1} <= levels
    np.testing.assert_allclose(ch.energy_per_length[ch.is_main], cfg.energy_per_length_main)


def test_with_incloud_preset():
    cfg = ChannelConfig(preset="with_incloud")
    for ch in _channels("with_incloud", 10, seed=9):
        seg_len = stats.segment_lengths(ch)
        assert seg_len[ch.is_incloud].sum() >= cfg.incloud_length_m[0] - cfg.segment_length_m
        a, b = ch.segment_endpoints()
        z_ref = ch.nodes[0, 2]
        assert np.abs(b[ch.is_incloud, 2] - z_ref).max() < 1000.0  # stays near-horizontal
        assert np.all(ch.refires == (ch.is_main | ch.is_incloud))


def test_multi_stroke_preset():
    cfg = ChannelConfig(preset="multi_stroke")
    for ch in _channels("multi_stroke", 20, seed=10):
        t = ch.stroke_times
        assert 2 <= len(t) <= 4 and t[0] == 0.0
        gaps = np.diff(t)
        assert np.all(gaps >= cfg.stroke_interval_s[0]) and np.all(gaps <= cfg.stroke_interval_s[1])


# --- config ---------------------------------------------------------------


def test_preset_defaults_and_overrides():
    assert ChannelConfig(preset="straight").branch_probability == 0.0
    assert ChannelConfig(preset="straight", branch_probability=0.1).branch_probability == 0.1
    assert ChannelConfig(preset="with_incloud").incloud
    assert ChannelConfig().turn_mean == pytest.approx(math.radians(16.0))


def test_bad_ranges_rejected():
    with pytest.raises(ValidationError):
        ChannelConfig(start_height_m=(7000, 4000))
    with pytest.raises(ValidationError):
        ChannelConfig(preset="nonsense")


# --- stats ----------------------------------------------------------------


def _line_channel(points: np.ndarray) -> Channel:
    n = len(points) - 1
    return Channel(
        nodes=points,
        segments=np.c_[np.arange(n), np.arange(1, n + 1)],
        energy_per_length=np.ones(n),
        branch_id=np.zeros(n, dtype=np.int64),
        is_main=np.ones(n, dtype=bool),
        is_incloud=np.zeros(n, dtype=bool),
    )


def test_fractal_dimension_of_straight_line_is_one():
    pts = np.linspace([0, 0, 5000], [800, 300, 0], 501)
    assert stats.fractal_dimension(_line_channel(pts)) == pytest.approx(1.0, abs=0.06)


def test_fractal_dimension_of_tortuous_channels_between_one_and_two():
    d = [stats.fractal_dimension(ch) for ch in _channels("branched", 5)]
    assert all(1.0 < x < 2.0 for x in d)


def test_basic_stats_on_known_shape():
    pts = np.array([[0, 0, 20.0], [0, 0, 10.0], [10, 0, 10.0]])
    ch = _line_channel(pts)
    assert stats.total_length(ch) == pytest.approx(20.0)
    assert stats.horizontal_extent(ch) == pytest.approx(10.0)
    np.testing.assert_allclose(stats.turn_angles(ch), [math.pi / 2])
    assert stats.branch_count(ch) == 0

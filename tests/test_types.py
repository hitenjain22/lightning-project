import numpy as np
import pytest

from thunder.types import (
    Channel,
    GroundTruth,
    MicArray,
    Reconstruction,
    Recording,
    UniformAtmosphere,
)


def _two_segment_channel() -> Channel:
    nodes = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 50.0], [0.0, 0.0, 0.0]])
    return Channel(
        nodes=nodes,
        segments=np.array([[0, 1], [1, 2]]),
        energy_per_length=np.full(2, 1e5),
        branch_id=np.zeros(2, dtype=np.int64),
        is_main=np.ones(2, dtype=bool),
        is_incloud=np.zeros(2, dtype=bool),
    )


def test_channel_valid():
    ch = _two_segment_channel()
    a, b = ch.segment_endpoints()
    assert ch.n_segments == 2
    np.testing.assert_array_equal(b[-1], [0, 0, 0])
    assert a.shape == (2, 3)


def test_channel_rejects_bad_shapes():
    ch = _two_segment_channel()
    with pytest.raises(ValueError):
        Channel(ch.nodes, ch.segments, np.ones(3), ch.branch_id, ch.is_main, ch.is_incloud)
    with pytest.raises(ValueError):
        one = np.ones(1, bool)
        Channel(ch.nodes, np.array([[0, 5]]), np.ones(1), np.zeros(1, int), one, ~one)


def test_mic_array_ideal():
    arr = MicArray.ideal(np.zeros((4, 3)))
    assert arr.n_mics == 4
    assert not np.shares_memory(arr.nominal_positions, arr.true_positions)
    with pytest.raises(ValueError):
        MicArray(np.zeros((4, 3)), np.zeros((3, 3)), np.zeros(4), np.zeros(4))


def test_recording_and_truth():
    truth = GroundTruth(t0=0.0, mic_positions=np.zeros((2, 3)), segment_arrival_times=np.zeros((5, 2)))
    rec = Recording(np.zeros((2, 800)), 8000.0, np.zeros((2, 3)), 0.0, truth)
    assert rec.n_mics == 2
    assert rec.duration == pytest.approx(0.1)
    with pytest.raises(ValueError):
        Recording(np.zeros((2, 800)), 8000.0, np.zeros((3, 3)), 0.0)


def test_reconstruction_empty():
    r = Reconstruction.empty("A", "abc")
    assert r.points.shape == (0, 3) and r.covariances.shape == (0, 3, 3)


def test_uniform_atmosphere_straight_line():
    atm = UniformAtmosphere(sound_speed=340.0, temperature=288.15)
    src = np.array([[0.0, 0.0, 3400.0], [340.0, 0.0, 0.0]])
    path = atm.propagate(src, np.zeros(3))
    np.testing.assert_allclose(path.travel_time, [10.0, 1.0])
    np.testing.assert_allclose(path.arrival_direction[0], [0, 0, -1])
    np.testing.assert_allclose(path.amplitude_factor, [1 / 3400, 1 / 340])
    assert atm.wind(np.array([0.0, 10.0])).shape == (2, 3)

"""M9 visualization: animation, uncertainty axes in the 3D view, waveform onset detection."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from thunder.viz.animate import animate_reconstruction
from thunder.viz.plot3d import ellipsoid_axes_trace, reconstruction_figure
from thunder.viz.plots import plot_waveform_stack


def toy_example(n_points: int = 20) -> dict:
    nodes = np.column_stack([np.full(11, 1000.0), np.zeros(11), np.linspace(3000, 0, 11)])
    segs = np.c_[np.arange(10), np.arange(1, 11)]
    rng = np.random.default_rng(0)
    pts = nodes[rng.integers(0, 11, n_points)] + rng.normal(0, 5, (n_points, 3))
    return {
        "nodes": nodes,
        "segments": segs,
        "is_main": np.ones(10, bool),
        "is_incloud": np.zeros(10, bool),
        "points": pts,
        "covariances": np.repeat(np.diag([4.0, 9.0, 25.0])[None], n_points, axis=0),
        "error_m": rng.uniform(0, 10, n_points),
        "window_times": np.sort(rng.uniform(3, 9, n_points)),
        "mics": np.array([[0.0, 0.0, 1.5], [10.0, 0.0, 1.5], [0.0, 10.0, 1.5]]),
        "signals": rng.standard_normal((1, 8000 * 10)),
        "sample_rate": 8000.0,
    }


def test_animation_writes_a_gif_with_every_frame(tmp_path):
    out = animate_reconstruction(toy_example(), tmp_path / "a.gif", "test", n_frames=5, hold_frames=2, dpi=40)
    with Image.open(out) as im:
        # Pillow merges identical consecutive frames, so the 2 held frames become a longer last
        # frame: check the total playing time (7 frames at 12 fps) instead of the frame count.
        total = 0
        for k in range(im.n_frames):
            im.seek(k)
            total += im.info["duration"]
    assert abs(total - 7 * 1000 / 12) < 7 * 10  # GIF durations are stored in 10 ms steps


def test_animation_handles_no_points(tmp_path):
    ex = toy_example()
    for k in ("points", "error_m", "window_times"):
        ex[k] = ex[k][:0]
    ex["covariances"] = ex["covariances"][:0]
    animate_reconstruction(ex, tmp_path / "b.gif", n_frames=3, hold_frames=0, dpi=40)


def test_ellipsoid_axes_have_two_sigma_half_lengths():
    p = np.array([[0.0, 0.0, 0.0]])
    trace = ellipsoid_axes_trace(p, np.diag([4.0, 9.0, 25.0])[None])
    ends = np.column_stack(
        [np.array([v for v in a if v is not None], dtype=float) for a in (trace.x, trace.y, trace.z)]
    )
    half = 0.5 * np.linalg.norm(ends[1::2] - ends[::2], axis=1)  # each axis is drawn as one segment
    np.testing.assert_allclose(np.sort(half), [4.0, 6.0, 10.0])  # 2 sigma: 2*2, 2*3, 2*5
    fig = reconstruction_figure(toy_example(), "t")
    assert any("uncertainty" in (t.name or "") for t in fig.data)


def test_waveform_onset_zoom_ignores_background_noise():
    fs = 8000
    rng = np.random.default_rng(0)
    x = 0.05 * rng.standard_normal((3, 20 * fs))
    x[:, 7 * fs : 7 * fs + 400] += 50 * np.exp(-np.arange(400) / 80)
    fig, ax = plt.subplots(1, 2)
    plot_waveform_stack((ax[0], ax[1]), x, fs, ["a", "b", "c"])
    lo, hi = ax[1].get_xlim()
    assert 6.9 < lo < 7.0 < hi
    plt.close(fig)

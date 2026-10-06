import json

import numpy as np
import pandas as pd
import pytest

from thunder.config import ChannelConfig, RunConfig
from thunder.experiments.montecarlo import channel_overrides, run_monte_carlo
from thunder.experiments.pipeline import (
    StageCache,
    child_seeds,
    iter_bolt_variants,
    merge,
    run_bolt,
    run_bolt_variants,
)
from thunder.experiments.reports import make_report
from thunder.experiments.runner import run_experiment
from thunder.experiments.sweeps import crossing, paired_ratio


def small_mc_config(tmp_path, workers: int) -> RunConfig:
    return RunConfig.model_validate(
        {
            "experiment": "mc_test",
            "seed": 7,
            "results_root": str(tmp_path),
            # short channels keep the test fast; the logic is the same at any size
            "channel": {"strike_distance_m": [1200.0, 1800.0], "start_height_m": [2000.0, 2500.0]},
            "array": {"layout": "square_center", "aperture_m": 50.0},
            "synthesis": {"write_wav": False},
            "reconstruction": {"method": "A"},
            "monte_carlo": {
                "n_bolts": 3,
                "presets": ["tortuous", "branched", "with_incloud"],
                "workers": workers,
                "bootstrap_samples": 200,
            },
        }
    )


def test_monte_carlo_is_independent_of_worker_count(tmp_path):
    a = run_monte_carlo(small_mc_config(tmp_path, 1))
    b = run_monte_carlo(small_mc_config(tmp_path, 3))
    cols = [c for c in a.table.columns if not c.startswith("time_")]
    pd.testing.assert_frame_equal(a.table[cols], b.table[cols])
    pd.testing.assert_frame_equal(a.points, b.points)
    assert list(a.table["preset"]) == ["tortuous", "branched", "with_incloud"]
    assert set(a.examples) == {"tortuous", "branched", "with_incloud"}


def test_channel_overrides_exclude_preset_defaults():
    cfg = RunConfig.model_validate(
        {"experiment": "x", "seed": 0, "channel": {"preset": "with_incloud", "strike_distance_m": [1, 2]}}
    )
    assert channel_overrides(cfg) == {"strike_distance_m": [1.0, 2.0]}
    cfg2 = RunConfig.model_validate(
        {"experiment": "x", "seed": 0, "channel": {"preset": "with_incloud", "incloud": False}}
    )
    assert channel_overrides(cfg2) == {"incloud": False}  # differs from the preset: a real override


def test_monte_carlo_run_folder_presets_and_report(tmp_path):
    run_dir = run_experiment(small_mc_config(tmp_path, 1))  # parallelism is covered by the test above
    t = pd.read_csv(run_dir / "bolts.csv").set_index("preset")
    assert t["strike_distance_m"].between(1200.0, 1800.0).all()  # explicit field kept for every preset
    assert t.loc["with_incloud", "incloud_length_m"] >= 2000.0  # each preset's own fields applied
    assert t.loc["tortuous", "incloud_length_m"] == 0.0 and t.loc["tortuous", "branch_count"] == 0
    for name in ("bolts.csv", "points.csv.gz", "examples.pkl", "metrics.json", "config.resolved.yaml"):
        assert (run_dir / name).exists(), name
    summary = json.loads((run_dir / "metrics.json").read_text())
    est, lo, hi = summary["overall"]["pooled_point_error_median_m"]
    assert lo <= est <= hi and np.isfinite(est)
    out = make_report(run_dir)
    assert "Pooled point error, median" in out.read_text()
    for name in (
        "examples_3d",
        "error_vs_range_altitude",
        "error_distribution",
        "coverage",
        "per_bolt_error",
    ):
        assert (run_dir / "figures" / f"{name}.png").exists()
    assert (run_dir / "figures" / "example_branched.html").exists()


# --- variants (common random numbers, stage cache) ---------------------------------


def test_child_seeds_match_spawn_and_do_not_mutate():
    ss = np.random.SeedSequence(42)
    a = child_seeds(ss)
    b = child_seeds(ss)  # same object again: same streams
    ref = np.random.default_rng(42).spawn(4)
    for x, y, r in zip(a, b, ref, strict=True):
        assert np.array_equal(np.random.default_rng(x).random(4), np.random.default_rng(y).random(4))
        assert np.array_equal(np.random.default_rng(x).random(4), r.random(4))
    assert ss.n_children_spawned == 0


def test_merge_is_recursive_and_does_not_alias():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    out = merge(base, {"a": {"b": 2}, "e": {"f": 3}})
    assert out == {"a": {"b": 2, "c": [1, 2]}, "d": 1, "e": {"f": 3}}
    out["a"]["c"].append(3)
    assert base["a"]["c"] == [1, 2]


VARIANTS = {
    "clean": {},
    "noisy": {"sensors": {"noise": {"background_snr_db": 10.0}}},
    "method_c": {"reconstruction": {"method": "C"}},
    "far": {"channel": {"strike_distance_m": [2500.0, 2500.0]}},
}


def variant_base(tmp_path) -> dict:
    d = small_mc_config(tmp_path, 1).model_dump(mode="json")
    d["sensors"] = {}
    return d


def test_variants_equal_separate_runs_and_share_stages(tmp_path):
    base = variant_base(tmp_path)
    channel = {
        "preset": "tortuous",
        "strike_distance_m": [1500.0, 1500.0],
        "start_height_m": [2000.0, 2000.0],
    }
    seed = np.random.SeedSequence(3).spawn(2)[1]
    together = run_bolt_variants(base, VARIANTS, seed, channel)
    for name, ov in VARIANTS.items():
        cfg = RunConfig.model_validate(merge(base, ov))
        alone = run_bolt(cfg, seed, ChannelConfig.model_validate(merge(channel, ov.get("channel", {}))))
        np.testing.assert_array_equal(together[name].recording.signals, alone.recording.signals)
        for k, v in alone.metrics.items():
            assert together[name].metrics[k] == pytest.approx(v, nan_ok=True), (name, k)
    # Same channel draws for the near variants; the far variant only moves the bolt.
    assert together["noisy"].channel is together["clean"].channel
    assert together["far"].channel is not together["clean"].channel
    assert together["method_c"].recording is together["clean"].recording  # recon-only change: reused
    assert together["noisy"].recording is not together["clean"].recording

    cache = StageCache()
    for ov in VARIANTS.values():
        run_bolt(
            RunConfig.model_validate(merge(base, ov)),
            seed,
            ChannelConfig.model_validate(merge(channel, ov.get("channel", {}))),
            cache,
        )
    assert len(cache.channels) == 2 and len(cache.clean) == 2  # near + far
    assert len(cache.corrupted) == 3  # near clean-sensor, near noisy, far

    # A one-entry cache recomputes evicted stages; results must not change.
    tiny = StageCache(max_recordings=1)
    for name, res in iter_bolt_variants(base, VARIANTS, seed, channel, tiny):
        np.testing.assert_array_equal(res.recording.signals, together[name].recording.signals)
        assert len(tiny.clean) == 1 and len(tiny.corrupted) == 1


def test_monte_carlo_with_variants(tmp_path):
    cfg = small_mc_config(tmp_path, 1)
    d = cfg.model_dump(mode="json")
    d["monte_carlo"].update(
        n_bolts=2, variants={k: VARIANTS[k] for k in ("clean", "noisy")}, keep_points=True
    )
    res = run_monte_carlo(RunConfig.model_validate(d))
    assert len(res.table) == 4 and set(res.table["variant"]) == {"clean", "noisy"}
    assert set(res.points["variant"]) <= {"clean", "noisy"}
    assert set(res.summary) == {"clean", "noisy"} and "overall" in res.summary["clean"]
    t = res.table.set_index(["variant", "bolt"])
    # Common random numbers: both variants see the same bolts.
    np.testing.assert_array_equal(t.loc["clean", "channel_length_m"], t.loc["noisy", "channel_length_m"])
    assert np.all(t.loc["noisy", "snr_band_db"].to_numpy() == pytest.approx(10.0))


# --- sweep analysis helpers -----------------------------------------------------------


def test_crossing_interpolates_and_reports_edges():
    levels = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    values = np.array([0.1, 0.2, 0.6, 2.0])
    x = crossing(levels, values, 0.5)
    assert 1e-5 < x < 1e-4
    # log-linear interpolation: 0.5 is 3/4 of the way from 0.2 to 0.6
    assert x == pytest.approx(10 ** (-5 + 0.75))
    assert crossing(levels, values, 0.05) is None  # fails already at the best level
    assert crossing(levels, values, 5.0) == float("inf")  # fine at every level
    assert crossing(np.array([0.0, 10.0]), np.array([1.0, 3.0]), 2.0, log_x=False) == pytest.approx(5.0)
    # A level with no result counts as failing; nothing is assumed between it and the last pass.
    assert crossing(levels, np.array([0.1, 0.2, np.nan, 2.0]), 0.5) == pytest.approx(1e-5)


def test_paired_ratio_removes_bolt_to_bolt_variation():
    rng = np.random.default_rng(0)
    scale = rng.lognormal(0, 1, 40)  # large bolt-to-bolt variation
    t = pd.DataFrame(
        {
            "bolt": np.tile(np.arange(40), 2),
            "variant": np.repeat(["base", "worse"], 40),
            "err": np.r_[scale, 2 * scale],
        }
    )
    r = paired_ratio(t, "err", "base", n_boot=200)
    assert r.loc["worse", "ratio"] == pytest.approx(2.0)
    assert r.loc["worse", "lo"] == pytest.approx(2.0) and r.loc["worse", "hi"] == pytest.approx(2.0)
    assert r.loc["base", "ratio"] == pytest.approx(1.0)

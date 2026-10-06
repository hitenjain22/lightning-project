import json

import numpy as np
import pandas as pd

from thunder.config import RunConfig
from thunder.experiments.montecarlo import channel_overrides, run_monte_carlo
from thunder.experiments.reports import make_report
from thunder.experiments.runner import run_experiment


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

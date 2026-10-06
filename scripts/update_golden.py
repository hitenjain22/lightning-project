"""Regenerate the fixed-seed golden metrics checked by tests/test_regression.py.

Usage: python scripts/update_golden.py

SPEC.md test layer 4: any change that moves these numbers beyond tolerance must be explained
in docs/log.md before the golden file is regenerated.
"""

import json
from pathlib import Path

from thunder.config import RunConfig
from thunder.experiments.pipeline import run_bolt

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "roundtrip_metrics.json"
KEYS = (
    "n_points",
    "point_error_median_m",
    "point_error_p90_m",
    "coverage_main_50m",
    "coverage_50m",
    "strike_error_m",
    "chamfer_m",
)

CASES = {
    "tortuous_2km_ideal": {"channel": {"preset": "tortuous", "strike_distance_m": [2000.0, 2000.0]}},
    "branched_1500m_ideal": {"channel": {"preset": "branched", "strike_distance_m": [1500.0, 1500.0]}},
    "branched_2km_noisy": {
        "channel": {"preset": "branched", "strike_distance_m": [2000.0, 2000.0]},
        "sensors": {
            "mic": {"preset": "measurement"},
            "timing": {"preset": "gps_synced"},
            "position": {"preset": "surveyed"},
            "noise": {"background_snr_db": 15.0},
        },
    },
}


def case_config(name: str, spec: dict) -> RunConfig:
    return RunConfig.model_validate(
        {
            "experiment": name,
            "seed": 424242,
            "array": {"layout": "square_center", "aperture_m": 50.0},
            "synthesis": {"write_wav": False},
            "reconstruction": {"method": "A"},
            **spec,
        }
    )


def compute() -> dict:
    out = {}
    for name, spec in CASES.items():
        m = run_bolt(case_config(name, spec), 424242).metrics
        out[name] = {k: m[k] for k in KEYS}
    return out


if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(compute(), indent=2) + "\n")
    print(GOLDEN.read_text())

"""Fixed-seed golden metrics (SPEC.md test layer 4). Regenerate with scripts/update_golden.py
only after explaining the change in docs/log.md."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update_golden", ROOT / "scripts" / "update_golden.py")
assert spec is not None and spec.loader is not None
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

EXPECTED = json.loads(golden.GOLDEN.read_text())


@pytest.mark.parametrize("name", list(golden.CASES))
def test_golden_metrics(name):
    got = golden.run_bolt(golden.case_config(name, golden.CASES[name]), 424242).metrics
    for key, want in EXPECTED[name].items():
        assert got[key] == pytest.approx(want, rel=0.01, abs=0.05), f"{name}: {key}"

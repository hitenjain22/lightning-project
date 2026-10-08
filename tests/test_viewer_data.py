"""The website's data file (docs/viewer/bolt.json) is complete and self-consistent."""

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "docs" / "viewer" / "bolt.json"


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/make_viewer_data.py")
def test_viewer_data_is_consistent():
    d = json.loads(DATA.read_text())
    ch, pts = d["channel"], d["points"]
    nodes = np.array(ch["nodes"])
    segs = np.array(ch["segments"])
    assert nodes.shape[1] == 3 and segs.max() < len(nodes)
    assert len(ch["kind"]) == len(ch["arrival_s"]) == len(segs)
    n = len(pts["t"])
    assert n == d["metrics"]["n_points"] == len(pts["xyz"]) == len(pts["error_m"]) == len(pts["sigma_m"])
    assert np.all(np.diff(pts["t"]) >= 0)  # the page relies on time-ordered points
    w = d["waveform"]
    assert len(w["min"]) == len(w["max"]) and max(w["max"]) <= 1.0 and min(w["min"]) >= -1.0
    arrivals = [a for a in ch["arrival_s"] if a is not None]
    assert 0 < min(arrivals) <= max(arrivals) <= w["duration_s"]
    assert (DATA.parent / "thunder.wav").exists()

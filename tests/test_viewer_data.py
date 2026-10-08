"""The website's strike library (docs/viewer/bolts/) is complete and self-consistent."""

import json
from pathlib import Path

import numpy as np
import pytest

LIB = Path(__file__).resolve().parents[1] / "docs" / "viewer" / "bolts"


@pytest.mark.skipif(not (LIB / "index.json").exists(), reason="run scripts/make_viewer_data.py")
def test_viewer_library_is_consistent():
    index = json.loads((LIB / "index.json").read_text())
    strikes = index["strikes"]
    assert index["summary"]["n_strikes"] == len(strikes) > 1
    assert [s["id"] for s in strikes] == list(range(len(strikes)))
    assert index["summary"]["total_points"] == sum(s["n_points"] for s in strikes)
    for s in strikes:
        d = json.loads((LIB / f"{s['id']}.json").read_text())
        assert (LIB / f"{s['id']}.mp3").stat().st_size > 1000
        ch, pts = d["channel"], d["points"]
        nodes, segs = np.array(ch["nodes"]), np.array(ch["segments"])
        assert nodes.shape[1] == 3 and segs.max() < len(nodes)
        assert len(ch["kind"]) == len(ch["arrival_s"]) == len(ch["leader_m"]) == len(segs)
        n = len(pts["t"])
        assert n == d["metrics"]["n_points"] == s["n_points"] == len(pts["xyz"]) == len(pts["error_m"])
        assert np.all(np.diff(pts["t"]) >= 0)  # the page relies on time-ordered points
        w = d["waveform"]
        assert len(w["min"]) == len(w["max"]) and max(w["max"]) <= 1.0 and min(w["min"]) >= -1.0
        arrivals = [a for a in ch["arrival_s"] if a is not None]
        assert 0 < min(arrivals) <= max(arrivals) <= w["duration_s"]
        assert d["about"]["id"] == s["id"] and d["about"]["distance_km"] == s["distance_km"]

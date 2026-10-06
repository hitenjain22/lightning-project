import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from thunder.config import RunConfig, load_config
from thunder.experiments.runner import run_pipeline

REPO = Path(__file__).resolve().parents[1]


def test_base_config_loads():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg.sample_rate_hz == 8000.0
    assert cfg.oversample == 8


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("experiment: x\nseed: 1\nsampel_rate_hz: 8000\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_config_hash_is_stable_and_sensitive():
    a = RunConfig(experiment="x", seed=1)
    assert a.config_hash() == RunConfig(experiment="x", seed=1).config_hash()
    assert a.config_hash() != RunConfig(experiment="x", seed=2).config_hash()


def test_empty_pipeline_writes_results_folder(tmp_path):
    cfg = load_config(REPO / "configs" / "base.yaml", {"results_root": str(tmp_path)})
    run_dir = run_pipeline(cfg)

    assert run_dir.parent == tmp_path / "base"
    for name in ("config.resolved.yaml", "meta.json", "metrics.json", "arrays.npz"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "figures").is_dir()

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == cfg.seed and meta["config_hash"] == cfg.config_hash()
    assert "commit" in meta["git"]

    # The resolved config round-trips to the same config.
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert RunConfig.model_validate(resolved) == cfg

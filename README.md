# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M0 (setup) is done. The pipeline runs from a config and writes a results folder, but none of the physics exists yet.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_experiment.py configs/base.yaml
pytest
ruff check .
```

Each run writes `results/<experiment>/<run_id>/` containing the resolved config, git commit, seed and metrics.

# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M2 (thunder synthesis) is done. Channels become multi-mic recordings and WAV files (`python scripts/make_synthesis_demo.py configs/experiments/synthesis_demo.yaml`). Sensors and arrays (M3) are next.

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

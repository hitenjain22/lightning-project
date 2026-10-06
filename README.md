# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M3 (sensors and arrays) is done: 10 array layouts and the full recorder chain (noise, mic response, clocks, positions, ADC, flash-time error). Demo: `python scripts/make_sensor_demo.py configs/experiments/sensor_demo.yaml`. Next is M4: Method A reconstruction and metrics, the MVP.

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

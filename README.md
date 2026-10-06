# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M1 (channel generator) is done: five channel presets, statistics, and a gallery (`python scripts/make_channel_gallery.py configs/experiments/channel_gallery.yaml`). Thunder synthesis (M2) is next.

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

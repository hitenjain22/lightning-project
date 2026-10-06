# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M4 (the MVP) is done. Method A reconstructs 3D lightning channels from simulated thunder, end to end.

**First result (E1, ideal conditions, oracle atmosphere, 200 bolts at 1–3 km):** median point error **1.4 m** (95% CI 1.4–1.5 m), 83% of the main channel within 50 m of a reconstructed point, and 3.8 s per bolt end to end. Details are in `docs/results/e1_sanity.md`. Next: M5, a realistic atmosphere, and the first mismatched-model (headline) runs.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_experiment.py configs/base.yaml                      # one bolt, full pipeline
python scripts/run_experiment.py configs/experiments/e1_sanity.yaml     # E1 Monte Carlo (~4 min, 4 cores)
python scripts/make_figures.py results/e1_sanity/<run_id>               # figures + summary.md
pytest                     # ~75 s; slow tests: pytest -m slow
ruff check . && mypy src
```

Each run writes `results/<experiment>/<run_id>/` containing the resolved config, git commit, seed and metrics.

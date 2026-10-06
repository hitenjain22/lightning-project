# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M6 is done. The project has three reconstruction methods: A (plane-wave TDOA), B (SRP-PHAT, several sources per window) and C (absolute-time multilateration). All run through a realistic atmosphere (lapse rate, wind, humidity, ISO absorption, ground reflection) with exact ray tracing.

**Results so far** (Method A, 5 mics, 50 m aperture, bolts at 1–3 km):

| Condition | Median point error |
| --- | --- |
| E1: still air, true atmosphere known (oracle) | **1.4 m** |
| M5: realistic atmosphere, true atmosphere known (oracle) | **1.7 m** |
| M5: realistic atmosphere, assumed with no wind and an approximate lapse rate (realistic) | **137 m** |
| M5: realistic atmosphere, straight rays | 153 m |
| M6: Method B, realistic atmosphere, oracle | 1.8 m, with 88% of the main channel within 50 m (A: 83%) |

Unknown wind dominates the realistic error. Details are in `docs/results/e1_sanity.md`, `docs/results/m5_atmosphere.md` and `docs/results/m6_methods.md`. Next: M7, experiments E2–E5.

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
python scripts/make_atmosphere_figures.py configs/experiments/atmosphere_figures.yaml  # ray fans
pytest                     # ~75 s; slow tests: pytest -m slow
ruff check . && mypy src
```

Each run writes `results/<experiment>/<run_id>/` containing the resolved config, git commit, seed and metrics.

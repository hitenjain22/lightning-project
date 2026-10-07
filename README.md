# thunder: acoustic lightning reconstruction

This simulation framework generates synthetic lightning channels, synthesizes the thunder that a microphone array would record, and reconstructs the 3D channel from those recordings. See `SPEC.md` for the full plan.

**Status:** M7 is done (experiments E2–E5). The project has three reconstruction methods: A (plane-wave TDOA), B (SRP-PHAT, several sources per window) and C (absolute-time multilateration). All run through a realistic atmosphere (lapse rate, wind, humidity, ISO absorption, ground reflection) with exact ray tracing.

**Results so far** (Method A, 5 mics, 50 m aperture, bolts at 1–3 km):

| Condition | Median point error |
| --- | --- |
| E1: still air, true atmosphere known (oracle) | **1.4 m** |
| M5: realistic atmosphere, true atmosphere known (oracle) | **1.7 m** |
| M5: realistic atmosphere, assumed with no wind and an approximate lapse rate (realistic) | **137 m** |
| M5: realistic atmosphere, straight rays | 153 m |
| M6: Method B, realistic atmosphere, oracle | 1.8 m, with 88% of the main channel within 50 m (A: 83%) |

Unknown wind dominates the realistic error. Details are in `docs/results/e1_sanity.md`, `docs/results/m5_atmosphere.md` and `docs/results/m6_methods.md`.

**M7 findings** (realistic sensors, paired Monte Carlo; full write-ups in `docs/results/e2`–`e5`):

| Experiment | Headline |
| --- | --- |
| **E2 array design** | Symmetric layouts win: square + center or rings at about 50 m (0.04°, about 3 m at 1–3 km). A surrogate built from geometry alone (Cramér–Rao bound over the channel's directions, plus the plane-wave curvature bias that asymmetric layouts suffer) ranks 20 layouts like full simulation (Spearman ρ = 0.98). Larger apertures are more accurate but lose most of the channel |
| **E3 error budget** | For 20 m median error: clocks synced to about 0.4 ms, mic positions to about 12 cm, temperature to about 2.5 K, wind known to about 0.7 m/s. Unknown wind (about 26 m per m/s) and position or sync errors dominate; flash time and SNR barely matter for accuracy (SNR costs coverage) |
| **E4 methods** | With the true atmosphere A, B and C reach 3.0–3.3 m; B recovers the most channel (86% main, 57% branches). Without the wind, every method sits near 130 m: the atmosphere matters about 40× more than the method |
| **E5 limits** | Angular accuracy holds at about 0.035° out to 15 km. Main-channel coverage drops below 50% at about 11 km, from the refraction shadow and overlapping arrivals, not from noise |

Next: M8, Method D (joint atmosphere estimation) and E6–E7.

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
python scripts/e2_array_design.py        # E2 (~50 min, 4 cores); E3-E5: scripts/e3_error_budget.py, e4_methods.py, e5_limits.py
python scripts/e4_methods.py --n-bolts 5  # any experiment, small (~5 min)
python scripts/e5_limits.py --analyze results/e5_limits/<run_id> --docs-prefix e5_   # re-analyze a run folder
pytest                     # ~2 min; slow tests: pytest -m slow
ruff check . && mypy src
```

Each run writes `results/<experiment>/<run_id>/` containing the resolved config, git commit, seed and metrics.

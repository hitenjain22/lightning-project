# CLAUDE.md

Project: acoustic lightning reconstruction (simulation only). The full spec is in `SPEC.md`; read it before starting any milestone.

## Working rules

- Work one milestone at a time. Do not start the next until the current one's tests pass and the user approves.
- At the start of each phase, propose the module interfaces and a short plan, then wait for approval before writing large amounts of code.
- Never invent physical constants silently. All constants live in `src/thunder/constants.py` with a comment naming the source; anything uncertain is marked `# VERIFY`.
- Record every modeling assumption in `docs/assumptions.md` and every design decision in `docs/log.md`.
- Pass a `numpy.random.Generator` explicitly everywhere; no global random state.
- Prefer clear vectorized NumPy over clever code. Optimize only after profiling.
- Commit after each working milestone with a descriptive message.

## Conventions (from SPEC.md)

- Coordinates: local East-North-Up (ENU), origin at the array centroid, z up, ground at z = 0.
- Units: SI only (m, s, Pa, K). Radians internally; degrees only in plots.
- Time: t = 0 is the return stroke.
- Default recording sample rate 8 kHz, with internal oversampling for fractional delays.

## Inverse-crime guard

- Reconstruction code (`src/thunder/recon/`) must never read `Recording.truth` or import `GroundTruth`. `tests/test_recon_isolation.py` enforces this.
- Synthesis always uses finer resolution and fuller physics than reconstruction assumes. Headline results come from mismatched-model runs; oracle runs are labeled as such.

## Commands

- Environment: `uv venv --python 3.12 && uv pip install -e ".[dev]"` (activate `.venv`).
- Tests: `pytest` (slow tests are skipped by default; run them with `pytest -m slow`).
- Lint: `ruff check .`; types: `mypy src`.
- Run a pipeline: `python scripts/run_experiment.py configs/base.yaml`.

# Lightning from Thunder

Reconstructing lightning channels in 3D from the sound of their thunder: a simulation framework,
four reconstruction methods, seven experiments and an interactive 3D website.

**[Try it in your browser →](https://hitenjain22.github.io/lightning-project/)**

[![The website rebuilding a strike from its thunder](docs/media/site/rebuild.jpg)](https://hitenjain22.github.io/lightning-project/)

*A simulated strike 2.9 km away being rebuilt from the thunder recorded by five microphones. Each amber line is a direction measured at the array; each point is placed from one 0.1 s slice of the recording and coloured by its distance from the true channel (blue).*

## The idea

Thunder is an acoustic recording of the lightning channel. Every metre of the channel sends out a pressure pulse when the return stroke heats it, and those pulses reach microphones a few tens of metres apart at slightly different times. The delays give each sound's direction; the time since the flash gives its distance. Together, slice by slice through the rumble, they trace the channel in 3D.

This project:

1. **generates** realistic lightning channels (tortuous, branched, in-cloud, multi-stroke);
2. **synthesizes** the thunder a microphone array would record: a physical source model, exact ray tracing through a stratified, windy atmosphere, absorption, ground reflection, and realistic sensors (mic responses, noise, clock and position errors);
3. **reconstructs** the 3D channel from the recordings with four methods; and
4. **evaluates** the result against the ground truth in paired Monte Carlo experiments with bootstrap confidence intervals.

The reconstruction code never sees the ground truth; a test enforces this. The [technical report](docs/report.md) explains the physics, the methods and every experiment.

## The website

The [website](https://hitenjain22.github.io/lightning-project/) walks through one strike in four steps: the leader and return stroke, the thunder spreading out (the part of the channel you are hearing lights up, from exact ray-traced arrival times, with the real simulated recording), the reconstruction playing in recording time, and the result. *New strike* picks another of 60 strikes; *All 60* shows them together.

| Result: rebuilt (amber) beside the true channel (blue) | All 60 strikes around the array |
| --- | --- |
| [![Side-by-side result](docs/media/site/result.jpg)](https://hitenjain22.github.io/lightning-project/#strike=12&step=4&side=1) | [![All strikes](docs/media/site/all.jpg)](https://hitenjain22.github.io/lightning-project/#storm) |

Every strike on the site was simulated and reconstructed by this repository's pipeline (`scripts/make_viewer_data.py`), with varied channel types, distances (1.5–6 km), winds and temperatures. None were left out: the median error per strike is **4.0 m** (range 2.6–9.2 m), with 83% of the main channel recovered, when the atmosphere is known. The site is plain HTML, CSS and JavaScript with Three.js (`docs/index.html`, `docs/viewer/`), served by GitHub Pages.

## Key results

All experiments use the same bolts across compared variants (paired), realistic sensors and 95% bootstrap CIs. Write-ups with figures are in [`docs/results/`](docs/results).

| | Finding |
| --- | --- |
| **E1** sanity | Ideal conditions: 1.4 m median point error |
| **E2** array design | Symmetric layouts (square + center, rings) at about 50 m are best: about 0.04° direction error, about 3 m at 1–3 km. A **surrogate built from geometry alone** (Cramér–Rao bound over the channel's directions, plus the wavefront-curvature bias of plane-wave methods on asymmetric layouts) ranks 20 layouts like full simulation (Spearman ρ = 0.98) |
| **E3** error budget | For 20 m median error: clocks within about 0.4 ms, mic positions within about 12 cm, temperature within about 2.5 K, wind known to about 0.7 m/s. Unknown wind costs about 26 m per m/s |
| **E4** methods | With the true atmosphere, Methods A–C reach 3.0–3.3 m; B recovers the most channel (86% of the main channel, 57% of branches). With the wind unknown, every method sits near 130 m: the atmosphere matters about 40× more than the method |
| **E5** limits | Direction accuracy holds to 15 km (about 0.035°). Coverage drops below 50% near 11 km, from the refraction shadow and overlapping arrivals, not noise |
| **E6** four phones | As-is (phone GPS positions, hand-clap sync): about 0.5 km error. Tape-measured positions plus 0.1 ms sync reach professional-kit quality (180 m with B; 163 m with Method D, which calibrates the atmosphere and the array); with the atmosphere known, about 25 m |
| **E7** self-calibration | Method D recovers the sound-speed profile and gives near-honest error bars on realistic data (2σ coverage 0.68–0.87 vs 0.74 nominal). It does **not** recover the wind from a few realistic bolts: after the sources absorb a cross-wind, its signature is microsecond-level, the size of millimetres of mic position. Independent wind data is the route to the 3 m oracle |

Method D's uncertainty is calibrated on controlled cases (1/2/3σ coverage 0.19 / 0.75 / 0.97 vs 0.20 / 0.74 / 0.97 nominal). Self-calibrating on a flat array is the one exception, with a light 3σ tail; `docs/log.md` explains why.

## What's inside

| Stage | Highlights |
| --- | --- |
| Channel generator (`thunder.channel`) | Biased random walk with the measured turn-angle distribution, recursive branching, in-cloud sections, multiple strokes |
| Thunder synthesis (`thunder.acoustics`) | Few's N-wave source with energy-calibrated amplitude, sub-segment tortuosity, exact continuous deposition on an oversampled grid |
| Atmosphere (`thunder.atmosphere`) | Stratified temperature, humidity and power-law wind; exact eigenray solver (validated against an ODE ray tracer); refraction shadow zones; ISO 9613-1 absorption; ground reflection |
| Sensors (`thunder.sensors`) | Array layouts; mic response presets (measurement to phone); coherent ambient, wind and rain noise; clock offsets and drift; position and flash-time errors |
| Reconstruction (`thunder.recon`) | **A** plane-wave TDOA (two-pass β-PHAT GCC, constrained slowness fit). **B** steered response power, several sources per window. **C** absolute-time multilateration through any atmosphere. **D** Bayesian: calibrated error ellipsoids, plus self-calibration of the sound speed, wind and the array itself, jointly over several bolts |
| Evaluation (`thunder.eval`, `thunder.experiments`) | Point error, coverage, angular, branch and strike-point metrics, uncertainty calibration; paired Monte Carlo with common random numbers and a stage cache; cluster bootstrap CIs |
| Website (`docs/index.html`, `docs/viewer/`) | Three.js viewer for a library of 60 strikes exported by `scripts/make_viewer_data.py`: playback synced to the recorded thunder, per-point error colours, 2σ uncertainty bars, all-strikes view |

## Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate   # or: python3.12 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"                           # or: pip install -e ".[dev]"
python scripts/run_experiment.py configs/base.yaml   # one bolt end to end, ~10 s (first run ~1-2 min: caches)
```

This writes `results/base/<run_id>/` (its `reconstruction.png` looks like [this](docs/figures/demo_reconstruction.png)):
- `figures/reconstruction.png` and an interactive `figures/reconstruction.html`;
- the recorded thunder as WAV files (`audio/`);
- all metrics (`metrics.json`);
- the resolved config, seed and git commit, for reproducibility.

```bash
pytest                    # 209 tests, ~3-8 min depending on the machine (slow ones: pytest -m slow)
ruff check . && mypy src  # lint and types
```

To view the website locally, run `python -m http.server -d docs` and open http://localhost:8000.

## Reproduce the experiments

```bash
python scripts/run_experiment.py configs/experiments/e1_sanity.yaml   # E1 Monte Carlo (~4 min, 4 cores)
python scripts/e2_array_design.py      # E2-E7: one script each (e3_error_budget, e4_methods, e5_limits,
python scripts/e4_methods.py --n-bolts 5   #   e6_consumer, e7_self_calibration); --n-bolts for a small run
python scripts/e5_limits.py --analyze results/e5_limits/<run_id>   # tables and figures from a run folder
```

The full experiments take from about 30 minutes (E4, E6) to several hours (E7) on 4 cores. Every run folder holds its resolved config, seed and git commit.

Regenerate the website's strike library with `python scripts/make_viewer_data.py --n 60` (about 40 minutes on 7 cores), and the report's media with `python scripts/make_media.py --docs`.

## More media

- [Animated reconstruction (GIF)](docs/media/hero.gif) and a [stand-alone Plotly 3D view with uncertainty](docs/media/reconstruction.html) (download and open in a browser) of the same 2.5 km strike.
- The thunder of one strike shape heard from [1 km](docs/audio/thunder_1km.wav), [3 km](docs/audio/thunder_3km.wav), [8 km](docs/audio/thunder_8km.wav) and [15 km](docs/audio/thunder_15km.wav).
- Every figure from the experiments is in [`docs/figures/`](docs/figures) and the result write-ups.

## Project documents

- [`docs/report.md`](docs/report.md): the technical report.
- [`SPEC.md`](SPEC.md): the full plan, physics background and milestones.
- [`docs/log.md`](docs/log.md): every design decision and bug found, with the evidence.
- [`docs/assumptions.md`](docs/assumptions.md): every modeling assumption. Uncertain physical values are marked VERIFY.

**Status:** version 1.1. Milestones M0–M9 are done (simulation, four methods, experiments E1–E7, media and report), plus the interactive website and its 60-strike library.

**Limitations:** simulation only. Field validation with a real array, ideally alongside lightning photographs as ground truth, is the natural next step. The acoustic source efficiency and several sensor and noise levels are literature-based estimates (see `docs/assumptions.md`).

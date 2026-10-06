# Acoustic Lightning Reconstruction — Project Spec

Oct 5, 2026 · @Hiten Jain

## Overview

Build a Python simulation framework that generates synthetic lightning channels, synthesizes the thunder a microphone array would record, and reconstructs the 3D channel from those recordings. The project is simulation-only: no field data. The headline result is not "it works" but *how well, under what conditions, and with what array design*.

**Goals**

1. A forward model realistic enough that reconstruction is not trivially easy (tortuous channels, directional emission, refracting atmosphere, real sensor flaws).
2. A reconstruction pipeline that outputs a 3D point cloud with per-point uncertainty.
3. Quantitative experiments on array geometry, robustness to errors, method comparison, and range limits.
4. Full reproducibility: every figure regenerates from a config file and a random seed.

**Success criteria**

- Ideal-conditions round trip (still air, no noise, 5+ mics): reconstructed points sit close to the true channel for bolts within 3 km. Provisional target: median error under \~50 m; revise after the first working pipeline.
- Error degrades smoothly and explainably as wind, noise, sync error and range increase, and every breaking point is documented.
- Headline numbers come from *mismatched-model* runs (see Testing), never from the reconstruction assuming the exact physics used to synthesize the data.

**Out of scope (for now):** real recordings, electromagnetic lightning sensing, detailed cloud electrification physics.

**How to use this spec with Claude Code**

- Save this doc as `SPEC.md` in the repo root. Copy the working rules below into `CLAUDE.md`.
- First prompt: "Read SPEC.md. Do Milestone M0 only, then stop and summarize what you built and what's next."
- Then go one milestone at a time.

**Working rules for Claude Code**

- Work one milestone at a time. Do not start the next until the current one's tests pass and the user approves.
- At the start of each phase, propose the module interfaces and a short plan, then wait for approval before writing large amounts of code.
- Never invent physical constants silently. All constants live in `constants.py` with a comment naming the source; anything uncertain is marked `# VERIFY`.
- Record every modeling assumption in `docs/assumptions.md` and every design decision in `docs/log.md`.
- Pass a `numpy.random.Generator` explicitly everywhere; no global random state.
- Prefer clear vectorized NumPy over clever code. Optimize only after profiling.
- Commit after each working milestone with a descriptive message.

## Physics background

Thunder is the sum of pressure pulses from every piece of the channel, arriving at different times because each piece is a different distance away. Reconstruction inverts that: arrival time gives distance, and time differences across mics give direction.

**Thunder generation**

- A return stroke heats the channel to roughly 30,000 K in microseconds. The pressure jump launches a shock that decays into an acoustic N-wave within meters to tens of meters.
- The stroke travels up the channel at a large fraction of light speed, so for acoustics the whole channel fires at effectively the same instant (t = 0).
- Each short straight segment radiates most strongly broadside (perpendicular to itself) and weakly end-on. Segments that happen to face the listener produce the loud "claps"; the spread of distances produces the rumble.
- Few's model ties the dominant frequency to the energy deposited per unit length through a relaxation radius. Typical thunder peaks in the tens to low hundreds of Hz. Treat exact constants as `# VERIFY`.

```latex
R_0 = \sqrt{\frac{E_\ell}{\pi p_0}}, \qquad f_{\text{peak}} \approx 0.63\,\frac{c_0}{R_0}
```

Here E\_ℓ is energy per unit length (J/m), p0 ambient pressure, c0 sound speed. Constants are from memory of Few (1969); verify before relying on them.

**Propagation**

- Sound speed depends on temperature: c ≈ 20.05 √T m/s with T in kelvin (≈ 343 m/s at 20 °C).
- Temperature usually drops \~6.5 K/km with height, so sound speed drops too. Rays bend upward, which creates an acoustic shadow zone; thunder is rarely heard beyond \~15–25 km.
- Wind adds vectorially. The effective sound speed along a ray is c plus the wind component along the propagation direction.
- Air absorbs high frequencies more than low (ISO 9613-1), so distant thunder sounds lower and smoother.
- The ground reflects sound, adding an image path.

**Why one microphone cannot recover shape**

With the flash time t0 known, one mic converts each arrival time into a range. Every point on that sphere produces the same arrival time, so one mic gives a histogram of range, never a shape.

```latex
r = c\,(t_{\text{arrival}} - t_0)
```

**Arrays and time difference of arrival (TDOA)**

With mics at positions m\_i, a source at position s reaches each mic at a slightly different time. The differences give direction; combined with the range from t0, each chunk of sound becomes a 3D point.

```latex
t_i = t_0 + \frac{\lVert \mathbf{s} - \mathbf{m}_i \rVert}{c}, \qquad \tau_{ij} = t_j - t_i
```

When the array is much smaller than the source distance (far field), arrivals are plane waves from unit direction u, and the delays become linear in the slowness vector u/c:

```latex
\tau_{ij} \approx -\frac{(\mathbf{m}_j - \mathbf{m}_i)\cdot \mathbf{u}}{c}
```

That linearity is what makes the baseline solver a simple least-squares problem.

**Conventions (use everywhere)**

- Coordinates: local East-North-Up (ENU), origin at the array centroid, z up, ground at z = 0.
- Units: SI only (meters, seconds, pascals, kelvin). Angles in radians internally, degrees only in plots.
- Time: t = 0 is the return stroke.
- Sample rate: default 8 kHz for recordings, with internal oversampling for fractional delays.

## Tech stack and repo structure

Python 3.11+, installable as a package (`pip install -e .`), with YAML configs driving every run.

| Need | Library |
| --- | --- |
| Arrays, linear algebra | numpy |
| Signal processing, ODEs, optimization, spatial queries | scipy (`signal`, `integrate.solve_ivp`, `optimize`, `spatial.cKDTree`) |
| Configs and data validation | pydantic + PyYAML |
| Results tables | pandas |
| Static plots | matplotlib |
| Interactive 3D | plotly (export to HTML) |
| Clustering | scikit-learn (DBSCAN) |
| MCMC (Phase 5, Method D) | emcee |
| Audio export | soundfile |
| Speed-ups if profiling demands | numba |
| Tests, lint, types | pytest, ruff, mypy (lenient) |

**Repo layout**

```
lightning-acoustics/
  pyproject.toml
  README.md
  SPEC.md                # this document
  CLAUDE.md              # working rules
  configs/
    base.yaml
    experiments/         # one YAML per experiment
  src/thunder/
    constants.py         # every physical constant, with source comments
    types.py             # Channel, Recording, Atmosphere, Array, Reconstruction dataclasses
    config.py            # pydantic models for YAML
    channel/             # Phase 1: generator.py, branching.py, stats.py
    acoustics/           # Phase 2: source.py, synth.py
    atmosphere/          # Phase 3: profiles.py, raytrace.py, absorption.py, ground.py
    sensors/             # Phase 4: arrays.py, corruption.py
    recon/               # Phase 5: preprocess.py, tdoa.py, srp.py, multilat.py, bayes.py, postprocess.py
    eval/                # Phase 6: metrics.py
    experiments/         # Phase 7: runner.py, sweeps.py, design.py
    viz/                 # Phase 8: plots.py, plot3d.py, animate.py
  scripts/               # CLI entry points (run_experiment.py, make_figures.py)
  tests/
  notebooks/             # exploration only, never the source of truth
  docs/
    assumptions.md
    log.md
    results/             # one markdown summary per experiment
  results/               # gitignored run outputs
```

**Core data contracts** (define these in M0 so phases plug together)

- `Channel`: node positions (N×3), segment index pairs, energy per unit length per segment, branch id, main-channel flag, metadata (seed, generator params).
- `Atmosphere`: temperature, wind and humidity profiles as functions of height, plus a method returning travel time and arrival direction between two points.
- `Array`: nominal mic positions (M×3), true mic positions, per-mic clock offset and drift, frequency response, noise settings.
- `Recording`: signals (M×samples), sample rate, nominal mic positions, reported flash time, plus a hidden ground-truth block (true t0, true positions, per-segment arrival times at each mic).
- `Reconstruction`: points (K×3), per-point covariance (K×3×3), window times, quality scores, method name, config hash.

**Run outputs**

Each run writes to `results/<experiment>/<run_id>/`: the resolved config, git commit hash, seed, arrays as `.npz`, metrics as CSV/JSON, and figures. Nothing in a figure should be unreproducible from that folder.

## Phase 1: Lightning channel generator

Produce realistic 3D cloud-to-ground channels whose tortuosity and branching match published statistics, with presets from simple to hard.

**Model**

- Biased random walk from the cloud downward to a ground strike point. Typical negative cloud-to-ground flashes start several km up (roughly 4–7 km); make the start height a config range.
- Each step: take the previous direction, rotate it by a random angle, then blend in a bias toward the target strike point so the channel actually reaches the ground.
- Turn-angle distribution: Hill (1968) reported a mean absolute direction change of about 16° between successive segments of tens of meters. Use that as the default, marked `# VERIFY`.
- Base segment length \~10 m (configurable). Phase 2 subdivides finer.
- Branching: at each step, spawn a branch with probability p\_branch. Branches leave at an angle offset, take shorter random lengths, never reach the ground, and carry a lower energy fraction. Allow sub-branches to a configurable depth.
- Optional in-cloud horizontal channel: a long near-horizontal segment at 5–7 km altitude feeding the top of the vertical channel. Real flashes often have kilometers of these.
- Optional multiple strokes: re-fire the main channel (not the branches) at later times, typically tens of ms apart.

**Presets**

1. `straight`: nearly vertical, low tortuosity, no branches (for debugging).
2. `tortuous`: realistic turn angles, no branches.
3. `branched`: realistic turn angles plus branches.
4. `with_incloud`: branched plus a long horizontal in-cloud section.
5. `multi_stroke`: branched with 2–4 strokes.

**Config parameters**

Start height range, strike distance range from the array (default 1–8 km), strike azimuth range, segment length, turn-angle distribution and scale, target bias strength, branch probability, branch angle and length distributions, max branch depth, energy per unit length (main vs branch), in-cloud length and altitude, number of strokes and spacing.

**Outputs**

A `Channel` object, plus `stats.py` functions computing turn-angle histogram, total length, branch count, horizontal extent, and fractal dimension via box counting.

**Tests**

- Segments form a connected tree; the main channel touches z = 0 within one segment length of the target.
- Turn-angle statistics on 500 generated channels fall within tolerance of the configured distribution.
- The same seed gives an identical channel; different seeds differ.
- No segment goes below ground.

**Deliverable for review:** a gallery figure of 12 channels across presets, plus the stats table.

## Phase 2: Thunder synthesis (forward model)

Turn a `Channel` into the pressure signal each mic would record, first in a uniform still atmosphere with straight-line propagation. Phase 3 swaps in the realistic atmosphere behind the same interface.

**Source model (string of pearls)**

- Subdivide every segment into point emitters spaced \~0.5–1 m apart, all firing at t = 0 (per stroke).
- Each emitter radiates an N-wave pulse. Its duration and dominant frequency follow from the segment's energy per unit length (Physics background); amplitude scales with energy and emitter spacing.
- Do not hand-code a directivity pattern. Summing many coherent point emitters along a straight segment naturally produces strong broadside and weak end-on radiation through interference. This is the core idea of the Ribner and Roy approach and keeps the physics honest.
- Ignore nonlinear shock propagation beyond the first few meters (log in `assumptions.md`; stretch goal later).

**Efficient summation**

A 5 km channel at 0.5 m spacing is \~10,000 emitters × M mics. Do not convolve per emitter.

1. For each emitter and mic, compute travel time and amplitude (1/r spreading in Phase 2).
2. Scatter-add amplitude impulses into an oversampled time grid (e.g., 8× the output rate) with linear fractional-delay weights.
3. Group emitters by pulse shape (bin by energy per unit length), and convolve each group's impulse train with its N-wave once.
4. Low-pass and decimate to the output sample rate.

**Ground truth to save**

For every segment and every mic: arrival time, amplitude, and arrival direction. Evaluation and debugging depend on this.

**Outputs**

Clean `Recording` objects; WAV files per mic (so you can actually listen to the synthetic thunder); spectrograms.

**Tests**

- Single point source: arrival time equals r/c within one output sample; amplitude scales as 1/r.
- Straight vertical line from height H at horizontal distance D: first arrival at D/c, last arrival at √(D² + H²)/c.
- Straight horizontal segment: received energy broadside is much higher than end-on.
- Rumble duration roughly equals (r\_max − r\_min)/c for any channel.
- Spectrum peak lands in the expected tens-to-hundreds of Hz band for default energy.

**Deliverable for review:** waveform stack and spectrogram for one bolt across 4 mics, plus WAV files.

## Phase 3: Atmosphere and propagation

Replace straight-line, constant-speed propagation with a horizontally stratified, moving atmosphere. Each effect is a separate toggle so experiments can isolate it.

**1. Sound speed profile**

- Temperature profile T(z): surface temperature (default 25 °C) with a lapse rate (default 6.5 K/km). Optional inversion layer (temperature rising with height over a configurable band).
- c(z) from T(z); include the small humidity correction.

**2. Wind profile**

- Power-law profile u(z) = u\_ref (z / z\_ref)^α with configurable speed, direction and exponent; optional direction shear with height.
- Effective sound speed along a ray: c(z) + w(z)·n, where n is the propagation direction.

**3. Ray tracing**

- Integrate the ray equations for a stratified moving medium with `scipy.integrate.solve_ivp`.
- Eigenrays (the ray from a given source that actually hits a given mic) need a shooting method, which is too slow per emitter. Instead precompute travel-time tables.
- No wind: a 2D table T(source height, horizontal distance). With wind: a 3D table T(source height, horizontal distance, azimuth relative to wind). Mics are all near ground, so one table serves every mic via offsets.
- Each table entry stores travel time, arrival elevation and azimuth at the receiver, and ray-tube spreading amplitude. Interpolate with `RegularGridInterpolator`.
- Mark shadow-zone cells (no eigenray). Default: no direct arrival there; optional weak diffracted arrival as a stretch goal.

**4. Absorption**

- Implement the ISO 9613-1 absorption coefficient α(f, T, relative humidity, pressure).
- Apply as a frequency-domain filter per path, binning emitters by path length so you filter groups, not individual emitters.

**5. Ground reflection**

- Mics at a configurable height (default 1.5 m). Add an image-source path.
- Start with a rigid ground (reflection coefficient ≈ 1); optional impedance model (Delany–Bazley) later.

**6. Turbulence (stretch)**

Random travel-time jitter correlated over space and time, plus amplitude scintillation. This is what makes real cross-correlations messy.

**Key design point**

Synthesis always uses the full, high-resolution atmosphere. Reconstruction may assume (a) constant c, (b) the true atmosphere (oracle, upper bound), or (c) a mismatched atmosphere such as the right surface temperature but wrong lapse rate and no wind knowledge. Option (c) is the realistic case and should drive headline results.

**Tests**

- Uniform still atmosphere: ray-traced travel times match straight-line r/c to within 0.1%.
- Standard lapse rate: rays curve upward and a shadow zone appears at the expected order of distance.
- Downwind arrivals are earlier than upwind arrivals at equal distance.
- Interpolated table values match direct ray traces at random test points within tolerance.

**Deliverable for review:** ray-fan plots for still air, lapse rate, and strong wind; travel-time error map of straight-line vs ray-traced.

## Phase 4: Sensor and array model

Corrupt clean recordings the way real hardware would, while keeping the true corruptions hidden in the ground-truth block so evaluation can attribute errors.

**Array geometry generators** (all take an aperture parameter, default range 5–500 m)

- Equilateral triangle (3 mics, the minimum).
- Square, and square plus center (4–5 mics).
- Circle of N mics.
- L-shape and cross.
- 3D: square plus one mic raised on a mast (\~10 m) to improve elevation resolution.
- Random positions within a disk.
- Distributed: two or three small sub-arrays separated by hundreds of meters to km, enabling triangulation between sub-arrays.
- Free-form: arbitrary positions (needed for Phase 7 optimization).

**Microphone model**

- Frequency response as a band-pass filter. Presets: `measurement` (≈2 Hz–2 kHz), `audio` (≈20 Hz–20 kHz), `phone` (high-pass ≈100 Hz, which removes much of thunder's energy).
- Self-noise floor, clipping level, and ADC bit depth.

**Noise**

- Background noise (pink or brown) at a configurable SNR.
- Wind noise: strong at low frequency, scales with wind speed, and incoherent between mics more than a few meters apart.
- Rain noise: broadband, configurable level.

**Timing errors**

- Per-mic constant clock offset, drift in ppm, and sample jitter.
- Presets: `gps_synced` (≈1 µs), `shared_interface` (≈0, one multichannel recorder), `hand_synced` (≈1–10 ms, separate recorders aligned by a clap).

**Position errors**

Per-mic Gaussian error on true positions (σ from 1 cm to 1 m). Reconstruction only sees nominal positions.

**Flash-time (t0) error**

- Gaussian error on the reported t0. Presets: `photodiode` (≈10 µs), `lightning_network` (≈1 ms), `video_30fps` (uniform within one 33 ms frame).
- A t0 error shifts every range estimate by c × error, so 33 ms ≈ 11 m. Worth showing explicitly.

**Tests**

- With all corruptions off, output equals the clean input exactly.
- A known clock offset shifts the cross-correlation peak by exactly that offset.
- Measured SNR matches configured SNR within 0.5 dB.

**Deliverable for review:** plot of each array preset; before/after waveforms for each corruption type.

## Phase 5: Reconstruction algorithms

Build four methods of increasing sophistication behind one interface, `reconstruct(recording, array_nominal, atmosphere_assumed, config) -> Reconstruction`, so experiments can swap them freely.

**Shared preprocessing**

1. Remove DC; band-pass (default 10–300 Hz, configurable).
2. Optional spectral whitening.
3. Slide windows over the recording from first arrival to end of rumble (default 100 ms windows, 50% overlap). Each window is one batch of channel segments arriving at roughly the same time.

**Method A: plane-wave TDOA (baseline, build first)**

1. For every mic pair in a window, compute the generalized cross-correlation with PHAT weighting (GCC-PHAT).
2. Find the peak lag, restricted to physically possible lags |τ\_ij| ≤ d\_ij / c, and refine to sub-sample precision with parabolic interpolation.
3. Solve the linear least-squares problem for the slowness vector (delays are linear in u/c; see Physics background). Normalize to get direction u.
4. Range R = c × (window center time − t0). Point = array centroid + R·u.
5. Quality gates: correlation peak height, least-squares residual, and whether the solved slowness magnitude is close to 1/c. Drop windows that fail.

**Method B: steered response power (SRP-PHAT)**

- For each window, grid-search candidate 3D points on the spherical shell at range R ± δ. Score each point by summing GCC-PHAT values at the delays that point predicts.
- Take multiple local maxima per window, which lets it find two branches arriving at the same time (Method A cannot).
- Coarse-to-fine grid for speed; this method handles near-field geometry automatically.

**Method C: absolute-time multilateration**

- Use per-mic absolute arrival times (with t0) rather than only differences. Each mic gives its own range; solve nonlinear least squares with `scipy.optimize.least_squares` and a robust loss (`soft_l1` or `huber`).
- Requires matching the same acoustic feature across mics; use cross-correlation alignment to pair them.

**Method D: Bayesian with uncertainty**

- Likelihood over source position given TDOAs, with noise terms for timing error, position error, and sound-speed uncertainty.
- Per window: Laplace approximation (fast) or MCMC with emcee (slow, reference) to produce a covariance ellipsoid per point.
- Self-calibration extension: treat sound speed and wind as unknown global parameters shared across all windows and estimate them jointly with the points. A big potential win, since the atmosphere is usually unknown.

**Refraction-aware variants**

Methods B, C and D accept an assumed `Atmosphere`. When it is not constant-speed, predicted delays come from the Phase 3 travel-time tables instead of straight lines.

**Post-processing**

1. Remove outliers with DBSCAN.
2. Build a channel skeleton: a minimum spanning tree over the cleaned points, pruned of short spurs.
3. Optional smoothing along the skeleton.

**Tests**

- Single point source, ideal conditions: Method A recovers direction within 0.5° and range within 1%.
- Two simultaneous sources at different directions: Method B finds both.
- Method D's posterior covers the truth at the nominal rate on simple synthetic cases.

## Phase 6: Evaluation metrics

Score each reconstruction on accuracy (are points near the true channel?) and completeness (is the whole channel found?), and break errors down by cause.

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Point error | Distance from each reconstructed point to the nearest point on the true channel polyline; report median and 90th percentile | Accuracy, like precision |
| Coverage at d | Fraction of true channel length (main + branches) within distance d of any reconstructed point; report the curve over d | Completeness, like recall |
| Chamfer distance | Mean nearest-neighbor distance in both directions between reconstructed points and densely sampled truth | One-number summary |
| Radial vs transverse error | Split each point's error into the component along the line of sight and the component perpendicular | Radial error comes from t0 and sound-speed errors; transverse from TDOA errors |
| Strike-point error | Distance between estimated and true ground contact point | The most practical single output |
| Uncertainty calibration | Fraction of true points inside each point's 1σ, 2σ, 3σ ellipsoid vs expected 19.9%, 73.9%, 97.1% (chi-square, 3 degrees of freedom) | Checks whether Method D's error bars are honest |
| Runtime | Seconds per bolt per method | Feasibility |

**Aggregation**

- Every configuration runs as a Monte Carlo over many random bolts (default 200) with varied range, azimuth and preset.
- Report means with bootstrap 95% confidence intervals.
- Always also report error vs range, vs source altitude, and vs SNR, not just averages.

**Tests**

- A reconstruction identical to the truth scores zero error and full coverage.
- For a straight-line channel, shifting all points 10 m perpendicular to it gives a point error of exactly 10 m.
- Calibration check on synthetic Gaussian samples returns the nominal fractions.

## Phase 7: Experiments

Seven experiments turn the framework into findings. Each gets its own YAML config, a script, a saved results table, figures, and a short summary in `docs/results/`.

**E1. Sanity benchmark**

Ideal conditions (still air, no noise, perfect sync, 5 mics, 50 m aperture), Method A, 200 bolts at 1–3 km. Confirms the pipeline works end to end before anything else.

**E2. Array design optimization (the main OR contribution)**

- Question: for a fixed number of mics and maximum aperture, which layout minimizes reconstruction error over a realistic distribution of bolts?
- Step 1, parametric sweep: number of mics (3–12) × aperture (5–500 m) × layout family (triangle, circle, L, 3D mast, distributed).
- Step 2, fast surrogate: derive the Fisher information matrix and Cramér–Rao lower bound for direction estimation given TDOA noise. This gives the best achievable error for a layout without running full simulations.
- Step 3, optimize free mic positions against the surrogate using optimal-design criteria: D-optimal (maximize the determinant of Fisher information) and A-optimal (minimize the trace of its inverse). Use `scipy.optimize.differential_evolution` or CMA-ES, with aperture and ground-placement constraints.
- Step 4: verify the optimized layouts with full simulation and compare against the best parametric layouts. Does the surrogate's ranking hold up?

**E3. Robustness and error budget**

- One-at-a-time sweeps over SNR, clock sync error, mic position error, t0 error, wind speed, and atmosphere mismatch; then pairwise sweeps for the most important pairs.
- Output an error budget, e.g. "to keep angular error under X°, sync must be under Y ms and positions known to Z cm." A tornado chart ranks which error source matters most.

**E4. Method comparison**

Methods A–D, each with straight-line and refraction-aware variants, across all presets and realistic corruption. Compare accuracy, coverage, branch recovery, and runtime.

**E5. Limits**

Error and coverage vs distance (1–15 km), branching depth, and presence of in-cloud horizontal channels. Find where reconstruction turns to mush and explain why (shadow zone, absorption, overlapping arrivals).

**E6. Consumer-hardware feasibility**

Could someone do this with 4 phones? Use the `phone` mic preset (100 Hz high-pass), `hand_synced` timing, `video_30fps` flash timing, and GPS-grade (several meter) position knowledge. Report what is still recoverable. This is a good hook for a write-up.

**E7. Self-calibration**

With Method D estimating sound speed and wind jointly, how well does it recover the true atmosphere, and how much does that improve point accuracy over assuming a standard atmosphere?

## Phase 8: Visualization and write-up

The project should be understandable in 30 seconds from the README and defensible in 30 minutes from the report.

**Figures and media**

- Interactive 3D plot (plotly HTML): true channel, reconstructed points colored by error, uncertainty ellipsoids, and mic positions.
- Animated reconstruction: points appear as sound reaches the array, with the waveform playing underneath. Export as GIF/MP4; this is the README hero image.
- Synthetic thunder WAV files for a few bolts at different distances.
- Waveform stack and spectrogram per mic; GCC-PHAT heatmap over time (lag vs window).
- Ray-fan plots for different atmospheres.
- Error vs range curves per method; coverage curves.
- Array layout plots, including the optimized layouts from E2.
- Tornado chart for the error budget (E3).

**Write-up**

- `README.md`: one-paragraph pitch, hero GIF, a results figure, how to install and reproduce.
- Technical report (6–10 pages, paper structure): introduction, related work, forward model, reconstruction methods, experiments, results, limitations (simulation only, model assumptions), future work (field validation with a real array, ideally paired with lightning photographs as ground truth).

**Optional demo**

A small Streamlit app: drag mics around, pick a bolt and conditions, and watch the reconstruction update.

**Resume framing (draft, revise with real numbers)**

Built a physics-based simulation and inverse-problem pipeline reconstructing 3D lightning channels from thunder recorded by microphone arrays; optimized array geometry using Fisher-information design criteria, cutting median localization error by X% vs standard layouts.

## Testing and validation strategy

The biggest risk in a simulation-only project is results that look great because the test is rigged. Guard against that first, then cover correctness.

**Avoid the "inverse crime"**

The inverse crime is synthesizing data and reconstructing it with the same model and discretization, which makes results unrealistically good. Rules:

- Synthesis always uses finer emitter spacing, higher oversampling, and fuller physics (ray tracing, absorption, ground reflection, noise) than reconstruction assumes.
- Reconstruction never sees ground-truth fields (true positions, true t0, true atmosphere) unless an experiment is explicitly labeled "oracle."
- Headline results use mismatched-model runs. Oracle runs are reported only as upper bounds.
- Add a unit test that fails if any `recon/` module imports from the ground-truth block.

**Test layers**

1. Unit tests per module (listed in each phase).
2. Analytic cases with closed-form answers: point source, straight vertical line, horizontal line at known height, uniform-atmosphere ray tracing.
3. Round-trip tests: generate → synthesize → corrupt (off) → reconstruct must recover the channel within tolerance. Keep a small, fast version that runs on every commit.
4. Regression tests: fixed-seed golden metrics for a handful of standard cases. Any change beyond tolerance fails and must be explained in `docs/log.md`.
5. Optional property-based tests (hypothesis) for geometry helpers.

**Physical sanity checks to automate**

- Rumble duration ≈ (r\_max − r\_min)/c.
- Spectrum peak in the expected band.
- Downwind arrivals earlier than upwind.
- Shadow zone appears under a standard lapse rate.

**Performance targets (provisional)**

- One bolt end-to-end (generate, synthesize, reconstruct with Method A) in under \~10 s on a laptop.
- Full test suite under \~2 minutes; slow Monte Carlo tests marked `@pytest.mark.slow` and skipped by default.

## Milestones and build order

The first real payoff is M4, a working ideal-conditions round trip; everything after that adds realism and findings. Each milestone ends with tests passing and a short summary to the user.

1. **M0 Setup.** Repo, `pyproject.toml`, `CLAUDE.md`, data contracts in `types.py`, pydantic config loading, `constants.py`, pytest + ruff running. *Gate: empty pipeline runs on a config and writes a results folder.*
2. **M1 Channel generator** (Phase 1). *Gate: gallery figure and stats tests pass.*
3. **M2 Thunder synthesis, uniform atmosphere** (Phase 2). *Gate: analytic tests pass; WAVs sound like thunder.*
4. **M3 Sensors and arrays** (Phase 4). *Gate: corruption tests pass; array preset plots.*
5. **M4 Method A + metrics, run E1** (Phases 5A and 6). *Gate: ideal round trip meets the provisional accuracy target. This is the MVP.*
6. **M5 Realistic atmosphere** (Phase 3). *Gate: ray-tracing tests pass; ray-fan figures.*
7. **M6 Methods B and C, refraction-aware variants.** *Gate: two-source test passes for B; mismatched-atmosphere runs work.*
8. **M7 Experiments E2–E5.** *Gate: each has a config, results table, figures and summary.*
9. **M8 Method D, uncertainty calibration, E6 and E7.** *Gate: calibration within tolerance on simple cases.*
10. **M9 Visualization polish and write-up** (Phase 8). *Gate: README with hero GIF; report draft.*

**Stretch goals (only after M9)**

- Nonlinear near-source propagation (weak-shock steepening).
- Turbulence model.
- Intracloud-only flashes.
- Separating multiple strokes in one recording.
- A learned reconstructor (neural net trained on simulated data) compared against the classical methods, with special attention to how it degrades under model mismatch.
- Streamlit demo.
- A small field pilot to test against real thunder.

## References to verify

These are from memory and have not been checked against the sources. Confirm titles, years and the specific claims used (constants, tortuosity statistics) before citing them or hard-coding values.

| Reference | Used for |
| --- | --- |
| Few, A. A. (1969), power spectrum of thunder, *J. Geophys. Res.* | Relaxation radius and dominant-frequency model |
| Few, A. A. (1970), lightning channel reconstruction from thunder measurements, *J. Geophys. Res.* | Original acoustic reconstruction method |
| Ribner, H. S. and Roy, D. (1982), acoustics of thunder with tortuous lightning, *J. Acoust. Soc. Am.* | String-of-pearls synthesis model |
| Hill, R. D. (1968), analysis of irregular paths of lightning channels, *J. Geophys. Res.* | Turn-angle statistics (≈16° default) |
| Arechiga, R. O. et al. (2011), acoustic localization of triggered lightning, *J. Geophys. Res.* | Modern array-based reconstruction |
| Lacroix, A., Farges, T. et al. (≈2018–2019), acoustic lightning reconstruction compared with a lightning mapping array, *J. Geophys. Res. Atmos.* | Validation approach and realistic error levels |
| Knapp, C. and Carter, G. (1976), the generalized correlation method for time-delay estimation, *IEEE Trans. ASSP* | GCC-PHAT |
| DiBiase, J. (2000), PhD thesis on SRP-PHAT, Brown University | Steered response power |
| ISO 9613-1 (1993), atmospheric absorption of sound | Absorption coefficient |
| Pierce, A. D., *Acoustics: An Introduction to Its Physical Principles and Applications* | Ray acoustics in moving media |
| Rakov, V. and Uman, M. (2003), *Lightning: Physics and Effects* | General lightning parameters |
| Kaipio, J. and Somersalo, E. (2005), *Statistical and Computational Inverse Problems* | The inverse-crime concept; Bayesian inversion |

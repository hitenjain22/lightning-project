# Design log

## 2026-10-05: M0 setup

- **Repo root is this folder**, not a nested `lightning-acoustics/`. The package is `thunder` under `src/` and installs with `pip install -e .`.
- **`MicArray` instead of `Array`.** The spec calls the contract `Array`, but that name is easily confused with numpy arrays.
- **`Atmosphere` is an abstract base class**, not a dataclass. It defines profile methods (`temperature`, `wind`, `relative_humidity`, `sound_speed`) and `propagate(sources, receivers) -> ArrivalPath`. `UniformAtmosphere` is the straight-line, constant-c implementation that Phase 2 will use. Phase 3 adds ray-table implementations behind the same interface.
- **Ground truth sits in `Recording.truth`** (`GroundTruth`). An AST-based test fails if anything under `recon/` imports `GroundTruth` or reads `.truth`.
- **Configs reject unknown keys** (`extra="forbid"`), so typos fail loudly. Each phase adds its own config section when it is built; M0 only has top-level run settings.
- **Run folder name** is `<UTC timestamp>_<config hash>_s<seed>`. The config hash is SHA-256 over the canonical JSON of the resolved config.
- **Constants:** Hill's 16° and Few's 0.63 are marked `# VERIFY`. `SOUND_SPEED_COEFF` is derived from gamma, R and M (about 20.047), not hard-coded as 20.05.

## 2026-10-05: M1 channel generator

- **Steering sets the direction of each turn, not its size.** Each turn angle is drawn exactly from the configured distribution. The bias only sets the direction of the turn around the channel, using a von Mises distribution centered on the goal. Blending in a pull toward the target, as the spec suggests, would shrink the turns and break the turn-angle test; this way the measured turn statistics match the configuration exactly (mean 16.0° measured against 16° configured, Kolmogorov–Smirnov test passes).
- **Exact landing.** The main channel walks until it crosses z = 0. The last segment is cut at the ground, then the whole main channel is shifted sideways so it lands exactly on the target. The shift is rigid, so shape and turn statistics are unchanged. Branches and the in-cloud section are grown after the shift.
- **Sideways start offset added** (`start_offset_m`, 0–1.5 km). Without it, every channel was nearly plumb: horizontal extent was only about 150 m over a 5.6 km drop.
- **Branches and in-cloud sections hold their initial heading.** The first version steered branches straight down and let in-cloud sections drift freely. Branches then fell back alongside the main channel, and in-cloud sections curled into tangles. Holding the initial heading fixed both (visible in the gallery).
- **Contract change:** added `Channel.is_incloud`, plus `Channel.refires = is_main | is_incloud`, the segments that fire on every stroke.
- **Modules:** the walk primitives live in `channel/walk.py`, so `generator.py` and `branching.py` can both use them without a circular import.
- **Angles** are written in degrees in YAML (`_deg` fields) and exposed in radians through config properties.
- **Turn angles are measured** with `atan2(|a×b|, a·b)`, which stays accurate for small angles where `arccos` loses precision. Only joints within the same branch count, because branch departure joints follow the branch-angle distribution instead.
- **Fractal dimension** comes from box counting on points sampled along the channel, over 8 geometric box sizes from 4 segment lengths up to a quarter of the largest bounding-box side. A straight line scores 0.95–1.0.
- **Performance:** the first version took about 0.12 s per channel. Computing the perpendicular basis once per step, without `np.cross`, brought this to about 0.03 s. The 500-channel turn test is marked slow (about 12 s); a 100-channel version runs by default.

## 2026-10-05: M2 thunder synthesis (uniform atmosphere)

- **Continuous line-element deposit instead of point emitters.** Each 0.5 m piece spreads its acoustic mass (q·length/r) uniformly over the interval between the arrival times of its two ends, deposited exactly onto the 64 kHz grid: four `bincount` ramp terms and one `cumsum`. This is the exact integral of a line source with linear arrival time. Point emitters 0.5 m apart would produce a spurious c/0.5 m ≈ 690 Hz pulse train end-on, inside the thunder band. Pieces whose arrival spread is under one grid sample fall back to linear fractional-delay point deposits, which is the spec's method. Sample k owns the cell [k − ½, k + ½)·dt, so timing is unbiased. Refining the emitter spacing from 1 m to 0.1 m changes the signal by under 1% (tested).
- **N-wave duration calibrated exactly to Few.** The N-wave spectrum is T·|j₁(ωT/2)|, so its peak sits at x* = 2.0816. Choosing T = x*/(π f_peak) puts the spectral peak exactly at Few's frequency (tested to 0.5%). At 1×10⁶ J/m and 25 °C this gives T = 5.39 ms and f_peak = 123 Hz.
- **Closed-form amplitude from energy conservation.** For a line source, the line Green's function plus Parseval give a radiated power per unit length of πq²T²/ρ₀, using ∫₀^∞ j₁²/x dx = ¼ (verified by quadrature). Hence q = sqrt(ηE_ℓρ₀/π)/T. Numerically, the energy flux through a cylinder around a 12 km line comes out within 0.5% of ηE_ℓ.
- **Cell-averaged N-wave kernel.** The pulse kernel is averaged over each grid cell using the pulse's antiderivative. That keeps its mean exactly zero (no DC) and its timing unbiased across the jumps at 0 and T.
- **Grouping** is by pulse duration in bins 0.5% wide, with one `oaconvolve` per group per mic. Decimation uses `resample_poly`, a zero-phase polyphase FIR, so it adds no delay.
- **Directivity emerges, with no hand-coded pattern.** A straight 100 m segment heard side-on carries 9,200× more energy than heard end-on. With micro-tortuosity on it is 3,070×.
- **Micro-tortuosity added (A14).** The first demo had isolated claps with about 50 dB of silence between them. Real thunder rumbles continuously. The cause was perfectly straight 10 m segments, which are much longer than the 1.9 m wavelength and cancel almost totally off-broadside. A Brownian bridge inside each segment, calibrated to 16° per 1 m (measured: 15.98°), fills in a continuous rumble about 30–40 dB below the claps, which looks like real thunder spectrograms. It changes no coarse geometry, and it makes synthesis strictly finer than any reconstruction model, which helps avoid the inverse crime. The analytic tests run with it off.
- **Contract additions:** `Atmosphere.pressure(z)` (abstract) and `Atmosphere.density(z)` (ideal gas). The source model needs pressure at the source height, and Phase 3 will make it vary with height.
- **Recording time origin:** the recording starts at the flash (`reported_t0 = 0`). Phase 4 adds t0 error.
- **Levels:** about 50 Pa peak at 3 km and about 30 Pa at 8 km (η = 0.002, VERIFY). The level falls slower than 1/r with distance because broadside segments act like line sources (cylindrical spreading). Phase 3 absorption will reduce distant levels.
- **Performance:** synthesizing a branched bolt for 4 mics takes about 1.1 s; one run of the whole pipeline takes about 3.5 s.

## 2026-10-05: M3 sensors and arrays

- **Two places to apply errors.** Position and clock errors are physical: they change when sound actually arrives and how it's timestamped. Synthesis therefore runs at the true positions and maps every arrival through τ = (1 + drift)·t + offset before the exact deposit. That reproduces fractional offsets with no resampling error: the clock-shifted recording matches an FFT fractional delay of the clean one to within 1%, and cross-correlation recovers the offset to within 0.05 samples. Everything else is applied afterward in physical order: acoustic noise, then mic response, then self-noise, then jitter, then clipping, then the ADC. Each step is skipped when disabled, so with everything off the output is bit-identical to the clean signal (tested with `np.array_equal`).
- **`MicArray` contract change:** the `frequency_response`/`noise` fields are replaced by hidden per-mic hardware values (`gain_db`, `corner_scale`), plus a `recorder_time()` method. `synthesize` accepts a `MicArray` and reports the nominal positions.
- **One aperture definition for every layout:** the largest horizontal distance between any two mics. Each layout is built at unit size, scaled to the exact aperture, and centered at the origin. `distributed` uses the aperture of each sub-array.
- **Noise synthesis method.** Each frequency bin gets an eigendecomposition of the coherence matrix, using V·sqrt(λ) because Cholesky fails on the nearly singular matrices at low frequency. Bins are processed in 8192-bin chunks to bound memory. Verified: PSD level within 2%; pink and brown slopes −1.00 and −1.99; diffuse coherence matches sinc² to within 0.01 at 1–5 Hz.
- **SNR set exactly.** Background noise is scaled so the band-power ratio equals the configured SNR exactly. An independent Welch-based estimate agrees within 0.5 dB, as the spec requires.
- **Pre-roll.** Noise is generated 2 s before the recording so causal filters reach steady state, and the extra is then dropped.
- **Independent random streams.** The runner gives the channel, array, synthesis and sensor stages separate child generators, and `build_mic_array` splits further per error type. Turning one error on never changes another error's realization (tested).
- **Fixes found during an audit before commit:** (1) Self-noise originally spread the datasheet's 20 Hz–20 kHz level over 0–4 kHz, overstating the noise density by about 7 dB. It's now density-based. (2) The `random_disk` spacing rule required 100% area coverage, which can loop forever. It's now 25% coverage, with an iteration guard. (3) `adc_bits` without a clip level is rejected, because the ADC full scale would be undefined. (4) Zero-σ draws produced −0.0, now normalized to 0.0.
- **Measured preset effects** (branched bolt): measurement mic −0.2% energy; audio mic −3.4% (−22% at 1 km, where it also clips); phone −45% at 3 km and −71% at 1 km (100 Hz high-pass, plus clipping on 0.17% of samples at 1 km).
- **Flash-time error expressed as range error:** photodiode 3 mm, lightning network 0.35 m, 30 fps video 3.3 m (σ).

## 2026-10-05: M4 Method A, metrics, and E1

**Method A design (`recon/tdoa.py`)**
- **Two-pass GCC.** On a 50 m array the same sound can arrive up to 146 ms apart at two mics, longer than a 100 ms window, so one fixed window per mic hears different chunks of the channel.
  - **Pass 1** pairs a short window on the reference mic with a long window on each other mic covering all physical lags.
  - **Measured problem:** PHAT across mismatched windows gave a lag σ of 0.37 samples, against 0.029 for plain cross-correlation, because the extra content in the long window gets whitened up.
  - **Pass 2** therefore puts equal Hann windows on the same sound at both mics of every pair and measures only the residual lag: σ = 0.002 samples, 160× better. Each mic's pass-2 spectrum is computed once per window and shared by all its pairs.
- **One constrained least-squares solve for direction.** min ‖W^½(Ap + τ)‖² subject to |p| = 1/c, solved exactly with the secular equation in the eigenbasis of H = AᵀWA.
  - **Planar arrays** are the trust-region "hard case": least squares for the horizontal slowness, plus the vertical component that restores |p| = 1/c, pointing upward.
  - **One code path** handles planar, mast and noisy cases. With exact plane-wave TDOAs, the direction error is below 10⁻⁶°.
- **Covariance (bug found by its own test).**
  - **The bug:** my first version projected the unconstrained covariance onto the tangent plane, which predicted 4.1× the Monte Carlo spread. That projection ignores the information the constraint itself carries.
  - **The fix:** restrict the normal matrix to the tangent plane first, cov(u) = c²σ²·T(TᵀHT)⁻¹Tᵀ. The residual variance now uses n − 2 degrees of freedom, since the fit has two free parameters.
  - **Result:** this matches Monte Carlo within 8% (trace), and it also covers planar arrays and near-horizontal arrivals.
- **Range** uses the energy centroid of the reference window, not its center. This removes up to ±17 m of within-window range ambiguity.
- **Band-pass** is zero-phase (forward-backward), so filtering never shifts timing (tested).
- **β-PHAT, β = 0.6.** On 18 development bolts at SNR ∞/25/15/5 dB, lowering β from 1 to 0.6 cut the median error at every SNR (clean: 1.51 → 1.37 m; 15 dB: 3.63 → 2.73 m). Applying β in pass 1 as well raised coverage (clean 0.80 → 0.83; 25 dB 0.53 → 0.60). β = 0.3 was marginally better still, but 0.6 keeps the whitening that multipath (ground reflection, M5) needs.

**Tuning protocol.** All gates were tuned on development bolts (seeds 5000+, `scripts/dev_tune_method_a.py`), never on E1's seeds. On a grid of 72 combinations over 30 bolts, the median error stayed between 1.0 and 1.6 m everywhere, so the method isn't fragile.
- **Detection dynamic range: 40 → 80 dB.** The 40 dB cut dropped faint but valid upper-channel sound (seed 1002: 0% coverage above 4.2 km). The noise floor and the quality gates now decide.
- **Residual gate: 2 ms.** Stricter gates traded coverage for a tiny accuracy gain.
- **`min_peak`** never binds when there's no noise; it stays at 0.3 as a safeguard for noisy recordings.
- **DBSCAN `min_samples` = 2, `eps` = 300 m.** At 3 / 150 m, isolated but correct points near the ground were deleted.
- **Window length: 100 ms (the spec default).** 50 ms windows were more accurate (1.0 vs 1.5 m median) but covered less (0.77 vs 0.80). Accuracy is already 30× better than the target, so coverage wins.

**Strike point.** The estimator moved into reconstruction (`Reconstruction.extra["strike_point"]`), because it's a method output that uses no truth. Capped extrapolation beat projecting the lowest points straight down on development bolts (median 39 vs 54 m, 75th percentile 55 vs 108 m). The cap of 0.5 is physical (about 27° lean), not tuned: 0.25 scored 1 m better.

**Metrics.**
- **Point error** is the exact point-to-segment distance against every segment (no sampling error).
- **Coverage and Chamfer** use 0.5 m truth samples weighted by length.
- **Pooled statistics** use a cluster bootstrap that resamples whole bolts, because points from one bolt are correlated. Resampling points understated the CI width by more than 3× in a test.

**Bugs caught by the new tests before any results were trusted.**
1. **Monte Carlo presets.** Workers rebuilt the config from a full dump, so every channel field counted as user-set and overrode each preset's own settings: `with_incloud` bolts had no in-cloud section. Overrides are now computed in the parent (`channel_overrides`).
2. **Tuning script.** It re-sent a 190 MB recording cache with every task. Workers now load it once.
3. **Slow test.** A brute-force reference took 143 s; a KD-tree gives the same answer. The full suite now takes about 75 s (spec: under 2 min).

**Regression goldens.** `tests/golden/roundtrip_metrics.json` holds three fixed-seed cases: tortuous at 2 km, branched at 1.5 km, and branched at 2 km with measurement mic, GPS clocks, surveyed positions and 15 dB SNR. Regenerate them with `scripts/update_golden.py`, only after explaining the change here.

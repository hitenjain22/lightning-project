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

## 2026-10-05: M5 realistic atmosphere (Phase 3)

**No travel-time tables: an exact eigenray solver instead.** The spec suggests precomputed tables T(h, r[, φ]). With wind, a table accurate to a fraction of a sample needs hundreds of MB and awkward 2D inverse interpolation near overhead sources. A layered medium has a closed-form structure instead: the horizontal slowness s_h is conserved, so travel time T, drift X and the Jacobian ∂X/∂s_h are integrals over height. The eigenray from a point to a mic is solved by:
- a bracketed 1D Newton iteration along the source-to-mic azimuth, using an analytic directional derivative;
- then a 2D Newton step for the sideways drift caused by crosswind.

This is exact for wind, including the lateral drift that the common "effective sound speed" approximation ignores. Shadow is detected exactly: a point is shadowed when even the limiting ray falls short.

**Numerics.** Each part was validated before the next was built.
- **Per-layer weights.** Assuming S = s_z² is linear in z, the factors 1/√S and 1/S^(3/2) integrate exactly. My first rule averaged the numerators. I then derived closed-form node weights exact when *both* the numerator and S are linear (`layer_weights`; checked against quadrature to 1e-9). With 50 m layers and no wind, this cut the worst error from 243 µs to 3.5 µs.
- **Graded height grid.** A regular 5 m grid gave 25 µs median error with wind, because the power-law wind changes fastest in the lowest meters. The grid now starts at 5 cm and grows by 15% per layer up to the step cap. That's the scale-free spacing a power law needs, at about 40 extra nodes. Inversion edges and the tropopause are inserted as nodes so no layer straddles a kink.
- **Step cap: 20 m.** Against a 0.5 m reference:

  | Case | Median | Worst |
  | --- | --- | --- |
  | Still air, with inversion | 0.27 µs | 2.6 µs |
  | Wind 8 m/s with 15°/km veer | 6.9 µs | 53 µs (about 2 cm of range) |

  These errors are smooth across the array, so mic-to-mic delays are far more accurate still.
- **Uniform limit.** Travel time matches R/c to 1e-11, direction to 1e-14, and ray-tube amplitude matches 1/R to 1e-8. The amplitude check needs the ρc impedance factor: a uniform-c stratified atmosphere still has hydrostatic density.
- **Shadow distance.** It matches the circular-ray prediction √(2R_cH − H²) to within 2% (test).
- **Independent reference.** The fast solver agrees with the `solve_ivp` Hamiltonian tracer to under 0.5 m landing error and 0.1 ms in time, with the dispersion relation conserved to 5e-6. The tracer's own derivative step had to shrink near the ground to reach that.

**Speed (no numba).** numba's llvmlite has no build for this Intel macOS with NumPy 2.5, and a dependency that doesn't install everywhere is unacceptable. Measured improvements:
- **Node weights:** evaluating each height node once, plus a scalar directional derivative in the 1D stage, made the solver 13× faster.
- **Initial guesses:** a cached windless fan per mic height cut the number of Newton passes.
- **Warm starts:** each mic starts from the previous mic's solution (2.3× faster with wind; results within 1 µs; the full solver is the fallback).
- **Absorption:** kernels cached across mics, 100 m path-length bins, and a vectorized attenuation lookup took synthesis from 26 s to 13 s per bolt with every effect on.
- **Synthesis:** rays are solved only at channel nodes. Emitter times use a cubic Hermite interpolation with known end slopes (∇T = −s), plus a first-order correction for the sub-meter wiggle. Against exact per-emitter rays: max 0.23 µs, and identical shadow sets (test).

**Bugs caught before any results were trusted.**
1. **Reflected path crash.** The 1D Newton loop's closure used full-length arrays on a subset of rays. The direct path only survived because every ray converged in one step.
2. **Fan builder.** The first version was O(levels²): it called the integrator once per height. It was rewritten as one cumulative pass.
3. **Warm start disabled.** The guess was discarded entirely if any single ray was shadowed. It's now applied per ray.
4. **Method A sign.** The fitted slowness points *toward* the source, while ray slowness points along propagation, so the back-trace was 5 km off until the sign was fixed. The oracle round trip through wind then gave about 2 m median.
5. **Bool inversion.** `~bool` in `ray_geometry` gave the right answer only by integer coincidence. It's now `not`.
6. **Test bugs.** A fixed FFT length missed a pulse arriving at 17 s; test frequencies sat on the N-wave's spectral zeros; and a circular-angle comparison wasn't circular.

**Humidity in the uniform model too.** The spec requires the humidity correction. With it in only one model, a zero-gradient stratified atmosphere wouldn't match the uniform one. Adding it changes c by about 0.2%, which moved the golden metrics (for example 1.51 → 1.37 m). With the old dry c restored, the refactored synthesis reproduces the previous goldens exactly (to within 1e-9), so the shift is only the intended physics change. The goldens were regenerated.

**Contract changes.**
- `ArrivalPath` gains `source_slowness` and `path_length`, and NaN marks shadow.
- `Atmosphere` gains `propagate_reflected`, `paths`, `locate`, `attenuation_db`, the effect toggles and `is_uniform`.
- `propagate` takes an optional `guess`.
- `RayTableConfig` was replaced by `RayConfig`, which sets the solver numerics.

## 2026-10-06: M6 Methods B and C

**Method B (`recon/srp.py`): steered response power with β-PHAT, several sources per window.**
- **Steering over horizontal slowness.** Steering runs over u = c·s_h, not over 3D points on a range shell (the spec's suggestion). Across a compact array, inter-mic delays depend only on the slowness at the array, s_h being the ray's conserved slowness, with s_z from the dispersion relation at mic height. So one grid serves any stratified atmosphere, and the assumed atmosphere enters only when detections are placed (straight rays, or `locate`). This is the far-field form; wavefront curvature is Method C's job.
- **Per window:** GCC-PHAT curves from the reference mic to each other mic; SRP on a 0.02 grid in u; local maxima with non-maximum suppression; a fine search at 10× resolution.
- **Per peak:** matched-window TDOAs searched only ±2 ms around the peak's predicted lags, so a weaker source isn't captured by a stronger one. The same constrained slowness fit as A. Arrival time comes from a delay-and-sum beam steered to that source, so two sources in one window get their own ranges.
- **Gate result:** two equally loud simultaneous sources 90° apart are both found (direction error 0.01°, range within 1%). A finds neither, because its single-direction fit fails.

**Method C (`recon/multilat.py`): absolute-time multilateration.**
- **Observed times:** per-mic delays come from all pair TDOAs by weighted least squares; with the reference arrival time and t0, they give absolute travel times.
- **Solve:** robust soft-ℓ1 Levenberg–Marquardt Gauss–Newton through the assumed atmosphere, batched over all windows, with the free gradient ∂T/∂x = −s_source and eigenrays warm-started from the previous iteration.
- **Verified:** converges from 40 m perturbations to under 5 cm through wind with veer.
- **Near field:** on a 300 m distributed array with a source 900 m away, C is within 1 m. The plane-wave fit's curvature misfit (up to ~146 ms) makes A reject the window entirely.
- **Seeding:** C seeds from every coherent window, not only windows passing A's plane-wave gates. My first version required A's gates, which made C useless exactly where it's needed.

**Bug: β-PHAT normalization (introduced in M4).** The correlation was scaled by nfft/(2·N_band), which is only correct for β = 1, where every weight has unit magnitude. With β = 0.6, peak heights scaled with signal amplitude to the power 2(1−β): peaks of 16–21 were observed, so every peak gate was meaningless. That's why M4's tuning found `min_peak` "never binds".
- **Fix:** divide by Σ|w_k|, so a perfect match peaks at exactly 1 for any β and level (tested).
- **Impact:** lags are unchanged; least-squares weights and gates change. Method A on the development bolts: 1.40 m median (was 1.37) and 0.83 main coverage (unchanged). `min_peak` still doesn't bind for A, clean or at 25 / 15 dB, because the residual gate does the filtering. The goldens were regenerated. The noisy golden dropped from 47 to 32 points, matching M4's development noise numbers; the earlier 47 was an artifact of the bug.

**B gates tuned on development bolts (seeds 5000+).** Once peaks were comparable, false B detections were SRP sidelobes, almost always secondary peaks.

| | Weakest pair peak | Fit residual (median) |
| --- | --- | --- |
| False detections | p90 0.43 | 0.5 ms |
| True detections | median 0.99 | 0.13 ms |

- **Why a combined gate:** a coherence floor of 0.6 alone kept 97% of true and rejected 98.6% of false detections. But two equally strong sources split the coherence (weakest pair 0.56–0.65), so it dropped real multi-source detections.
- **Chosen:** weakest pair ≥ 0.4 AND residual ≤ 0.5 ms. It gave the best coverage clean, at 25 dB and at 15 dB, with accuracy close to A's. Development bolts, main coverage within 50 m:

  | Condition | A | B |
  | --- | --- | --- |
  | Clean | 0.830 | 0.877 |
  | 25 dB SNR | 0.598 | 0.648 |
  | 15 dB SNR | 0.272 | 0.324 |

  Clean all-channel coverage rose from 0.75 to 0.82. The SRP power floor was split from `min_peak`, because pass-1 (mismatched-window) power runs lower than matched pair peaks.

**Refactor.** A, B and C share `prepare`, `coarse_shifts`, `matched_pairs`, `energy_time`, `place_points` and `assemble`. Method A was bit-identical (to 1e-12 on all goldens) before the normalization fix.

**M6 comparison (clean commit `43805f8`):** the same 60 realistic-atmosphere bolts as M5. With the true atmosphere, B raises all-channel coverage within 50 m from 76% to 82% and main-channel coverage from 83% to 88%, at 1.8 m vs 1.7 m median error. C equals A on this compact array, where wavefront curvature is negligible. With the mismatched atmosphere, all three methods sit at about 136 m, because unknown wind dominates. Speed per bolt: A 2.2 s, B 6.1 s, C 19.6 s. C's batched multilateration solves eigenrays for every mic on every iteration, which is the price of exact curvature in a stratified atmosphere.

## M7: Experiments E2–E5 (2026-10-06)

**Variant engine (common random numbers).** Each experiment is one Monte Carlo over *variants*: overrides of one base config, all run on the same bolts (`monte_carlo.variants`, recorded in the resolved config).
- **Stage cache:** keyed by the config fields each stage depends on (channel; array + position/timing/mic errors; synthesis inputs; corruption inputs). A reconstruction-only variant reuses the recording, and a noise-only variant reuses the clean synthesis.
- **Independent streams:** each stage draws from its own child seed, and the array builder draws layout, positions, clocks and mic tolerances from separate spawned streams. A mic-tolerance change therefore can't shift the true positions behind a cached synthesis (checked before relying on it).
- **Bounded memory:** the cache keeps the 4 most recent recordings per stage, and variants are ordered so shared stages are adjacent. Results stream out of each bolt instead of piling up. Eviction never changes results (tested with a one-entry cache).
- **Why:** paired comparisons (per-bolt ratios against a baseline) remove bolt-to-bolt variation, which dominates independent medians. The cache makes a 50-variant sweep cost roughly one synthesis per distinct array or atmosphere.
- **`child_seeds`:** replaces `default_rng(seed).spawn(4)`, which advances a counter on a shared SeedSequence. Repeated calls (one per variant) would otherwise get different streams. The streams are identical to the old ones (checked), so E1–M6 results are unchanged.

**E2 surrogate (Cramér–Rao bound).**
- **Model:** a plane wave from direction (az, el) with independent arrival-time noise σt, and an unknown emission time removed by projection.
- **Fisher information:** F = (σt c)⁻² Σ gₘgₘᵀ, where gₘ = Dᵀ(mₘ − m̄) and D = ∂u/∂(az, el).
- **Bound on the angular error variance:** tr(W F⁻¹), with W = diag(cos² el, 1).
- **Verified:**
  - matches finite differences to 1e-6;
  - for an isotropic planar array it reduces to the closed form (σt c)²/s · (1 + 1/sin² el), tested;
  - scales as 1/aperture and as σt.
- **Consequences:**
  - A planar array's elevation information vanishes at the horizon.
  - The optimal planar shape doesn't depend on aperture, so each n and criterion is optimized once at the design aperture.
  - Layouts with the same horizontal second moment score identically: triangle, square and square + center all give 0.223° at 50 m and σt = 0.1 ms.

**E2 optimizer.**
- **Method:** differential evolution over free (x, y), plus the mast height when the layout has a mast.
- **Aperture constraint:** a penalty during the search, then exact enforcement by rescaling.
- **Search grid:** coarse (azimuth every 30°, elevation every 10°) during the search; the final score uses the full grid.
- **Starting population:** seeded with regular polygons and polygon + center at four rotations.
- **Guarantee:** the result is the best of the optimum and those parametric candidates on the full grid, so it can never lose to them (tested).
- **First version's problems:** unseeded and on the full grid, it took 80–260 s and lost to the circle for 8 mics.

**Absolute ambient noise.** `noise.background_db_spl` sets the ambient band level in dB SPL (mutually exclusive with `background_snr_db`). With a relative SNR, every bolt gets the same SNR regardless of distance, which hides the propagation loss that E5 measures. The realized band SNR is recorded per bolt (`snr_band_db`).

**Realistic field kit (E2–E5):**
- measurement mics;
- GPS-synced clocks (1 µs);
- surveyed positions (1–2 cm);
- photodiode flash time (10 µs);
- 45 dB SPL ambient (VERIFY);
- 3 m/s wind noise.

**New metrics:** angular error (transverse error over range, as seen from the array), and side-branch coverage (NaN when a channel has no branches, so branchless presets don't count as 0).

**Large compact apertures fail for a physical reason, not a solver bug** (diagnosed before E2).
- **Measured:** at 150 m aperture, only 36 of 166 windows have multilateration times that any point of the true channel explains within 2 ms. Gauss–Newton converges on 38 windows, so the optimizer finds every explainable solution. At 500 m it is 4 of 180.
- **Cause:** a window holds sound from an extended stretch of channel, and the inter-mic delay changes along that stretch at a rate proportional to aperture/(c·R). Across large baselines the waveforms decorrelate, so pair TDOAs are mutually consistent (closure misfit 0.14 ms) but correspond to no single source.
- **What the gates do:** C's residual gate rejects these windows correctly.
- **Consequence:** this is exactly where the point-source CRB stops ranking layouts (E2 step 4).

**E2 finding → optimizer constraint: the independent-noise bound rewards stacked mics.**
- **What happened:** the unconstrained A-optimal 5-mic mast layout put two mics 0.45 m apart. Its grid bound was 0.130° (the best 5-mic score), but in full simulation it was 2× worse than the plain mast (0.080° vs 0.043°, 30 paired bolts).
- **Why:** the bound treats each mic's timing error as independent, so a co-located copy looks like a √2 gain. Real mics that close hear the same waveform and nearly the same noise, so their errors are correlated.
- **Decision:** `optimize_layout` requires every pair at least `min_separation_frac` × aperture apart (3-D; default 0.2), and never returns an infeasible layout.
- **Why 0.2:** it excludes clustering, while the 12-mic circle (spacing 0.26 × aperture) stays feasible. The constrained optimum's bound is insensitive to the value: 0.131–0.132° for 0.1–0.3, vs 0.136° for the plain mast.
- **Provenance:** the E2 main run used the unconstrained optimizer (commit `a31e59c`); the constrained layout was verified in a follow-up run on the same bolts.
- **The constraint was not the whole story.** The constrained optimum was still 1.72× worse than square + center, which led to the curvature finding below.

**E2 finding → analysis: weight the bound by where the channel is.**
- **What happened:** the bound averaged over a uniform direction grid ranked the 20 layouts at 50 m with Spearman ρ = 0.72. Averaged over the directions of the simulated channels (regenerated from the run's seeds; per-bolt length-weighted median of the per-point bound, then the median over bolts, which mirrors the simulated metric), it ranks them with ρ = 0.94.
- **Why:** a planar array's bound grows as 1/sin²(elevation), so a uniform grid is dominated by near-horizon directions where few channel points are. That inflated the mast's apparent advantage.
- **Decision:** E2 reports both. The channel-weighted bound is the recommended surrogate.
- **Calibration:** simulated error ≈ 0.52 × bound at σt = 100 µs, i.e. about 52 µs effective timing noise (0.4 samples at 8 kHz).

**E5 crash → synthesis fix: a fully shadowed channel is silence, not an error.**
- **What happened:** the first E5 run aborted after 46 minutes on a bolt whose entire channel had no eigenray to the array. Synthesis raised `ValueError`, a leftover from M2, when that case was unreachable in uniform air.
- **Decision:** synthesis now returns a silent recording, as long as the sound would have taken along straight lines, so sensor noise still applies. Reconstruction finds nothing, and the bolt counts with zero points and 100% shadow. Tested end to end; E5 was rerun from a clean commit.
- **Lesson:** an experiment that probes limits must treat "nothing heard" as data. The same principle already applied to bolts with no reconstructed points (NaN errors, coverage 0).

**Runtime measurement.** Wall-clock stage timings include waiting for other processes and sleep (a laptop closed during a smoke test inflated them severalfold). Reconstruction now also records process CPU time, which E2 and E4 report.

**Overlapping arrivals diagnostic.**
- **What it measures:** `arrival_s_per_km` is the length-weighted 5–95% span of the true segment arrival times at the reference mic, per km of heard channel.
- **Why:** it tests the E5 hypothesis that coverage falls with distance because channel parts arrive together.
- **Development bolt:** 1.9 s/km at 1 km vs 0.8 s/km at 15 km.

**E2 finding: plane-wave methods carry a wavefront-curvature bias on asymmetric layouts.**
- **Diagnosis, step by step:**
  - The direction fit is efficient: with ideal per-mic noise it is within 3% of the bound on every layout.
  - Measured time differences are 3–5 µs accurate on every layout.
  - Yet point sources gave 0.10–0.30° errors on asymmetric layouts.
  - Exact noise-free spherical arrivals reproduce those errors as pure bias, ∝ 1/R (mast 0.11°, triangle 0.19°, L-shape 0.29° at 2 km). Symmetric layouts give 0.003°, ∝ 1/R².
- **Mechanism:** the curvature term of the arrival times is quadratic in mic position, so it biases the fitted direction through the layout's third moments. These vanish for centrally symmetric layouts and for regular n-gons with n not a multiple of 3.
- **Method C** (absolute times, curvature modeled) removes most of it.
- **Decision:** keep A and B far-field (assumption A39) and report the bias.
  - `design.plane_wave_bias` computes it exactly from geometry.
  - The E2 surrogate adds it in quadrature to the calibrated bound, √((k·bound)² + bias²), with k fit on the 13 bias-free layouts. Rank agreement: ρ = 0.98 (bound alone over channel directions: 0.94; grid bound: 0.72).
- **Candidate for M8:** a curvature-corrected Method A. Re-fit with time differences corrected for the spherical-wave residual at the first-pass point; range is known from t0, so this costs one iteration.

## M8: Method D, uncertainty calibration, E6 and E7 (2026-10-07)

**Method D design.**
- **Likelihood:** each window has per-mic travel times. The covariance is a·I + b·11ᵀ:
  - a: independent per-mic terms (measurement noise from the window's own pair misfit with a 50 µs floor, plus clock, plus position/c);
  - b: terms common to the window (the window's time spread, plus the flash-time error when it isn't estimated).
- **Why these terms:** they are the SPEC's noise terms, expressed as the user's *knowledge* of their hardware.
- **The 50 µs floor:** E2's calibrated effective timing noise.

**Self-calibration model: an effective moving medium.**
- **Model:** straight rays, c(h) = c0 + c1·h and w(h) = w0 + w1·h, *path-averaged* over source height h. Closed-form travel time with analytic derivatives (tested exact to 10⁻⁸ against finite differences).
- **Why not ray-trace candidate atmospheres:** that would cost one full eigenray solve per mic, per window, per parameter, per iteration.
- **Adequacy:** with the parameters fitted to the true stratified atmosphere it gives 1.4 m (still air) and 7.7 m (6 m/s wind) on a development bolt, against 160 m assuming still air. The approximation is not the bottleneck.

**Inference: variable projection.** Three solvers were tried before this one.
1. **Hand-written joint Levenberg–Marquardt with a Schur complement:** zig-zagged in the flat, ill-conditioned valley and stalled (some trials at 200+ iterations, or at the wrong optimum).
2. **scipy trust region, sparse LSMR inner solver:** stalled just above the optimum.
3. **scipy trust region, dense exact solves:** correct (matches an independent least-squares fit to 3 decimals) but 40 s for 300 windows.
4. **Chosen, variable projection:**
   - **Structure:** given the medium, windows separate into 3-parameter problems (solved vectorized). The outer problem has 6 + G parameters (one dt0 per recording) with the exactly projected Jacobian.
   - **Accuracy:** 6/6 trials at the exact optimum.
   - **Speed:** 2.7 s for 300 windows.
   - **Multi-start:** windows that fit badly are re-solved from their original start. Windows pushed to the horizon while the medium was far off (cos el = c·|s_h| > 1) otherwise stay stuck after it improves.

**Calibration gate.**
- **Test:** synthetic observations drawn from the likelihood, with the medium drawn from the prior (a Bayesian consistency check, 25–30 trials).

| Case | 1σ / 2σ / 3σ coverage | Medium z-scores |
| --- | --- | --- |
| Fixed atmosphere, flat array | 0.191 / 0.749 / 0.974 | — |
| Self-calibrating, mast array | 0.204 / 0.742 / 0.974 | 0.9–1.1 |
| Self-calibrating, flat array (documented limitation, slow test) | 0.21 / 0.75 / 0.91 | 0.8–1.3 |
| Nominal | 0.199 / 0.739 / 0.971 | 1 |

- **Diagnosis of the flat-array case:**
  - **Mechanism:** a flat array measures horizontal slowness only, and elevation follows from cos(el) = c·|s_h|, which is extremely steep near the horizon. A profile likelihood in c confirmed a one-sided, non-Gaussian posterior.
  - **Rejected fixes:**
    - Nested MCMC over the medium: far too slow, minutes per trial.
    - Importance sampling from the Laplace proposal: weight degeneracy, because near-horizon windows make the Laplace-integrated likelihood itself unreliable.
  - **Decision:** Laplace, with this limitation documented.
- **Earlier failure:** the first gate run (0.07 / 0.26 / 0.46) was an unconverged solver, not the method. Converged fits are what the table reports.

**Single-bolt vs storm self-calibration.**
- **Single bolt:** sound from one bolt reaches the array over a narrow range of azimuths. Only the wind component along the line of sight changes anything measurable (through the speed of sound along the ray and wavefront curvature). A cross-wind shifts every apparent source sideways in proportion to its travel time, which looks exactly like a different channel.
- **Consequence:** single-bolt self-calibration recovers the sound-speed profile (lapse within about 0.1 m/s per km on the development bolt) but leaves the cross-wind at its prior, with honest uncertainty. In still air with a correct prior it drifts by about 1 m/s within a ±6 m/s posterior, which costs about 20 m sideways, and the error bars cover it (2σ coverage 0.98).
- **Storm self-calibration (new):** several recordings share the medium, each with its own dt0. Bolts at other azimuths see the cross-wind along their line of sight.
  - **Development storm** (4 bolts at 0, 90, 180 and 270°, 6 m/s wind, still-air prior): 13–24 m error vs 80–98 m per bolt and 82–317 m with still air assumed. Recovered wind (9.6, 3.2) m/s vs true path-averaged (8.5, 3.1).
  - **Requirement:** azimuth diversity, not just bolt count. In an E7 smoke storm with bolts at 90, 150 and 156°, the objective at the found and the true wind differed by less than 0.2%: the data cannot tell them apart. E7 reports the wind error against the azimuth range.

**Fixed-atmosphere Method D:** windows are independent, so there's no outer solve. The flash-time error goes into each window's covariance, and the speed is close to Method C's.

**Experiment pitfall (E6), fixed before the run.**
- **Problem:** variant overrides that give only a preset name (`{"position": {"preset": "tape"}}`) are merged into a resolved config whose fields are all explicit. The preset only fills missing fields, so the base's values survived. Tape positions and the field kit silently equaled the phones.
- **Fix:** overrides are now built from the config classes, with every field set.
- **Earlier experiments:** E2–E5 always set explicit values, so they were not affected (checked).

**Array self-calibration (E7 isolation, then fix).**
- **Diagnosis:** the first E7 run had storm self-calibration failing on realistic recordings. Isolation on one calm storm (true wind 1 m/s) showed that mic position errors *alone*, at survey grade (1–2 cm, about 60 µs), invent several m/s of wind and turn 6–15 m into 94–159 m.
- **Mechanism:** these errors are the same in every window. Treated as independent noise, they look like signal once pooled, and the shared medium absorbs them.
- **Decision:** when self-calibrating, per-mic clock offsets and 3-D position offsets are parameters, with priors from the stated hardware accuracy, and they are removed from the per-window noise.
- **Synthetic test:** recovers 2 cm offsets and the wind.
- **E7 rerun (commit `f581407`):**
  - error-bar coverage 0.40 / 0.73 / 0.86 (was 0.16 / 0.40 / 0.63); medium z-scores 1.0–1.2;
  - calm storms are no longer damaged (35 → 62 m, was 35 → 170 m);
  - windy storms gain little (225 → 209 m, was 225 → 104 m).
- **Why the windy gain shrank:** mic offsets and the wind both shift arrival times in direction-dependent ways, so they share the signal. With honest priors, a few realistic bolts can't separate them.
- **Conclusion recorded:** self-calibration recovers the sound-speed profile and gives honest uncertainty, but not the wind. Independent wind information (anemometer, sounding) is the practical route to the oracle's 3 m.

**Experiment logistics.** E7's first run took 4 h (16 CPU-hours).
- **Cost driver:** fixed-atmosphere Method D, with ray tracing to a 10 m mast mic about 5× slower than to ground mics.
- **Rerun:** the array-calibration rerun used the flat array only (75 min).
- **Mast results:** they come from the first run.

## M9: Visualization and write-up (2026-10-08)

- **Hero animation** (`thunder.viz.animate`): points appear at their window time (the recorder time at the reference mic), over the faint true channel, while a cursor runs along the waveform. The view turns 50° over the clip. Output is a GIF through Pillow (no ffmpeg on this machine); 600 px, 60 frames plus a hold on the last, about 2 MB.
  - **Pillow merges identical frames:** the hold becomes one longer final frame, so the test checks the total duration rather than the frame count.
- **Interactive view:** uncertainty ellipsoids are drawn as their 2σ principal axes, one line trace for all points, hidden until enabled in the legend. Full ellipsoid meshes for hundreds of points would make the HTML heavy and unreadable.
- **GCC-PHAT heatmap:** computed with Method A's own preprocessing and first-pass correlator (`prepare`, `_phat_correlation`), so it shows exactly what the method sees.
- **Bug: the waveform-stack onset detector zoomed to t = 0 on noisy recordings.** Its threshold was 0.1% of the peak, which noise exceeds. It now also requires 8× a robust noise level (median absolute value; thunder is sparse in the recording). Tested.
- **Audio:** the same bolt shape at 1, 3, 8 and 15 km in the realistic atmosphere with the field-kit sensors, one mic, each file normalized (absolute levels in `docs/audio/levels.json`): peak 137 → 108 dB SPL, rumble 21 → 49 s.
- **Single runs** (`run_experiment.py` without a Monte Carlo section) save `reconstruction.png` and `.html`: the README's quick demo.
- **Report:** `docs/report.md`. The related-work citations carry the SPEC's caveat that their bibliographic details are unverified.

## Final audit (2026-10-08)

**Bug: Method D was not bit-for-bit reproducible.** The slow storm test failed after array calibration was added. Investigating it showed two runs of the same script giving different winds: (5.6, 1.7) and (1.9, 1.9) m/s.
- **Cause:** multithreaded BLAS reorders floating-point sums. The near-flat self-calibration posterior amplifies those last-digit differences into different optima. Single-threaded, two runs are bit-identical (checked).
- **Fix:** Method D's inference, every Monte Carlo worker and every E7 storm worker run with one linear-algebra thread (`threadpoolctl`, now a declared dependency). This also avoids oversubscription, because the worker processes already use every core.
- **Tested:** two self-calibrating reconstructions of the same recording are identical.
- **Methods A–C were reproducible already:** the regenerated goldens are byte-identical.
- **E6 and E7 rerun** with the deterministic code; their documents report the rerun.

**Finding: a wind estimate needs millimetre-level array knowledge.** The storm test (ideal sensors, 6 m/s wind, still-air prior) gave 30 m with array calibration off, but about 150 m with array calibration at its default 2 cm position prior. The fitted offsets were only 0.3–0.5 cm.
- **Why:** most of a cross-wind's effect is absorbed by moving the sources (the apparent channel shifts sideways). What remains observable is microsecond-level, the same size as a few millimetres of mic position.
- **Consequence:** with a centimetre-level position prior, honest inference must report the wind as unknown, which is what E7 found on realistic data.
- **The test now encodes ideal hardware known to be ideal** (1 mm, 0.1 µs priors).
- **Practical implication:** storm self-calibration of the wind needs a millimetre-surveyed array (total station) and sub-microsecond clocks. Otherwise, measure the wind independently.

**Final reproducible runs.**
- **E7:** commit `a0ff2aa`. Conclusions unchanged, with slightly different numbers now that inference is deterministic: 6-bolt storm 150 m (was 146 m), 2σ coverage 0.68.
- **E6, first rerun:** also at `a0ff2aa`. It exposed that array calibration with phone-grade priors (±4 m, ±3 ms) is ill-posed from one bolt: 1,548 m and overconfident.
- **Decision:** new switch `d_array_calibration` (default on). With it off, per-mic errors stay noise terms. E6's raw-phone Method D variant turns it off.
- **E6, final run** (commit `4b1925f`):
  - raw phones with D: 528 m, bars 0.25 / 0.54 / 0.71;
  - upgraded phones with D, array calibration on: **163 m, bars 0.53 / 0.87 / 0.93**, better than B (180 m) and equal to the field kit.

## Interactive website (2026-10-08)

**What:** `docs/index.html`, served by GitHub Pages from `/docs`. Act 1 shows the simulated strike: the channel, sound spreading from it, the parts of the channel being heard at each moment, and the recorded thunder. Act 2 plays the reconstruction in recording time: each 0.1 s window's point appears with a ray from the array, and the true channel lights up where that window's sound really came from. The last step gives honest metrics, with truth, reconstruction, 2σ uncertainty and side-by-side toggles.
- **Data:** `scripts/make_viewer_data.py` runs the same pipeline as the hero GIF (`configs/experiments/media.yaml`, seed 3: Method B, realistic atmosphere and sensors, atmosphere known). It exports `docs/viewer/bolt.json` (87 kB) and the reference mic's recording as `thunder.wav`. Ground truth goes into the JSON only for display and scoring; the reconstruction never reads it.
- **"Being heard now" highlight:** uses each segment's true ray-traced arrival time at the reference mic (`segment_arrival_times`), not straight-line distance, so it follows the real refraction. Shadowed segments never light up.
- **Wavefront shells are illustrative:** spheres at 347 m/s from a few emitters on the main channel. The real fronts are refracted; the highlight above is the exact part.
- **No build step:** Three.js 0.160 from jsDelivr through an import map. Fat lines (`LineSegments2`) and bloom make the channel readable; reconstructed points are fixed-pixel-size so they stay visible at any zoom; low error is drawn bright (reversed viridis).
- **Honesty:** the final panel quotes this bolt's numbers and the 60-bolt E4 result (3.3 m, 86% main coverage for B with the true atmosphere), and states that the unknown wind raises errors to about 130 m.
- **Deep links:** `#step=N&t=T` opens a step paused at a time (`&side=1` on the last step). Used for headless screenshot checks at desktop and phone widths.
- **Decision: keep the GitHub repo and add Pages,** rather than deleting it and making a new one. Deleting is irreversible and loses history, stars and links; Pages gives the site a URL from the same repo.


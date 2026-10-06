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

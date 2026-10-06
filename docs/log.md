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

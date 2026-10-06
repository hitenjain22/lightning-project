# Modeling assumptions

Every modeling assumption goes here, with the phase it affects and how it could be relaxed.

| # | Assumption | Phase | Why / how to relax |
| --- | --- | --- | --- |
| A1 | Sound speed constants are for dry air (c = sqrt(gamma R T), gamma = 1.4). A humidity correction is added in Phase 3. | 2, 3 | Gives about 343 m/s at 20 °C, which is enough for the uniform-atmosphere phases. |
| A2 | The whole channel fires at t = 0 for each stroke, because the return stroke moves at a large fraction of light speed. | 1, 2 | The error is microseconds against acoustic times of seconds. |
| A3 | Atmosphere is horizontally stratified: profiles depend only on height. | 3 | This is standard for ray tables. Turbulence (a stretch goal) adds the 3D variation. |
| A4 | The channel is built from straight segments about 10 m long. Tortuosity is a random walk whose turn angles are independent and identically distributed: half-normal with a 16° mean (Hill 1968, VERIFY). The shape of the distribution is assumed; Hill's mean is the only constraint. | 1 | Fit the shape to Hill's histogram once the paper is checked. `exponential` is available as an alternative. |
| A5 | Turn directions are steered toward a goal by a von Mises distribution on the turn's direction around the channel (concentration κ = 2). κ is a tuning parameter, not a physical value. | 1 | Tune κ against published channel statistics (horizontal extent, tortuosity ratio). |
| A6 | The cloud end of the channel sits 0–1.5 km sideways of the strike point (VERIFY). Without this offset, channels are unrealistically plumb vertical. | 1 | Replace with a published distribution of channel inclination. |
| A7 | Each branch is steered toward its own departure direction (κ = 2) and never comes within one segment length of the ground. Branch angle U(20°, 60°), median length 300 m (log-normal, σ = 0.75), length halving at each level of sub-branching, and 0.01 branches per 10 m step are all unsourced placeholders (VERIFY). | 1 | Calibrate against photographic branch statistics. |
| A8 | Energy per unit length is 1×10⁶ J/m on the main channel and in-cloud section; each branch level multiplies it by 0.2 (VERIFY). It is constant along each path. | 1, 2 | Few's model puts the spectral peak near 120 Hz at this energy. Energy decaying with height or branch length could be added. |
| A9 | The in-cloud section is a single near-horizontal walk at the height where the channel starts. It holds its initial heading (κ = 2), a restoring pull keeps it near that altitude (45° tilt per 1 km of error), and it has no branches of its own. It refires on every stroke, like the main channel. | 1, 2 | Branched in-cloud networks are a stretch goal. Check whether in-cloud sections should refire on later strokes. |
| A10 | Later strokes reuse the exact geometry of the main channel (and in-cloud section). Stroke count and interstroke interval are uniform within the configured ranges. | 1, 2 | Real subsequent strokes sometimes take a new ground path. |

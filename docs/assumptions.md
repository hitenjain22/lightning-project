# Modeling assumptions

Every modeling assumption goes here, with the phase it affects and how it could be relaxed.

| # | Assumption | Phase | Why / how to relax |
| --- | --- | --- | --- |
| A1 | Sound speed constants are for dry air (c = sqrt(gamma R T), gamma = 1.4). A humidity correction is added in Phase 3. | 2, 3 | Gives about 343 m/s at 20 °C, which is enough for the uniform-atmosphere phases. |
| A2 | The whole channel fires at t = 0 for each stroke, because the return stroke moves at a large fraction of light speed. | 1, 2 | The error is microseconds against acoustic times of seconds. |
| A3 | Atmosphere is horizontally stratified: profiles depend only on height. | 3 | This is standard for ray tables. Turbulence (a stretch goal) adds the 3D variation. |

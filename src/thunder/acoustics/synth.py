"""Phase 2 thunder synthesis: Channel -> pressure signal at each microphone.

Source discretization. Every segment is split into short pieces ("emitters", default 0.5 m).
Each piece is treated as a *continuous* line element, not a point: its acoustic mass
(q * length / r) is spread uniformly over the interval between the arrival times of its two
ends, and deposited exactly onto an oversampled time grid. This is the exact integral of a
line source whose arrival time is linear along the piece, so it has no grating artifacts at
any emitter spacing (point emitters 0.5 m apart would produce a spurious c / 0.5 m ~ 690 Hz
pulse train for end-on geometry). Pieces whose arrival spread is under one grid step reduce
to an ordinary point deposit with linear fractional-delay weights. Broadside directivity
then emerges naturally from interference, as in Ribner & Roy (1982).

Efficient summation (spec Phase 2):
  1. per mic, travel times of every piece's ends and its spreading factor (via Atmosphere);
  2. deposit masses onto the oversampled grid (vectorized, np.bincount + one cumsum);
  3. group pieces by pulse duration and convolve each group's mass train with its N-wave once;
  4. anti-alias low-pass and decimate to the output rate (zero-phase polyphase FIR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from thunder.acoustics.source import line_amplitude, nwave_kernel, pulse_duration
from thunder.types import Atmosphere, Channel, FloatArray, GroundTruth, IntArray, Recording

# Pulse-duration group width (relative). Pieces in one group share the group's N-wave;
# 0.5% duration error shifts the spectral peak by 0.5%, far below model uncertainty.
DURATION_GROUP_STEP = math.log(1.005)
# Silence kept after the last pulse ends, s.
TAIL_PAD_S = 0.25


@dataclass(frozen=True)
class Emitters:
    """Short straight pieces of channel (K of them)."""

    start: FloatArray  # (K, 3)
    end: FloatArray  # (K, 3)
    length: FloatArray  # (K,)
    energy_per_length: FloatArray  # (K,)
    segment: IntArray  # (K,) parent segment index
    refires: np.ndarray  # (K,) bool, fires on every stroke

    @property
    def midpoint(self) -> FloatArray:
        return 0.5 * (self.start + self.end)


def _lateral_basis(u: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Row-wise unit vectors e1, e2 perpendicular to each unit row of u (vectorized)."""
    helper = np.where(np.abs(u[:, 2:3]) > 0.9, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
    e1 = np.cross(helper, u)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(u, e1)


def micro_diffusivity(turn_mean: float, scale: float) -> float:
    """Brownian-bridge lateral variance per unit length (m) giving `turn_mean` between pieces of `scale`.

    Lateral increments over a piece of length l are N(0, s2 * l) per axis, so the turn between
    consecutive pieces is Rayleigh with mean sqrt(pi * s2 / l) (small angles). Solve for s2.
    """
    return turn_mean**2 * scale / math.pi


def discretize(
    ch: Channel,
    spacing: float,
    rng: np.random.Generator | None = None,
    micro_turn_mean: float | None = None,
    micro_scale: float = 1.0,
) -> Emitters:
    """Split every segment into ceil(L / spacing) pieces.

    With `micro_turn_mean` set, each segment gets sub-segment tortuosity: a 2D Brownian bridge
    perpendicular to the segment, pinned at both ends (so the coarse geometry, its joints and
    the ground truth are unchanged), with diffusivity chosen so pieces of length `micro_scale`
    turn by `micro_turn_mean` on average.
    """
    a, b = ch.segment_endpoints()
    seg_len = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(seg_len / spacing).astype(np.int64))
    seg = np.repeat(np.arange(ch.n_segments), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    f0 = (k / n[seg])[:, None]
    f1 = ((k + 1) / n[seg])[:, None]
    d = (b - a)[seg]
    start = a[seg] + f0 * d
    end = a[seg] + f1 * d

    if micro_turn_mean is not None and micro_turn_mean > 0:
        if rng is None:
            raise ValueError("micro-tortuosity needs an explicit rng")
        piece = (seg_len / n)[seg]
        s2 = micro_diffusivity(micro_turn_mean, micro_scale)
        inc = rng.standard_normal((len(seg), 2)) * np.sqrt(s2 * piece)[:, None]
        c = np.cumsum(inc, axis=0)
        first = np.cumsum(n) - n
        before = np.vstack([np.zeros((1, 2)), c])[first]  # cumsum just before each segment
        walk = c - np.repeat(before, n, axis=0)  # W at the end of each piece
        total = walk[np.cumsum(n) - 1]  # W_n per segment
        b_end = walk - f1 * total[seg]  # bridge: zero at both ends of the segment
        b_start = np.vstack([np.zeros((1, 2)), b_end[:-1]])
        b_start[first] = 0.0
        e1, e2 = _lateral_basis(d / seg_len[seg][:, None])
        start = start + b_start[:, :1] * e1 + b_start[:, 1:] * e2
        end = end + b_end[:, :1] * e1 + b_end[:, 1:] * e2

    return Emitters(
        start=start,
        end=end,
        length=np.linalg.norm(end - start, axis=1),
        energy_per_length=ch.energy_per_length[seg],
        segment=seg,
        refires=ch.refires[seg],
    )


def deposit(t_a: FloatArray, t_b: FloatArray, mass: FloatArray, n_grid: int, dt: float) -> FloatArray:
    """Spread each mass uniformly over [t_a, t_b] onto samples at k * dt.

    Sample k owns the cell [(k - 1/2) dt, (k + 1/2) dt), so timing is unbiased. Exact for a
    uniform spread; total mass is conserved. Times must lie in [0, (n_grid - 2) dt].
    """
    lo = np.minimum(t_a, t_b) / dt
    hi = np.maximum(t_a, t_b) / dt
    out = np.zeros(n_grid)

    short = (hi - lo) < 1.0
    if short.any():  # point deposit at the midpoint, linear fractional-delay weights
        u = 0.5 * (lo[short] + hi[short])
        i = np.floor(u).astype(np.int64)
        f = u - i
        w = mass[short]
        out += np.bincount(i, w * (1 - f), minlength=n_grid + 1)[:n_grid]
        out += np.bincount(i + 1, w * f, minlength=n_grid + 1)[:n_grid]

    long_ = ~short
    if long_.any():  # uniform spread: two ramps per piece, written as first differences
        va, vb = lo[long_] + 0.5, hi[long_] + 0.5  # cell coordinates: cell k is [k, k + 1)
        slope = mass[long_] / (vb - va)
        diff = np.zeros(n_grid + 2)
        for v, sign in ((va, 1.0), (vb, -1.0)):
            n0 = np.ceil(v).astype(np.int64)
            frac = n0 - v
            diff += np.bincount(n0 - 1, sign * slope * frac, minlength=n_grid + 2)[: n_grid + 2]
            diff += np.bincount(n0, sign * slope * (1 - frac), minlength=n_grid + 2)[: n_grid + 2]
        out += np.cumsum(diff)[:n_grid]
    return out


def synthesize(
    ch: Channel,
    atmosphere: Atmosphere,
    mic_positions: FloatArray,
    sample_rate: float,
    oversample: int,
    emitter_spacing: float,
    efficiency: float,
    rng: np.random.Generator | None = None,
    micro_turn_mean: float | None = None,
    micro_scale: float = 1.0,
) -> Recording:
    """Clean recording of the thunder from `ch` at each microphone. The flash is at t = 0.

    micro_turn_mean (rad) enables sub-segment tortuosity (see `discretize`); None gives
    perfectly straight segments, as the analytic tests need.
    """
    mics = np.atleast_2d(np.asarray(mic_positions, dtype=float))
    em = discretize(ch, emitter_spacing, rng, micro_turn_mean, micro_scale)
    z = em.midpoint[:, 2]
    duration = pulse_duration(em.energy_per_length, atmosphere.pressure(z), atmosphere.sound_speed(z))
    q = line_amplitude(em.energy_per_length, atmosphere.density(z), duration, efficiency)
    group = np.round(np.log(duration) / DURATION_GROUP_STEP).astype(np.int64)
    groups = np.unique(group)

    # Every stroke re-fires the refiring pieces; branches fire on the first stroke only.
    strokes = ch.stroke_times
    fire_idx = np.concatenate([np.arange(len(z))] + [np.flatnonzero(em.refires)] * (len(strokes) - 1))
    fire_shift = np.concatenate(
        [np.full(len(z), strokes[0])] + [np.full(int(em.refires.sum()), t) for t in strokes[1:]]
    )

    paths = [
        tuple(atmosphere.propagate(pts, m) for pts in (em.start, em.end, em.midpoint)) for m in mics
    ]
    t_last = max(float(np.max(np.maximum(pa.travel_time, pb.travel_time))) for pa, pb, _ in paths)
    t_end = t_last + float(strokes[-1]) + float(np.max(duration)) + TAIL_PAD_S
    n_out = int(math.ceil(t_end * sample_rate))
    n_grid = n_out * oversample
    dt = 1.0 / (sample_rate * oversample)

    signals = np.zeros((len(mics), n_out))
    for i, (pa, pb, pm) in enumerate(paths):
        mass = q * em.length * pm.amplitude_factor
        t_a = pa.travel_time[fire_idx] + fire_shift
        t_b = pb.travel_time[fire_idx] + fire_shift
        m_all = mass[fire_idx]
        g_all = group[fire_idx]
        fine = np.zeros(n_grid)
        for g in groups:
            sel = g_all == g
            train = deposit(t_a[sel], t_b[sel], m_all[sel], n_grid, dt)
            kernel = nwave_kernel(math.exp(g * DURATION_GROUP_STEP), dt)
            fine += signal.oaconvolve(train, kernel)[:n_grid]
        signals[i] = signal.resample_poly(fine, 1, oversample)[:n_out]

    return Recording(
        signals=signals,
        sample_rate=float(sample_rate),
        nominal_mic_positions=mics.copy(),
        reported_t0=0.0,
        truth=_ground_truth(ch, atmosphere, mics, efficiency),
    )


def _ground_truth(ch: Channel, atmosphere: Atmosphere, mics: FloatArray, efficiency: float) -> GroundTruth:
    """Per segment and mic: arrival time and direction (from the midpoint, first stroke) and amplitude."""
    a, b = ch.segment_endpoints()
    mid = 0.5 * (a + b)
    z = mid[:, 2]
    duration = pulse_duration(ch.energy_per_length, atmosphere.pressure(z), atmosphere.sound_speed(z))
    q = line_amplitude(ch.energy_per_length, atmosphere.density(z), duration, efficiency)
    seg_len = np.linalg.norm(b - a, axis=1)
    paths = [atmosphere.propagate(mid, m) for m in mics]
    return GroundTruth(
        t0=0.0,
        mic_positions=mics.copy(),
        segment_arrival_times=np.stack([p.travel_time for p in paths], axis=1),
        extra={
            "segment_amplitude": np.stack([q * seg_len * p.amplitude_factor for p in paths], axis=1),
            "segment_arrival_direction": np.stack([p.arrival_direction for p in paths], axis=1),
            "segment_pulse_duration": duration,
            "stroke_times": ch.stroke_times.copy(),
        },
    )

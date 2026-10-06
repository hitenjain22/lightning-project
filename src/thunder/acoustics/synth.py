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
from thunder.types import (
    ArrivalPath,
    Atmosphere,
    Channel,
    FloatArray,
    GroundTruth,
    IntArray,
    MicArray,
    Recording,
)

# Pulse-duration group width (relative). Pieces in one group share the group's N-wave;
# 0.5% duration error shifts the spectral peak by 0.5%, far below model uncertainty.
DURATION_GROUP_STEP = math.log(1.005)
# Silence kept after the last pulse ends, s.
TAIL_PAD_S = 0.25
# Absorption grouping: emitters whose path lengths differ by < this share one filter. Even at
# 4 kHz (~25-35 dB/km, where thunder has almost no energy) a 100 m bin spans < +-1.8 dB.
ABSORPTION_LENGTH_BIN_M = 100.0
# Absorption filters are shared (cached) between groups whose mean source heights fall in the
# same band; alpha varies slowly with height.
ABSORPTION_HEIGHT_BIN_M = 250.0
# Longest absorption impulse response considered (samples of the fine grid) and the extra
# recording tail allowed for it.
ABSORPTION_KERNEL_SAMPLES = 16384
ABSORPTION_PAD_S = 0.1


@dataclass(frozen=True)
class Emitters:
    """Short straight pieces of channel (K of them)."""

    start: FloatArray  # (K, 3)
    end: FloatArray  # (K, 3)
    length: FloatArray  # (K,)
    energy_per_length: FloatArray  # (K,)
    segment: IntArray  # (K,) parent segment index
    refires: np.ndarray  # (K,) bool, fires on every stroke
    frac_start: FloatArray  # (K,) position of `start` along its segment, 0..1 (before micro offsets)
    frac_end: FloatArray  # (K,)

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
        frac_start=f0[:, 0],
        frac_end=f1[:, 0],
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


@dataclass(frozen=True)
class NodePaths:
    """One propagation path (direct or reflected) evaluated at the channel nodes for one mic."""

    T: FloatArray  # (N,) travel time, NaN in shadow
    slowness: FloatArray  # (N, 3) source slowness (grad of T w.r.t. source = -slowness)
    amplitude: FloatArray  # (N,)
    direction: FloatArray  # (N, 3) arrival wave normal at the mic
    path: ArrivalPath  # raw result (also the warm start for the next mic)


def _hermite_time(
    nodes: NodePaths, ch: Channel, seg: IntArray, frac: FloatArray, points: FloatArray
) -> FloatArray:
    """Travel time from points on (or within ~1 m of) segments, interpolated from the nodes.

    Along the segment: cubic Hermite in the fraction f with end slopes dT/df = -s . (b - a)
    (exact to third order). Off the segment line (sub-segment tortuosity): first-order
    correction -s(f) . offset. NaN if either end node is in the shadow.
    """
    ia, ib = ch.segments[seg, 0], ch.segments[seg, 1]
    a, b = ch.nodes[ia], ch.nodes[ib]
    d = b - a
    ta, tb = nodes.T[ia], nodes.T[ib]
    sa, sb = nodes.slowness[ia], nodes.slowness[ib]
    da = -np.einsum("ij,ij->i", sa, d)
    db = -np.einsum("ij,ij->i", sb, d)
    f = frac
    f2, f3 = f * f, f * f * f
    t_line = (2 * f3 - 3 * f2 + 1) * ta + (f3 - 2 * f2 + f) * da + (-2 * f3 + 3 * f2) * tb + (f3 - f2) * db
    s_f = (1 - f)[:, None] * sa + f[:, None] * sb
    offset = points - (a + f[:, None] * d)
    return t_line - np.einsum("ij,ij->i", s_f, offset)


def _node_paths(
    atmosphere: Atmosphere, ch: Channel, mic: FloatArray, guesses: list | None
) -> list[NodePaths]:
    """Every modeled path (direct, then reflected if enabled) from all channel nodes to one mic."""
    raw = []
    kinds = [False] + ([True] if atmosphere.ground_reflection else [])
    for k, reflected in enumerate(kinds):
        guess = guesses[k].path if guesses is not None else None
        fn = atmosphere.propagate_reflected if reflected else atmosphere.propagate
        p = fn(ch.nodes, mic, guess=guess)
        amp = p.amplitude_factor * (atmosphere.reflection_coefficient if reflected else 1.0)
        raw.append(NodePaths(p.travel_time, p.source_slowness, amp, p.arrival_direction, p))
    return raw


def _emitter_times(
    atmosphere: Atmosphere,
    ch: Channel,
    em: Emitters,
    mic: FloatArray,
    node: NodePaths | None,
    reflected: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """(t_start, t_end, amplitude at the midpoint, path length at the midpoint) for every emitter."""
    if atmosphere.is_uniform:  # straight rays: exact per emitter
        fn = atmosphere.propagate_reflected if reflected else atmosphere.propagate
        pa, pb, pm = (fn(pts, mic) for pts in (em.start, em.end, em.midpoint))
        amp = pm.amplitude_factor * (atmosphere.reflection_coefficient if reflected else 1.0)
        return pa.travel_time, pb.travel_time, amp, pm.path_length
    assert node is not None
    t_a = _hermite_time(node, ch, em.segment, em.frac_start, em.start)
    t_b = _hermite_time(node, ch, em.segment, em.frac_end, em.end)
    f_mid = 0.5 * (em.frac_start + em.frac_end)
    ia, ib = ch.segments[em.segment, 0], ch.segments[em.segment, 1]
    amp = (1 - f_mid) * node.amplitude[ia] + f_mid * node.amplitude[ib]
    image = em.midpoint * np.array([1.0, 1.0, -1.0]) if reflected else em.midpoint
    length = np.linalg.norm(np.asarray(mic) - image, axis=1)
    return t_a, t_b, amp, length


def _absorption_kernel(base: FloatArray, attenuation_db: FloatArray, nfft: int) -> tuple[FloatArray, int]:
    """Kernel = base pulse convolved with a zero-phase absorption filter. Returns (kernel, lead),
    where lead samples precede the pulse start (the filter is non-causal by symmetry)."""
    spec = np.fft.rfft(base, nfft) * 10.0 ** (-attenuation_db / 20.0)
    h = np.fft.irfft(spec, nfft)
    h = np.roll(h, nfft // 2)  # pulse start now at nfft // 2
    mag = np.abs(h)
    keep = np.flatnonzero(mag > 1e-7 * mag.max())
    lo, hi = int(keep[0]), int(keep[-1]) + 1
    return h[lo:hi], nfft // 2 - lo


def synthesize(
    ch: Channel,
    atmosphere: Atmosphere,
    mics: FloatArray | MicArray,
    sample_rate: float,
    oversample: int,
    emitter_spacing: float,
    efficiency: float,
    rng: np.random.Generator | None = None,
    micro_turn_mean: float | None = None,
    micro_scale: float = 1.0,
) -> Recording:
    """Clean recording of the thunder from `ch` at each microphone. The flash is at t = 0.

    mics: plain positions (an ideal array), or a MicArray whose hidden truth is applied
    exactly: sound is synthesized at the true positions, and every arrival time t is mapped
    to the recorder's clock tau = (1 + drift) t + offset before deposition (the ppm change
    of pulse length is neglected). The Recording carries the nominal positions.

    Paths: the direct ray, plus the ground reflection if the atmosphere enables it. In a
    stratified atmosphere rays are solved at the channel nodes and interpolated to the
    emitters (`_hermite_time`); emitters in the acoustic shadow are silent. With absorption
    on, emitters are grouped by pulse duration and path length (ABSORPTION_LENGTH_BIN_M) and
    each group's pulse is filtered by its ISO 9613-1 attenuation (zero-phase).

    micro_turn_mean (rad) enables sub-segment tortuosity (see `discretize`); None gives
    perfectly straight segments, as the analytic tests need.
    """
    if isinstance(mics, MicArray):
        array = mics
    else:
        array = MicArray.ideal(np.atleast_2d(np.asarray(mics, dtype=float)))
    true_pos = array.true_positions
    em = discretize(ch, emitter_spacing, rng, micro_turn_mean, micro_scale)
    z = em.midpoint[:, 2]
    duration = pulse_duration(em.energy_per_length, atmosphere.pressure(z), atmosphere.sound_speed(z))
    q = line_amplitude(em.energy_per_length, atmosphere.density(z), duration, efficiency)
    dur_group = np.round(np.log(duration) / DURATION_GROUP_STEP).astype(np.int64)

    # Every stroke re-fires the refiring pieces; branches fire on the first stroke only.
    strokes = ch.stroke_times
    fire_idx = np.concatenate([np.arange(len(z))] + [np.flatnonzero(em.refires)] * (len(strokes) - 1))
    fire_shift = np.concatenate(
        [np.full(len(z), strokes[0])] + [np.full(int(em.refires.sum()), t) for t in strokes[1:]]
    )

    # Paths at the channel nodes (stratified only), warm-started from the previous mic.
    node_paths: list[list[NodePaths] | None] = []  # per mic; None for uniform atmospheres
    prev = None
    for m in true_pos:
        cur = None if atmosphere.is_uniform else _node_paths(atmosphere, ch, m, prev)
        node_paths.append(cur)
        prev = cur
    n_paths = 1 + int(atmosphere.ground_reflection)

    per_mic = []
    for i, m in enumerate(true_pos):
        entries = []
        for k in range(n_paths):
            mic_nodes = node_paths[i]
            node = mic_nodes[k] if mic_nodes is not None else None
            t_a, t_b, amp, length = _emitter_times(atmosphere, ch, em, m, node, reflected=k == 1)
            ok = np.isfinite(t_a) & np.isfinite(t_b) & (amp > 0)
            sel = fire_idx[ok[fire_idx]]
            shift = fire_shift[ok[fire_idx]]
            entries.append(
                (
                    array.recorder_time(t_a[sel] + shift, i),
                    array.recorder_time(t_b[sel] + shift, i),
                    (q * em.length * amp)[sel],
                    dur_group[sel],
                    length[sel],
                    z[sel],
                    k == 1,
                )
            )
        per_mic.append(entries)

    starts = [np.minimum(e[0], e[1]) for entries in per_mic for e in entries if len(e[0])]
    ends = [np.maximum(e[0], e[1]) for entries in per_mic for e in entries if len(e[0])]
    if starts:
        if min(float(np.min(t)) for t in starts) < 0:
            raise ValueError("a clock offset makes an arrival precede the recording start (t < 0)")
        t_last = max(float(np.max(t)) for t in ends)
    else:
        # The whole channel is in the acoustic shadow: a silent recording (a real outcome at
        # long range, which experiments must count, not crash on). Its length is when the
        # sound would have arrived along straight lines, so sensors still add their noise.
        far = max(float(np.max(np.linalg.norm(em.midpoint - m, axis=1))) for m in true_pos)
        c_ground = float(atmosphere.sound_speed(np.array([float(np.min(true_pos[:, 2]))]))[0])
        t_last = far / c_ground + float(strokes[-1])
    t_end = (
        t_last + float(np.max(duration)) + TAIL_PAD_S + (ABSORPTION_PAD_S if atmosphere.absorption else 0.0)
    )
    n_out = int(math.ceil(t_end * sample_rate))
    n_grid = n_out * oversample
    dt = 1.0 / (sample_rate * oversample)

    signals = np.zeros((len(true_pos), n_out))
    kernel_cache: dict = {}
    for i, entries in enumerate(per_mic):
        fine = np.zeros(n_grid)
        for t_a, t_b, mass, dgroup, length, z_src, reflected in entries:
            _accumulate(
                fine,
                t_a,
                t_b,
                mass,
                dgroup,
                length,
                z_src,
                float(true_pos[i, 2]),
                reflected,
                atmosphere,
                dt,
                kernel_cache,
            )
        signals[i] = signal.resample_poly(fine, 1, oversample)[:n_out]

    return Recording(
        signals=signals,
        sample_rate=float(sample_rate),
        nominal_mic_positions=array.nominal_positions.copy(),
        reported_t0=0.0,
        truth=_ground_truth(ch, atmosphere, array, efficiency, node_paths),
    )


def _accumulate(fine, t_a, t_b, mass, dgroup, length, z_src, z_mic, reflected, atmosphere, dt, cache) -> None:
    """Deposit, shape and add every (duration, path-length) group into the fine grid.

    Each group is processed on its own time span only (groups are short in time because path
    length and arrival time go together). Absorption kernels are cached in `cache`, shared by
    all mics and paths of one synthesis.
    """
    n_grid = len(fine)
    if atmosphere.absorption:
        lgroup = np.floor(length / ABSORPTION_LENGTH_BIN_M).astype(np.int64)
        key = dgroup * 1_000_003 + lgroup
    else:
        key = dgroup
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g)]
    nfft = 0
    att = None
    if atmosphere.absorption and groups:
        # One vectorized attenuation call for every group (common FFT grid).
        longest = max(
            len(nwave_kernel(math.exp(int(dgroup[g[0]]) * DURATION_GROUP_STEP), dt)) for g in groups
        )
        nfft = int(2 ** math.ceil(math.log2(longest + ABSORPTION_KERNEL_SAMPLES)))
        z_mean = np.array([float(np.mean(z_src[g])) for g in groups])
        l_mean = np.array([float(np.mean(length[g])) for g in groups])
        z_key = np.round(z_mean / ABSORPTION_HEIGHT_BIN_M) * ABSORPTION_HEIGHT_BIN_M
        l_key = (np.floor(l_mean / ABSORPTION_LENGTH_BIN_M) + 0.5) * ABSORPTION_LENGTH_BIN_M
        att = atmosphere.attenuation_db(np.fft.rfftfreq(nfft, dt), z_key, z_mic, l_key, reflected)
    for gi, members in enumerate(groups):
        g = int(dgroup[members[0]])
        base = nwave_kernel(math.exp(g * DURATION_GROUP_STEP), dt)
        lead = 0
        if att is not None:
            ck = (g, int(lgroup[members[0]]), round(float(z_key[gi])), bool(reflected), nfft)
            if ck not in cache:
                cache[ck] = _absorption_kernel(base, att[gi], nfft)
            kernel, lead = cache[ck]
        else:
            kernel = base
        lo_t = float(np.min(np.minimum(t_a[members], t_b[members])))
        hi_t = float(np.max(np.maximum(t_a[members], t_b[members])))
        i0 = max(int(math.floor(lo_t / dt)) - 2, 0)
        span = int(math.ceil(hi_t / dt)) + 3 - i0
        local = deposit(t_a[members] - i0 * dt, t_b[members] - i0 * dt, mass[members], span + 2, dt)
        shaped = signal.oaconvolve(local, kernel)
        start = i0 - lead
        lo = max(start, 0)
        hi = min(start + len(shaped), n_grid)
        if hi > lo:
            fine[lo:hi] += shaped[lo - start : hi - start]


def _ground_truth(
    ch: Channel, atmosphere: Atmosphere, array: MicArray, efficiency: float, node_paths: list | None = None
) -> GroundTruth:
    """Per segment and mic, direct path from the segment midpoint (first stroke): arrival time
    (NaN in the acoustic shadow), amplitude and arrival direction.

    `segment_arrival_times` are true physical times; `segment_arrival_times_recorder` are the
    same events read on each mic's clock.
    """
    mics = array.true_positions
    a, b = ch.segment_endpoints()
    mid = 0.5 * (a + b)
    z = mid[:, 2]
    duration = pulse_duration(ch.energy_per_length, atmosphere.pressure(z), atmosphere.sound_speed(z))
    q = line_amplitude(ch.energy_per_length, atmosphere.density(z), duration, efficiency)
    seg_len = np.linalg.norm(b - a, axis=1)
    times, amps, dirs = [], [], []
    for i, m in enumerate(mics):
        if atmosphere.is_uniform or node_paths is None or node_paths[i] is None:
            p = atmosphere.propagate(mid, m)
            times.append(p.travel_time)
            amps.append(p.amplitude_factor)
            dirs.append(p.arrival_direction)
        else:
            node = node_paths[i][0]
            seg = np.arange(ch.n_segments)
            half = np.full(ch.n_segments, 0.5)
            times.append(_hermite_time(node, ch, seg, half, mid))
            ia, ib = ch.segments[:, 0], ch.segments[:, 1]
            amps.append(0.5 * (node.amplitude[ia] + node.amplitude[ib]))
            d = node.direction[ia] + node.direction[ib]
            dirs.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    arrival = np.stack(times, axis=1)
    return GroundTruth(
        t0=0.0,
        mic_positions=mics.copy(),
        segment_arrival_times=arrival,
        extra={
            "segment_arrival_times_recorder": np.stack(
                [array.recorder_time(arrival[:, i], i) for i in range(len(mics))], axis=1
            ),
            "segment_amplitude": np.stack([q * seg_len * amp for amp in amps], axis=1),
            "segment_arrival_direction": np.stack(dirs, axis=1),
            "segment_pulse_duration": duration,
            "stroke_times": ch.stroke_times.copy(),
        },
    )

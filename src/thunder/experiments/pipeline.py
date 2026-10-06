"""One bolt end to end: generate -> synthesize -> sensors -> reconstruct -> evaluate.

Shared by the single-run pipeline (runner.py) and the Monte Carlo runner (montecarlo.py), so
both always do exactly the same thing.

Random streams: the bolt seed is split into four independent child seeds (channel, array,
synthesis, sensors). Every stage builds a fresh generator from its child seed, so a stage's
draws do not depend on which other stages ran before it. Reconstruction is deterministic.

Variants (`run_bolt_variants`): several configs on the same bolt. Stage results are cached by
the config fields they depend on, so variants that differ only in, say, the reconstruction
reuse one synthesized and corrupted recording, and variants that differ in noise level reuse
one clean synthesis. All variants see the same random draws (common random numbers), which
removes bolt-to-bolt noise from comparisons between variants.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from thunder.acoustics.synth import synthesize
from thunder.atmosphere.profiles import build_atmosphere
from thunder.channel.generator import generate_channel
from thunder.config import ChannelConfig, RunConfig
from thunder.eval.metrics import evaluate
from thunder.recon import reconstruct
from thunder.sensors.arrays import build_mic_array
from thunder.sensors.corruption import corrupt
from thunder.types import Channel, FloatArray, MicArray, Reconstruction, Recording


@dataclass
class BoltResult:
    channel: Channel
    array: MicArray | None = None
    recording: Recording | None = None
    reconstruction: Reconstruction | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    per_point: dict[str, FloatArray] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    stages: list[str] = field(default_factory=list)


def child_seeds(seed: int | np.random.SeedSequence) -> list[np.random.SeedSequence]:
    """The four stage seeds of a bolt, derived without mutating `seed`.

    Identical to `np.random.default_rng(seed).spawn(4)` on a fresh seed (what earlier versions
    used), but safe to call any number of times on the same SeedSequence object (`spawn`
    advances a counter on the object, so repeated calls would give different streams).
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    start = ss.n_children_spawned
    return [
        np.random.SeedSequence(ss.entropy, spawn_key=(*ss.spawn_key, start + i), pool_size=ss.pool_size)
        for i in range(4)
    ]


def _key(*parts: Any) -> str:
    return json.dumps(parts, sort_keys=True, default=str)


@dataclass
class StageCache:
    """Stage results of one bolt, keyed by the config fields each stage depends on.

    Recordings are large (tens of MB for many mics or distant bolts), so the two recording
    stages keep only the `max_recordings` most recently used entries; order variants so that
    those sharing a recording are adjacent. Eviction never changes results (every stage
    rebuilds its generator from the bolt seed), only how much is recomputed.
    """

    channels: dict[str, Channel] = field(default_factory=dict)
    arrays: dict[str, MicArray] = field(default_factory=dict)
    clean: dict[str, tuple[Recording, float]] = field(default_factory=dict)
    corrupted: dict[str, tuple[Recording, float]] = field(default_factory=dict)
    max_recordings: int | None = 4

    def get(self, store: dict[str, Any], key: str) -> Any:
        """Entry or None; marks it most recently used."""
        if key not in store:
            return None
        store[key] = store.pop(key)
        return store[key]

    def put(self, store: dict[str, Any], key: str, value: Any) -> Any:
        store[key] = value
        if self.max_recordings is not None and store is not self.channels and store is not self.arrays:
            while len(store) > self.max_recordings:
                store.pop(next(iter(store)))
        return value


def run_bolt(
    cfg: RunConfig,
    seed: int | np.random.SeedSequence,
    channel_cfg: ChannelConfig | None = None,
    cache: StageCache | None = None,
) -> BoltResult:
    """Run every configured stage for one bolt (reusing `cache` entries when inputs match)."""
    channel_cfg = channel_cfg or cfg.channel
    if channel_cfg is None:
        raise ValueError("run_bolt needs a channel config")
    cache = cache if cache is not None else StageCache()
    s_channel, s_array, s_synth, s_sensor = child_seeds(seed)
    seed_int = seed if isinstance(seed, int) else None
    d = cfg.model_dump(mode="json")
    sensors = d.get("sensors") or {}

    t = time.perf_counter()
    k_ch = _key(channel_cfg.model_dump(mode="json"))
    if k_ch not in cache.channels:
        cache.channels[k_ch] = generate_channel(channel_cfg, np.random.default_rng(s_channel), seed=seed_int)
    ch = cache.channels[k_ch]
    res = BoltResult(channel=ch, stages=["generate"])
    res.timings_s["generate"] = time.perf_counter() - t
    if cfg.synthesis is None:
        return res

    syn = cfg.synthesis
    atm = build_atmosphere(cfg.atmosphere)
    k_arr = _key(d["array"], sensors.get("position"), sensors.get("timing"), sensors.get("mic"))
    if k_arr not in cache.arrays:
        cache.arrays[k_arr] = build_mic_array(cfg.array, cfg.sensors, np.random.default_rng(s_array))
    array = cache.arrays[k_arr]
    # Synthesis depends on the true positions and clocks (position + timing errors), not on the
    # mic response or the noise.
    k_syn = _key(
        k_ch,
        d["atmosphere"],
        d["array"],
        d["synthesis"],
        d["sample_rate_hz"],
        d["oversample"],
        sensors.get("position"),
        sensors.get("timing"),
    )
    if cache.get(cache.clean, k_syn) is None:
        t = time.perf_counter()
        rec = synthesize(
            ch,
            atm,
            array,
            cfg.sample_rate_hz,
            cfg.oversample,
            syn.emitter_spacing_m,
            syn.acoustic_efficiency,
            np.random.default_rng(s_synth),
            syn.micro_turn_mean,
            syn.micro_scale_m,
        )
        cache.put(cache.clean, k_syn, (rec, time.perf_counter() - t))
    rec, res.timings_s["synthesize"] = cache.clean[k_syn]  # cost of the (possibly shared) stage
    res.stages.append("synthesize")
    if cfg.sensors is not None:
        k_cor = _key(k_syn, k_arr, sensors)
        if cache.get(cache.corrupted, k_cor) is None:
            z_mic = np.array([float(np.mean(array.true_positions[:, 2]))])
            t = time.perf_counter()
            out = corrupt(
                rec,
                array,
                cfg.sensors,
                np.random.default_rng(s_sensor),
                sound_speed=float(atm.sound_speed(z_mic)[0]),
                air_density=float(atm.density(z_mic)[0]),
                wind_at_mics_mps=float(np.linalg.norm(atm.wind(z_mic)[0])),
            )
            cache.put(cache.corrupted, k_cor, (out, time.perf_counter() - t))
        rec, res.timings_s["sensors"] = cache.corrupted[k_cor]
        res.stages.append("sensors")
    res.array, res.recording = array, rec

    if cfg.reconstruction is not None:
        rcfg = cfg.reconstruction
        assumed = build_atmosphere(rcfg.atmosphere if rcfg.atmosphere is not None else cfg.atmosphere)
        t, cpu = time.perf_counter(), time.process_time()
        recon = reconstruct(rec, rec.nominal_mic_positions, assumed, rcfg)
        res.timings_s["reconstruct"] = time.perf_counter() - t
        # CPU time: comparable across runs regardless of machine load or sleep (wall time is not).
        res.timings_s["reconstruct_cpu"] = time.process_time() - cpu
        res.stages.append("reconstruct")
        res.reconstruction = recon
        t = time.perf_counter()
        metrics, per_point = evaluate(
            recon.points,
            recon.covariances,
            ch,
            rec.nominal_mic_positions.mean(axis=0),
            cfg.evaluation,
            recon.extra.get("strike_point"),
        )
        res.timings_s["evaluate"] = time.perf_counter() - t
        res.stages.append("evaluate")
        metrics["oracle_atmosphere"] = rcfg.atmosphere is None
        metrics["n_windows_active"] = recon.extra.get("n_windows_active", 0)
        metrics["n_windows_passed"] = int(np.sum(recon.extra.get("window_passed", [])))
        metrics.update(diagnostics(ch, rec))
        res.metrics, res.per_point = metrics, per_point
    return res


def diagnostics(ch: Channel, rec: Recording) -> dict[str, float]:
    """Physical context for interpreting a bolt's metrics (uses ground truth: evaluation only)."""
    out: dict[str, float] = {"peak_pa": float(np.max(np.abs(rec.signals)))}
    if rec.truth is not None:
        a, b = ch.segment_endpoints()
        length = np.linalg.norm(b - a, axis=1)
        heard = np.isfinite(rec.truth.segment_arrival_times[:, 0])
        out["shadow_fraction"] = float(length[~heard].sum() / length.sum())
        # How spread out the channel's sound is in time at the reference mic: the length-weighted
        # 5-95% span of segment arrival times, per km of heard channel. Small values mean many
        # channel parts arrive at once (overlapping arrivals), e.g. at long range.
        t_arr = rec.truth.segment_arrival_times[heard, 0]
        if t_arr.size >= 2:
            w = length[heard]
            order = np.argsort(t_arr)
            cdf = np.cumsum(w[order]) / w.sum()
            span = float(np.interp(0.95, cdf, t_arr[order]) - np.interp(0.05, cdf, t_arr[order]))
            out["arrival_span_s"] = span
            out["arrival_s_per_km"] = span / (0.9 * w.sum() / 1000.0)
        info = rec.truth.extra.get("corruption_info", {})
        if "snr_band_db" in info:
            out["snr_band_db"] = float(info["snr_band_db"])
    return out


def merge(base: dict, overrides: dict) -> dict:
    """Recursive dict merge (overrides win; nested dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def iter_bolt_variants(
    base: dict,
    variants: dict[str, dict],
    seed: int | np.random.SeedSequence,
    channel_dict: dict | None = None,
    cache: StageCache | None = None,
) -> Iterator[tuple[str, BoltResult]]:
    """Run every variant (overrides of the `base` config dict) on the same bolt, one at a time.

    `channel_dict`: the bolt's channel config (preset and explicit fields); a variant's own
    `channel` overrides are merged on top of it. Yielding lets callers drop each result's
    recordings before the next variant runs.
    """
    cache = cache if cache is not None else StageCache()
    for name, overrides in variants.items():
        cfg = RunConfig.model_validate(merge(base, overrides))
        ch_cfg = None
        if channel_dict is not None:
            ch_cfg = ChannelConfig.model_validate(merge(channel_dict, overrides.get("channel", {})))
        yield name, run_bolt(cfg, seed, ch_cfg, cache)


def run_bolt_variants(
    base: dict,
    variants: dict[str, dict],
    seed: int | np.random.SeedSequence,
    channel_dict: dict | None = None,
) -> dict[str, BoltResult]:
    """All variants of one bolt as a dict (keeps every result in memory; see iter_bolt_variants)."""
    return dict(iter_bolt_variants(base, variants, seed, channel_dict, StageCache(max_recordings=None)))

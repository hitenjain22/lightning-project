"""One bolt end to end: generate -> synthesize -> sensors -> reconstruct -> evaluate.

Shared by the single-run pipeline (runner.py) and the Monte Carlo runner (montecarlo.py), so
both always do exactly the same thing. Random streams: the bolt seed is split into four
independent children (channel, array, synthesis, sensors); reconstruction is deterministic.
"""

from __future__ import annotations

import time
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


def run_bolt(
    cfg: RunConfig, seed: int | np.random.SeedSequence, channel_cfg: ChannelConfig | None = None
) -> BoltResult:
    """Run every configured stage for one bolt."""
    channel_cfg = channel_cfg or cfg.channel
    if channel_cfg is None:
        raise ValueError("run_bolt needs a channel config")
    channel_rng, array_rng, synth_rng, sensor_rng = np.random.default_rng(seed).spawn(4)
    seed_int = seed if isinstance(seed, int) else None

    t = time.perf_counter()
    ch = generate_channel(channel_cfg, channel_rng, seed=seed_int)
    res = BoltResult(channel=ch, stages=["generate"])
    res.timings_s["generate"] = time.perf_counter() - t
    if cfg.synthesis is None:
        return res

    syn = cfg.synthesis
    atm = build_atmosphere(cfg.atmosphere)
    array = build_mic_array(cfg.array, cfg.sensors, array_rng)
    t = time.perf_counter()
    rec = synthesize(
        ch,
        atm,
        array,
        cfg.sample_rate_hz,
        cfg.oversample,
        syn.emitter_spacing_m,
        syn.acoustic_efficiency,
        synth_rng,
        syn.micro_turn_mean,
        syn.micro_scale_m,
    )
    res.timings_s["synthesize"] = time.perf_counter() - t
    res.stages.append("synthesize")
    if cfg.sensors is not None:
        z_mic = np.array([float(np.mean(array.true_positions[:, 2]))])
        t = time.perf_counter()
        rec = corrupt(
            rec,
            array,
            cfg.sensors,
            sensor_rng,
            sound_speed=float(atm.sound_speed(z_mic)[0]),
            air_density=float(atm.density(z_mic)[0]),
        )
        res.timings_s["sensors"] = time.perf_counter() - t
        res.stages.append("sensors")
    res.array, res.recording = array, rec

    if cfg.reconstruction is not None:
        rcfg = cfg.reconstruction
        assumed = build_atmosphere(rcfg.atmosphere if rcfg.atmosphere is not None else cfg.atmosphere)
        t = time.perf_counter()
        recon = reconstruct(rec, rec.nominal_mic_positions, assumed, rcfg)
        res.timings_s["reconstruct"] = time.perf_counter() - t
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
        res.metrics, res.per_point = metrics, per_point
    return res

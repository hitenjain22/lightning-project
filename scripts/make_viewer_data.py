"""Data for the interactive website (docs/index.html): a library of simulated strikes, each with
its thunder and its reconstruction from sound, exported as JSON plus the recorded audio.

Usage: python scripts/make_viewer_data.py [configs/experiments/media.yaml] [--n 60] [--workers 8]

Writes docs/viewer/bolts/index.json and, per strike k, docs/viewer/bolts/k.json and k.mp3.
Everything comes from the same pipeline as the experiments: the channel generator, the ray-traced
synthesis through a realistic stratified atmosphere, realistic sensors, and Method B with the true
atmosphere (the same "oracle" setup as E4's best row; the site says so). Strike 0 is the README's
hero bolt (the config's own seed). The others vary the channel type, distance, direction, wind and
temperature; every strike that was generated is published, whatever its score. Ground truth (true
channel, true arrival times) is exported only for display and scoring, as in the experiments; the
reconstruction itself never sees it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from thunder.config import load_config
from thunder.experiments.pipeline import run_bolt
from thunder.experiments.reports import describe_atmosphere

OUT = Path("docs/viewer/bolts")
ENVELOPE_DT_S = 0.01  # waveform drawn as min/max per 10 ms
AUDIO_RATE_HZ = 16000  # MP3 at 16 kHz plays everywhere; the recording itself is 8 kHz
LIBRARY_SEED = 20261008

# Channel types and how often they appear (multi-stroke flashes are left out: the site shows one
# return stroke per strike).
PRESETS = {"branched": 0.55, "with_incloud": 0.30, "tortuous": 0.15}


def r(a: np.ndarray, d: int = 1) -> list:
    """Rounded nested lists (keeps the JSON small; 0.1 m is far below the errors shown)."""
    return np.round(np.asarray(a, dtype=float), d).tolist()


def strike_overrides(k: int, base: dict) -> tuple[int, dict]:
    """Seed and config overrides for strike k (k = 0 is the config itself)."""
    if k == 0:
        return int(base["seed"]), {}
    rng = np.random.default_rng([LIBRARY_SEED, k])
    preset = str(rng.choice(list(PRESETS), p=list(PRESETS.values())))
    dist = float(np.exp(rng.uniform(math.log(1500.0), math.log(6000.0))))  # log-uniform 1.5-6 km
    atm = dict(base["atmosphere"])
    atm["temperature_c"] = round(float(rng.uniform(15.0, 32.0)), 1)
    atm["wind"] = {
        "speed_mps": round(float(rng.uniform(0.0, 8.0)), 1),
        "direction_from_deg": round(float(rng.uniform(0.0, 360.0))),
    }
    channel = {"preset": preset, "strike_distance_m": [dist, dist]}
    return int(rng.integers(1, 2**31 - 1)), {"channel": channel, "atmosphere": atm}


def leader_distance(nodes: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Path length (m) from the channel's origin in the cloud to the end of each segment, for the
    leader animation. Segments are (parent, child), so walk parents from the root(s)."""
    seg_len = np.linalg.norm(nodes[segs[:, 1]] - nodes[segs[:, 0]], axis=1)
    node_d = np.full(len(nodes), np.nan)
    roots = np.setdiff1d(segs[:, 0], segs[:, 1])
    node_d[roots] = 0.0
    children: dict[int, list[int]] = {}
    for i, (a, _) in enumerate(segs):
        children.setdefault(int(a), []).append(i)
    stack = [int(x) for x in roots]
    while stack:
        a = stack.pop()
        for i in children.get(a, []):
            b = int(segs[i, 1])
            node_d[b] = node_d[a] + seg_len[i]
            stack.append(b)
    return np.nan_to_num(node_d[segs[:, 1]], nan=0.0)


def sort_points_by_time(points: dict) -> None:
    """Order the reconstructed points by window time (Method B returns up to three sources per
    window, not always in time order across windows); the page plays them back in this order."""
    order = np.argsort(points["t"], kind="stable")
    for key, values in points.items():
        points[key] = [values[i] for i in order]


def build_strike(args: tuple[str, int]) -> dict:
    """Simulate and reconstruct strike k; write its JSON and MP3; return its summary."""
    config, k = args
    base = load_config(config).model_dump(mode="json")
    seed, over = strike_overrides(k, base)
    cfg = load_config(config, {**over, "seed": seed})
    assert cfg.reconstruction is not None and cfg.channel is not None
    with threadpool_limits(1):
        res = run_bolt(cfg, cfg.seed)
    ch, rec, recon = res.channel, res.recording, res.reconstruction
    assert rec is not None and rec.truth is not None and recon is not None
    m = res.metrics
    ref = int(recon.extra.get("reference_mic", 0))

    kind = np.where(ch.is_main, 0, np.where(ch.is_incloud, 2, 1))  # 0 main, 1 branch, 2 in-cloud
    arrival = rec.truth.segment_arrival_times[:, ref]  # true arrival at the reference mic (NaN: shadow)
    seg_len = np.linalg.norm(np.diff(ch.nodes[ch.segments], axis=1)[:, 0], axis=1)

    sig = rec.signals[ref]
    fs = rec.sample_rate
    n = max(1, int(round(ENVELOPE_DT_S * fs)))
    blocks = sig[: (len(sig) // n) * n].reshape(-1, n)
    peak = float(np.max(np.abs(sig)))
    audio = resample_poly(0.9 * sig / peak, AUDIO_RATE_HZ, int(round(fs)))
    sf.write(OUT / f"{k}.mp3", np.clip(audio, -1, 1), AUDIO_RATE_HZ, format="MP3")

    def num(x: float, d: int = 3) -> float | None:
        return None if x is None or not np.isfinite(x) else round(float(x), d)

    strike = np.asarray(ch.metadata["strike_point"], dtype=float)
    atm = cfg.model_dump(mode="json")["atmosphere"]
    about = {
        "id": k,
        "seed": cfg.seed,
        "preset": cfg.channel.preset,
        "distance_km": round(float(np.hypot(strike[0], strike[1])) / 1000, 2),
        "azimuth_deg": round(math.degrees(math.atan2(strike[0], strike[1])) % 360),
        "top_km": round(float(ch.nodes[:, 2].max()) / 1000, 2),
        "length_km": round(float(seg_len.sum()) / 1000, 2),
        "n_branches": int(len(np.unique(ch.branch_id[kind == 1]))),
        "wind_mps": atm["wind"]["speed_mps"],
        "wind_from_deg": atm["wind"]["direction_from_deg"],
        "temperature_c": atm["temperature_c"],
        "duration_s": round(len(sig) / fs, 2),
        "first_arrival_s": num(np.nanmin(arrival), 2),
        "n_points": int(m["n_points"]),
        "median_error_m": num(m["point_error_median_m"], 2),
        "coverage_main_50m": num(m["coverage_main_50m"]),
    }
    data = {
        "about": {
            **about,
            "atmosphere": describe_atmosphere(atm),
            "array": f"{len(rec.nominal_mic_positions)} microphones, {cfg.array.layout}, "
            f"{cfg.array.aperture_m:g} m",
            "method": cfg.reconstruction.method,
            "assumed_atmosphere": "true (oracle)" if cfg.reconstruction.atmosphere is None else "assumed",
            "sound_speed_mps": round(331.3 * math.sqrt(1 + atm["temperature_c"] / 273.15), 1),
        },
        "metrics": {
            "n_points": int(m["n_points"]),
            "median_error_m": num(m["point_error_median_m"]),
            "p90_error_m": num(m["point_error_p90_m"]),
            "angular_error_deg": num(m["angular_error_median_deg"], 4),
            "coverage_main_50m": num(m["coverage_main_50m"]),
            "coverage_all_50m": num(m["coverage_50m"]),
            "coverage_branch_50m": num(m["coverage_branch_50m"]),
            "strike_error_m": num(m["strike_error_m"], 1),
            "shadow_fraction": num(m.get("shadow_fraction", 0.0)),
        },
        "channel": {
            "nodes": r(ch.nodes),
            "segments": ch.segments.tolist(),
            "kind": kind.tolist(),
            "arrival_s": [None if not np.isfinite(t) else round(float(t), 3) for t in arrival],
            "leader_m": r(leader_distance(ch.nodes, ch.segments), 0),
            "strike": r(strike),
        },
        "mics": r(rec.nominal_mic_positions, 2),
        "reference_mic": ref,
        "waveform": {
            "dt_s": n / fs,
            "min": r(blocks.min(axis=1) / peak, 3),
            "max": r(blocks.max(axis=1) / peak, 3),
            "peak_pa": peak,
            "duration_s": len(sig) / fs,
        },
        "window_s": cfg.reconstruction.window_s,
        "points": {
            "xyz": r(recon.points),
            "t": r(recon.window_times, 3),
            "error_m": r(res.per_point.get("error_m", np.zeros(len(recon.points)))),
            "sigma_m": r(np.sqrt(np.clip(np.einsum("kii->ki", recon.covariances), 0, None)), 2),
        },
    }
    sort_points_by_time(data["points"])
    (OUT / f"{k}.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"strike {k:3d}: {about['preset']:12s} {about['distance_km']:4.1f} km  "
          f"{about['n_points']:4d} points  median {about['median_error_m']} m", flush=True)
    return about


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/experiments/media.yaml")
    ap.add_argument("--n", type=int, default=60, help="number of strikes in the library")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        strikes = list(pool.map(build_strike, [(args.config, k) for k in range(args.n)]))

    med = np.array([s["median_error_m"] for s in strikes if s["median_error_m"] is not None])
    cov = np.array([s["coverage_main_50m"] for s in strikes if s["coverage_main_50m"] is not None])
    index = {
        "config": args.config,
        "library_seed": LIBRARY_SEED,
        "method": "B",
        "atmosphere_known": True,
        "summary": {
            "n_strikes": len(strikes),
            "n_reconstructed": int(sum(s["n_points"] > 0 for s in strikes)),
            "median_of_medians_m": round(float(np.median(med)), 2),
            "median_coverage_main": round(float(np.median(cov)), 3),
            "total_points": int(sum(s["n_points"] for s in strikes)),
        },
        "strikes": strikes,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index["summary"], indent=2))


if __name__ == "__main__":
    main()

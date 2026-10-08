"""Data for the interactive website (docs/index.html): one simulated bolt, its thunder and its
reconstruction, exported as JSON plus the recorded audio.

Usage: python scripts/make_viewer_data.py [configs/experiments/media.yaml] [--seed N]

Writes docs/viewer/bolt.json and docs/viewer/thunder.wav. Everything comes from the same pipeline
as the experiments (scripts/make_media.py uses the same config): the channel generator, the
ray-traced synthesis through a realistic atmosphere, realistic sensors, and Method B. Ground truth
(true channel, true arrival times) is exported only for display and scoring, as in the
experiments; the reconstruction itself never sees it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from thunder.config import load_config
from thunder.experiments.pipeline import run_bolt
from thunder.experiments.reports import describe_atmosphere

OUT = Path("docs/viewer")
ENVELOPE_DT_S = 0.01  # waveform drawn as min/max per 10 ms


def r1(a: np.ndarray, d: int = 1) -> list:
    """Rounded nested lists (keeps the JSON small; 0.1 m is far below the errors shown)."""
    return np.round(np.asarray(a, dtype=float), d).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/experiments/media.yaml")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config, {"seed": args.seed} if args.seed is not None else None)
    assert cfg.reconstruction is not None and cfg.channel is not None

    res = run_bolt(cfg, cfg.seed)
    ch, rec, recon = res.channel, res.recording, res.reconstruction
    assert rec is not None and rec.truth is not None and recon is not None
    m = res.metrics
    ref = int(recon.extra.get("reference_mic", 0))

    kind = np.where(ch.is_main, 0, np.where(ch.is_incloud, 2, 1))  # 0 main, 1 branch, 2 in-cloud
    arrival = rec.truth.segment_arrival_times[:, ref]  # true arrival at the reference mic (NaN: shadow)

    sig = rec.signals[ref]
    fs = rec.sample_rate
    n = max(1, int(round(ENVELOPE_DT_S * fs)))
    k = len(sig) // n
    blocks = sig[: k * n].reshape(k, n)
    peak = float(np.max(np.abs(sig)))

    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / "thunder.wav", 0.9 * sig / peak, int(round(fs)), subtype="PCM_16")

    cov = recon.covariances
    data = {
        "about": {
            "config": args.config,
            "seed": cfg.seed,
            "atmosphere": describe_atmosphere(cfg.model_dump(mode="json")["atmosphere"]),
            "array": f"{len(rec.nominal_mic_positions)} microphones, {cfg.array.layout}, "
            f"{cfg.array.aperture_m:g} m",
            "method": cfg.reconstruction.method,
            "assumed_atmosphere": "true (oracle)" if cfg.reconstruction.atmosphere is None else "assumed",
            "sound_speed_mps": 347.0,
        },
        "metrics": {
            "n_points": int(m["n_points"]),
            "median_error_m": float(m["point_error_median_m"]),
            "p90_error_m": float(m["point_error_p90_m"]),
            "angular_error_deg": float(m["angular_error_median_deg"]),
            "coverage_main_50m": float(m["coverage_main_50m"]),
            "coverage_all_50m": float(m["coverage_50m"]),
            "coverage_branch_50m": None
            if np.isnan(m["coverage_branch_50m"])
            else float(m["coverage_branch_50m"]),
            "strike_error_m": float(m["strike_error_m"]),
            "shadow_fraction": float(m.get("shadow_fraction", 0.0)),
        },
        "channel": {
            "nodes": r1(ch.nodes),
            "segments": ch.segments.tolist(),
            "kind": kind.tolist(),
            "arrival_s": [None if not np.isfinite(t) else round(float(t), 4) for t in arrival],
            "length_km": float(
                np.linalg.norm(np.diff(ch.nodes[ch.segments], axis=1)[:, 0], axis=1).sum() / 1000
            ),
            "strike": r1(ch.metadata["strike_point"]),
        },
        "mics": r1(rec.nominal_mic_positions, 2),
        "reference_mic": ref,
        "waveform": {
            "dt_s": n / fs,
            "min": r1(blocks.min(axis=1) / peak, 3),
            "max": r1(blocks.max(axis=1) / peak, 3),
            "peak_pa": peak,
            "duration_s": len(sig) / fs,
        },
        "window_s": cfg.reconstruction.window_s,
        "points": {
            "xyz": r1(recon.points),
            "t": r1(recon.window_times, 3),
            "error_m": r1(res.per_point.get("error_m", np.zeros(len(recon.points)))),
            "sigma_m": r1(np.sqrt(np.clip(np.einsum("kii->ki", cov), 0, None)), 2),
        },
    }
    (OUT / "bolt.json").write_text(json.dumps(data, separators=(",", ":")))
    size = (OUT / "bolt.json").stat().st_size
    print(json.dumps(data["metrics"], indent=2))
    print(f"Wrote {OUT / 'bolt.json'} ({size / 1e3:.0f} kB) and {OUT / 'thunder.wav'}")


if __name__ == "__main__":
    main()

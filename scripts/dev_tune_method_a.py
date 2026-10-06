"""Tune Method A gates on DEVELOPMENT bolts (seeds disjoint from every experiment seed).

Usage: python scripts/dev_tune_method_a.py [--bolts 30]

Synthesizes ideal-condition recordings once (cached in results/dev_tune/), then grid-searches
reconstruction settings in parallel. Prints a table; the chosen defaults and the reasoning
are recorded in docs/log.md. Development seeds start at 5000; E1 uses its own config seed.
"""

import argparse
import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from thunder.acoustics.synth import synthesize
from thunder.channel.generator import generate_channel
from thunder.config import ChannelConfig, EvaluationConfig, ReconstructionConfig
from thunder.constants import ACOUSTIC_EFFICIENCY, ZERO_CELSIUS_K, sound_speed_dry
from thunder.eval.metrics import evaluate
from thunder.recon import reconstruct
from thunder.sensors.arrays import square
from thunder.types import MicArray, UniformAtmosphere

DEV_SEED0 = 5000
CACHE = Path("results/dev_tune/recordings.pkl")
T_AIR = ZERO_CELSIUS_K + 25.0
ATM = UniformAtmosphere(sound_speed_dry(T_AIR), T_AIR)
ARRAY = MicArray.ideal(square(50.0, 1.5, center=True))
PRESETS = ("tortuous", "branched", "with_incloud")

GRID = {
    "window_s": [0.05, 0.1],
    "max_residual_s": [5e-4, 1e-3, 2e-3],
    "min_peak": [0.2, 0.3, 0.4],
    "dbscan_min_samples": [2, 3],
    "dbscan_eps_m": [150.0, 300.0],
}


def make_bolts(n: int):
    if CACHE.exists():
        bolts = pickle.loads(CACHE.read_bytes())
        if len(bolts) >= n:
            return bolts[:n]
    bolts = []
    for k in range(n):
        seed = DEV_SEED0 + k
        cfg = ChannelConfig(preset=PRESETS[k % 3], strike_distance_m=(1000.0, 3000.0))
        ch = generate_channel(cfg, np.random.default_rng(seed), seed=seed)
        rec = synthesize(ch, ATM, ARRAY, 8000.0, 8, 0.5, ACOUSTIC_EFFICIENCY, np.random.default_rng(seed + 1),
                         np.radians(16.0), 1.0)
        bolts.append((ch, rec))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(bolts))
    return bolts


_BOLTS: list = []


def _init_worker(n: int) -> None:
    """Load the cached recordings once per worker process (not once per task)."""
    _BOLTS.extend(pickle.loads(CACHE.read_bytes())[:n])


def score(params: dict) -> dict:
    bolts = _BOLTS
    cfg = ReconstructionConfig(detection_dynamic_range_db=80.0, **params)
    rows = []
    for ch, rec in bolts:
        r = reconstruct(rec, ARRAY.nominal_positions, ATM, cfg)
        m, pp = evaluate(r.points, r.covariances, ch, ARRAY.nominal_positions.mean(0), EvaluationConfig())
        rows.append({**m, "errors": pp.get("error_m", np.zeros(0))})
    pooled = np.concatenate([r["errors"] for r in rows])
    return {
        **params,
        "pooled_median_m": float(np.median(pooled)) if pooled.size else np.nan,
        "pooled_p90_m": float(np.percentile(pooled, 90)) if pooled.size else np.nan,
        "coverage_main_50m": float(np.mean([r["coverage_main_50m"] for r in rows])),
        "coverage_50m": float(np.mean([r["coverage_50m"] for r in rows])),
        "strike_median_m": float(np.nanmedian([r["strike_error_m"] for r in rows])),
        "points_per_bolt": float(np.mean([r["n_points"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bolts", type=int, default=30)
    args = parser.parse_args()
    bolts = make_bolts(args.bolts)
    keys = list(GRID)
    combos = [dict(zip(keys, v, strict=True)) for v in itertools.product(*GRID.values())]
    del bolts  # workers read the cache themselves
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(args.bolts,)) as pool:
        results = list(pool.map(score, combos))
    df = pd.DataFrame(results).sort_values("coverage_main_50m", ascending=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    out = CACHE.parent / "grid.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

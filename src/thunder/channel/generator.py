"""Phase 1: cloud-to-ground lightning channel generator.

The main channel walks (see walk.py) from a start point directly above the target strike
point until it crosses z = 0; the last segment is cut at the ground and the whole main channel
is then translated horizontally so it lands exactly on the target. The in-cloud section and
branches are grown afterwards from the translated main channel.
"""

from __future__ import annotations

import math

import numpy as np

from thunder.channel.branching import grow_branches
from thunder.channel.walk import DOWN, Goal, TreeBuilder, random_walk, turn_angle_distribution, unit
from thunder.config import ChannelConfig
from thunder.types import Channel, FloatArray

INCLOUD_ALTITUDE_SCALE_M = 1000.0  # altitude error that tilts the in-cloud goal by 45 deg


def _incloud_goal(heading: FloatArray, z_ref: float) -> Goal:
    """Keep the initial horizontal heading, with a restoring pull toward altitude z_ref."""

    def goal(pos: FloatArray, d: FloatArray) -> FloatArray:
        g = heading.copy()
        g[2] = (z_ref - pos[2]) / INCLOUD_ALTITUDE_SCALE_M
        return unit(g)

    return goal


def generate_channel(cfg: ChannelConfig, rng: np.random.Generator, seed: int | None = None) -> Channel:
    """Generate one cloud-to-ground channel. `seed` is recorded in metadata only."""
    step = cfg.segment_length_m
    turn_dist = turn_angle_distribution(cfg.turn_distribution, cfg.turn_mean)

    # Target strike point, and a start point above it offset horizontally in a random direction.
    dist = rng.uniform(*cfg.strike_distance_m)
    az = rng.uniform(*cfg.strike_azimuth)  # clockwise from north (+y)
    target = np.array([dist * math.sin(az), dist * math.cos(az), 0.0])
    height = rng.uniform(*cfg.start_height_m)
    offset = rng.uniform(*cfg.start_offset_m)
    offset_az = rng.uniform(-math.pi, math.pi)
    start = target + np.array([offset * math.sin(offset_az), offset * math.cos(offset_az), height])

    n_max = int(math.ceil(20 * height / step))
    walk = random_walk(
        start, unit(target - start), n_max, step, turn_dist, cfg.bias_strength,
        lambda p, d: unit(target - p) if np.linalg.norm(target - p) > 0 else DOWN, rng, land=True,
    )
    if not walk.landed:
        raise RuntimeError("main channel did not reach the ground; increase bias_strength")
    main = walk.nodes.copy()
    main[:, :2] += target[:2] - main[-1, :2]  # land exactly on the target
    main[-1] = target

    tree = TreeBuilder(main[0])
    main_idx = tree.add_path(main, 0, cfg.energy_per_length_main, main=True)

    if cfg.incloud:
        length = rng.uniform(*cfg.incloud_length_m)
        heading = rng.uniform(-math.pi, math.pi)
        d0 = np.array([math.cos(heading), math.sin(heading), 0.0])
        ic = random_walk(
            main[0], d0, max(1, round(length / step)), step, turn_dist, cfg.bias_strength,
            _incloud_goal(d0, main[0, 2]), rng, floor=step,
        )
        tree.add_path(ic.nodes, 0, cfg.energy_per_length_main, incloud=True)

    grow_branches(tree, main_idx, 1, cfg.energy_per_length_main, cfg, turn_dist, rng)

    n_strokes = int(rng.integers(cfg.n_strokes[0], cfg.n_strokes[1] + 1))
    intervals = rng.uniform(*cfg.stroke_interval_s, size=n_strokes - 1)
    stroke_times = np.concatenate([[0.0], np.cumsum(intervals)])

    metadata = {
        "seed": seed,
        "preset": cfg.preset,
        "params": cfg.model_dump(mode="json"),
        "strike_point": target.tolist(),
        "start_point": main[0].tolist(),
    }
    return tree.build(stroke_times, metadata)

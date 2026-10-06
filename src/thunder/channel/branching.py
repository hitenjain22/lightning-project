"""Branch growth: side channels that leave the parent at an angle and never reach the ground.

A branch leaves its parent at the branch angle and is then steered (von Mises turn azimuth,
see walk.py) toward its own departure direction, so it keeps heading down and outward the
way branches do in photographs instead of collapsing back alongside the parent.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from thunder.channel.walk import Goal, TreeBuilder, random_walk, rotate, unit
from thunder.config import ChannelConfig
from thunder.types import FloatArray


def _fixed_goal(direction: FloatArray) -> Goal:
    return lambda pos, d: direction


def grow_branches(
    tree: TreeBuilder,
    parent_nodes: list[int],
    depth: int,
    energy: float,
    cfg: ChannelConfig,
    turn_dist: stats.rv_continuous,
    rng: np.random.Generator,
) -> None:
    """Spawn branches from interior nodes of a path, recursing to cfg.branch_max_depth."""
    if depth > cfg.branch_max_depth or cfg.branch_probability == 0 or len(parent_nodes) < 3:
        return
    step = cfg.segment_length_m
    interior = parent_nodes[1:-1]
    hits = rng.random(len(interior)) < cfg.branch_probability
    sites = [k for k, hit in zip(interior, hits, strict=True) if hit]
    branch_energy = energy * cfg.branch_energy_fraction
    median = cfg.branch_length_median_m * cfg.branch_length_decay ** (depth - 1)
    lo, hi = cfg.branch_angle
    for site in sites:
        pos = tree.nodes[site]
        if pos[2] < 2 * step:
            continue
        # Incoming direction at the site (the segment ending at `site`).
        idx = parent_nodes.index(site)
        incoming = unit(pos - tree.nodes[parent_nodes[idx - 1]])
        d0 = rotate(incoming, rng.uniform(lo, hi), rng.uniform(-math.pi, math.pi))
        length = median * math.exp(cfg.branch_length_sigma * rng.standard_normal())
        n_steps = max(1, round(length / step))
        walk = random_walk(
            pos,
            d0,
            n_steps,
            step,
            turn_dist,
            cfg.branch_bias_strength,
            _fixed_goal(d0),
            rng,
            floor=step,
        )
        if len(walk.nodes) < 2:
            continue
        idxs = tree.add_path(walk.nodes, site, branch_energy)
        grow_branches(tree, idxs, depth + 1, branch_energy, cfg, turn_dist, rng)

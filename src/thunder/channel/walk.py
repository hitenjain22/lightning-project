"""Random-walk primitives shared by the main channel, branches and in-cloud sections.

Walk model. Each step keeps the previous direction d and turns it by an angle theta drawn
exactly from the configured turn-angle distribution. The steering bias only chooses the
*direction* of the turn: the turn azimuth phi (around d) is von Mises distributed, centered
on the projection of the goal direction onto the plane perpendicular to d, with concentration
kappa. So measured turn angles match the configured distribution exactly (Hill statistics),
while the walk still drifts toward its goal. kappa = 0 is an unbiased walk.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from thunder.types import Channel, FloatArray

DOWN = np.array([0.0, 0.0, -1.0])

Goal = Callable[[FloatArray, FloatArray], FloatArray]  # (position, direction) -> unit goal dir


def turn_angle_distribution(name: str, mean: float) -> stats.rv_continuous:
    """Frozen scipy distribution of turn angles (rad) with the given mean."""
    if name == "halfnormal":
        return stats.halfnorm(scale=mean * math.sqrt(math.pi / 2))
    if name == "exponential":
        return stats.expon(scale=mean)
    raise ValueError(f"unknown turn distribution {name!r}")


def perpendicular_basis(d: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Two unit vectors e1, e2 with (e1, e2, d) right-handed and orthonormal. d must be unit."""
    x, y, z = d
    # e1 = helper x d, with helper = x-hat near vertical, else z-hat (avoids a degenerate cross).
    e1 = np.array([0.0, -z, y]) if abs(z) > 0.9 else np.array([-y, x, 0.0])
    e1 /= math.sqrt(e1 @ e1)
    e2 = np.array([d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]])
    return e1, e2


def _tilt(d: FloatArray, e1: FloatArray, e2: FloatArray, angle: float, azimuth: float) -> FloatArray:
    out = math.cos(angle) * d + math.sin(angle) * (math.cos(azimuth) * e1 + math.sin(azimuth) * e2)
    return out / math.sqrt(out @ out)


def rotate(d: FloatArray, angle: float, azimuth: float) -> FloatArray:
    """Tilt unit vector d by `angle`, toward azimuth `azimuth` in d's perpendicular plane."""
    e1, e2 = perpendicular_basis(d)
    return _tilt(d, e1, e2, angle, azimuth)


def turn(d: FloatArray, angle: float, goal: FloatArray, offset: float, fallback: float) -> FloatArray:
    """Turn d by exactly `angle`, toward the goal's azimuth around d plus `offset`.

    If the goal is parallel to d its azimuth is undefined and `fallback` is used instead.
    """
    e1, e2 = perpendicular_basis(d)
    a, b = float(goal @ e1), float(goal @ e2)
    azimuth = math.atan2(b, a) + offset if math.hypot(a, b) > 1e-9 else fallback
    return _tilt(d, e1, e2, angle, azimuth)


@dataclass(frozen=True)
class WalkResult:
    nodes: FloatArray  # (n + 1, 3), first row is the start point
    landed: bool  # True if the walk was cut at z = 0


def random_walk(
    start: FloatArray,
    d0: FloatArray,
    n_steps: int,
    step: float,
    turn_dist: stats.rv_continuous,
    kappa: float,
    goal: Goal,
    rng: np.random.Generator,
    floor: float | None = None,
    land: bool = False,
) -> WalkResult:
    """Biased random walk. The first step goes along d0; each later step turns.

    land:  stop when a step crosses z = 0, cutting that step exactly at the ground.
    floor: stop before any step that would end below this height (step is discarded).
    """
    thetas = np.minimum(turn_dist.rvs(size=n_steps, random_state=rng), math.pi)
    offsets = rng.vonmises(0.0, kappa, n_steps) if kappa > 0 else rng.uniform(-math.pi, math.pi, n_steps)
    uniform = rng.uniform(-math.pi, math.pi, n_steps)

    pos = np.asarray(start, dtype=float)
    d = np.asarray(d0, dtype=float) / np.linalg.norm(d0)
    nodes = [pos]
    for i in range(n_steps):
        if i > 0:
            d = turn(d, float(thetas[i]), goal(pos, d), float(offsets[i]), float(uniform[i]))
        nxt = pos + step * d
        if land and nxt[2] <= 0.0:
            frac = pos[2] / (pos[2] - nxt[2])
            nxt = pos + frac * (nxt - pos)
            nxt[2] = 0.0
            nodes.append(nxt)
            return WalkResult(np.array(nodes), True)
        if floor is not None and nxt[2] < floor:
            break
        nodes.append(nxt)
        pos = nxt
    return WalkResult(np.array(nodes), False)


def unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


class TreeBuilder:
    """Accumulates paths into the flat node/segment arrays of a Channel."""

    def __init__(self, root: FloatArray):
        self.nodes: list[FloatArray] = [np.asarray(root, dtype=float)]
        self.segments: list[tuple[int, int]] = []
        self.energy: list[float] = []
        self.branch_id: list[int] = []
        self.is_main: list[bool] = []
        self.is_incloud: list[bool] = []
        self.next_branch_id = 0

    def add_path(
        self, path: FloatArray, parent: int, energy: float, main: bool = False, incloud: bool = False
    ) -> list[int]:
        """Append path[1:] as new nodes hanging off node `parent`. Returns all node indices."""
        bid = self.next_branch_id
        self.next_branch_id += 1
        indices = [parent]
        for p in path[1:]:
            self.nodes.append(p)
            child = len(self.nodes) - 1
            self.segments.append((indices[-1], child))
            self.energy.append(energy)
            self.branch_id.append(bid)
            self.is_main.append(main)
            self.is_incloud.append(incloud)
            indices.append(child)
        return indices

    def build(self, stroke_times: FloatArray, metadata: dict) -> Channel:
        n = len(self.segments)
        return Channel(
            nodes=np.array(self.nodes),
            segments=np.array(self.segments, dtype=np.int64).reshape(n, 2),
            energy_per_length=np.array(self.energy, dtype=float),
            branch_id=np.array(self.branch_id, dtype=np.int64),
            is_main=np.array(self.is_main, dtype=bool),
            is_incloud=np.array(self.is_incloud, dtype=bool),
            stroke_times=stroke_times,
            metadata=metadata,
        )

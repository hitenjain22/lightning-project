"""Phase 4 array geometry generators and realization of hidden sensor errors.

Every generator builds a unit-size horizontal shape, scales it so the largest horizontal
distance between any two mics equals the aperture exactly, and centers it so the horizontal
centroid is the origin (ENU convention). Mics sit at `mic_height` unless stated otherwise.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import pdist

from thunder.config import ArrayConfig, SensorsConfig
from thunder.types import FloatArray, MicArray


def horizontal_aperture(positions: FloatArray) -> float:
    """Largest horizontal distance between any two mics, m."""
    p = np.asarray(positions)
    return float(np.max(pdist(p[:, :2]))) if len(p) > 1 else 0.0


def _finish(xy: FloatArray, aperture: float, z: FloatArray | float) -> FloatArray:
    """Scale xy to the aperture, center it, and attach heights."""
    xy = np.asarray(xy, dtype=float)
    xy = xy * (aperture / float(np.max(pdist(xy))))
    xy = xy - xy.mean(axis=0)
    z = np.broadcast_to(np.asarray(z, dtype=float), (len(xy),))
    return np.column_stack([xy, z])


def _ring(n: int, phase: float = math.pi / 2) -> FloatArray:
    ang = phase + 2 * math.pi * np.arange(n) / n
    return np.column_stack([np.cos(ang), np.sin(ang)])


def triangle(aperture: float, height: float = 1.5) -> FloatArray:
    """Equilateral triangle; side = aperture. The 3-mic minimum."""
    return _finish(_ring(3), aperture, height)


def square(aperture: float, height: float = 1.5, center: bool = False) -> FloatArray:
    """Square; diagonal = aperture. With center=True a fifth mic sits in the middle."""
    xy = _ring(4, math.pi / 4)
    if center:
        xy = np.vstack([xy, [0.0, 0.0]])
    return _finish(xy, aperture, height)


def circle(n: int, aperture: float, height: float = 1.5) -> FloatArray:
    """n mics evenly on a circle (for odd n the circle is slightly larger than the aperture)."""
    if n < 3:
        raise ValueError("circle needs at least 3 mics")
    return _finish(_ring(n), aperture, height)


def l_shape(n: int, aperture: float, height: float = 1.5) -> FloatArray:
    """Corner mic plus two equal perpendicular arms (n odd, >= 3)."""
    if n < 3 or n % 2 == 0:
        raise ValueError("l_shape needs an odd number of mics >= 3")
    k = (n - 1) // 2
    arm = np.arange(1, k + 1, dtype=float)
    xy = np.vstack([[0.0, 0.0], np.column_stack([arm, 0 * arm]), np.column_stack([0 * arm, arm])])
    return _finish(xy, aperture, height)


def cross(n: int, aperture: float, height: float = 1.5) -> FloatArray:
    """Four equal perpendicular arms; n = 4k (no center) or 4k + 1 (with a center mic)."""
    if n < 4 or n % 4 not in (0, 1):
        raise ValueError("cross needs n = 4k or 4k + 1 mics, n >= 4")
    k = n // 4
    arm = np.arange(1, k + 1, dtype=float)
    xy = [np.column_stack([sx * arm, sy * arm]) for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    if n % 4 == 1:
        xy.insert(0, np.zeros((1, 2)))
    return _finish(np.vstack(xy), aperture, height)


def mast(aperture: float, height: float = 1.5, mast_height: float = 10.0) -> FloatArray:
    """Square at mic height plus a fifth mic on a central mast (improves elevation resolution)."""
    p = square(aperture, height, center=True)
    p[-1, 2] = mast_height
    return p


def random_disk(n: int, aperture: float, rng: np.random.Generator, height: float = 1.5) -> FloatArray:
    """n mics uniform in a disk, no two closer than aperture / (2 sqrt(n)), then scaled.

    The exclusion disks cover 25% of the array disk, well below the ~55% jamming limit of
    random sequential packing, so rejection sampling always terminates quickly.
    """
    if n < 3:
        raise ValueError("random_disk needs at least 3 mics")
    min_sep = 0.5 / math.sqrt(n)  # in units of the unit-diameter disk
    pts: list[FloatArray] = []
    for _ in range(1000 * n):
        if len(pts) == n:
            break
        r = 0.5 * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        cand = np.array([r * math.cos(a), r * math.sin(a)])
        if all(np.linalg.norm(cand - q) >= min_sep for q in pts):
            pts.append(cand)
    if len(pts) < n:
        raise RuntimeError("random_disk rejection sampling did not converge")
    return _finish(np.array(pts), aperture, height)


def distributed(
    sub_layout: str, n_sub: int, sub_aperture: float, separation: float, height: float = 1.5
) -> FloatArray:
    """Two (on a line) or three (equilateral triangle) identical sub-arrays `separation` apart."""
    sub = {"triangle": triangle, "square": square, "square_center": lambda a, h: square(a, h, True)}[
        sub_layout
    ](sub_aperture, height)
    centers = np.array([[-0.5, 0.0], [0.5, 0.0]]) if n_sub == 2 else _ring(3) / math.sqrt(3)
    centers = centers * separation
    centers -= centers.mean(axis=0)
    out = np.vstack([sub + [c[0], c[1], 0.0] for c in centers])
    out[:, :2] -= out[:, :2].mean(axis=0)
    return out


def nominal_positions(cfg: ArrayConfig, rng: np.random.Generator) -> FloatArray:
    """Nominal mic positions for an array config."""
    a, h, n = cfg.aperture_m, cfg.mic_height_m, cfg.n_mics
    layout = cfg.layout
    if layout == "free_form":
        assert cfg.positions_m is not None
        return np.array(cfg.positions_m, dtype=float)
    if layout == "triangle":
        return triangle(a, h)
    if layout in ("square", "square_center"):
        return square(a, h, center=layout == "square_center")
    if layout == "mast":
        return mast(a, h, cfg.mast_height_m)
    if layout == "distributed":
        return distributed(cfg.subarray_layout, cfg.n_subarrays, a, cfg.subarray_separation_m, h)
    if n is None:
        raise ValueError(f"layout {layout!r} needs n_mics")
    if layout == "circle":
        return circle(n, a, h)
    if layout == "l_shape":
        return l_shape(n, a, h)
    if layout == "cross":
        return cross(n, a, h)
    if layout == "random_disk":
        return random_disk(n, a, rng, h)
    raise ValueError(f"unknown layout {layout!r}")


def build_mic_array(
    array_cfg: ArrayConfig, sensors: SensorsConfig | None, rng: np.random.Generator
) -> MicArray:
    """Nominal layout plus a realization of the hidden hardware errors.

    Random draws come from independent child streams per error type, so turning one error
    on or off never changes the realization of another.
    """
    layout_rng, pos_rng, clock_rng, mic_rng = rng.spawn(4)
    nominal = nominal_positions(array_cfg, layout_rng)
    m = len(nominal)
    if sensors is None:
        return MicArray.ideal(nominal)

    pos = sensors.position
    sigma = np.array([pos.horizontal_std_m, pos.horizontal_std_m, pos.vertical_std_m])
    err = pos_rng.standard_normal((m, 3)) * sigma + 0.0  # + 0.0 turns -0.0 into 0.0
    true = nominal + err
    if np.any(true[:, 2] < 0):
        true[:, 2] = np.maximum(true[:, 2], 0.0)  # a mic cannot be underground

    tim = sensors.timing
    k = 1 if tim.common_clock else m
    offset = np.broadcast_to(clock_rng.standard_normal(k) * tim.offset_std_s + 0.0, (m,)).copy()
    drift = np.broadcast_to(clock_rng.standard_normal(k) * tim.drift_std_ppm + 0.0, (m,)).copy()

    mic = sensors.mic
    gain_db = mic_rng.standard_normal(m) * mic.gain_tolerance_db + 0.0
    corner_scale = np.exp(mic_rng.standard_normal(m) * mic.corner_tolerance)
    return MicArray(nominal, true, offset, drift, gain_db, corner_scale)

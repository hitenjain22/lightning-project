"""Phase 5 reconstruction. Every method shares one interface:

    reconstruct(recording, array_nominal, atmosphere_assumed, config) -> Reconstruction

Inverse-crime guard: nothing in this package may read `Recording.truth` or import
`GroundTruth` (enforced by tests/test_recon_isolation.py). Methods see only the signals,
the nominal mic positions, the reported flash time and an *assumed* atmosphere.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from thunder.config import ReconstructionConfig
from thunder.recon.bayes import reconstruct_bayes, reconstruct_bayes_storm
from thunder.recon.multilat import reconstruct_multilateration
from thunder.recon.postprocess import dbscan_inliers, estimate_strike_point, skeleton
from thunder.recon.srp import reconstruct_srp
from thunder.recon.tdoa import reconstruct_plane_wave
from thunder.types import Atmosphere, FloatArray, Reconstruction, Recording


def reconstruct(
    recording: Recording,
    array_nominal: FloatArray,
    atmosphere_assumed: Atmosphere,
    config: ReconstructionConfig,
) -> Reconstruction:
    """Reconstruct the channel, then remove outliers, build its skeleton and estimate the strike point."""
    mics = np.asarray(array_nominal, dtype=float)
    if config.method == "A":
        raw = reconstruct_plane_wave(recording, mics, atmosphere_assumed, config)
    elif config.method == "B":
        raw = reconstruct_srp(recording, mics, atmosphere_assumed, config)
    elif config.method == "C":
        raw = reconstruct_multilateration(recording, mics, atmosphere_assumed, config)
    elif config.method == "D":
        raw = reconstruct_bayes(recording, mics, atmosphere_assumed, config)
    else:  # pragma: no cover - guarded by the config Literal
        raise ValueError(f"unknown method {config.method!r}")

    return postprocess(raw, config)


def reconstruct_storm(
    recordings: list[Recording],
    array_nominal: FloatArray,
    atmosphere_assumed: Atmosphere,
    config: ReconstructionConfig,
) -> list[Reconstruction]:
    """Method D on several recordings that share the atmosphere (storm self-calibration)."""
    if config.method != "D":
        raise ValueError("storm reconstruction is Method D")
    raws = reconstruct_bayes_storm(
        recordings, np.asarray(array_nominal, dtype=float), atmosphere_assumed, config
    )
    return [postprocess(r, config) for r in raws]


def postprocess(raw: Reconstruction, config: ReconstructionConfig) -> Reconstruction:
    """Remove outliers (DBSCAN), build the channel skeleton and estimate the strike point."""
    keep = dbscan_inliers(raw.points, config.dbscan_eps_m, config.dbscan_min_samples)
    pts = raw.points[keep]
    return dataclasses.replace(
        raw,
        points=pts,
        covariances=raw.covariances[keep],
        window_times=raw.window_times[keep],
        quality=raw.quality[keep],
        extra={
            **raw.extra,
            "gated_points": raw.points,
            "dbscan_inlier": keep,
            "skeleton_edges": skeleton(pts, config.skeleton_spur_m),
            "strike_point": estimate_strike_point(pts),
        },
    )

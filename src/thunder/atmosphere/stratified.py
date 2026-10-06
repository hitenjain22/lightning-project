"""Horizontally stratified, optionally windy atmosphere with traced rays (SPEC.md Phase 3)."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from thunder import constants as C
from thunder.atmosphere import raytrace as rt
from thunder.atmosphere.profiles import StratifiedProfiles
from thunder.config import AtmosphereConfig
from thunder.types import ArrivalPath, Atmosphere, FloatArray

AMPLITUDE_CAP = 10.0  # caustics: the ray-tube amplitude is capped at 10x the straight-line 1/R


class StratifiedAtmosphere(Atmosphere):
    is_uniform = False

    def __init__(self, cfg: AtmosphereConfig):
        if cfg.model != "stratified":
            raise ValueError("StratifiedAtmosphere needs model: stratified")
        self.cfg = cfg
        self.profiles = StratifiedProfiles(cfg)
        self.absorption = cfg.absorption
        self.ground_reflection = cfg.ground_reflection
        self.reflection_coefficient = cfg.reflection_coefficient
        kinks = [C.TROPOPAUSE_M] + ([cfg.inversion.base_m, cfg.inversion.top_m] if cfg.inversion else [])
        self.grid = rt.RayGrid.from_atmosphere(
            self, cfg.rays.max_height_m, cfg.rays.integration_step_m, tuple(kinks)
        )

    @classmethod
    def cached(cls, cfg: AtmosphereConfig) -> StratifiedAtmosphere:
        return _cached(json.dumps(cfg.model_dump(mode="json"), sort_keys=True))

    # profiles
    def temperature(self, z: FloatArray) -> FloatArray:
        return self.profiles.temperature(z)

    def wind(self, z: FloatArray) -> FloatArray:
        return self.profiles.wind(z)

    def relative_humidity(self, z: FloatArray) -> FloatArray:
        return self.profiles.relative_humidity(z)

    def pressure(self, z: FloatArray) -> FloatArray:
        return self.profiles.pressure(z)

    def sound_speed(self, z: FloatArray) -> FloatArray:
        return self.profiles.sound_speed(z)

    # propagation
    def _paths(
        self, sources: FloatArray, receiver: FloatArray, reflected: bool, guess: ArrivalPath | None = None
    ) -> ArrivalPath:
        src = np.atleast_2d(np.asarray(sources, dtype=float))
        rcv = np.asarray(receiver, dtype=float)
        initial = None
        if guess is not None:  # e.g. the same sources' paths to a nearby receiver (NaN = no guess)
            initial = np.where(guess.valid[:, None], guess.source_slowness[:, :2], np.nan)
        rays = rt.solve_eigenrays(
            self.grid,
            src,
            rcv,
            reflected,
            self.cfg.rays.tolerance_m,
            self.cfg.rays.max_iterations,
            self._fan(float(rcv[2]), reflected),
            initial,
        )
        n_r, slow_s, amp = rt.ray_geometry(self.grid, rays, src, rcv, reflected)
        image = src * np.array([1.0, 1.0, -1.0]) if reflected else src
        length = np.linalg.norm(rcv - image, axis=1)
        amp = np.where(
            rays.valid, np.minimum(np.nan_to_num(amp, nan=0.0, posinf=0.0), AMPLITUDE_CAP / length), 0.0
        )
        return ArrivalPath(
            travel_time=np.where(rays.valid, rays.T, np.nan),
            arrival_direction=n_r,
            amplitude_factor=amp,
            source_slowness=slow_s,
            path_length=length,
        )

    def _fan(self, z_receiver: float, reflected: bool) -> rt.Fan:
        """Windless initial-guess fan for a receiver height (cached per height, 1 cm resolution)."""
        key = (round(z_receiver, 2), reflected)
        fans = self.__dict__.setdefault("_fans", {})
        if key not in fans:
            fans[key] = rt.build_fan(self.grid, z_receiver, reflected)
        return fans[key]

    def propagate(
        self, sources: FloatArray, receiver: FloatArray, guess: ArrivalPath | None = None
    ) -> ArrivalPath:
        return self._paths(sources, receiver, reflected=False, guess=guess)

    def propagate_reflected(
        self, sources: FloatArray, receiver: FloatArray, guess: ArrivalPath | None = None
    ) -> ArrivalPath:
        return self._paths(sources, receiver, reflected=True, guess=guess)

    def locate(
        self, receiver: FloatArray, slowness_h: FloatArray, travel_time: FloatArray
    ) -> tuple[FloatArray, np.ndarray]:
        return rt.locate(self.grid, np.asarray(receiver, dtype=float), slowness_h, travel_time)


@lru_cache(maxsize=8)
def _cached(cfg_json: str) -> StratifiedAtmosphere:
    return StratifiedAtmosphere(AtmosphereConfig.model_validate_json(cfg_json))

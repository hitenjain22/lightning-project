"""Build Atmosphere objects from config. Phase 3 adds stratified and windy profiles here."""

from __future__ import annotations

from thunder.config import AtmosphereConfig
from thunder.constants import ZERO_CELSIUS_K, sound_speed_dry
from thunder.types import Atmosphere, UniformAtmosphere


def build_atmosphere(cfg: AtmosphereConfig) -> Atmosphere:
    if cfg.model == "uniform":
        t = ZERO_CELSIUS_K + cfg.temperature_c
        return UniformAtmosphere(sound_speed_dry(t), t, cfg.relative_humidity, cfg.pressure_pa)
    raise ValueError(f"unknown atmosphere model {cfg.model!r}")

"""Atmospheric profiles and the factory that builds an Atmosphere from config.

Temperature: T(z) = T0 - Gamma z (+ an optional inversion ramp of delta_k over [base, top]),
constant above the tropopause (11 km). Pressure: hydrostatic, ln p(z) = ln p0 -
(g / R_d) int_0^z dz' / T(z'), integrated on a fine grid (exact barometric formula for a pure
lapse rate, which the tests check). Humidity: constant relative humidity. Sound speed:
c = sqrt(gamma R_d T (1 + 0.51 q)) with q the specific humidity (sonic temperature, Kaimal &
Gaynor 1991; Magnus saturation pressure, Alduchov & Eskridge 1996). Wind: power law
u(z) = u_ref ((z + z0) / (z_ref + z0))^alpha blowing FROM a direction that may veer with height.
"""

from __future__ import annotations

import math

import numpy as np

from thunder import constants as C
from thunder.config import AtmosphereConfig
from thunder.types import Atmosphere, FloatArray, FloatLike, UniformAtmosphere

_PRESSURE_GRID_STEP_M = 1.0


def saturation_vapor_pressure(temperature_k: FloatLike) -> FloatArray:
    """Saturation vapor pressure over water (Pa), Magnus formula."""
    t = np.asarray(temperature_k, dtype=float) - C.ZERO_CELSIUS_K
    return C.MAGNUS_A_PA * np.exp(C.MAGNUS_B * t / (t + C.MAGNUS_C_DEGC))


def specific_humidity(
    temperature_k: FloatLike, relative_humidity: FloatLike, pressure_pa: FloatLike
) -> FloatArray:
    """Specific humidity q (kg/kg) from relative humidity (fraction)."""
    e = np.asarray(relative_humidity) * saturation_vapor_pressure(temperature_k)
    p = np.asarray(pressure_pa, dtype=float)
    return C.EPSILON_WATER * e / (p - (1.0 - C.EPSILON_WATER) * e)


def sound_speed_moist(
    temperature_k: FloatLike, relative_humidity: FloatLike, pressure_pa: FloatLike
) -> FloatArray:
    """Sound speed (m/s) in humid air via the sonic temperature T (1 + 0.51 q)."""
    q = specific_humidity(temperature_k, relative_humidity, pressure_pa)
    return C.SOUND_SPEED_COEFF * np.sqrt(np.asarray(temperature_k) * (1.0 + C.SONIC_HUMIDITY_COEFF * q))


class StratifiedProfiles:
    """Height profiles of temperature, pressure, humidity, sound speed and wind."""

    def __init__(self, cfg: AtmosphereConfig, z_top: float | None = None):
        self.cfg = cfg
        self.t0 = C.ZERO_CELSIUS_K + cfg.temperature_c
        self.gamma = cfg.lapse_rate_k_per_km / 1000.0
        self.rh = cfg.relative_humidity
        top = float(z_top if z_top is not None else max(cfg.rays.max_height_m, C.TROPOPAUSE_M) + 1000.0)
        self._z = np.arange(0.0, top + _PRESSURE_GRID_STEP_M, _PRESSURE_GRID_STEP_M)
        inv_t = 1.0 / self.temperature(self._z)
        integral = np.concatenate([[0.0], np.cumsum(0.5 * (inv_t[1:] + inv_t[:-1]) * np.diff(self._z))])
        self._ln_p = math.log(cfg.pressure_pa) - C.G_STANDARD / C.R_DRY_AIR * integral

    def temperature(self, z: FloatArray) -> FloatArray:
        zc = np.clip(np.asarray(z, dtype=float), 0.0, C.TROPOPAUSE_M)
        t = self.t0 - self.gamma * zc
        inv = self.cfg.inversion
        if inv is not None:
            t = t + inv.delta_k * np.clip(
                (np.asarray(z, dtype=float) - inv.base_m) / (inv.top_m - inv.base_m), 0, 1
            )
        return t

    def pressure(self, z: FloatArray) -> FloatArray:
        zq = np.asarray(z, dtype=float)
        ln_p = np.interp(np.clip(zq, 0.0, self._z[-1]), self._z, self._ln_p)
        above = zq > self._z[-1]  # isothermal extrapolation beyond the grid
        if np.any(above):
            t_top = float(self.temperature(self._z[-1]))
            ln_p = np.where(above, ln_p - C.G_STANDARD / (C.R_DRY_AIR * t_top) * (zq - self._z[-1]), ln_p)
        return np.exp(ln_p)

    def relative_humidity(self, z: FloatArray) -> FloatArray:
        return np.full_like(np.asarray(z, dtype=float), self.rh)

    def sound_speed(self, z: FloatArray) -> FloatArray:
        return sound_speed_moist(self.temperature(z), self.rh, self.pressure(z))

    def wind(self, z: FloatArray) -> FloatArray:
        """Wind vector (..., 3), m/s (the direction the air moves TO; zero vertical component)."""
        zq = np.asarray(z, dtype=float)
        w = self.cfg.wind
        out = np.zeros((*zq.shape, 3))
        if w is None or w.speed_mps == 0:
            return out
        zc = np.clip(zq, 0.0, None)
        speed = (
            w.speed_mps
            * ((zc + w.height_offset_m) / (w.reference_height_m + w.height_offset_m)) ** w.exponent
        )
        theta = np.radians(w.direction_from_deg + w.shear_deg_per_km * (zc - w.reference_height_m) / 1000.0)
        out[..., 0] = -speed * np.sin(theta)
        out[..., 1] = -speed * np.cos(theta)
        return out


def build_atmosphere(cfg: AtmosphereConfig) -> Atmosphere:
    """Atmosphere object for a config (stratified atmospheres are cached by config)."""
    if cfg.model == "uniform":
        t = C.ZERO_CELSIUS_K + cfg.temperature_c
        c = float(sound_speed_moist(t, cfg.relative_humidity, cfg.pressure_pa))
        return UniformAtmosphere(
            c,
            t,
            cfg.relative_humidity,
            cfg.pressure_pa,
            cfg.absorption,
            cfg.ground_reflection,
            cfg.reflection_coefficient,
        )
    from thunder.atmosphere.stratified import StratifiedAtmosphere

    return StratifiedAtmosphere.cached(cfg)

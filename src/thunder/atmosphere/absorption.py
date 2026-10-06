"""Pure-tone atmospheric absorption coefficient, ISO 9613-1:1993 (equations 3-5 and B.1).

alpha(f) [dB/m] = 8.686 f^2 { 1.84e-11 (p_a/p_r)^-1 (T/T0)^1/2
                   + (T/T0)^-5/2 [ 0.01275 exp(-2239.1/T) / (f_rO + f^2/f_rO)
                                 + 0.1068  exp(-3352.0/T) / (f_rN + f^2/f_rN) ] }
f_rO = (p_a/p_r) (24 + 4.04e4 h (0.02 + h) / (0.391 + h))
f_rN = (p_a/p_r) (T/T0)^-1/2 (9 + 280 h exp{-4.170 [(T/T0)^-1/3 - 1]})
h    = molar concentration of water vapor in percent
     = RH(%) (p_sat/p_r) / (p_a/p_r),  p_sat/p_r = 10^C,  C = -6.8346 (T01/T)^1.261 + 4.6151
"""

from __future__ import annotations

import numpy as np

from thunder.constants import ISO_PR_PA, ISO_T0_K, ISO_T01_K

# Local alias (this module must not import thunder.types, which imports it lazily).
FloatArray = float | np.ndarray


def water_vapor_molar_percent(
    temperature_k: FloatArray, relative_humidity: FloatArray, pressure_pa: FloatArray
) -> FloatArray:
    """Molar concentration of water vapor h (%), ISO 9613-1 Annex B."""
    t = np.asarray(temperature_k, dtype=float)
    c = -6.8346 * (ISO_T01_K / t) ** 1.261 + 4.6151
    return 100.0 * np.asarray(relative_humidity) * 10.0**c / (np.asarray(pressure_pa) / ISO_PR_PA)


def absorption_db_per_m(
    f: FloatArray, temperature_k: FloatArray, relative_humidity: FloatArray, pressure_pa: FloatArray
) -> FloatArray:
    """ISO 9613-1 absorption (dB/m). Inputs broadcast; relative_humidity is a fraction (0-1)."""
    f = np.asarray(f, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    pa = np.asarray(pressure_pa, dtype=float) / ISO_PR_PA
    h = water_vapor_molar_percent(t, relative_humidity, pressure_pa)
    tr = t / ISO_T0_K
    fro = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = pa * tr**-0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1.0)))
    return (
        8.686
        * f**2
        * (
            1.84e-11 / pa * tr**0.5
            + tr**-2.5
            * (
                0.01275 * np.exp(-2239.1 / t) / (fro + f**2 / fro)
                + 0.1068 * np.exp(-3352.0 / t) / (frn + f**2 / frn)
            )
        )
    )


# Frequencies (Hz) on which the height profile of alpha is tabulated; other frequencies are
# interpolated in log-log space (alpha is smooth there, ~f^2 at high frequency).
_TABLE_FREQS = np.logspace(0.0, np.log10(64000.0), 96)
_Z_STEP_M = 10.0


def _cumulative_alpha(atmosphere, z_top: float) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative integral over height of alpha(f, z): (z grid, C[f, z]) in dB, cached per atmosphere."""
    cache = atmosphere.__dict__.setdefault("_absorption_cache", {})
    key = float(np.ceil(z_top / 1000.0) * 1000.0)  # reuse one table for all heights below key
    if key not in cache:
        z = np.arange(0.0, key + _Z_STEP_M, _Z_STEP_M)
        alpha = np.asarray(
            absorption_db_per_m(
                _TABLE_FREQS[:, None],
                atmosphere.temperature(z),
                atmosphere.relative_humidity(z),
                atmosphere.pressure(z),
            )
        )
        c = np.concatenate(
            [
                np.zeros((len(_TABLE_FREQS), 1)),
                np.cumsum(0.5 * (alpha[:, 1:] + alpha[:, :-1]) * np.diff(z), axis=1),
            ],
            axis=1,
        )
        cache[key] = (z, c, alpha)
    return cache[key][0], cache[key][1]


def path_absorption_db(atmosphere, freqs, z_source, z_receiver: float, path_length, reflected: bool):
    """Absorption (dB), shape (K, F): path length times alpha averaged over the heights spanned.

    Direct path: heights between source and receiver. Ground-reflected path: from the source
    down to the ground and back up to the receiver.
    """
    zs = np.clip(np.asarray(z_source, dtype=float), 0.0, None)
    zr = max(float(z_receiver), 0.0)
    z_grid, cum = _cumulative_alpha(atmosphere, float(max(zs.max(initial=0.0), zr)) + 1.0)

    def integral(zq):  # (F, K) integral of alpha from 0 to zq (linear interpolation in height)
        zq = np.clip(np.asarray(zq, dtype=float), z_grid[0], z_grid[-1])
        i = np.clip(np.searchsorted(z_grid, zq) - 1, 0, len(z_grid) - 2)
        w = (zq - z_grid[i]) / (z_grid[i + 1] - z_grid[i])
        return cum[:, i] * (1 - w) + cum[:, i + 1] * w

    if reflected:
        span = zs + zr
        total = integral(zs) + integral(np.full_like(zs, zr))
    else:
        span = np.abs(zs - zr)
        total = np.abs(integral(zs) - integral(np.full_like(zs, zr)))
    alpha_point = absorption_db_per_m(
        _TABLE_FREQS[:, None],
        atmosphere.temperature(zs),
        atmosphere.relative_humidity(zs),
        atmosphere.pressure(zs),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_alpha = np.where(span > 1.0, total / np.maximum(span, 1e-300), alpha_point)  # (F0, K)
    # Log-log interpolation to the requested frequencies (same weights for every path).
    lt = np.log(_TABLE_FREQS)
    lf = np.log(np.clip(np.asarray(freqs, dtype=float), _TABLE_FREQS[0], _TABLE_FREQS[-1]))
    i = np.clip(np.searchsorted(lt, lf) - 1, 0, len(lt) - 2)
    w = (lf - lt[i]) / (lt[i + 1] - lt[i])
    la = np.log(np.maximum(mean_alpha, 1e-300))  # (F0, K)
    out = la[i].T * (1 - w) + la[i + 1].T * w  # (K, F)
    return np.exp(out) * np.asarray(path_length, dtype=float)[:, None]

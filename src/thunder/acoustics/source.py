"""Phase 2 source model: the N-wave each piece of channel radiates, and its calibration.

Pulse shape. Every channel element radiates a unit-peak N-wave
    n(t) = 1 - 2 t / T   for 0 <= t < T  (compression first), 0 otherwise.
Its spectrum is |N(w)| = T |j1(w T / 2)|, with j1 the spherical Bessel function of order 1,
so the spectral peak sits at w T / 2 = x*, the maximum of j1 (x* ~ 2.0816).

Duration from Few's model. The relaxation radius R0 = sqrt(E_l / (pi p0)) and the dominant
frequency f_peak = 0.63 c0 / R0 (both # VERIFY in constants.py). We choose T so the N-wave's
spectral peak equals f_peak:  T = x* / (pi f_peak).  Pressure and sound speed are taken at
the source height, so pulses from high in the channel are longer (lower ambient pressure).

Amplitude from energy conservation. A line source with pressure density q (Pa) radiates
    p(t) = integral q / r * n(t - r / c) ds.
For a long straight line the far field is a cylindrical wave; using the line-source Green's
function (-i pi H0^(2)(k rho)) and Parseval, the acoustic energy radiated per unit length is
    E_ac = (4 pi q^2 / rho0) * integral_0^inf |N(w)|^2 / w dw = pi q^2 T^2 / rho0,
because integral_0^inf j1(x)^2 / x dx = 1/4.  Setting E_ac = eta * E_l gives
    q = sqrt(eta * E_l * rho0 / pi) / T.
The test suite checks both the 1/4 integral and the resulting energy flux numerically.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from scipy import optimize

from thunder.constants import FEW_FPEAK_COEFF
from thunder.types import FloatArray


def spherical_j1(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=float)
    return (np.sin(x) - x * np.cos(x)) / x**2


@cache
def nwave_peak_x() -> float:
    """Location x* of the maximum of j1 (the N-wave spectral peak is at w T / 2 = x*)."""
    res = optimize.minimize_scalar(lambda x: -spherical_j1(x), bounds=(1.0, 3.0), method="bounded",
                                   options={"xatol": 1e-12})
    return float(res.x)


def relaxation_radius(energy_per_length: FloatArray, pressure: FloatArray) -> FloatArray:
    """Few's relaxation radius R0 = sqrt(E_l / (pi p0)), m."""
    return np.sqrt(np.asarray(energy_per_length) / (math.pi * np.asarray(pressure)))


def peak_frequency(
    energy_per_length: FloatArray, pressure: FloatArray, sound_speed: FloatArray
) -> FloatArray:
    """Few's dominant frequency f_peak = 0.63 c0 / R0, Hz."""
    return FEW_FPEAK_COEFF * np.asarray(sound_speed) / relaxation_radius(energy_per_length, pressure)


def pulse_duration(
    energy_per_length: FloatArray, pressure: FloatArray, sound_speed: FloatArray
) -> FloatArray:
    """N-wave duration T (s) whose spectral peak equals Few's f_peak."""
    return nwave_peak_x() / (math.pi * peak_frequency(energy_per_length, pressure, sound_speed))


def line_amplitude(
    energy_per_length: FloatArray, density: FloatArray, duration: FloatArray, efficiency: float
) -> FloatArray:
    """Pressure density q (Pa) so a long straight line radiates efficiency * E_l per unit length."""
    return np.sqrt(efficiency * np.asarray(energy_per_length) * np.asarray(density) / math.pi) / np.asarray(
        duration
    )


def nwave_kernel(duration: float, dt: float) -> FloatArray:
    """Cell-averaged unit-peak N-wave on a grid of spacing dt, starting at t = 0.

    kernel[k] is the mean of n(t) over [(k - 1/2) dt, (k + 1/2) dt], which keeps the pulse's
    timing unbiased and its integral exactly zero (no DC) despite the jumps at 0 and T.
    """
    T = float(duration)

    def antiderivative(t: np.ndarray) -> np.ndarray:
        tc = np.clip(t, 0.0, T)
        return tc - tc**2 / T  # zero at both t = 0 and t = T

    k = np.arange(int(math.ceil(T / dt)) + 2)
    return (antiderivative((k + 0.5) * dt) - antiderivative((k - 0.5) * dt)) / dt

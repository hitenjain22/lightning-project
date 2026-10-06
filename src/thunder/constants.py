"""Physical constants. Every value names its source; uncertain values are marked # VERIFY.

All quantities are SI. Add new constants here, never inline in other modules.
"""

import math

# --- Exact / defined values -------------------------------------------------

# Molar gas constant, J/(mol K). Exact since the 2019 SI redefinition (CODATA 2018).
R_UNIVERSAL = 8.314462618

# 0 degC in kelvin. Exact by definition of the Celsius scale.
ZERO_CELSIUS_K = 273.15

# Standard sea-level pressure, Pa. Defined value (ISO 2533 / U.S. Standard Atmosphere 1976).
P_STANDARD = 101_325.0

# --- Air properties ---------------------------------------------------------

# Mean molar mass of dry air, kg/mol. U.S. Standard Atmosphere 1976 (M0 = 28.9644 kg/kmol).
M_DRY_AIR = 0.0289644

# Specific gas constant of dry air, J/(kg K). Derived: R_UNIVERSAL / M_DRY_AIR.
R_DRY_AIR = R_UNIVERSAL / M_DRY_AIR

# Ratio of specific heats for dry air (ideal diatomic gas), dimensionless.
# Standard textbook value, e.g. Pierce, "Acoustics", ch. 1.
GAMMA_AIR = 1.4

# Coefficient in c = SOUND_SPEED_COEFF * sqrt(T), m/(s sqrt(K)), for dry air.
# Derived: sqrt(GAMMA_AIR * R_DRY_AIR), about 20.05. SPEC.md quotes 20.05.
SOUND_SPEED_COEFF = math.sqrt(GAMMA_AIR * R_DRY_AIR)

# Standard tropospheric temperature lapse rate, K/m (6.5 K/km).
# ISO 2533 / U.S. Standard Atmosphere 1976, layer 0-11 km.
LAPSE_RATE_STANDARD = 0.0065

# --- Lightning / thunder (literature values, unverified) ---------------------

# Mean absolute direction change between successive channel segments of tens of meters, rad.
# Hill (1968), J. Geophys. Res., as recalled in SPEC.md (~16 deg).  # VERIFY
HILL_MEAN_TURN_ANGLE = math.radians(16.0)

# Few's dominant-frequency coefficient: f_peak ~= FEW_FPEAK_COEFF * c0 / R0,
# with relaxation radius R0 = sqrt(E_l / (pi * p0)).
# Few (1969), J. Geophys. Res., as recalled in SPEC.md.  # VERIFY
FEW_FPEAK_COEFF = 0.63


def sound_speed_dry(temperature_k: float) -> float:
    """Adiabatic sound speed in dry air (m/s) at temperature T (K): c = sqrt(gamma R T)."""
    return SOUND_SPEED_COEFF * math.sqrt(temperature_k)

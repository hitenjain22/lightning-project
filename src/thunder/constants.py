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

# --- Lightning channel statistics (Phase 1 defaults, all unverified) ----------

# Typical negative cloud-to-ground flash start height range, m. SPEC.md ("roughly 4-7 km").  # VERIFY
CG_START_HEIGHT_RANGE = (4000.0, 7000.0)

# Horizontal offset between where the channel leaves the cloud and its ground strike point, m.
# No source identified; placeholder so channels lean as in photographs instead of being
# plumb vertical. Up to ~0.3 of the start height.  # VERIFY
CG_START_OFFSET_RANGE = (0.0, 1500.0)

# Return-stroke energy per unit channel length, J/m. Literature values recalled as roughly
# 1e5-1e6 J/m (Few 1969; Rakov & Uman 2003). 1e6 is chosen because Few's model then gives
# f_peak ~ 120 Hz (R0 ~ 1.8 m), inside the observed thunder band; 1e5 would give ~390 Hz.  # VERIFY
ENERGY_PER_LENGTH_MAIN = 1.0e6

# Energy per unit length of a branch relative to its parent. No source identified;
# placeholder reflecting that branches carry much less current than the main channel.  # VERIFY
BRANCH_ENERGY_FRACTION = 0.2

# Number of return strokes per multi-stroke flash. SPEC.md ("2-4 strokes").  # VERIFY
STROKES_PER_FLASH_RANGE = (2, 4)

# Interstroke interval range, s. SPEC.md ("tens of ms"); Rakov & Uman (2003) recalled as a
# geometric mean near 60 ms.  # VERIFY
INTERSTROKE_INTERVAL_RANGE = (0.020, 0.100)

# In-cloud horizontal channel length range, m. SPEC.md ("kilometers"); placeholder range.  # VERIFY
INCLOUD_LENGTH_RANGE = (2000.0, 8000.0)

# In-cloud channel altitude range, m. SPEC.md ("5-7 km altitude").  # VERIFY
INCLOUD_ALTITUDE_RANGE = (5000.0, 7000.0)

# Branch departure angle from the parent channel, rad. No source identified; placeholder.  # VERIFY
BRANCH_ANGLE_RANGE = (math.radians(20.0), math.radians(60.0))

# Median branch length, m, and log-normal shape (sigma of ln L). No source identified;
# placeholder giving branches of a few hundred meters.  # VERIFY
BRANCH_LENGTH_MEDIAN = 300.0
# Log-normal sigma for branch length (dimensionless). Placeholder.  # VERIFY
BRANCH_LENGTH_SIGMA = 0.75

# Branch-initiation probability per main-channel step of ~10 m. Placeholder chosen to give
# a handful of visible branches per flash, as in photographs.  # VERIFY
BRANCH_PROBABILITY_PER_STEP = 0.01


def sound_speed_dry(temperature_k: float) -> float:
    """Adiabatic sound speed in dry air (m/s) at temperature T (K): c = sqrt(gamma R T)."""
    return SOUND_SPEED_COEFF * math.sqrt(temperature_k)

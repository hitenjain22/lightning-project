import math
import re
from pathlib import Path

from thunder import constants as C


def test_sound_speed_at_20c_is_about_343():
    c = C.sound_speed_dry(C.ZERO_CELSIUS_K + 20.0)
    assert math.isclose(c, 343.2, abs_tol=0.3)


def test_sound_speed_coeff_matches_spec_value():
    assert math.isclose(C.SOUND_SPEED_COEFF, 20.05, abs_tol=0.01)


def test_every_constant_has_a_source_comment():
    """Each UPPER_CASE assignment must be preceded by a comment line."""
    lines = Path(C.__file__).read_text().splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^[A-Z][A-Z0-9_]* = ", line):
            assert i > 0 and lines[i - 1].lstrip().startswith("#"), f"no source comment: {line}"

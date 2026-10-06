"""Pydantic models for YAML run configs.

Each phase adds its own section when it is built. Unknown keys are rejected so typos fail loudly.
Angles are written in degrees in YAML (fields ending in `_deg`) and exposed in radians via
properties; everything else is SI.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thunder import constants as C


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _deg(r: tuple[float, float]) -> tuple[float, float]:
    return (math.degrees(r[0]), math.degrees(r[1]))


ChannelPreset = Literal["straight", "tortuous", "branched", "with_incloud", "multi_stroke"]

# Each preset is a set of field defaults; explicit config fields override them.
CHANNEL_PRESETS: dict[str, dict[str, Any]] = {
    "straight": {
        "turn_mean_deg": 1.0,
        "bias_strength": 8.0,
        "branch_probability": 0.0,
        "start_offset_m": (0.0, 0.0),
    },
    "tortuous": {"branch_probability": 0.0},
    "branched": {},
    "with_incloud": {"incloud": True, "start_height_m": C.INCLOUD_ALTITUDE_RANGE},
    "multi_stroke": {"n_strokes": C.STROKES_PER_FLASH_RANGE},
}


class ChannelConfig(StrictModel):
    """Phase 1 lightning channel generator. Ranges are (low, high), sampled uniformly."""

    preset: ChannelPreset = "branched"

    # Geometry of the main channel
    start_height_m: tuple[float, float] = C.CG_START_HEIGHT_RANGE
    strike_distance_m: tuple[float, float] = (1000.0, 8000.0)  # from the array centroid
    strike_azimuth_deg: tuple[float, float] = (0.0, 360.0)  # clockwise from north
    # Horizontal distance of the start point from the strike point (random direction).
    start_offset_m: tuple[float, float] = C.CG_START_OFFSET_RANGE
    segment_length_m: float = Field(default=10.0, gt=0)

    # Tortuosity: turn angle between successive segments
    turn_distribution: Literal["halfnormal", "exponential"] = "halfnormal"
    turn_mean_deg: float = Field(default=math.degrees(C.HILL_MEAN_TURN_ANGLE), gt=0, lt=90)
    # von Mises concentration of the turn *direction* toward the goal (tuning parameter, not physics).
    bias_strength: float = Field(default=2.0, ge=0)

    # Branching
    branch_probability: float = Field(default=C.BRANCH_PROBABILITY_PER_STEP, ge=0, le=1)
    branch_angle_deg: tuple[float, float] = _deg(C.BRANCH_ANGLE_RANGE)
    branch_length_median_m: float = Field(default=C.BRANCH_LENGTH_MEDIAN, gt=0)
    branch_length_sigma: float = Field(default=C.BRANCH_LENGTH_SIGMA, ge=0)
    branch_length_decay: float = Field(default=0.5, gt=0, le=1)  # median multiplier per depth
    branch_bias_strength: float = Field(default=2.0, ge=0)  # pull toward the departure direction
    branch_max_depth: int = Field(default=2, ge=0)

    # Energy
    energy_per_length_main: float = Field(default=C.ENERGY_PER_LENGTH_MAIN, gt=0)
    branch_energy_fraction: float = Field(default=C.BRANCH_ENERGY_FRACTION, gt=0, le=1)

    # In-cloud horizontal section (feeds the top of the main channel)
    incloud: bool = False
    incloud_length_m: tuple[float, float] = C.INCLOUD_LENGTH_RANGE

    # Strokes
    n_strokes: tuple[int, int] = (1, 1)
    stroke_interval_s: tuple[float, float] = C.INTERSTROKE_INTERVAL_RANGE

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if isinstance(data, dict):
            defaults = CHANNEL_PRESETS.get(data.get("preset", "branched"), {})
            return {**defaults, **data}
        return data

    @field_validator(
        "start_height_m",
        "strike_distance_m",
        "strike_azimuth_deg",
        "start_offset_m",
        "branch_angle_deg",
        "incloud_length_m",
        "n_strokes",
        "stroke_interval_s",
    )
    @classmethod
    def _ordered_nonnegative(cls, v: tuple) -> tuple:
        if v[0] > v[1] or v[0] < 0:
            raise ValueError(f"range must satisfy 0 <= low <= high, got {v}")
        return v

    @property
    def turn_mean(self) -> float:
        return math.radians(self.turn_mean_deg)

    @property
    def strike_azimuth(self) -> tuple[float, float]:
        return (math.radians(self.strike_azimuth_deg[0]), math.radians(self.strike_azimuth_deg[1]))

    @property
    def branch_angle(self) -> tuple[float, float]:
        return (math.radians(self.branch_angle_deg[0]), math.radians(self.branch_angle_deg[1]))


class RunConfig(StrictModel):
    """Top-level run configuration."""

    experiment: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    seed: int = Field(ge=0)
    results_root: Path = Path("results")
    sample_rate_hz: float = Field(default=8000.0, gt=0)
    oversample: int = Field(default=8, ge=1)
    description: str = ""
    channel: ChannelConfig | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """Short, stable hash of the resolved config (key order independent)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path, overrides: dict | None = None) -> RunConfig:
    """Load a YAML config, optionally applying top-level overrides."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update(overrides)
    return RunConfig.model_validate(data)

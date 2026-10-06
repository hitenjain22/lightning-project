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


class AtmosphereConfig(StrictModel):
    """Atmosphere used for synthesis. Phase 2: uniform still air. Phase 3 adds profiles and wind."""

    model: Literal["uniform"] = "uniform"
    temperature_c: float = 25.0  # SPEC.md Phase 3 default surface temperature
    relative_humidity: float = Field(default=0.5, ge=0, le=1)
    pressure_pa: float = Field(default=C.P_STANDARD, gt=0)


ArrayLayout = Literal[
    "triangle", "square", "square_center", "circle", "l_shape", "cross",
    "mast", "random_disk", "distributed", "free_form",
]


class ArrayConfig(StrictModel):
    """Nominal microphone layout, ENU meters, horizontal centroid at the origin.

    `aperture_m` is always the largest horizontal distance between any two mics (for
    `distributed`, the aperture of each sub-array). Giving `positions_m` selects `free_form`.
    """

    layout: ArrayLayout = "square_center"
    aperture_m: float = Field(default=50.0, gt=0)
    n_mics: int | None = Field(default=None, ge=3)  # circle, l_shape, cross, random_disk
    mic_height_m: float = Field(default=1.5, ge=0)  # SPEC.md Phase 3 default
    mast_height_m: float = Field(default=10.0, gt=0)  # height of the raised mic (layout "mast")
    subarray_layout: Literal["triangle", "square", "square_center"] = "triangle"
    n_subarrays: int = Field(default=3, ge=2, le=3)
    subarray_separation_m: float = Field(default=500.0, gt=0)
    positions_m: list[tuple[float, float, float]] | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _positions_imply_free_form(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("positions_m") is not None and "layout" not in data:
            return {**data, "layout": "free_form"}
        return data

    @model_validator(mode="after")
    def _check_layout(self) -> ArrayConfig:
        if self.layout == "free_form" and self.positions_m is None:
            raise ValueError("layout free_form needs positions_m")
        if self.layout != "free_form" and self.positions_m is not None:
            raise ValueError("positions_m is only used with layout free_form")
        return self


# --- Phase 4 sensor presets. Hardware numbers are typical-datasheet assumptions (# VERIFY),
# --- recorded in docs/assumptions.md.

MIC_PRESETS: dict[str, dict[str, Any]] = {
    "ideal": {},
    # Measurement / infrasound-capable condenser mic with 24-bit recorder.
    "measurement": {
        "highpass_hz": 2.0, "lowpass_hz": 2000.0, "gain_tolerance_db": 0.2, "corner_tolerance": 0.02,
        "self_noise_db_spl": 20.0, "clip_db_spl": 140.0, "adc_bits": 24,
    },
    # Ordinary audio condenser/dynamic mic with 24-bit recorder.
    "audio": {
        "highpass_hz": 20.0, "lowpass_hz": 20000.0, "gain_tolerance_db": 1.0, "corner_tolerance": 0.10,
        "self_noise_db_spl": 15.0, "clip_db_spl": 130.0, "adc_bits": 24,
    },
    # Smartphone MEMS mic: ~100 Hz high-pass in the audio path, ~120 dB SPL overload, 16-bit.
    "phone": {
        "highpass_hz": 100.0, "lowpass_hz": None, "gain_tolerance_db": 2.0, "corner_tolerance": 0.20,
        "self_noise_db_spl": 32.0, "clip_db_spl": 120.0, "adc_bits": 16,
    },
}


class MicConfig(StrictModel):
    """Microphone + recorder chain. Filters are causal Butterworth (analog-like phase)."""

    preset: Literal["ideal", "measurement", "audio", "phone"] = "ideal"
    highpass_hz: float | None = Field(default=None, gt=0)
    lowpass_hz: float | None = Field(default=None, gt=0)  # ignored if >= 0.45 * sample rate
    filter_order: int = Field(default=2, ge=1, le=8)  # per band edge
    gain_tolerance_db: float = Field(default=0.0, ge=0)  # per-mic sensitivity error std
    corner_tolerance: float = Field(default=0.0, ge=0, lt=0.5)  # per-mic relative corner error std
    self_noise_db_spl: float | None = None  # datasheet level over 20 Hz-20 kHz (white)
    clip_db_spl: float | None = None  # full scale of the recorder
    adc_bits: int | None = Field(default=None, ge=4, le=32)

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {**MIC_PRESETS.get(data.get("preset", "ideal"), {}), **data}
        return data

    @model_validator(mode="after")
    def _adc_needs_full_scale(self) -> MicConfig:
        if self.adc_bits is not None and self.clip_db_spl is None:
            raise ValueError("adc_bits needs clip_db_spl (the ADC full scale)")
        return self


TIMING_PRESETS: dict[str, dict[str, Any]] = {
    "perfect": {},
    # One multichannel interface: one clock for all channels, crystal error ~10 ppm.
    "shared_interface": {"common_clock": True, "drift_std_ppm": 10.0},
    # GPS-disciplined recorders: ~1 us residual offset, no drift.
    "gps_synced": {"offset_std_s": 1e-6},
    # Separate recorders aligned by a hand clap: ms-level offsets, independent crystals.
    "hand_synced": {"offset_std_s": 3e-3, "drift_std_ppm": 20.0},
}


class TimingConfig(StrictModel):
    """Per-mic recorder clock errors (hidden): tau = (1 + drift) * t + offset, plus sample jitter."""

    preset: Literal["perfect", "shared_interface", "gps_synced", "hand_synced"] = "perfect"
    offset_std_s: float = Field(default=0.0, ge=0)
    drift_std_ppm: float = Field(default=0.0, ge=0)
    jitter_std_s: float = Field(default=0.0, ge=0)  # per-sample aperture jitter
    common_clock: bool = False  # True: all mics share one offset and drift realization

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {**TIMING_PRESETS.get(data.get("preset", "perfect"), {}), **data}
        return data


POSITION_PRESETS: dict[str, dict[str, Any]] = {
    "exact": {},
    "surveyed": {"horizontal_std_m": 0.01, "vertical_std_m": 0.02},  # RTK GNSS / total station
    "tape": {"horizontal_std_m": 0.10, "vertical_std_m": 0.05},  # tape measure and compass
    "consumer_gps": {"horizontal_std_m": 3.0, "vertical_std_m": 5.0},  # phone GPS
}


class PositionConfig(StrictModel):
    """Gaussian error of true mic positions about the nominal ones (per axis)."""

    preset: Literal["exact", "surveyed", "tape", "consumer_gps"] = "exact"
    horizontal_std_m: float = Field(default=0.0, ge=0)
    vertical_std_m: float = Field(default=0.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {**POSITION_PRESETS.get(data.get("preset", "exact"), {}), **data}
        return data


FLASH_TIME_PRESETS: dict[str, dict[str, Any]] = {
    "exact": {},
    "photodiode": {"distribution": "normal", "scale_s": 10e-6},
    "lightning_network": {"distribution": "normal", "scale_s": 1e-3},
    # Flash time reported as the center of the first 30 fps frame showing it.
    "video_30fps": {"distribution": "uniform", "scale_s": 1.0 / 60.0},
}


class FlashTimeConfig(StrictModel):
    """Error of the reported flash time t0. normal: std = scale; uniform: +-scale."""

    preset: Literal["exact", "photodiode", "lightning_network", "video_30fps"] = "exact"
    distribution: Literal["normal", "uniform"] = "normal"
    scale_s: float = Field(default=0.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {**FLASH_TIME_PRESETS.get(data.get("preset", "exact"), {}), **data}
        return data


class NoiseConfig(StrictModel):
    """Acoustic noise at the mics (added before the mic response)."""

    background_snr_db: float | None = None  # None: no background noise
    background_color: Literal["white", "pink", "brown"] = "pink"
    background_field: Literal["diffuse", "incoherent"] = "diffuse"
    snr_band_hz: tuple[float, float] = (10.0, 300.0)  # SPEC.md default analysis band
    wind_speed_mps: float = Field(default=0.0, ge=0)  # at mic height; 0 = no wind noise
    rain_db_spl: float | None = None  # broadband rain-impact noise level; None = no rain

    @field_validator("snr_band_hz")
    @classmethod
    def _band(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not 0 < v[0] < v[1]:
            raise ValueError(f"snr_band_hz must satisfy 0 < low < high, got {v}")
        return v


class SensorsConfig(StrictModel):
    """Phase 4: everything that makes a real recording differ from the clean synthetic one."""

    mic: MicConfig = Field(default_factory=MicConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    flash_time: FlashTimeConfig = Field(default_factory=FlashTimeConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


class SynthesisConfig(StrictModel):
    """Phase 2 forward model."""

    emitter_spacing_m: float = Field(default=0.5, gt=0)  # finer than reconstruction ever assumes
    # Sub-segment tortuosity (Brownian bridge inside each segment); null disables it.
    micro_turn_mean_deg: float | None = Field(default=math.degrees(C.HILL_MEAN_TURN_ANGLE), ge=0, lt=90)
    micro_scale_m: float = Field(default=C.MICRO_TORTUOSITY_SCALE, gt=0)
    acoustic_efficiency: float = Field(default=C.ACOUSTIC_EFFICIENCY, gt=0, le=1)
    write_wav: bool = True

    @property
    def micro_turn_mean(self) -> float | None:
        return None if self.micro_turn_mean_deg is None else math.radians(self.micro_turn_mean_deg)


class RunConfig(StrictModel):
    """Top-level run configuration."""

    experiment: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    seed: int = Field(ge=0)
    results_root: Path = Path("results")
    sample_rate_hz: float = Field(default=8000.0, gt=0)
    oversample: int = Field(default=8, ge=1)
    description: str = ""
    channel: ChannelConfig | None = None
    atmosphere: AtmosphereConfig = Field(default_factory=AtmosphereConfig)
    array: ArrayConfig = Field(default_factory=ArrayConfig)
    synthesis: SynthesisConfig | None = None
    sensors: SensorsConfig | None = None

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

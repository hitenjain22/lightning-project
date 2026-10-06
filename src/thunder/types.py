"""Core data contracts shared by every phase.

These are plain containers with shape validation. Physics lives in the phase modules.
Shapes: N nodes, S segments, M mics, K reconstructed points.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from thunder.constants import R_DRY_AIR

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _check_shape(name: str, arr: np.ndarray, shape: tuple[int | None, ...]) -> None:
    """Raise ValueError unless arr matches shape (None matches any length)."""
    ok = arr.ndim == len(shape) and all(
        want is None or got == want for got, want in zip(arr.shape, shape, strict=True)
    )
    if not ok:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")


@dataclass(frozen=True)
class Channel:
    """A lightning channel as a tree of straight segments in ENU coordinates (m)."""

    nodes: FloatArray  # (N, 3) node positions
    segments: IntArray  # (S, 2) node index pairs (parent, child)
    energy_per_length: FloatArray  # (S,) J/m
    branch_id: IntArray  # (S,) 0 = main channel
    is_main: BoolArray  # (S,) True for main (ground-reaching) channel segments
    is_incloud: BoolArray  # (S,) True for the near-horizontal in-cloud section
    stroke_times: FloatArray = field(default_factory=lambda: np.zeros(1))  # (strokes,) s
    metadata: dict[str, Any] = field(default_factory=dict)  # seed, generator params

    def __post_init__(self) -> None:
        _check_shape("nodes", self.nodes, (None, 3))
        _check_shape("segments", self.segments, (None, 2))
        s = self.segments.shape[0]
        for name in ("energy_per_length", "branch_id", "is_main", "is_incloud"):
            _check_shape(name, getattr(self, name), (s,))
        _check_shape("stroke_times", self.stroke_times, (None,))
        if s and (np.min(self.segments) < 0 or np.max(self.segments) >= self.nodes.shape[0]):
            raise ValueError("segments reference nodes out of range")

    @property
    def n_segments(self) -> int:
        return int(self.segments.shape[0])

    @property
    def refires(self) -> BoolArray:
        """Segments that fire on every stroke (main + in-cloud); branches fire on the first only."""
        return self.is_main | self.is_incloud

    def segment_endpoints(self) -> tuple[FloatArray, FloatArray]:
        """Start and end positions of every segment, each (S, 3)."""
        return self.nodes[self.segments[:, 0]], self.nodes[self.segments[:, 1]]


@dataclass(frozen=True)
class ArrivalPath:
    """Result of propagating from a source to a receiver."""

    travel_time: FloatArray  # (...,) s
    arrival_direction: FloatArray  # (..., 3) unit vector of propagation at the receiver
    amplitude_factor: FloatArray  # (...,) geometric spreading factor (1/r in uniform air)


class Atmosphere(abc.ABC):
    """Horizontally stratified atmosphere. Profiles are functions of height z (m)."""

    @abc.abstractmethod
    def temperature(self, z: FloatArray) -> FloatArray:
        """Temperature (K) at height z."""

    @abc.abstractmethod
    def wind(self, z: FloatArray) -> FloatArray:
        """Wind vector (..., 3) in m/s at height z."""

    @abc.abstractmethod
    def relative_humidity(self, z: FloatArray) -> FloatArray:
        """Relative humidity (0-1) at height z."""

    @abc.abstractmethod
    def pressure(self, z: FloatArray) -> FloatArray:
        """Ambient pressure (Pa) at height z."""

    @abc.abstractmethod
    def sound_speed(self, z: FloatArray) -> FloatArray:
        """Sound speed (m/s) at height z, excluding wind."""

    def density(self, z: FloatArray) -> FloatArray:
        """Air density (kg/m^3) at height z from the ideal gas law (dry air)."""
        return self.pressure(z) / (R_DRY_AIR * self.temperature(z))

    @abc.abstractmethod
    def propagate(self, sources: FloatArray, receivers: FloatArray) -> ArrivalPath:
        """Travel time, arrival direction and spreading between broadcastable point sets (..., 3)."""


class UniformAtmosphere(Atmosphere):
    """Still air with constant sound speed, temperature and pressure; straight-line propagation."""

    def __init__(
        self,
        sound_speed: float,
        temperature: float,
        relative_humidity: float = 0.5,
        pressure: float = 101_325.0,
    ):
        self.c = float(sound_speed)
        self.t = float(temperature)
        self.rh = float(relative_humidity)
        self.p = float(pressure)

    def temperature(self, z: FloatArray) -> FloatArray:
        return np.full_like(np.asarray(z, dtype=float), self.t)

    def wind(self, z: FloatArray) -> FloatArray:
        return np.zeros((*np.shape(z), 3))

    def relative_humidity(self, z: FloatArray) -> FloatArray:
        return np.full_like(np.asarray(z, dtype=float), self.rh)

    def pressure(self, z: FloatArray) -> FloatArray:
        return np.full_like(np.asarray(z, dtype=float), self.p)

    def sound_speed(self, z: FloatArray) -> FloatArray:
        return np.full_like(np.asarray(z, dtype=float), self.c)

    def propagate(self, sources: FloatArray, receivers: FloatArray) -> ArrivalPath:
        d = np.asarray(receivers, dtype=float) - np.asarray(sources, dtype=float)
        r = np.linalg.norm(d, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = d / r[..., None]
            amp = 1.0 / r
        return ArrivalPath(travel_time=r / self.c, arrival_direction=direction, amplitude_factor=amp)


@dataclass(frozen=True)
class MicArray:
    """Microphone array: what the experimenter believes (nominal) and the hidden hardware truth.

    Named MicArray (not Array) to avoid confusion with numpy arrays. Recorder clock model:
    a pulse arriving at true time t (since the flash) is recorded at clock time
        tau = (1 + clock_drift_ppm * 1e-6) * t + clock_offset.
    """

    nominal_positions: FloatArray  # (M, 3) m, what reconstruction sees
    true_positions: FloatArray  # (M, 3) m, used only by synthesis
    clock_offset: FloatArray  # (M,) s
    clock_drift_ppm: FloatArray  # (M,)
    gain_db: FloatArray | None = None  # (M,) sensitivity error; None = 0 dB
    corner_scale: FloatArray | None = None  # (M,) multiplier on filter corner frequencies; None = 1

    def __post_init__(self) -> None:
        _check_shape("nominal_positions", self.nominal_positions, (None, 3))
        m = self.nominal_positions.shape[0]
        _check_shape("true_positions", self.true_positions, (m, 3))
        _check_shape("clock_offset", self.clock_offset, (m,))
        _check_shape("clock_drift_ppm", self.clock_drift_ppm, (m,))
        if self.gain_db is None:
            object.__setattr__(self, "gain_db", np.zeros(m))
        if self.corner_scale is None:
            object.__setattr__(self, "corner_scale", np.ones(m))
        _check_shape("gain_db", np.asarray(self.gain_db), (m,))
        _check_shape("corner_scale", np.asarray(self.corner_scale), (m,))

    @property
    def n_mics(self) -> int:
        return int(self.nominal_positions.shape[0])

    @property
    def has_clock_error(self) -> bool:
        return bool(np.any(self.clock_offset != 0) or np.any(self.clock_drift_ppm != 0))

    def recorder_time(self, t: FloatArray, mic: int) -> FloatArray:
        """Clock time at which mic `mic` records an event at true time t."""
        return (1.0 + self.clock_drift_ppm[mic] * 1e-6) * np.asarray(t) + self.clock_offset[mic]

    @classmethod
    def ideal(cls, positions: FloatArray) -> MicArray:
        """Array with perfectly known positions, perfect clocks and identical mics."""
        p = np.asarray(positions, dtype=float)
        m = p.shape[0]
        return cls(p, p.copy(), np.zeros(m), np.zeros(m))


@dataclass(frozen=True)
class GroundTruth:
    """Hidden truth attached to a Recording. Reconstruction must never read this."""

    t0: float  # s, true flash time
    mic_positions: FloatArray  # (M, 3)
    segment_arrival_times: FloatArray  # (S, M) s
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recording:
    """What a field team would have: signals, sample rate, nominal geometry, reported t0."""

    signals: FloatArray  # (M, samples) Pa
    sample_rate: float  # Hz
    nominal_mic_positions: FloatArray  # (M, 3)
    reported_t0: float  # s
    truth: GroundTruth | None = None

    def __post_init__(self) -> None:
        _check_shape("signals", self.signals, (None, None))
        _check_shape("nominal_mic_positions", self.nominal_mic_positions, (self.signals.shape[0], 3))
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def n_mics(self) -> int:
        return int(self.signals.shape[0])

    @property
    def duration(self) -> float:
        return self.signals.shape[1] / self.sample_rate


@dataclass(frozen=True)
class Reconstruction:
    """Output of any reconstruction method."""

    points: FloatArray  # (K, 3) m
    covariances: FloatArray  # (K, 3, 3) m^2
    window_times: FloatArray  # (K,) s, window centers
    quality: FloatArray  # (K,) method-specific score, higher is better
    method: str
    config_hash: str

    def __post_init__(self) -> None:
        _check_shape("points", self.points, (None, 3))
        k = self.points.shape[0]
        _check_shape("covariances", self.covariances, (k, 3, 3))
        _check_shape("window_times", self.window_times, (k,))
        _check_shape("quality", self.quality, (k,))

    @classmethod
    def empty(cls, method: str, config_hash: str) -> Reconstruction:
        return cls(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0), np.zeros(0), method, config_hash)

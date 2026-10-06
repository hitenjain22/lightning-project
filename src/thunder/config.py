"""Pydantic models for YAML run configs.

Each phase adds its own section when it is built. Unknown keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    """Top-level run configuration."""

    experiment: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    seed: int = Field(ge=0)
    results_root: Path = Path("results")
    sample_rate_hz: float = Field(default=8000.0, gt=0)
    oversample: int = Field(default=8, ge=1)
    description: str = ""

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

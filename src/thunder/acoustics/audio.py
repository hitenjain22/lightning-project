"""WAV export so synthetic thunder can be listened to."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from thunder.types import Recording


def write_wavs(rec: Recording, out_dir: Path, peak: float = 0.9) -> float:
    """Write one 24-bit WAV per mic, all scaled by one common gain so relative levels are kept.

    Returns the gain (WAV units per pascal) so absolute pressure can be recovered.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    max_abs = float(np.max(np.abs(rec.signals)))
    gain = peak / max_abs if max_abs > 0 else 1.0
    for i, x in enumerate(rec.signals):
        sf.write(out_dir / f"mic{i:02d}.wav", x * gain, int(round(rec.sample_rate)), subtype="PCM_24")
    return gain

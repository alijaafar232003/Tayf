"""Receiver health of one block of samples: ADC level and clipping."""
from __future__ import annotations

import numpy as np

from ..core.models import IQBlock

# ADC level guide for an 8-bit receiver (total power of the samples, dBFS)
LEVEL_LOW_DBFS = -35.0      # below this, quantization noise starts to dominate
CLIP_WARN = 0.001           # more than 0.1 % of samples at full scale


def measure_health(block: IQBlock) -> dict:
    """ADC level and clipping of one block. Clipping is when I or Q hits full scale.
    The level leaves out the DC offset: on this HackRF it is a fixed ~-39 dBFS that
    does not follow the gain, and at low gain it would hide how much noise and
    signal actually reach the ADC."""
    x = block.samples
    clipped = (np.abs(x.real) >= 127 / 128) | (np.abs(x.imag) >= 127 / 128)
    level = 10 * np.log10(np.mean(np.abs(x - x.mean()) ** 2) + 1e-20)
    clip = float(clipped.mean())
    hint = "clipping" if clip > CLIP_WARN else "low" if level < LEVEL_LOW_DBFS else "ok"
    return {"level_dbfs": round(float(level), 1), "clip_pct": round(100 * clip, 3), "level_hint": hint}

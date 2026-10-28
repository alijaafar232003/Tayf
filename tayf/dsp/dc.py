"""DC offset removal.

A direct-conversion receiver (like the HackRF) mixes the RF straight down to 0 Hz.
Small imbalances in the mixer and ADC leave a constant offset in I and Q, which
the FFT shows as a spike at the center frequency.

Subtracting the mean of each processing block removes it. The notch this makes
is only about 1 / (block duration) wide, e.g. ~20 Hz for a 50 ms block, so real
signals are barely touched. Recordings keep the raw samples; this is applied
during processing only.
"""
import numpy as np


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - x.mean()

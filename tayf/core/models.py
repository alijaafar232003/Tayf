"""Core data objects shared by every part of the pipeline.

Power convention: samples are complex baseband scaled so that |x| = 1 is the
converter's full scale. Power in dB relative to that is dBFS (0 dBFS = clipping).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReceiverSettings:
    """How the receiver was set up. Saved with every recording and session so
    results from different sessions can be compared."""

    center_freq_hz: float
    sample_rate_hz: float
    amp_on: bool | None = None          # HackRF RF amp (+~11 dB), None = unknown
    lna_gain_db: float | None = None    # HackRF IF gain, 0-40 dB in 8 dB steps
    vga_gain_db: float | None = None    # HackRF baseband gain, 0-62 dB in 2 dB steps
    antenna_id: str | None = None
    bias_tee: bool | None = None        # 3.3 V antenna power for an external LNA
    hardware: str | None = None         # e.g. "HackRF One"

    @property
    def gains_known(self) -> bool:
        return None not in (self.amp_on, self.lna_gain_db, self.vga_gain_db)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IQBlock:
    """A contiguous chunk of complex baseband samples.

    The pipeline only ever sees IQBlocks, so it cannot tell whether they came
    from a file or from live reception."""

    samples: np.ndarray                 # complex64, full scale = 1.0
    sample_rate_hz: float
    center_freq_hz: float
    start_time: datetime                # wall-clock time of the first sample
    start_index: int = 0                # position of the first sample in its stream

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.complex64)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate_hz

    @property
    def start_offset_s(self) -> float:
        """Seconds from the start of the stream to the first sample."""
        return self.start_index / self.sample_rate_hz

    @property
    def span_hz(self) -> tuple[float, float]:
        half = self.sample_rate_hz / 2
        return self.center_freq_hz - half, self.center_freq_hz + half

    def time_at(self, sample: int) -> datetime:
        return self.start_time + timedelta(seconds=sample / self.sample_rate_hz)


@dataclass
class SpectrumFrame:
    """Power versus absolute frequency at one moment: one waterfall row."""

    freqs_hz: np.ndarray                # absolute frequency of each bin, ascending
    power_dbfs: np.ndarray              # power per bin; a full-scale tone reads 0 dBFS
    rbw_hz: float                       # resolution bandwidth (equivalent noise bandwidth of one bin)
    center_freq_hz: float
    time: datetime
    offset_s: float = 0.0               # position in the stream (recordings)
    n_averages: int = 1


@dataclass
class SweepFrame:
    """Power versus frequency from hackrf_sweep. No I/Q, so it cannot be
    investigated or classified, only used to find candidates."""

    freqs_hz: np.ndarray
    power_db: np.ndarray
    bin_width_hz: float
    time: datetime = field(default_factory=utc_now)

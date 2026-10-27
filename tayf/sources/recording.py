"""A SigMF recording played as an I/Q source, with seeking."""
from __future__ import annotations

from pathlib import Path

from ..core.models import IQBlock
from ..core.sigmf_io import SigMFRecording


class RecordingSource:
    kind = "recording"
    seekable = True

    def __init__(self, path: str | Path):
        self.recording = SigMFRecording.open(path)
        self.settings = self.recording.settings
        self.position = 0                       # next sample to read

    @property
    def duration_s(self) -> float:
        return self.recording.duration_s

    @property
    def position_s(self) -> float:
        return self.position / self.settings.sample_rate_hz

    def seek(self, t_s: float) -> None:
        n = int(round(t_s * self.settings.sample_rate_hz))
        self.position = max(0, min(n, self.recording.n_samples))

    def read_block(self, n_samples: int) -> IQBlock | None:
        if self.position >= self.recording.n_samples:
            return None
        block = self.recording.read(self.position, n_samples)
        self.position += len(block)
        return block

    def read_at(self, t_s: float, n_samples: int) -> IQBlock:
        """Random access without moving the play position."""
        start = int(round(t_s * self.settings.sample_rate_hz))
        return self.recording.read(start, n_samples)

    def close(self) -> None:
        pass

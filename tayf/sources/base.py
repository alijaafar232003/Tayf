"""The interface every I/Q source implements: recording file now, HackRF live in Phase 2."""
from __future__ import annotations

from typing import Protocol

from ..core.models import IQBlock, ReceiverSettings


class IQSource(Protocol):
    kind: str                       # "recording" | "live"
    settings: ReceiverSettings
    seekable: bool

    def read_block(self, n_samples: int) -> IQBlock | None:
        """Next n_samples of I/Q, or None when the source has ended."""
        ...

    def close(self) -> None: ...

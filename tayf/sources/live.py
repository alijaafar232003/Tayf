"""Live I/Q from a radio.

A reader thread pulls samples from the radio continuously into a ring buffer, so
the USB stream never stalls while the app is busy. The display takes the newest
samples from the buffer; when it is slower than the radio, older samples are
simply overwritten (that is a display skip, not a hardware loss).

Offset tuning: the radio's local oscillator (LO) is tuned a little away from the
frequency you ask for, so the DC spike (always at the LO) does not land on the
signal you are looking at. The spectrum is drawn at true frequencies.
"""
from __future__ import annotations

import threading
from datetime import timedelta

import numpy as np

from ..core.models import IQBlock, ReceiverSettings, utc_now
from ..core.sigmf_io import to_complex
from .radio import FREQ_RANGE, Gains, Radio

SETTLE_S = 0.01          # samples just after a retune or gain change are discarded


def offset_for(sample_rate_hz: float) -> float:
    """LO offset: 10 % of the sample rate, at least 100 kHz (200 kHz at 2 MS/s)."""
    return max(100e3, round(0.1 * sample_rate_hz / 10e3) * 10e3)


class LiveSource:
    kind = "live"
    seekable = False

    def __init__(self, radio: Radio, tuned_hz: float, sample_rate_hz: float, gains: Gains,
                 antenna_id: str | None = None, offset_tuning: bool = True,
                 buffer_s: float = 0.5):
        self.radio = radio
        self.antenna_id = antenna_id
        self.offset_tuning = offset_tuning
        self.buffer_s = buffer_s
        self.gains = gains
        self.overflows = 0                # times the radio reported lost samples
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.sample_rate_hz = float(sample_rate_hz)
        self._alloc()
        self.tuned_hz = float(tuned_hz)
        radio.set_sample_rate(self.sample_rate_hz)
        radio.set_gains(gains)
        self._apply_frequency()

    # --- settings -------------------------------------------------------
    @property
    def offset_hz(self) -> float:
        return offset_for(self.sample_rate_hz) if self.offset_tuning else 0.0

    @property
    def lo_hz(self) -> float:
        return self.tuned_hz + self.offset_hz

    @property
    def settings(self) -> ReceiverSettings:
        return ReceiverSettings(self.lo_hz, self.sample_rate_hz, self.gains.amp,
                                self.gains.lna_db, self.gains.vga_db, self.antenna_id,
                                hardware=self.radio.name)

    def _alloc(self) -> None:
        self._size = int(self.buffer_s * self.sample_rate_hz)
        self._buf = np.zeros(2 * self._size, dtype=np.int8)
        self._total = 0                   # samples written since start
        self._valid_from = 0              # first sample after the last setting change

    def _invalidate(self) -> None:
        """Discard samples taken before a setting change, plus a short settling time."""
        self._valid_from = self._total + int(SETTLE_S * self.sample_rate_hz)

    def _apply_frequency(self) -> None:
        self.radio.set_frequency(self.lo_hz)

    def tune(self, hz: float) -> None:
        lo, hi = FREQ_RANGE
        with self._lock:
            self.tuned_hz = float(np.clip(hz, lo, hi - self.offset_hz))
            self._apply_frequency()
            self._invalidate()

    def set_offset_tuning(self, on: bool) -> None:
        with self._lock:
            self.offset_tuning = bool(on)
            self._apply_frequency()
            self._invalidate()

    def set_gains(self, gains: Gains) -> None:
        with self._lock:
            self.gains = gains
            self.radio.set_gains(gains)
            self._invalidate()

    def set_sample_rate(self, hz: float) -> None:
        # The stream must be stopped to change the rate safely.
        was_running = self._running
        self.stop()
        with self._lock:
            self.sample_rate_hz = float(hz)
            self.radio.set_sample_rate(self.sample_rate_hz)
            self._alloc()
            self._apply_frequency()       # the offset depends on the rate
            self._invalidate()
        if was_running:
            self.start()

    # --- streaming ------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.radio.start()
        self._thread = threading.Thread(target=self._reader, name="radio-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        chunk = np.empty(2 * 131_072, dtype=np.int8)
        while self._running:
            n = self.radio.read(chunk)
            if n < 0:
                self.overflows += 1
                continue
            if n == 0:
                continue
            with self._lock:
                if 2 * n > len(self._buf):        # rate changed under us; drop this chunk
                    continue
                start = (self._total % self._size) * 2
                end = start + 2 * n
                if end <= len(self._buf):
                    self._buf[start:end] = chunk[:2 * n]
                else:
                    k = len(self._buf) - start
                    self._buf[start:] = chunk[:k]
                    self._buf[:end - len(self._buf)] = chunk[k:2 * n]
                self._total += n

    def latest(self, n: int) -> IQBlock | None:
        """The newest n samples, or None if fewer than n have arrived since the
        last setting change."""
        with self._lock:
            n = min(n, self._size)
            if self._total - self._valid_from < n:
                return None
            end = (self._total % self._size) * 2
            start = end - 2 * n
            raw = self._buf[start:end].copy() if start >= 0 else \
                np.concatenate([self._buf[start:], self._buf[:end]])
            total, fs, lo = self._total, self.sample_rate_hz, self.lo_hz
        return IQBlock(to_complex(raw, "ci8"), fs, lo,
                       utc_now() - timedelta(seconds=n / fs), total - n)

    def read_block(self, n_samples: int) -> IQBlock | None:
        return self.latest(n_samples)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self.radio.stop()

    def close(self) -> None:
        self.stop()
        self.radio.close()

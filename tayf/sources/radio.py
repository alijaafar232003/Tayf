"""Radio hardware behind one small interface: the real HackRF (SoapySDR) and a
simulated HackRF for development and tests without hardware.

A radio only tunes, sets gains and delivers raw int8 I/Q chunks (the HackRF's
native format). Buffering, offset tuning and DC removal live in LiveSource.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# HackRF gain stages (receive)
AMP_DB = 14                      # RF amp: on/off, ~14 dB
LNA_STEPS = list(range(0, 41, 8))   # IF gain 0-40 dB, 8 dB steps
VGA_STEPS = list(range(0, 63, 2))   # baseband gain 0-62 dB, 2 dB steps
SAMPLE_RATES = [2e6, 4e6, 8e6, 10e6, 12.5e6, 16e6, 20e6]
FREQ_RANGE = (1e6, 6e9)


def snap_gain(value: float, steps: list[int]) -> int:
    return min(steps, key=lambda s: abs(s - value))


@dataclass
class Gains:
    amp: bool = False
    lna_db: int = 16
    vga_db: int = 20

    @property
    def total_db(self) -> int:
        return (AMP_DB if self.amp else 0) + self.lna_db + self.vga_db


class Radio(Protocol):
    name: str

    def set_frequency(self, hz: float) -> None: ...
    def set_sample_rate(self, hz: float) -> None: ...
    def set_gains(self, gains: Gains) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, out: np.ndarray) -> int:
        """Fill `out` (int8, interleaved I/Q) with up to len(out)//2 samples.
        Returns samples read, 0 on timeout, -1 if the device reported an overflow
        (samples were lost before we read them)."""
        ...
    def close(self) -> None: ...


# --- real HackRF ------------------------------------------------------------
# (SoapySDR's HackRF module needs the env's DLL folder: tayf/__init__.py adds it.)

def find_hackrf() -> list[dict]:
    """HackRFs that SoapySDR can see (empty if none is plugged in)."""
    import SoapySDR
    return [dict(d) for d in SoapySDR.Device.enumerate() if dict(d).get("driver") == "hackrf"]


class HackRFRadio:
    name = "HackRF One"

    def __init__(self):
        import SoapySDR
        from SoapySDR import SOAPY_SDR_CS8, SOAPY_SDR_RX
        self._soapy, self._rx = SoapySDR, SOAPY_SDR_RX
        # Device() with no arguments: on this PC, Device(dict(driver="hackrf")) failed.
        self.dev = SoapySDR.Device()
        if self.dev.getDriverKey().lower() != "hackrf":
            raise RuntimeError(f"Found {self.dev.getDriverKey()}, not a HackRF")
        self.stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS8, [0])
        self.mtu = self.dev.getStreamMTU(self.stream)
        self.serial = dict(self.dev.getHardwareInfo()).get("serial", "")
        self._active = False

    def set_frequency(self, hz: float) -> None:
        self.dev.setFrequency(self._rx, 0, float(hz))

    def set_sample_rate(self, hz: float) -> None:
        # Changing the rate while streaming is unreliable: pause the stream.
        was = self._active
        if was:
            self.stop()
        self.dev.setSampleRate(self._rx, 0, float(hz))
        self.dev.setBandwidth(self._rx, 0, 0.75 * float(hz))   # anti-alias filter
        if was:
            self.start()

    def set_gains(self, g: Gains) -> None:
        self.dev.setGain(self._rx, 0, "AMP", AMP_DB if g.amp else 0)
        self.dev.setGain(self._rx, 0, "LNA", g.lna_db)
        self.dev.setGain(self._rx, 0, "VGA", g.vga_db)

    def start(self) -> None:
        if not self._active:
            self.dev.activateStream(self.stream)
            self._active = True

    def stop(self) -> None:
        if self._active:
            self.dev.deactivateStream(self.stream)
            self._active = False

    def read(self, out: np.ndarray) -> int:
        n = min(len(out) // 2, self.mtu)
        sr = self.dev.readStream(self.stream, [out], n, timeoutUs=200_000)
        if sr.ret == self._soapy.SOAPY_SDR_OVERFLOW:
            return -1
        return max(sr.ret, 0)

    def close(self) -> None:
        self.stop()
        self.dev.closeStream(self.stream)
        self.dev = None


# --- simulated HackRF -------------------------------------------------------

# A small made-up environment, loosely based on the first real FM recording.
SIM_SIGNALS = [
    # (center Hz, kind, signal power relative to the total noise power in a 2 MHz span, dB)
    (97.1e6, "wbfm", -2.0), (98.5e6, "wbfm", 2.0), (98.9e6, "wbfm", -4.0),
    (101.3e6, "wbfm", 8.0), (104.6e6, "wbfm", 4.0),
    (124.0e6, "am", -5.0),
    (433.92e6, "ook", 5.0),
]


class SimRadio:
    """Behaves like a HackRF: real-time pacing, gain, 8-bit quantization and
    clipping, a DC offset at the tuned frequency, and overflow if read too slowly."""

    name = "Simulated HackRF"
    # Antenna noise in a 2 MHz span at 0 dB gain (dBFS). At LNA 16 + VGA 20 this
    # gives ~-40 dBFS, matching the first real FM recording.
    noise_db = -76.0
    dc_offset = 0.008 + 0.004j

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.fc, self.fs, self.gains = 100e6, 2e6, Gains()
        self._t0 = None
        self._n = 0                       # samples delivered since start
        self._phase = {}                  # per-signal FM phase, kept continuous

    def set_frequency(self, hz: float) -> None:
        self.fc = float(hz)

    def set_sample_rate(self, hz: float) -> None:
        self.fs = float(hz)
        self._restart_clock()

    def set_gains(self, gains: Gains) -> None:
        self.gains = gains

    def start(self) -> None:
        self._restart_clock()

    def stop(self) -> None:
        self._t0 = None

    def _restart_clock(self) -> None:
        self._t0, self._n = time.perf_counter(), 0

    def read(self, out: np.ndarray) -> int:
        if self._t0 is None:
            time.sleep(0.05)
            return 0
        # Deliver samples no faster than real time, like the hardware.
        due = int((time.perf_counter() - self._t0) * self.fs) - self._n
        if due > self.fs * 0.5:            # reader fell >0.5 s behind: the device drops data
            self._n += due
            return -1
        n = min(len(out) // 2, due)
        if n < 4096:
            time.sleep(0.005)
            return 0
        x = self._generate(n)
        g = 10 ** (self.gains.total_db / 20)
        x = x * g + self.dc_offset
        iq = np.empty(2 * n, dtype=np.float32)
        iq[0::2], iq[1::2] = x.real, x.imag
        out[:2 * n] = np.clip(np.round(iq * 128), -128, 127).astype(np.int8)
        self._n += n
        return n

    def _generate(self, n: int) -> np.ndarray:
        fs, t = self.fs, (self._n + np.arange(n)) / self.fs
        # Noise density is fixed, so a wider span collects proportionally more noise.
        noise_db = self.noise_db + 10 * np.log10(fs / 2e6)
        std = np.sqrt(10 ** (noise_db / 10) / 2)
        x = std * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        for f, kind, p in SIM_SIGNALS:
            off = f - self.fc
            if abs(off) > 0.5 * fs:
                continue
            a = 10 ** ((p + self.noise_db) / 20)
            carrier = np.exp(2j * np.pi * off * t)
            if kind == "wbfm":
                audio = 0.6 * np.sin(2 * np.pi * 1e3 * t + f) + 0.4 * np.sin(2 * np.pi * 2.3e3 * t)
                ph = self._phase.get(f, 0.0) + np.cumsum(2 * np.pi * 75e3 * audio / fs)
                self._phase[f] = ph[-1]
                x += a * carrier * np.exp(1j * ph)
            elif kind == "am":
                x += a * carrier * (1 + 0.5 * np.sin(2 * np.pi * 800 * t))
            elif kind == "ook":
                on = (t % 1.0) < 0.15
                x += a * carrier * (on & ((t * 2e3).astype(int) % 3 != 0))
        return x

    def close(self) -> None:
        self.stop()

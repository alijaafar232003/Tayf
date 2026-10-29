"""Band scan: power versus frequency over a wide range, from hackrf_sweep.

Live mode sees at most 20 MHz at once. hackrf_sweep lets the HackRF firmware hop
across a range in 20 MHz steps (about 8 GHz per second) and returns an FFT power
spectrum for each step. We stitch the steps of one pass into one SweepFrame.
There is no I/Q, so a scan finds where signals are; live mode looks at them.

hackrf_sweep runs as its own process with binary output (-B). Each record is:
  uint32 record_length, uint64 hz_low, uint64 hz_high, float32 power_db[n]
where record_length counts the bytes after itself.
"""
from __future__ import annotations

import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

from ..core.models import ReceiverSettings, SweepFrame, utc_now
from .radio import AMP_DB, FREQ_RANGE, SIM_SIGNALS, Gains

SWEEP_FS = 20e6                     # hackrf_sweep always samples at 20 MS/s
BIN_WIDTHS = [1e6, 500e3, 250e3, 100e3, 50e3, 25e3, 10e3, 5e3]
MAX_BINS = 200_000                  # keeps one sweep small enough to draw and send


def check_range(f_lo: float, f_hi: float, bin_hz: float) -> None:
    if not FREQ_RANGE[0] <= f_lo < f_hi <= FREQ_RANGE[1]:
        raise ValueError(f"Scan range must lie within {FREQ_RANGE[0] / 1e6:.0f}-{FREQ_RANGE[1] / 1e6:.0f} MHz")
    if bin_hz not in BIN_WIDTHS:
        raise ValueError(f"bin_hz must be one of {BIN_WIDTHS}")
    if (f_hi - f_lo) / bin_hz > MAX_BINS:
        raise ValueError(f"{(f_hi - f_lo) / bin_hz:.0f} bins is too many: use a wider bin or a narrower range")


class SweepAssembler:
    """Collects hackrf_sweep segments into whole sweeps on a fixed frequency grid.

    A sweep is complete when a segment's start frequency repeats (the firmware
    has started the next pass). Bins outside [f_lo, f_hi] are dropped; bins never
    filled stay NaN."""

    def __init__(self, f_lo: float, f_hi: float):
        self.f_lo, self.f_hi = f_lo, f_hi
        self.bin_hz: float | None = None
        self._grid: np.ndarray | None = None
        self._seen: set[int] = set()

    def add(self, hz_low: int, hz_high: int, power: np.ndarray) -> SweepFrame | None:
        done = None
        if hz_low in self._seen:
            done = self._finish()
        if self._grid is None:
            self.bin_hz = (hz_high - hz_low) / len(power)
            n = int(round((self.f_hi - self.f_lo) / self.bin_hz))
            self._grid = np.full(n, np.nan, dtype=np.float32)
        self._seen.add(hz_low)
        # Bin i of this segment is centred at hz_low + (i + 0.5) * bin
        start = int(round((hz_low - self.f_lo) / self.bin_hz))
        a, b = max(start, 0), min(start + len(power), len(self._grid))
        if a < b:
            self._grid[a:b] = power[a - start:b - start]
        return done

    def _finish(self) -> SweepFrame:
        g, self._grid, self._seen = self._grid, None, set()
        freqs = self.f_lo + (np.arange(len(g)) + 0.5) * self.bin_hz
        return SweepFrame(freqs, g, self.bin_hz, utc_now())


class SweepSource:
    """Common part of the real and simulated sweep: keeps recent sweeps and
    hands the display the average of all sweeps since its last frame."""

    kind = "scan"
    radio_name = "?"

    def __init__(self, f_lo: float, f_hi: float, bin_hz: float, gains: Gains,
                 antenna_id: str | None = None):
        check_range(f_lo, f_hi, bin_hz)
        self.f_lo, self.f_hi, self.bin_hz = float(f_lo), float(f_hi), float(bin_hz)
        self.gains, self.antenna_id = gains, antenna_id
        self.sweeps = 0                       # sweeps completed since start
        self.error: str | None = None
        self._lock = threading.Lock()
        self._pending: list[SweepFrame] = []  # sweeps the display has not taken yet
        self._last: SweepFrame | None = None
        self._times: deque[float] = deque(maxlen=50)
        self._t0 = time.time()

    @property
    def settings(self) -> ReceiverSettings:
        return ReceiverSettings((self.f_lo + self.f_hi) / 2, SWEEP_FS, self.gains.amp,
                                self.gains.lna_db, self.gains.vga_db, self.antenna_id,
                                hardware="HackRF One (hackrf_sweep)")

    @property
    def sweep_rate(self) -> float:
        """Completed sweeps per second, over the last few sweeps."""
        t = self._times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def _deliver(self, frame: SweepFrame) -> None:
        with self._lock:
            self.sweeps += 1
            self._times.append(time.time())
            self._pending.append(frame)
            del self._pending[:-200]          # the display is not reading: keep it bounded
            self._last = frame

    def take(self) -> SweepFrame | None:
        """Mean power (linear) of all sweeps since the last call; None if no new sweep.
        Averaging lowers the noise variance, so weak steady signals stand out."""
        with self._lock:
            got, self._pending = self._pending, []
        if not got:
            return None
        if len(got) == 1:
            return got[0]
        with np.errstate(invalid="ignore"):
            lin = np.nanmean([10 ** (f.power_db / 10) for f in got], axis=0)
            power = (10 * np.log10(lin)).astype(np.float32)
        return SweepFrame(got[-1].freqs_hz, power, got[-1].bin_width_hz, got[-1].time)

    def latest(self) -> SweepFrame | None:
        with self._lock:
            return self._last

    @property
    def elapsed_s(self) -> float:
        return time.time() - self._t0

    def start(self) -> None: ...
    def close(self) -> None: ...


class HackRFSweep(SweepSource):
    radio_name = "HackRF One"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exe = shutil.which("hackrf_sweep") or os.path.join(sys.prefix, "Library", "bin", "hackrf_sweep.exe")
        self._proc: subprocess.Popen | None = None
        self._stderr: deque[str] = deque(maxlen=20)

    def command(self) -> list[str]:
        # hackrf_sweep takes whole MHz; the extra edges are cut off by the assembler.
        lo, hi = math.floor(self.f_lo / 1e6), math.ceil(self.f_hi / 1e6)
        g = self.gains
        return [self.exe, "-f", f"{lo}:{hi}", "-w", str(int(self.bin_hz)), "-B",
                "-a", "1" if g.amp else "0", "-l", str(g.lna_db), "-g", str(g.vga_db)]

    def start(self, timeout_s: float = 4.0) -> None:
        if not os.path.exists(self.exe):
            raise RuntimeError("hackrf_sweep was not found (it comes with the HackRF tools)")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      creationflags=flags, bufsize=0)
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True, name="hackrf_sweep out").start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True, name="hackrf_sweep err").start()
        # Wait for the first sweep, so "no HackRF" or "busy" is reported at once.
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            if self.sweeps:
                return
            if self._proc.poll() is not None:
                time.sleep(0.1)               # let the stderr thread catch the message
                code, self._proc = self._proc.returncode, None
                raise RuntimeError(self._error_text() or f"hackrf_sweep exited with code {code}")
            time.sleep(0.02)
        self.close()
        raise RuntimeError("hackrf_sweep gave no data: " + (self._error_text() or "timed out"))

    def _error_text(self) -> str:
        lines = [l for l in self._stderr if "fail" in l.lower() or "error" in l.lower() or "not found" in l.lower()]
        return "; ".join(lines or list(self._stderr)[-2:])

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        out, asm = proc.stdout, SweepAssembler(self.f_lo, self.f_hi)

        def read_exact(n: int) -> bytes | None:
            buf = b""
            while len(buf) < n:
                chunk = out.read(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            return buf

        while True:
            head = read_exact(4)
            if head is None:
                break
            (length,) = struct.unpack("<I", head)
            rec = read_exact(length)
            if rec is None:
                break
            hz_low, hz_high = struct.unpack_from("<QQ", rec)
            frame = asm.add(hz_low, hz_high, np.frombuffer(rec, dtype="<f4", offset=16))
            if frame is not None:
                self._deliver(frame)
        if self._proc is proc and proc.wait() != 0:     # it stopped on its own, not through close()
            self.error = self._error_text() or "hackrf_sweep stopped"

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            self._stderr.append(raw.decode(errors="replace").strip())

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is None or p.poll() is not None:
            return
        # Ask nicely first (Ctrl+Break), so hackrf_sweep releases the HackRF cleanly.
        try:
            p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            p.wait(timeout=2.0)
        except Exception:
            p.kill()
            p.wait(timeout=2.0)


# --- simulated sweep -----------------------------------------------------------

# Wideband signals the live simulator does not have: (low Hz, high Hz, dB re noise in 2 MHz, duty)
SIM_BANDS = [
    (538e6, 546e6, 6.0, 1.0), (594e6, 602e6, 3.0, 1.0),             # DVB-T multiplexes
    (935.2e6, 935.4e6, 10.0, 1.0), (947.6e6, 947.8e6, 6.0, 1.0),     # GSM carriers
    (950e6, 960e6, 4.0, 1.0),                                         # UMTS/LTE 900
    (1805e6, 1825e6, 8.0, 1.0), (2110e6, 2125e6, 5.0, 1.0),           # LTE 1800, UMTS 2100
    (2402e6, 2422e6, 6.0, 0.3), (2427e6, 2447e6, 2.0, 0.2),           # Wi-Fi channels 1 and 6
    (2620e6, 2640e6, 4.0, 1.0),                                       # LTE 2600
]


class SimSweep(SweepSource):
    """Behaves like hackrf_sweep on a made-up environment: same grid, ~8 GHz/s pacing,
    gain, and noise that fluctuates like a short FFT average."""

    radio_name = "Simulated HackRF"
    noise_density_db = -76.0 - 10 * math.log10(2e6)   # same antenna noise as SimRadio, dBFS/Hz at 0 dB gain

    def __init__(self, *args, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._run, daemon=True, name="sim sweep").start()
        while not self.sweeps:
            time.sleep(0.01)

    def _run(self) -> None:
        steps = max(1, math.ceil((self.f_hi - self.f_lo) / 20e6))
        sweep_s = max(0.01, steps * 0.0025)            # 20 MHz per 2.5 ms = 8 GHz/s
        while self._running:
            self._deliver(self._sweep())
            time.sleep(sweep_s)

    def _sweep(self) -> SweepFrame:
        bw = self.bin_hz
        n = int(round((self.f_hi - self.f_lo) / bw))
        f = self.f_lo + (np.arange(n) + 0.5) * bw
        g = self.gains.lna_db + self.gains.vga_db + (AMP_DB if self.gains.amp else 0)
        noise = 10 ** ((self.noise_density_db + 10 * np.log10(bw)) / 10)
        lin = np.full(n, noise)
        ref = 10 ** (-76.0 / 10)                         # noise power in 2 MHz, the SIM_SIGNALS reference
        t = time.time()
        narrow = {"wbfm": 180e3, "am": 8e3, "ook": 30e3}
        sigs = [(fc - narrow[k] / 2, fc + narrow[k] / 2, p, 0.15 if k == "ook" else 1.0) for fc, k, p in SIM_SIGNALS]
        for lo, hi, p, duty in sigs + SIM_BANDS:
            if hi < self.f_lo or lo > self.f_hi or (duty < 1 and (t % 1.0) > duty):
                continue
            density = ref * 10 ** (p / 10) / (hi - lo)
            # Power per bin = density x overlap of the bin with the signal
            overlap = np.clip(np.minimum(f + bw / 2, hi) - np.maximum(f - bw / 2, lo), 0, None)
            lin += density * overlap
        lin *= 10 ** (g / 10)
        lin *= self.rng.gamma(8, 1 / 8, n)              # a few FFTs averaged per bin
        lin = np.minimum(lin, 1.0)                       # nothing reads above full scale
        return SweepFrame(f, (10 * np.log10(lin)).astype(np.float32), bw, utc_now())

    def close(self) -> None:
        self._running = False

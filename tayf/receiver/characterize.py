"""Receiver characterization: what the HackRF adds by itself.

Run with nothing on the antenna port (best: a 50 ohm load), so everything
measured comes from the receiver. Used by notebook 02, which saves the results
as the receiver profile (profile.py). The profile is what later stages compare
against: the detector flattens the floor with it, the Receiver Check tells
"my receiver" from "the environment".

Measurements:
  floor_vs_gain   noise floor and ADC use at each LNA/VGA setting
  span_shape      how the floor varies across the span (filter edges, DC spike)
  spur_map        signals the receiver makes itself, over 1 MHz - 6 GHz
  check_carrier   is a carrier fixed in absolute frequency, or tied to the LO?
  image_rejection how strongly a real signal appears mirrored around the LO
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import median_filter

from ..core.sigmf_io import to_complex
from ..dsp.dc import remove_dc
from ..dsp.spectrum import rbw_hz, welch_power
from ..sources.radio import Gains, HackRFRadio
from ..sources.sweep import HackRFSweep


# --- capture -------------------------------------------------------------------

class Rx:
    """Holds the HackRF for the measurements.

    The stream runs continuously; each capture first drains what the driver
    buffered meanwhile. A sample-rate change reopens the device: on this driver,
    stopping and restarting the stream after a pause can replay old buffers
    endlessly (hundreds of MS/s of "samples") instead of delivering new ones."""

    def __init__(self):
        self.radio: HackRFRadio | None = None
        self.fs: float | None = None

    def at_rate(self, fs: float) -> HackRFRadio:
        if fs != self.fs:
            self.close()
            self.radio = HackRFRadio()
            self.radio.set_sample_rate(fs)
            self.fs = fs
        return self.radio

    def close(self) -> None:
        if self.radio is not None:
            self.radio.close()
        self.radio, self.fs = None, None


def _drain(radio, fs: float, settle_s: float = 0.02) -> None:
    """Read and throw away samples until they arrive no faster than real time.
    Until the driver's buffer is empty, what we read was received before the
    last change (old frequency or gain)."""
    junk = np.empty(2 * 131_072, dtype=np.int8)
    batch = max(262_144, int(0.02 * fs))
    t_end = time.time() + 5
    while True:
        # Time a batch of reads: buffered data comes back at once, live data
        # only as fast as the radio makes it.
        t, got = time.perf_counter(), 0
        while got < batch:
            got += max(radio.read(junk), 0)
        if time.perf_counter() - t > 0.6 * got / fs:
            break
        if time.time() > t_end:
            raise RuntimeError("The HackRF keeps returning buffered samples faster than real time")
    t1 = time.time()
    while time.time() - t1 < settle_s:                    # let the PLL and gains settle
        radio.read(junk)


def capture(rx: Rx, fc: float, fs: float, gains: Gains, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n fresh samples at fc. Returns (complex64 full scale 1.0, raw int8 interleaved)."""
    radio = rx.at_rate(fs)
    radio.set_frequency(fc)
    radio.set_gains(gains)
    radio.start()
    _drain(radio, fs)
    buf = np.empty(2 * n, dtype=np.int8)
    got, t_end = 0, time.time() + 5 + 2 * n / fs
    t0 = time.perf_counter()
    while got < n and time.time() < t_end:
        got += max(radio.read(buf[2 * got:]), 0)
    if got < n:
        raise RuntimeError(f"Only {got} of {n} samples arrived")
    if time.perf_counter() - t0 < 0.6 * n / fs:
        raise RuntimeError("Samples arrived faster than real time: they are not fresh")
    return to_complex(buf, "ci8"), buf


def _floor_db_hz(f: np.ndarray, p_db: np.ndarray, fs: float, nfft: int) -> float:
    """Median noise density (dBFS/Hz) in the flat middle of the span, DC excluded."""
    keep = (np.abs(f) > 0.02 * fs) & (np.abs(f) < 0.35 * fs)
    # Median of averaged noise ~ its mean when many FFTs are averaged.
    return float(np.median(p_db[keep]) - 10 * np.log10(rbw_hz(fs, nfft)))


# --- 1. floor vs gain ----------------------------------------------------------

def floor_vs_gain(rx: Rx, fc: float, fs: float, lna_list, vga_list, amp: bool = False,
                  n: int = 262_144, nfft: int = 1024) -> list[dict]:
    rows = []
    for lna in lna_list:
        for vga in vga_list:
            g = Gains(amp, lna, vga)
            x, raw = capture(rx, fc, fs, g, n)
            f, p, _ = welch_power(remove_dc(x), fs, nfft)
            clip = float(np.mean(np.abs(raw.astype(np.int16)) >= 127))
            rows.append({"fc_hz": fc, "amp": amp, "lna_db": lna, "vga_db": vga,
                         "total_db": g.total_db,
                         "floor_dbfs_hz": round(_floor_db_hz(f, p, fs, nfft), 2),
                         "level_dbfs": round(float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20)), 2),
                         "rms_lsb": round(float(np.std(raw.astype(np.float32))), 2),
                         "clip_pct": round(100 * clip, 3)})
    return rows


def floor_model(lna, vga, q, n_vga, n_front):
    """Noise floor (dBFS/Hz) as the power sum of three sources along the chain:
         q        the ADC's own floor (quantization): fixed, gain does not move it
         n_vga    noise entering at the VGA input: amplified by the VGA only
         n_front  noise entering at the front end (before the LNA): amplified by LNA + VGA
    n_vga and n_front are referred to 0 dB gain (where they would read at 0 dB)."""
    lna, vga = np.asarray(lna, float), np.asarray(vga, float)
    return 10 * np.log10(10 ** (q / 10) + 10 ** ((n_vga + vga) / 10) + 10 ** ((n_front + lna + vga) / 10))


def fit_floor(rows: list[dict]) -> dict:
    """Fit floor_model to measured rows (amp off). The two cross-over points are
    the gain-staging rule (Friis: the first stage with enough gain sets the noise):
      lna_db_front_dominates  LNA gain where front-end noise passes the VGA-stage
                              noise: below it, the LNA is wasting sensitivity
      vga_db_clears_adc       VGA gain (at that LNA) where receiver noise passes
                              the ADC floor: below it, quantization hides signals"""
    from scipy.optimize import least_squares
    lna = np.array([r["lna_db"] for r in rows], float)
    vga = np.array([r["vga_db"] for r in rows], float)
    y = np.array([r["floor_dbfs_hz"] for r in rows], float)
    res = least_squares(lambda p: floor_model(lna, vga, *p) - y, x0=[y.min(), y.max() - 62, y.max() - 100])
    q, n_vga, n_front = res.x
    return {"adc_floor_dbfs_hz": round(float(q), 2), "n_vga_dbfs_hz": round(float(n_vga), 2),
            "n_front_dbfs_hz": round(float(n_front), 2),
            "lna_db_front_dominates": round(float(n_vga - n_front), 1),
            "vga_db_clears_adc_at_lna32": round(float(q - 10 * np.log10(10 ** (n_vga / 10) + 10 ** ((n_front + 32) / 10))), 1),
            "rms_error_db": round(float(np.sqrt(np.mean(res.fun ** 2))), 2)}


# --- 2. floor shape across the span ----------------------------------------------

def span_shape(rx: Rx, fc: float, fs: float, gains: Gains, n: int = 2_000_000,
               nfft: int = 1024, points: int = 256) -> dict:
    """Average spectrum of the receiver's own noise, relative to its middle.
    Without DC removal (to see the spike) and with it (what the app shows)."""
    x, _ = capture(rx, fc, fs, gains, n)
    f, raw_db, k = welch_power(x, fs, nfft)
    _, dc_db, _ = welch_power(remove_dc(x), fs, nfft)
    mid = np.median(dc_db[(np.abs(f) > 0.02 * fs) & (np.abs(f) < 0.25 * fs)])
    shape = dc_db - mid
    c = nfft // 2                                     # the DC bin (f = 0)
    near = np.r_[dc_db[c - 12:c - 4], dc_db[c + 5:c + 13]]
    near_raw = np.r_[raw_db[c - 12:c - 4], raw_db[c + 5:c + 13]]
    # Downsample the shape for storage: mean (in dB) of groups of bins
    g = nfft // points
    small = shape.reshape(points, g).mean(axis=1)
    edge = lambda frac: float(np.interp(frac * fs, f, shape))
    flat = np.abs(f[shape > -1.0])
    return {"fs_hz": fs, "fc_hz": fc, "gains": [gains.amp, gains.lna_db, gains.vga_db], "n_averages": k,
            "offset_frac": np.round(np.linspace(-0.5, 0.5, points, endpoint=False) + 0.5 / points, 5).tolist(),
            "shape_db": np.round(small, 2).tolist(),
            "dc_spike_raw_db": round(float(raw_db[c] - np.median(near_raw)), 1),
            "dc_spike_removed_db": round(float(dc_db[c] - np.median(near)), 1),
            "edge_db_at_0p45": round(min(edge(-0.45), edge(0.45)), 1),
            "edge_db_at_0p40": round(min(edge(-0.40), edge(0.40)), 1),
            "flat_1db_frac": round(float(2 * np.percentile(flat, 99) / fs), 3),
            "_full": (f, shape, raw_db - mid)}         # for plotting only (not saved)


# --- 3. spurs over the whole range ------------------------------------------------

def spur_map(f_lo: float = 1e6, f_hi: float = 6000e6, bin_hz: float = 100e3, gains: Gains = Gains(False, 24, 30),
             seconds: float = 10.0, threshold_db: float = 6.0) -> dict:
    """Mean of all hackrf_sweep passes in `seconds`; a spur is a narrow peak at
    least threshold_db above the local floor (running median over ~10 MHz)."""
    s = HackRFSweep(f_lo, f_hi, bin_hz, gains)
    s.start()
    try:
        time.sleep(seconds)
        frame = s.take()
        sweeps = s.sweeps
    finally:
        s.close()
    p = frame.power_db.astype(float)
    p = np.where(np.isnan(p), np.nanmedian(p), p)
    win = max(11, int(10e6 / bin_hz) | 1)
    local = median_filter(p, size=win, mode="nearest")
    excess = p - local
    spurs, i = [], 0
    while i < len(p):
        if excess[i] >= threshold_db:
            j = i
            while j + 1 < len(p) and excess[j + 1] >= threshold_db:
                j += 1
            k = i + int(np.argmax(excess[i:j + 1]))
            spurs.append({"freq_hz": float(frame.freqs_hz[k]), "height_db": round(float(excess[k]), 1),
                          "width_hz": float((j - i + 1) * frame.bin_width_hz)})
            i = j + 1
        else:
            i += 1
    return {"f_lo_hz": f_lo, "f_hi_hz": f_hi, "bin_hz": frame.bin_width_hz, "sweeps": sweeps,
            "gains": [gains.amp, gains.lna_db, gains.vga_db], "threshold_db": threshold_db,
            "spurs": sorted(spurs, key=lambda s: -s["height_db"]),
            "_full": (frame.freqs_hz, p, local)}


# --- 4. is a carrier fixed, or tied to the LO? ---------------------------------------

def _band_power(f_abs, p_db, f0, half_bw):
    m = np.abs(f_abs - f0) <= half_bw
    return float(10 * np.log10(np.sum(10 ** (p_db[m] / 10))))


def check_carrier(rx: Rx, f_hz: float, gains: Gains, fs: float = 2e6, shifts=(-300e3, 300e3),
                  n: int = 1_000_000, nfft: int = 8192, search_hz: float = 20e3) -> dict:
    """Tune the LO to two different places near f_hz. A carrier that stays at the
    same absolute frequency is really at f_hz (a clock harmonic inside the HackRF
    or a real signal); one that moves with the LO is an artifact of the receiver."""
    out = []
    for s in shifts:
        lo = f_hz + s
        x, _ = capture(rx, lo, fs, gains, n)
        f, p, _ = welch_power(remove_dc(x), fs, nfft)
        fa = f + lo
        floor = float(np.median(p))
        m = np.abs(fa - f_hz) <= search_hz
        k = int(np.argmax(np.where(m, p, -999)))
        out.append({"lo_hz": lo, "peak_hz": float(fa[k]), "above_floor_db": round(float(p[k] - floor), 1)})
    fixed = all(o["above_floor_db"] > 6 for o in out) and abs(out[0]["peak_hz"] - out[1]["peak_hz"]) < 3 * fs / nfft
    return {"f_hz": f_hz, "present_at_both_lo": all(o["above_floor_db"] > 6 for o in out),
            "fixed_frequency": fixed, "captures": out}


# --- 5. image rejection (needs a real signal: antenna on) ---------------------------

def image_rejection(rx: Rx, f_signal: float, offsets, gains: Gains, fs: float = 10e6, shift: float = 0.7e6,
                    signal_bw: float = 150e3, n: int = 2_000_000, nfft: int = 4096) -> list[dict]:
    """A receiver with imperfect I/Q balance shows every signal a second time,
    mirrored around the LO, weaker by the image rejection ratio (IRR).

    Capture A: LO = f - d, so the signal is at +d and its image at f - 2d.
    Capture B: LO moved by `shift`, so the image moves away but whatever really
    sits at f - 2d (the FM band is crowded) stays. Power at f - 2d in A minus B
    is the image. If A is not clearly above B, the image is below what we can
    see and only a lower bound for the IRR is given."""
    rows = []
    for d in offsets:
        res = {}
        for name, lo in (("A", f_signal - d), ("B", f_signal - d - shift)):
            x, _ = capture(rx, lo, fs, gains, n)
            f, p, _ = welch_power(remove_dc(x), fs, nfft)
            fa = f + lo
            res[name] = {"sig": _band_power(fa, p, f_signal, signal_bw / 2),
                         "mirror": _band_power(fa, p, f_signal - 2 * d, signal_bw / 2)}
        a, b = 10 ** (res["A"]["mirror"] / 10), 10 ** (res["B"]["mirror"] / 10)
        sig = res["A"]["sig"]
        seen = a > 1.26 * b                           # > 1 dB above what is really there
        image_db = 10 * np.log10(a - b) if seen else 10 * np.log10(0.26 * b)
        rows.append({"offset_hz": d, "signal_db": round(sig, 1),
                     "signal_db_at_B_offset": round(res["B"]["sig"], 1), "b_offset_hz": d + shift,
                     "mirror_A_db": round(res["A"]["mirror"], 1), "mirror_B_db": round(res["B"]["mirror"], 1),
                     "image_seen": bool(seen), "irr_db": round(float(sig - image_db), 1),
                     "irr_is_lower_bound": not seen})
    return rows

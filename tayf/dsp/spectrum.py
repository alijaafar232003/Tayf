"""Spectrum estimation (Welch PSD) and waterfall rows.

Welch: split the block into overlapping windowed segments of nfft samples, FFT each,
average the powers. Averaging K segments does not lower the noise floor's mean, it
lowers its fluctuation (by about 1/sqrt(K)), which makes weak signals easier to see.

Scaling: a full-scale complex tone (|x| = 1) reads 0 dBFS at its peak, so a tone
reads its true power. Noise spreads over the span, so each bin holds only the
noise inside one resolution bandwidth: noise per bin = total noise * RBW / fs.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core.models import IQBlock, SpectrumFrame

EPS = 1e-20


def rbw_hz(fs: float, nfft: int, window: str = "hann") -> float:
    """Resolution bandwidth = equivalent noise bandwidth of one FFT bin."""
    w = signal.get_window(window, nfft)
    return fs * np.sum(w ** 2) / np.sum(w) ** 2


def welch_power(x: np.ndarray, fs: float, nfft: int = 2048, overlap: float = 0.5,
                window: str = "hann") -> tuple[np.ndarray, np.ndarray, int]:
    """Return (baseband freqs ascending, power per bin in dBFS, number of averages)."""
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    noverlap = int(nfft * overlap)
    f, p = signal.welch(x, fs=fs, window=window, nperseg=nfft, noverlap=noverlap,
                        return_onesided=False, scaling="spectrum", detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    step = nfft - noverlap
    n_avg = max(1, (len(x) - noverlap) // step)
    return f, 10 * np.log10(p + EPS), n_avg


def spectrum_frame(block: IQBlock, nfft: int = 2048, overlap: float = 0.5,
                   window: str = "hann") -> SpectrumFrame:
    f, p, n_avg = welch_power(block.samples, block.sample_rate_hz, nfft, overlap, window)
    return SpectrumFrame(
        freqs_hz=f + block.center_freq_hz,
        power_dbfs=p.astype(np.float32),
        rbw_hz=rbw_hz(block.sample_rate_hz, nfft, window),
        center_freq_hz=block.center_freq_hz,
        time=block.start_time,
        offset_s=block.start_offset_s,
        n_averages=n_avg,
    )


def reduce_bins(power_db: np.ndarray, n_out: int) -> np.ndarray:
    """Shrink a spectrum to n_out display bins by max-hold, so narrow signals
    stay visible when there are more bins than pixels."""
    n = power_db.shape[-1]
    if n <= n_out:
        return power_db
    factor = int(np.ceil(n / n_out))
    pad = factor * int(np.ceil(n / factor)) - n
    p = np.pad(power_db, [(0, 0)] * (power_db.ndim - 1) + [(0, pad)], mode="edge")
    return p.reshape(*p.shape[:-1], -1, factor).max(axis=-1)


def overview_rows(read_at, duration_s: float, fs: float, rows: int = 300,
                  nfft: int = 1024, averages: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A whole-recording waterfall for the timeline: `rows` spectra at evenly
    spaced times, each averaging `averages` segments. Returns (times_s, power[rows, nfft]).

    It samples the recording rather than covering every sample, so a burst
    shorter than the gap between rows can be missed; the playback waterfall
    covers everything."""
    n = nfft + (averages - 1) * nfft // 2       # 50 % overlap
    times = np.linspace(0, max(duration_s - n / fs, 0), rows)
    out = np.empty((rows, nfft), dtype=np.float32)
    for i, t in enumerate(times):
        _, p, _ = welch_power(read_at(t, n).samples, fs, nfft)
        out[i] = p
    return times, out

"""Signal Inspector: measure one signal from its I/Q.

Steps:
  1. channelize: shift the signal to 0 Hz and decimate to a narrow channel around it
     (sample rate >= 3x its width), so everything after works on that signal alone;
  2. spectrum of the channel: noise level from its quietest bins, centre (power-
     weighted), occupied bandwidth (99 % of the signal power, the ITU definition),
     -26 dB bandwidth, channel power and SNR;
  3. bursts: the envelope against the noise level (on at +6 dB, off at +3 dB):
     count, lengths, gaps, repeat period, duty cycle; or "continuous";
  4. instantaneous frequency (the phase step between samples): shows FM and FSK
     directly; its histogram during bursts shows the FSK levels;
  5. symbol rate and constellation, only where they make sense: a linear digital
     signal (PSK/QAM) has a spectral line at its symbol rate in |x|^2; then the
     signal is resampled at the symbol instants and its residual rotation removed.
     Constant-envelope signals (FM, FSK) have no such line: no constellation.

Strength is in dBFS (relative to the ADC's full scale); dBm needs calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage

from .detect import fcme, threshold_factor
from scipy import signal as sps

MIN_CHANNEL_HZ = 20e3           # narrowest channel kept (tones, narrow bursts)
ON_DB, OFF_DB = 6.0, 3.0        # burst detection on the envelope (over the noise)
ENV_WIN = 4                     # envelope smoothing (channel samples; the channel is >= 3x the signal's width)
MIN_RUN = 3                     # bursts or gaps shorter than this many samples are noise
CONST_ENV_CV = 0.15             # |x| varying less than this (std/mean) = constant envelope (FM, FSK)
MAX_POINTS = 2000               # longest array sent to the dashboard per view


@dataclass
class Bursts:
    continuous: bool
    count: int = 0
    duty: float = 1.0
    length_ms: dict = field(default_factory=dict)    # min / median / max
    gap_ms: dict = field(default_factory=dict)
    period_ms: float | None = None                    # median start-to-start
    on_power_dbfs: float | None = None                # power while on (the average includes the gaps)
    packets: int = 0                                  # groups of bursts split by long silences (key fobs)
    packet_ms: dict = field(default_factory=dict)
    packet_period_ms: float | None = None


@dataclass
class Inspection:
    f_center_hz: float
    offset_hz: float                # centre minus the frequency we were asked to look at
    obw_hz: float                   # 99 % occupied bandwidth
    bw26_hz: float                  # -26 dB bandwidth
    power_dbfs: float               # signal power in the channel (noise removed)
    noise_dbfs_hz: float            # noise density around it
    snr_db: float                   # signal power over the noise in the occupied bandwidth
    fs_channel_hz: float
    duration_s: float
    bursts: Bursts
    fsk_levels_hz: list[float]      # instantaneous-frequency clusters during bursts (FSK hint)
    symbol_rate_hz: float | None    # from the |x|^2 spectral line (linear modulations)
    constellation_order: int | None  # residual-rotation fit: 2 (BPSK-like) or 4 (QPSK-like), a hint
    notes: list[str]
    views: dict                     # downsampled arrays for the dashboard

    def to_dict(self) -> dict:
        return asdict(self)


def _thin(a: np.ndarray, n: int = MAX_POINTS) -> np.ndarray:
    return a[:: max(1, len(a) // n)][:n]


def channelize(x: np.ndarray, fs: float, offset_hz: float, width_hz: float) -> tuple[np.ndarray, float]:
    """Shift offset_hz to 0 Hz and decimate so the channel is >= 3x the signal's width.

    Done in the frequency domain: one FFT of the whole capture, keep the bins around
    the signal (that is the shift), inverse FFT of just those (that is the decimation).
    Much faster than mixing + a long decimation filter on millions of samples; the
    outer 10 % of the kept band is tapered so the cut does not ring."""
    n = len(x)
    want = max(3 * width_hz, MIN_CHANNEL_HZ)
    dec = max(1, int(fs // want))
    m = n // dec
    X = sp_fft.fft(x, workers=-1)
    k0 = int(round(offset_hz / fs * n))
    idx = (k0 + np.arange(-(m // 2), m - m // 2)) % n
    taper = sps.windows.tukey(m, 0.2)
    Y = np.fft.ifftshift(X[idx] * taper)
    y = sp_fft.ifft(Y, workers=-1) * (m / n)
    return y.astype(np.complex64), fs / dec


def own_signal(f: np.ndarray, sig: np.ndarray, width_hz: float) -> np.ndarray:
    """Of the signal bins in the channel, keep the pieces that reach into the requested
    width (with 50 % margin) around 0 Hz. One signal can be several pieces (an FM
    comb of Bessel lines, the two tones of FSK, OOK side-lobes); a neighbour in a
    crowded band lies wholly outside and must not be added into the measurement."""
    lab, n = ndimage.label(sig)
    if n <= 1:
        return sig
    window = np.abs(f) <= max(0.75 * width_hz, 2 * (f[1] - f[0]))
    keep = np.unique(lab[window & (lab > 0)])
    return np.isin(lab, keep)


def _bandwidths(f: np.ndarray, p: np.ndarray, noise: float, sig: np.ndarray) -> tuple[float, float, float, float]:
    """(centre, 99 % OBW, -26 dB BW, signal power) from a channel PSD (linear, per Hz).
    Only the bins FCME judged to be signal count: summing (power - noise) over every
    bin would add up the noise's own fluctuations and inflate the bandwidth."""
    df = f[1] - f[0]
    s = np.where(sig, np.clip(p - noise, 0, None), 0.0)
    total = s.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    c = np.cumsum(s) / total
    lo, hi = np.interp([0.005, 0.995], c, f)
    centre = float(np.sum(f * s) / total)
    pk = np.where(sig, p, 0.0)
    above = np.where(sig & (p >= pk.max() * 10 ** (-26 / 10)))[0]
    bw26 = float((above[-1] - above[0] + 1) * df)
    return centre, float(hi - lo), bw26, float(total * df)


def _bursts(env: np.ndarray, fs_env: float, noise_power: float, min_run: int = MIN_RUN) -> tuple[Bursts, np.ndarray]:
    on_thr, off_thr = noise_power * 10 ** (ON_DB / 10), noise_power * 10 ** (OFF_DB / 10)
    # Hysteresis without a loop: every stretch above +3 dB that reaches +6 dB somewhere is "on"
    lab, _ = ndimage.label(env > off_thr)
    hit = np.unique(lab[env > on_thr])
    on = np.isin(lab, hit[hit > 0])
    # Edges at half power (-3 dB) of each burst's own level, not near the noise:
    # a strong burst's edges ramp down over several samples (filter transition), and a
    # threshold just above the noise would make every burst look longer.
    lab, nb = ndimage.label(on)
    if nb:
        # the burst's level: a high percentile, so a region that spans several bursts and
        # the gaps between them is still judged by the bursts
        level = ndimage.labeled_comprehension(env, lab, np.arange(1, nb + 1), lambda v: np.percentile(v, 90),
                                              float, 0.0)
        cut = np.r_[0.0, level / 2][lab]
        on &= env >= cut
    # Blips and dips shorter than the envelope can resolve are noise, not bursts or gaps
    on = ndimage.binary_closing(on, np.ones(min_run)) if on.any() else on
    on = ndimage.binary_opening(on, np.ones(min_run)) if on.any() else on
    duty = float(on.mean())
    on_power = round(float(10 * np.log10(env[on].mean() + 1e-30)), 1) if on.any() else None
    # Continuous or not is judged on a 1 ms average: a weak continuous signal flickers
    # around the threshold sample by sample, but its average stays clearly above the noise.
    w = max(1, int(1e-3 * fs_env))
    slow = np.convolve(env, np.ones(w) / w, mode="same")
    if duty > 0.97 or (slow > off_thr).mean() > 0.97:
        return Bursts(True, 0, duty, on_power_dbfs=on_power), np.ones(len(env), bool)
    edges = np.diff(on.astype(int))
    starts, ends = np.where(edges == 1)[0] + 1, np.where(edges == -1)[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        ends = np.r_[ends, len(on)]
    # ignore a burst cut by the start or end of the capture when measuring lengths
    full = [(a, b) for a, b in zip(starts, ends) if a > 0 and b < len(on)]
    ms = lambda v: round(float(v) / fs_env * 1e3, 3)
    stat = lambda a: {"min": ms(np.min(a)), "median": ms(np.median(a)), "max": ms(np.max(a))} if len(a) else {}
    lengths = [b - a for a, b in full]
    gaps = [s - e for e, s in zip(ends[:-1], starts[1:])]
    period = ms(np.median(np.diff(starts))) if len(starts) > 2 else None
    # Packets: bursts separated by silences much longer than the typical gap
    packets, packet_len, packet_period = 0, {}, None
    if len(gaps) >= 2:
        long_gap = max(5 * np.median(gaps), 2 * min_run)
        splits = [i + 1 for i, g in enumerate(gaps) if g > long_gap]
        if splits:
            bounds = [0] + splits + [len(starts)]
            groups = [(starts[a], ends[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]
            packets = len(groups)
            whole = [(a, b) for a, b in groups if a > 0 and b < len(on)]
            packet_len = stat([b - a for a, b in whole])
            packet_period = ms(np.median(np.diff([a for a, _ in groups]))) if len(groups) > 2 else None
    return Bursts(False, int(len(starts)), duty, stat(lengths), stat(gaps), period, on_power,
                  packets, packet_len, packet_period), on


def _fsk_levels(inst_f: np.ndarray, on: np.ndarray, width_hz: float) -> list[float]:
    """Clusters of the instantaneous frequency while the signal is on (2 or more = FSK hint)."""
    v = inst_f[on[: len(inst_f)]]
    if len(v) < 50:
        return []
    lim = max(width_hz / 2, 1.0)
    hist, edges = np.histogram(np.clip(v, -lim, lim), bins=64, range=(-lim, lim))
    hist = np.convolve(hist, [1, 2, 1], mode="same")
    peaks, _ = sps.find_peaks(hist, height=0.15 * hist.max(), distance=6, prominence=0.1 * hist.max())
    levels = [round(float((edges[i] + edges[i + 1]) / 2), 1) for i in peaks]
    # levels closer than a quarter of the signal's width are one level smeared by noise
    return levels if len(levels) >= 2 and min(np.diff(levels)) > 0.25 * width_hz else levels[:1]


def _symbol_rate(y: np.ndarray, fs: float, obw: float) -> float | None:
    """The spectral line of |y|^2 at the symbol rate (linear modulations only)."""
    e = np.abs(y) ** 2
    e = e - e.mean()
    n = 1 << int(np.ceil(np.log2(len(e))))
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e)), n))
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f > 0.25 * obw) & (f < 1.3 * obw) & (f < 0.45 * fs)
    if band.sum() < 8:
        return None
    k = np.argmax(np.where(band, spec, 0))
    line_db = 20 * np.log10(spec[k] / (np.median(spec[band]) + 1e-20))
    return float(f[k]) if line_db > 15 else None


def _constellation(y: np.ndarray, fs: float, rs: float) -> tuple[np.ndarray, int]:
    """Sample at the symbol instants, then remove the residual frequency and phase.

    M-th power method: raising M-PSK symbols to the power M removes the data, leaving
    a pure tone at M x the residual frequency. The M (2 or 4) whose tone is sharpest
    is the order hint, and the tone's position gives the residual frequency."""
    sps_target = 8
    frac = Fraction(sps_target * rs / fs).limit_denominator(200)
    z = sps.resample_poly(y, frac.numerator, frac.denominator)
    phase = int(np.argmax([np.mean(np.abs(z[k::sps_target]) ** 2) for k in range(sps_target)]))
    pts = z[phase::sps_target]
    pts = pts / (np.sqrt(np.mean(np.abs(pts) ** 2)) + 1e-20)
    n = 1 << int(np.ceil(np.log2(len(pts)) + 2))
    best, best_ratio, best_f = 2, 0.0, 0.0
    for m in (2, 4):
        spec = np.abs(np.fft.fft(pts ** m, n))
        k = int(np.argmax(spec))
        ratio = spec[k] / (np.mean(spec) + 1e-20)
        if ratio > best_ratio:
            best, best_ratio, best_f = m, ratio, np.fft.fftfreq(n)[k] / m      # cycles per symbol
    pts = pts * np.exp(-2j * np.pi * best_f * np.arange(len(pts)))
    # what is left: a slow phase wander, followed with a running average of the M-th power
    ph = np.unwrap(np.angle(sps.lfilter(np.ones(64) / 64, 1, pts ** best))) / best
    pts = pts * np.exp(-1j * ph)
    return pts, best


def inspect(x: np.ndarray, fs: float, offset_hz: float, width_hz: float, f_tuned_hz: float = 0.0) -> Inspection:
    """Measure the signal at `offset_hz` (relative to the I/Q's centre), about `width_hz` wide."""
    y, fsc = channelize(x, fs, offset_hz, width_hz)
    notes = []
    # 2. spectrum of the channel
    nfft = int(min(4096, max(256, 2 ** np.floor(np.log2(len(y) / 8)))))
    f, p = sps.welch(y, fsc, nperseg=nfft, return_onesided=False, scaling="density", detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    flat = np.abs(f) < 0.4 * fsc                       # the channel's tapered edges are not noise
    f, p = f[flat], p[flat]
    k_eff = max(1.0, 0.95 * (2 * len(y) / nfft - 1))  # Welch segments (50 % overlap)
    noise, clean = fcme(p, threshold_factor(1e-3, k_eff))   # the detector's noise estimate, inside the channel
    mine = own_signal(f, ~clean, width_hz)
    if (~clean & ~mine).any():
        notes.append("Other signals share this channel; only the one at the requested frequency is measured.")
    centre, obw, bw26, sig_power = _bandwidths(f, p, noise, mine)
    if obw >= 0.8 * fsc:
        notes.append("The signal fills most of the channel: the bandwidths may be underestimated.")
    noise_in_obw = noise * max(obw, fsc / nfft)
    snr = 10 * np.log10(sig_power / noise_in_obw) if sig_power > 0 else -np.inf
    # 3. bursts, on the envelope of the signal's own band only (a narrow signal then
    #    competes with little noise; its edges can't be sharper than ~1/bandwidth anyway)
    cutoff = max(0.8 * obw, 2 * fsc / len(y), 500.0)
    Y = np.fft.fft(y * np.exp(-2j * np.pi * centre * np.arange(len(y)) / fsc))
    fa = np.abs(np.fft.fftfreq(len(y), 1 / fsc))
    # flat to the cutoff, then a cosine roll-off to 2x: a sharp edge would ring between bursts
    Y *= np.where(fa <= cutoff, 1.0, np.where(fa < 2 * cutoff, np.cos(np.pi / 2 * (fa - cutoff) / cutoff) ** 2, 0.0))
    yb = np.fft.ifft(Y)                                # the signal's band only, centred at 0 Hz
    env = np.convolve(np.abs(yb) ** 2, np.ones(ENV_WIN) / ENV_WIN, mode="same")
    noise_power = noise * 3 * cutoff                   # noise power in that band (flat 2x cutoff + roll-off)
    # the envelope cannot change faster than ~1/(2 x cutoff): shorter gaps are noise
    bursts, on = _bursts(env, fsc, noise_power, min_run=max(MIN_RUN, int(fsc / cutoff)))
    if snr < 10:
        notes.append(f"Weak signal ({snr:.0f} dB SNR): bandwidth and burst timing are less reliable.")
    # 4. instantaneous frequency (Hz from the centre), on the signal's own band: outside
    #    it there is only noise, which would scramble the phase steps
    inst = np.angle(yb[1:] * np.conj(yb[:-1])) * fsc / (2 * np.pi)
    inst = np.convolve(inst, np.ones(ENV_WIN) / ENV_WIN, mode="same")
    levels = _fsk_levels(inst, on, max(obw, 1e3))
    # 5. symbol rate + constellation (linear modulations only)
    yc = y * np.exp(-2j * np.pi * centre * np.arange(len(y)) / fsc)
    ys = yc[on] if not bursts.continuous and on.sum() > 1000 else yc
    mag = np.abs(ys)
    const_env = mag.std() / (mag.mean() + 1e-20) < CONST_ENV_CV and snr > 15
    rs = None if const_env else _symbol_rate(ys, fsc, max(obw, 1e3))
    const, order = (None, None)
    if rs:
        const, order = _constellation(ys, fsc, rs)
    elif const_env:
        notes.append("Constant envelope (FM, FSK, GMSK…): the information is in the frequency, so there is no "
                     "constellation to show; see the instantaneous frequency.")
    else:
        notes.append("No symbol-rate line in |x|^2 (analog, OFDM, noise-like or too weak): no constellation.")
    # views for the dashboard (downsampled)
    t = np.arange(len(y)) / fsc
    nseg = 128
    fspec, tspec, sxx = sps.spectrogram(y, fsc, nperseg=min(256, max(32, len(y) // nseg)), return_onesided=False,
                                        noverlap=0, mode="psd")
    sxx = np.fft.fftshift(sxx, axes=0)
    cols = sxx[:, :: max(1, sxx.shape[1] // nseg)][:, :nseg]
    first_on = int(np.argmax(on)) if on.any() else 0
    seg = slice(first_on, first_on + min(len(y), MAX_POINTS))      # raw I/Q from the first burst on
    views = {
        "spectrum": {"f_hz": _thin(f + offset_hz + f_tuned_hz, 512).tolist(),
                     "db": _thin(10 * np.log10(p + 1e-30), 512).round(2).tolist(),
                     "noise_db": round(10 * np.log10(noise), 2)},
        "spectrogram": {"db": np.round(10 * np.log10(cols.T + 1e-30), 1).tolist(),
                        "f_lo_hz": float(fspec.min() + offset_hz + f_tuned_hz), "f_hi_hz": float(fspec.max() + offset_hz + f_tuned_hz),
                        "t_s": float(t[-1])},
        "envelope": {"t_ms": _thin(t * 1e3).round(3).tolist(), "db": _thin(10 * np.log10(env + 1e-30)).round(2).tolist(),
                     "on_db": round(10 * np.log10(noise_power) + ON_DB, 2)},
        "iq": {"t_ms": (t[seg] * 1e3).round(4).tolist(), "i": y[seg].real.round(5).tolist(), "q": y[seg].imag.round(5).tolist()},
        "inst_freq": {"t_ms": _thin(t[:-1] * 1e3).round(3).tolist(), "hz": _thin(inst).round(1).tolist()},
        "constellation": None if const is None else {"i": _thin(const.real).round(3).tolist(), "q": _thin(const.imag).round(3).tolist()},
    }
    r = Inspection(f_center_hz=float(f_tuned_hz + offset_hz + centre), offset_hz=float(centre),
                      obw_hz=round(obw, 1), bw26_hz=round(bw26, 1),
                      power_dbfs=round(float(10 * np.log10(sig_power + 1e-30)), 1),
                      noise_dbfs_hz=round(float(10 * np.log10(noise)), 1), snr_db=round(float(snr), 1),
                      fs_channel_hz=fsc, duration_s=round(len(y) / fsc, 3), bursts=bursts, fsk_levels_hz=levels,
                      symbol_rate_hz=None if rs is None else round(rs, 1), constellation_order=order,
                      notes=notes, views=views)
    # The channel I/Q (centred on the requested frequency) is kept for saving, not in to_dict
    r.channel, r.channel_center_hz = y, f_tuned_hz + offset_hz
    return r

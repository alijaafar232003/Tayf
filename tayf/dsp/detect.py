"""Signal detection in one spectrum: FCME (forward consecutive mean excision).

The floor level is needed to say what is a signal, but averaging all bins does not
give it: the signals pull the average up. FCME starts from the quietest bins only,
takes their mean, sets a threshold a known factor above it, adds every bin under
the threshold to the noise set, and repeats until the set stops changing. The
bins left above the final threshold are signals.

The factor comes from the noise statistics: the power of one bin, averaged over K
FFTs, follows a Gamma(K) distribution around the true floor, so the threshold for
a chosen false-alarm probability is gamma.ppf(1 - pfa, K) / K.

FCME assumes a flat floor (decision D8). Before it runs:
  - the receiver's own floor shape (receiver profile: filter edges, DC bump) is
    divided out, when the spectrum comes from this HackRF at a profiled rate;
  - the DC bins at the LO are excluded;
  - a slow trend (a running 25th percentile over a quarter of the span) is taken
    out too: a tilted floor (antenna, environment) would otherwise put its high
    side above FCME's single threshold. Signals narrower than about 3/4 of that
    window barely move a low percentile;
and after the noise set is found, a local floor (a running median of the noise
bins) follows what is left.

Detections then need min_snr_db over the local floor (practical guard: with ~250
averages the statistical threshold alone is only ~1 dB, so floor ripples would
count), and their edges extend down to edge_snr_db (hysteresis).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter, percentile_filter, uniform_filter1d
from scipy.stats import gamma

from ..core.models import SpectrumFrame


@dataclass
class DetectorConfig:
    pfa: float = 1e-3               # false-alarm probability per bin (noise only)
    min_snr_db: float = 6.0         # a detection's peak must stand this far over the local floor
    edge_snr_db: float = 3.0        # its edges extend down to this
    merge_gap_hz: float = 0.0       # join detections closer than this
    split_db: float = 6.0           # split a detection at a dip this far below the peaks on both sides
    floor_window_frac: float = 1 / 32   # local floor: running median over this share of the span
    trend_window_frac: float = 1 / 4    # slow trend taken out before FCME (0 = off)
    dc_guard_hz: float = 2e3        # bins this close to the LO are not detected (DC spike;
    dc_guard_bins: float = 3.0      # whichever is wider: the residue after DC removal spreads ~3 bins)
    start_frac: float = 0.25        # FCME starts from this share of the quietest bins
    welch_overlap_k: float = 0.95   # independent averages per Welch average (Hann, 50 % overlap)


@dataclass
class Detection:
    f_lo_hz: float
    f_hi_hz: float
    f_center_hz: float              # power-weighted centre
    bandwidth_hz: float             # between the edges (edge_snr_db)
    peak_dbfs: float                # strongest bin (dBFS per bin)
    snr_db: float                   # strongest bin over the local floor
    n_bins: int
    spur: bool = False              # at one of the HackRF's own carriers (receiver profile)

    def to_dict(self) -> dict:
        return asdict(self)


def threshold_factor(pfa: float, k_eff: float) -> float:
    """Power ratio over the noise mean that averaged noise exceeds with probability pfa."""
    k = max(k_eff, 1.0)
    return float(gamma.ppf(1 - pfa, a=k, scale=1 / k))


def fcme(x: np.ndarray, factor: float, start_frac: float = 0.25, valid: np.ndarray | None = None,
         max_iter: int = 30) -> tuple[float, np.ndarray]:
    """FCME on linear powers x. Returns (noise mean, mask of the noise bins).
    `valid` marks bins that may be used at all (e.g. not the DC bins)."""
    valid = np.ones(len(x), bool) if valid is None else valid
    xs = np.sort(x[valid])
    clean = x <= xs[max(1, int(len(xs) * start_frac)) - 1]
    clean &= valid
    for _ in range(max_iter):
        mu = float(x[clean].mean())
        new = (x <= factor * mu) & valid
        if np.array_equal(new, clean):
            break
        clean = new
    return float(x[clean].mean()), clean


def shape_for(profile: dict | None, frame: SpectrumFrame) -> np.ndarray | None:
    """The receiver's floor shape (dB, per bin of this frame), if the profile has
    the frame's sample rate. Frames from other receivers or rates: None."""
    if not profile:
        return None
    df = frame.freqs_hz[1] - frame.freqs_hz[0]
    fs = df * len(frame.freqs_hz)
    s = next((s for s in profile.get("span_shape", []) if abs(s["fs_hz"] - fs) < 0.01 * fs), None)
    if s is None:
        return None
    off = (frame.freqs_hz - frame.center_freq_hz) / fs
    return np.interp(off, s["offset_frac"], s["shape_db"])


def local_floor(p_db: np.ndarray, clean: np.ndarray, window: int) -> np.ndarray:
    """Floor per bin from the noise bins only: gaps (signals) are filled by
    interpolation, then a running median follows slow changes."""
    idx = np.arange(len(p_db))
    filled = np.interp(idx, idx[clean], p_db[clean]) if clean.any() else p_db.copy()
    return median_filter(filled, size=max(3, window | 1), mode="nearest")


def detect(frame: SpectrumFrame, cfg: DetectorConfig | None = None, shape_db: np.ndarray | None = None,
           spurs_hz: list[float] = (), lo_hz: float | None = None) -> tuple[list[Detection], dict]:
    """Detections in one spectrum, plus the intermediate results (for plots and the notebook).

    shape_db: receiver floor shape to divide out (shape_for()); lo_hz: where the DC
    spike sits (None: no DC exclusion, e.g. scans); spurs_hz: the HackRF's own carriers."""
    cfg = cfg or DetectorConfig()
    f, p_db = frame.freqs_hz, frame.power_dbfs.astype(float)
    df = f[1] - f[0]
    valid = ~np.isnan(p_db)
    p_db = np.where(valid, p_db, np.nanmedian(p_db))
    flat_db = p_db - shape_db if shape_db is not None else p_db
    if lo_hz is not None:
        valid &= np.abs(f - lo_hz) > max(cfg.dc_guard_hz, cfg.dc_guard_bins * df)
    k_eff = max(1.0, frame.n_averages * cfg.welch_overlap_k)
    factor = threshold_factor(cfg.pfa, k_eff)
    trend = np.zeros_like(flat_db)
    if cfg.trend_window_frac > 0:
        w = max(5, int(len(f) * cfg.trend_window_frac))
        trend = uniform_filter1d(percentile_filter(np.where(valid, flat_db, np.median(flat_db)), 25, size=w,
                                                   mode="nearest"), size=max(3, w // 4), mode="nearest")
        trend -= np.median(trend)
    x = 10 ** ((flat_db - trend) / 10)
    _, clean = fcme(x, factor, cfg.start_frac, valid)
    floor_db = local_floor(flat_db, clean, int(len(f) * cfg.floor_window_frac))
    snr = np.where(valid, flat_db - floor_db, 0.0)
    thr_db = max(10 * np.log10(factor), cfg.min_snr_db)

    # Cores above thr_db, grown to their edges (edge_snr_db), then merged
    above_edge = snr >= cfg.edge_snr_db
    det: list[list[int]] = []
    i, n = 0, len(f)
    while i < n:
        if snr[i] >= thr_db:
            a = i
            while a > 0 and above_edge[a - 1]:
                a -= 1
            b = i
            while b + 1 < n and above_edge[b + 1]:
                b += 1
            if det and (a - det[-1][1]) * df <= cfg.merge_gap_hz + df:
                det[-1][1] = max(det[-1][1], b)
            else:
                det.append([a, b])
            i = b + 1
        else:
            i += 1

    # Neighbours whose skirts touch come out as one detection: split at deep dips.
    sm = np.convolve(snr, np.ones(3) / 3, mode="same")

    def split(a: int, b: int) -> list[list[int]]:
        if b - a < 4:
            return [[a, b]]
        seg = sm[a:b + 1]
        best, best_depth = None, cfg.split_db
        for m in range(1, len(seg) - 1):
            if seg[m] <= seg[m - 1] and seg[m] <= seg[m + 1]:
                depth = min(seg[:m].max(), seg[m + 1:].max()) - seg[m]
                if depth >= best_depth:
                    best, best_depth = m, depth
        if best is None:
            return [[a, b]]
        return split(a, a + best - 1) + split(a + best + 1, b)

    det = [p for a, b in det for p in split(a, b)]
    out = []
    for a, b in det:
        seg = slice(a, b + 1)
        w = 10 ** (p_db[seg] / 10)
        k = a + int(np.argmax(snr[seg]))
        fc = float(np.sum(f[seg] * w) / np.sum(w))
        spur = any(abs(fc - s) <= max(3 * df, 2e3) for s in spurs_hz)
        out.append(Detection(f_lo_hz=float(f[a] - df / 2), f_hi_hz=float(f[b] + df / 2), f_center_hz=fc,
                             bandwidth_hz=float((b - a + 1) * df), peak_dbfs=round(float(p_db[k]), 1),
                             snr_db=round(float(snr[k]), 1), n_bins=b - a + 1, spur=spur))
    info = {"flat_db": flat_db, "floor_db": floor_db, "clean": clean, "snr_db": snr, "valid": valid,
            "threshold_db": thr_db, "statistical_threshold_db": 10 * np.log10(factor), "k_eff": k_eff,
            "floor_dbfs": float(np.median(floor_db))}
    return out, info

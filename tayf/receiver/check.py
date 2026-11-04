"""Receiver Check: is the receiver set up well for this band, right now?

In a few seconds, on the live stream:
  1. try a handful of gain settings picked by the gain rule (from the receiver profile),
  2. measure each: ADC level, clipping, noise floor, and how far the strongest
     signals stand above the floor (SNR),
  3. suggest the setting that shows signals best without clipping or overload,
  4. explain the floor: how much of it comes from outside (antenna) versus from
     the HackRF itself (the profile says what the HackRF makes with nothing connected),
  5. list the HackRF's own carriers inside the span, so they are not taken for signals.

The gain rule (Friis, measured in notebook 02): enough LNA that the front end sets
the noise, then enough VGA that the noise clears the ADC floor; more only brings
clipping closer. The same functions will drive automatic gain control later.
"""
from __future__ import annotations

import time

import numpy as np

from ..dsp.dc import remove_dc
from ..dsp.spectrum import rbw_hz, welch_power
from ..session.manager import measure_health
from ..sources.radio import LNA_STEPS, VGA_STEPS, Gains
from .characterize import floor_model

ADC_MARGIN_DB = 6.0        # receiver noise this far above the ADC floor: the ADC adds < 1 dB
CLIP_LIMIT_PCT = 0.1       # more clipping than this disqualifies a setting
OVERLOAD_DB = 3.0          # floor rising this much more than the gain added = overload
PEAK_MIN_DB = 10.0         # a "signal" stands at least this far above the floor
SIGNAL_DB = 15.0           # judge by signals only if one stands this far out; otherwise by sensitivity
SAME_DB = 1.0              # settings within this of the best count as equally good


# --- the gain rule (from the receiver profile) -----------------------------------

def fit_for(profile: dict | None, fc_hz: float) -> dict | None:
    """The floor-vs-gain fit measured nearest to fc (in log frequency)."""
    fits = (profile or {}).get("floor_vs_gain", {}).get("fits", {})
    if not fits:
        return None
    best = min(fits, key=lambda k: abs(np.log10(float(k)) - np.log10(fc_hz)))
    return {**fits[best], "fc_hz": float(best)}


def receiver_floor(fit: dict, lna: int, vga: int) -> float:
    """What the floor would be with nothing connected (dBFS/Hz)."""
    return float(floor_model(lna, vga, fit["adc_floor_dbfs_hz"], fit["n_vga_dbfs_hz"], fit["n_front_dbfs_hz"]))


def vga_to_clear_adc(fit: dict | None, lna: int, margin_db: float = ADC_MARGIN_DB) -> int:
    """Least VGA gain that lifts the receiver's own noise margin_db above the ADC floor."""
    if fit is None:
        return 20
    rx_at_0 = 10 * np.log10(10 ** (fit["n_vga_dbfs_hz"] / 10) + 10 ** ((fit["n_front_dbfs_hz"] + lna) / 10))
    need = fit["adc_floor_dbfs_hz"] + margin_db - rx_at_0
    return next((v for v in VGA_STEPS if v >= need), VGA_STEPS[-1])


def candidate_gains(fit: dict | None) -> list[Gains]:
    """For each LNA step worth trying: the VGA the rule asks for, and 10 dB more
    (in case outside noise is weak and the extra helps)."""
    out = []
    for lna in (16, 24, 32, 40):
        v = vga_to_clear_adc(fit, lna)
        for vga in (v, min(v + 10, VGA_STEPS[-1])):
            g = Gains(False, lna, vga)
            if g not in out:
                out.append(g)
    return out


# --- measuring one setting --------------------------------------------------------

def measure(src, gains: Gains, n: int, nfft: int = 4096, timeout_s: float = 3.0) -> dict:
    """Switch to `gains` and measure a fresh block (the source drops samples taken
    before the change and while it settles)."""
    src.set_gains(gains)
    t_end, block = time.time() + timeout_s, None
    while block is None and time.time() < t_end:
        time.sleep(0.02)
        block = src.latest(n)
    if block is None:
        raise RuntimeError("No samples arrived from the radio")
    health = measure_health(block)
    fs = block.sample_rate_hz
    f, p, _ = welch_power(remove_dc(block.samples), fs, nfft)
    # Skip DC and the outer 15 % at each side: near the filter edges the HackRF shows
    # artifacts that do not grow with gain like real signals do (seen on 125 and 434 MHz).
    keep = (np.abs(f) > 0.02 * fs) & (np.abs(f) < 0.35 * fs)
    floor_bin = float(np.median(p[keep]))
    floor = floor_bin - 10 * np.log10(rbw_hz(fs, nfft))
    # Strongest peaks: local maxima PEAK_MIN_DB above the floor, at least 50 kHz apart
    idx = np.where(keep[1:-1] & (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > floor_bin + PEAK_MIN_DB))[0] + 1
    peaks = []
    for i in idx[np.argsort(p[idx])[::-1]]:
        if all(abs(f[i] - f[j]) > 50e3 for j in peaks):
            peaks.append(i)
        if len(peaks) == 5:
            break
    return {"amp": gains.amp, "lna_db": gains.lna_db, "vga_db": gains.vga_db, "total_db": gains.total_db,
            "level_dbfs": health["level_dbfs"], "clip_pct": health["clip_pct"],
            "floor_dbfs_hz": round(floor, 1),
            "peaks": [{"freq_hz": float(block.center_freq_hz + f[i]), "snr_db": round(float(p[i] - floor_bin), 1)}
                      for i in peaks],
            "_spec": (block.center_freq_hz + f, p, floor_bin)}     # for comparing settings (not returned)


# --- choosing ----------------------------------------------------------------------

def reference_signals(rows: list[dict], k: int = 3) -> list[float]:
    """The k strongest signals seen at any setting (frequencies, at least 50 kHz apart).
    Every setting is then judged on these same signals, so the comparison is fair."""
    ref = []
    for p in sorted((p for r in rows for p in r["peaks"]), key=lambda p: -p["snr_db"]):
        if p["snr_db"] >= SIGNAL_DB and all(abs(p["freq_hz"] - f) > 50e3 for f in ref):
            ref.append(p["freq_hz"])
        if len(ref) == k:
            break
    return ref


def snr_at(row: dict, freqs: list[float]) -> list[float]:
    """How far each reference signal stands above this setting's floor (dB)."""
    fa, p, floor_bin = row["_spec"]
    out = []
    for f in freqs:
        i = int(np.argmin(np.abs(fa - f)))
        out.append(round(float(p[max(0, i - 2):i + 3].max() - floor_bin), 1))
    return out


def choose(rows: list[dict], fit: dict | None) -> tuple[dict, list[str]]:
    """Pick the setting that shows signals best; mark clipping and overload."""
    for r in rows:
        r["rx_floor_dbfs_hz"] = round(receiver_floor(fit, r["lna_db"], r["vga_db"]), 1) if fit else None
        # Outside noise over the receiver's own floor (power difference, dB)
        r["outside_db"] = round(r["floor_dbfs_hz"] - r["rx_floor_dbfs_hz"], 1) if fit else None
        r["problems"] = []
        if r["clip_pct"] > CLIP_LIMIT_PCT:
            r["problems"].append("clipping")
    # Overload: with the same LNA, more VGA should raise the floor dB for dB. Between
    # LNA steps, the floor should not rise more than the gain did. Anything faster is
    # the receiver making its own distortion.
    by_gain = sorted(rows, key=lambda r: r["total_db"])
    for lo_r in by_gain:
        for hi_r in by_gain:
            dg = hi_r["total_db"] - lo_r["total_db"]
            if dg > 0 and hi_r["lna_db"] >= lo_r["lna_db"] and hi_r["vga_db"] >= lo_r["vga_db"] and \
                    hi_r["floor_dbfs_hz"] - lo_r["floor_dbfs_hz"] > dg + OVERLOAD_DB and "overload" not in hi_r["problems"]:
                hi_r["problems"].append("overload")
    ok = [r for r in rows if not r["problems"]] or sorted(rows, key=lambda r: r["clip_pct"])[:1]
    why = []
    ref = reference_signals(rows)
    for r in rows:
        r["ref_snr_db"] = snr_at(r, ref)
    score = lambda r: float(np.mean(r["ref_snr_db"]))
    if ref:
        best_snr = max(score(r) for r in ok)
        good = [r for r in ok if score(r) >= best_snr - SAME_DB]
        best = min(good, key=lambda r: r["total_db"])            # least gain = most headroom
        names = ", ".join(f"{f / 1e6:.3f}" for f in ref)
        why.append(f"Judged on the {len(ref)} strongest signal(s) ({names} MHz): this setting shows them "
                   f"{score(best):.0f} dB above the floor on average. Settings within {SAME_DB:.0f} dB of the best "
                   f"({best_snr:.0f} dB) count as equal; the one with the least gain keeps the most headroom.")
    else:
        # No signal to judge by: go for sensitivity. The floor referred to the input
        # (floor minus total gain) is the weakest signal level you could still see;
        # lower is better. It stops improving once outside noise dominates.
        sens = lambda r: r["floor_dbfs_hz"] - r["total_db"]
        top = min(sens(r) for r in ok)
        good = [r for r in ok if sens(r) <= top + SAME_DB]
        best = min(good, key=lambda r: r["total_db"])
        why.append(f"No signal stands {SIGNAL_DB:.0f} dB out here, so the choice is for sensitivity: the least gain "
                   f"whose noise floor, referred to the input, is within {SAME_DB:.0f} dB of the lowest "
                   f"({top:.1f} dBFS/Hz at 0 dB gain).")
    bad = [r for r in rows if r["problems"]]
    if bad:
        why.append("Ruled out: " + "; ".join(f"LNA {r['lna_db']}/VGA {r['vga_db']} ({', '.join(r['problems'])})" for r in bad) + ".")
    return best, why


def explain_floor(best: dict) -> str:
    o = best.get("outside_db")
    if o is None:
        return "No receiver profile yet (run notebook 02), so outside and receiver noise cannot be told apart."
    if o >= 6:
        return (f"The floor is {o:.0f} dB above what the HackRF makes by itself: it is noise picked up by the "
                "antenna. Sensitivity is set by the environment, so more gain would not reveal weaker signals.")
    if o >= 1:
        return (f"The floor is {o:.0f} dB above the HackRF's own: outside and receiver noise are both in it. "
                "A little more LNA (or a low-noise preamp at the antenna) could still help weak signals.")
    return ("The floor is the HackRF's own noise: the antenna brings in little here. Weak signals are limited by "
            "the receiver: more LNA, the amp, or a preamp at the antenna would help.")


def known_carriers(profile: dict | None, lo_hz: float, hi_hz: float) -> list[dict]:
    """The HackRF's own carriers (from the profile) that fall inside [lo_hz, hi_hz]."""
    return [c for c in (profile or {}).get("internal_carriers", []) if lo_hz <= c["freq_hz"] <= hi_hz]


def run_check(src, profile: dict | None, n: int | None = None) -> dict:
    """Try the candidate settings on a live source and return the findings.
    The source's gains are put back afterwards; applying the suggestion is up to the user."""
    fc, fs = src.tuned_hz, src.sample_rate_hz
    fit = fit_for(profile, fc)
    n = n or max(262_144, int(0.05 * fs))
    start = src.gains
    t0 = time.time()
    try:
        rows = [measure(src, g, n) for g in candidate_gains(fit)]
    finally:
        src.set_gains(start)
    best, why = choose(rows, fit)
    for r in rows:
        r.pop("_spec", None)
    lo, hi = src.lo_hz - fs / 2, src.lo_hz + fs / 2
    return {"tuned_hz": fc, "sample_rate_hz": fs, "seconds": round(time.time() - t0, 1),
            "profile": None if not profile else {"condition": profile.get("condition"), "created": profile.get("created"),
                                                 "fit_at_hz": fit["fc_hz"] if fit else None},
            "before": {"amp": start.amp, "lna_db": start.lna_db, "vga_db": start.vga_db},
            "suggestion": {"amp": best["amp"], "lna_db": best["lna_db"], "vga_db": best["vga_db"]},
            "best": best, "why": why, "floor": explain_floor(best),
            "known_carriers": known_carriers(profile, lo, hi), "rows": rows}

"""History: is this the same signal as one seen before, and has it changed?

A *signal profile* is one signal as known over time (e.g. "the station at 99.9 MHz",
"the key fob at 314.4 MHz"). Every inspection is an *observation*; it is linked to
the profile it matches, or starts a new one. The detector's closed tracks are kept
as light *candidate summaries* (D7) and count as sightings too.

Matching is by frequency and width only (no classifier yet):
  - centres closer than a tolerance: a quarter of the width, but at least the
    frequency error two cheap receivers can have (10 ppm) and 1 kHz;
  - widths within a factor of 3 (a 10 kHz signal next to a 200 kHz one is another signal).

Real and simulated data never match each other (origin "real" / "sim").
"""
from __future__ import annotations

PPM_TOL = 10e-6
MIN_TOL_HZ = 1e3
MAX_WIDTH_RATIO = 3.0
SIM_HARDWARE = ("synthetic", "Simulated HackRF")


def origin_of(hardware: str | None) -> str:
    return "sim" if (hardware or "") in SIM_HARDWARE else "real"


def tolerance_hz(f_hz: float, bw_hz: float) -> float:
    return max(0.25 * bw_hz, PPM_TOL * abs(f_hz), MIN_TOL_HZ)


def matches(f_hz: float, bw_hz: float, pf_hz: float, pbw_hz: float) -> bool:
    bw, pbw = max(bw_hz, 1e3), max(pbw_hz, 1e3)
    if not 1 / MAX_WIDTH_RATIO <= bw / pbw <= MAX_WIDTH_RATIO:
        return False
    return abs(f_hz - pf_hz) <= tolerance_hz(f_hz, max(bw, pbw))


def search_window(f_hz: float, bw_hz: float) -> tuple[float, float]:
    """A frequency range that surely contains every possible match (for the SQL pre-filter)."""
    w = tolerance_hz(f_hz, MAX_WIDTH_RATIO * max(bw_hz, 1e3))
    return f_hz - w, f_hz + w


def changes(prev: dict, now: dict) -> list[str]:
    """What differs between two observations of the same signal (empty = unchanged).
    Strength is compared only when the gains and antenna were the same."""
    out = []
    a, b = max(prev["obw_hz"], 1.0), max(now["obw_hz"], 1.0)
    if not 0.75 <= b / a <= 1 / 0.75:
        out.append(f"bandwidth {a / 1e3:.1f} → {b / 1e3:.1f} kHz")
    pc, nc = prev.get("continuous"), now.get("continuous")
    if pc is not None and nc is not None and pc != nc:
        out.append("was continuous, now bursty" if pc else "was bursty, now continuous")
    pp, np_ = prev.get("period_ms"), now.get("period_ms")
    if pp and np_ and not 0.8 <= np_ / pp <= 1.25:
        out.append(f"burst period {pp:.1f} → {np_:.1f} ms")
    if same_setup(prev.get("settings"), now.get("settings")):
        d = now["power_dbfs"] - prev["power_dbfs"]
        if abs(d) > 10:
            out.append(f"strength {d:+.0f} dB at the same gains")
    return out


def same_setup(s1: dict | None, s2: dict | None) -> bool:
    if not s1 or not s2:
        return False
    keys = ("amp_on", "lna_gain_db", "vga_gain_db", "antenna_id", "hardware")
    return all(s1.get(k) == s2.get(k) for k in keys) and s1.get("lna_gain_db") is not None

"""Automatic gain control (AGC): keep the ADC level in a good window, calmly.

Every gain change shifts the whole spectrum, splits recordings into segments and
disturbs the detector's floor estimate, so this AGC acts only when the level is
clearly wrong, one step at a time, and never flips back and forth:
  - too hot (clipping, or level above HIGH_DBFS): step down at once;
  - too low (level below LOW_DBFS, or the VGA below what the gain rule needs to lift
    the noise over the ADC floor while there is headroom: strong stations can fill the
    level while weak signals are still hidden by the ADC): step up, but only if it
    stayed low for a while and the last step down was long enough ago;
  - in between: leave the gain alone, whatever the signals do.

Which knob to turn follows the gain rule (notebook 02, Receiver Check): going up,
LNA first (it sets sensitivity), with just enough VGA to clear the ADC floor;
going down, VGA first (down to that minimum), then LNA.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..sources.radio import LNA_STEPS, VGA_STEPS, Gains
from .check import vga_to_clear_adc

LOW_DBFS = -35.0          # below this the 8-bit ADC barely sees the noise
HIGH_DBFS = -10.0         # above this, peaks of noise-like signals reach full scale
CLIP_PCT = 0.05           # clipping on more samples than this: step down at once
LOW_FOR_S = 2.0           # the level must stay low this long before stepping up...
FAR_LOW_FOR_S = 0.3       # ...but far too low (more than 10 dB under LOW_DBFS) needs no patience
STEP_EVERY_S = 1.0        # at most one step per second
NO_UP_AFTER_DOWN_S = 10.0 # after stepping down, wait this long before any step up
VGA_DOWN_DB = 6
VGA_UP_DB = 4


def _snap(v: float, steps: list[int]) -> int:
    return min(steps, key=lambda s: abs(s - v))


def step_down(g: Gains, fit: dict | None) -> Gains | None:
    """One step less gain: VGA first (down to what the rule needs), then LNA."""
    floor_vga = vga_to_clear_adc(fit, g.lna_db)
    if g.vga_db - VGA_DOWN_DB >= floor_vga:
        return Gains(g.amp, g.lna_db, _snap(g.vga_db - VGA_DOWN_DB, VGA_STEPS))
    if g.lna_db > LNA_STEPS[0]:
        lna = LNA_STEPS[LNA_STEPS.index(g.lna_db) - 1]
        return Gains(g.amp, lna, min(g.vga_db, max(vga_to_clear_adc(fit, lna), 0)))
    if g.vga_db > VGA_STEPS[0]:
        return Gains(g.amp, g.lna_db, _snap(max(g.vga_db - VGA_DOWN_DB, 0), VGA_STEPS))
    return None                                      # already at the minimum


def front_lna(fit: dict | None) -> int:
    """Least LNA step at which the front end, not the VGA stage, sets the noise
    (notebook 02: ~25-34 dB on this HackRF). Below it, LNA gain buys sensitivity."""
    need = fit["n_vga_dbfs_hz"] - fit["n_front_dbfs_hz"] if fit else 32
    return next((s for s in LNA_STEPS if s >= need), LNA_STEPS[-1])


def step_up(g: Gains, fit: dict | None) -> Gains | None:
    """One step more gain, in the order that helps sensitivity most:
    LNA up to where the front end dominates, then VGA up to what the rule asks for,
    then the rest of the LNA, then VGA."""
    nxt = lambda: LNA_STEPS[LNA_STEPS.index(g.lna_db) + 1]
    if g.lna_db < front_lna(fit):
        return Gains(g.amp, nxt(), g.vga_db)
    if g.vga_db < vga_to_clear_adc(fit, g.lna_db):
        return Gains(g.amp, g.lna_db, _snap(min(g.vga_db + VGA_UP_DB, VGA_STEPS[-1]), VGA_STEPS))
    if g.lna_db < LNA_STEPS[-1]:
        return Gains(g.amp, nxt(), g.vga_db)
    if g.vga_db < VGA_STEPS[-1]:
        return Gains(g.amp, g.lna_db, _snap(min(g.vga_db + VGA_UP_DB, VGA_STEPS[-1]), VGA_STEPS))
    return None                                      # already at the maximum (the amp stays manual)


@dataclass
class AGC:
    """The AGC's memory between frames. Call update() with each frame's health."""
    profile: dict | None = None      # receiver profile (notebook 02), loaded once
    fit: dict | None = None          # its floor-vs-gain fit nearest the tuned frequency
    fit_for_hz: float | None = None
    low_since: float | None = None
    last_step: float = -1e9
    last_down: float = -1e9
    changes: int = 0
    last_action: str = "watching"
    log: list[dict] = field(default_factory=list)

    def update(self, now: float, health: dict, g: Gains) -> tuple[Gains | None, str]:
        level, clip = health.get("level_dbfs"), health.get("clip_pct", 0.0)
        if level is None or now - self.last_step < STEP_EVERY_S:
            return None, ""
        new, why = None, ""
        if clip > CLIP_PCT or level > HIGH_DBFS:
            self.low_since = None
            new = step_down(g, self.fit)
            why = f"clipping {clip:.2f} %" if clip > CLIP_PCT else f"level {level:.1f} dBFS (too hot)"
            if new:
                self.last_down = now
        elif level < LOW_DBFS or (self.fit is not None and level < HIGH_DBFS - 10
                                  and g.vga_db < vga_to_clear_adc(self.fit, g.lna_db)):
            self.low_since = self.low_since if self.low_since is not None else now
            wait = FAR_LOW_FOR_S if level < LOW_DBFS - 10 else LOW_FOR_S
            if now - self.low_since >= wait and now - self.last_down >= NO_UP_AFTER_DOWN_S:
                new = step_up(g, self.fit)
                why = (f"level {level:.1f} dBFS for {now - self.low_since:.1f} s (too low)" if level < LOW_DBFS else
                       f"VGA below the {vga_to_clear_adc(self.fit, g.lna_db)} dB the gain rule needs (weak signals hidden by the ADC)")
        else:
            self.low_since = None                    # in the window: leave it alone
        if new is None:
            return None, ""
        self.last_step, self.low_since = now, None
        self.changes += 1
        self.last_action = f"LNA {g.lna_db}→{new.lna_db}, VGA {g.vga_db}→{new.vga_db}: {why}"
        self.log.append({"t": now, "from": [g.lna_db, g.vga_db], "to": [new.lna_db, new.vga_db], "why": why})
        del self.log[:-50]
        return new, self.last_action

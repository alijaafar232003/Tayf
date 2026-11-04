"""Tracker: links detections over time into tracks with stable IDs.

The detector sees one spectrum at a time; the tracker decides that the station
at 98.5 MHz now is the same one as a moment ago. Two detections are the same
signal when their frequency ranges overlap (or their centres are within half a
bandwidth): most signals stay put in frequency, so no motion model is needed.

Flicker is handled with counts, not single frames:
  - a new track is tentative; it is confirmed once seen in CONFIRM_M of the last
    CONFIRM_N frames in which its frequency was in view (one-frame blips never show);
  - a confirmed track survives gaps (bursts: a car key keeps its ID between presses);
  - it is closed after CLOSE_AFTER_S of silence, and its summary is handed over for
    the history (decision D7: a light summary of every candidate is saved).
Frames where the track's frequency is outside the span (after a retune) do not
count as misses.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detect import Detection

CONFIRM_M, CONFIRM_N = 3, 5
CLOSE_AFTER_S = 5.0
EMA = 0.2                           # smoothing of the reported centre and edges
FRAGMENT_DB = 6.0                   # a piece this much weaker than an overlapping signal is part of it


@dataclass
class Track:
    id: int
    f_lo_hz: float
    f_hi_hz: float
    f_center_hz: float
    bandwidth_hz: float             # between the smoothed edges
    max_bandwidth_hz: float
    snr_db: float                   # latest
    max_snr_db: float
    peak_dbfs: float
    first_seen: float               # seconds (wall clock live; recording time for recordings)
    last_seen: float
    spur: bool = False
    confirmed: bool = False
    hits: int = 1                   # frames with a detection
    in_view: int = 1                # frames in which its frequency was in view
    recent: deque = field(default_factory=lambda: deque([True], maxlen=CONFIRM_N))
    present: bool = True            # detected in the latest frame

    @property
    def duty(self) -> float:
        """Share of the frames in view in which it was detected (1.0 = always on)."""
        return self.hits / max(self.in_view, 1)

    def to_dict(self) -> dict:
        return {"id": self.id, "f_lo_hz": self.f_lo_hz, "f_hi_hz": self.f_hi_hz,
                "f_center_hz": round(self.f_center_hz, 1), "bandwidth_hz": round(self.bandwidth_hz, 1),
                "max_bandwidth_hz": self.max_bandwidth_hz,
                "snr_db": self.snr_db, "max_snr_db": self.max_snr_db, "peak_dbfs": self.peak_dbfs,
                "first_seen": self.first_seen, "last_seen": self.last_seen, "spur": self.spur,
                "confirmed": self.confirmed, "present": self.present, "hits": self.hits,
                "duty": round(self.duty, 3)}

    def summary(self) -> dict:
        """The light summary saved when the track closes (D7)."""
        return {**self.to_dict(), "duration_s": round(self.last_seen - self.first_seen, 2)}


def _same(t: Track, d: Detection) -> bool:
    if d.f_lo_hz <= t.f_hi_hz and t.f_lo_hz <= d.f_hi_hz:
        return True
    return abs(d.f_center_hz - t.f_center_hz) <= max(t.bandwidth_hz, d.bandwidth_hz) / 2


class Tracker:
    def __init__(self, close_after_s: float = CLOSE_AFTER_S):
        self.close_after_s = close_after_s
        self.tracks: dict[int, Track] = {}
        self.closed: list[dict] = []        # summaries of closed confirmed tracks, not yet saved
        self._next_id = 1
        self._last_t: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self._last_t = None

    def update(self, detections: list[Detection], t: float, span: tuple[float, float]) -> list[tuple[Detection, int]]:
        """Feed one frame's detections (time t, frequency span of the frame).
        Returns each detection with its track ID."""
        if self._last_t is not None and t < self._last_t:
            self.close_all(self._last_t)    # time went backwards (seek in a recording): start over
        self._last_t = t
        out, matched = [], set()
        for d in sorted(detections, key=lambda d: -d.snr_db):    # strongest first claim their tracks
            cands = [tr for tr in self.tracks.values() if _same(tr, d)]
            free = [tr for tr in cands if tr.id not in matched]
            # A much weaker piece overlapping a signal already matched this frame is a fragment of
            # it (an FM station whose edge dipped under the threshold): same track, nothing new.
            # A piece of similar strength is a neighbour and gets its own track.
            frag = [tr for tr in self.tracks.values() if tr.id in matched and d.snr_db < tr.snr_db - FRAGMENT_DB
                    and abs(d.f_center_hz - tr.f_center_hz) <= 0.75 * tr.max_bandwidth_hz + d.bandwidth_hz / 2]
            if not free and frag:
                out.append((d, frag[0].id))
                continue
            tr = min(free, key=lambda tr: abs(tr.f_center_hz - d.f_center_hz)) if free else None
            if tr is None:
                tr = Track(self._next_id, d.f_lo_hz, d.f_hi_hz, d.f_center_hz, d.bandwidth_hz, d.bandwidth_hz,
                           d.snr_db, d.snr_db, d.peak_dbfs, t, t, spur=d.spur)
                self._next_id += 1
                self.tracks[tr.id] = tr
            else:
                tr.hits += 1
                tr.in_view += 1
                tr.recent.append(True)
                # Smoothed, not a union: a union keeps widening until it swallows the neighbours
                tr.f_center_hz += EMA * (d.f_center_hz - tr.f_center_hz)
                tr.f_lo_hz += EMA * (d.f_lo_hz - tr.f_lo_hz)
                tr.f_hi_hz += EMA * (d.f_hi_hz - tr.f_hi_hz)
                tr.bandwidth_hz = tr.f_hi_hz - tr.f_lo_hz
                tr.max_bandwidth_hz = max(tr.max_bandwidth_hz, d.bandwidth_hz)
                tr.snr_db, tr.peak_dbfs, tr.last_seen = d.snr_db, d.peak_dbfs, t
                tr.max_snr_db = max(tr.max_snr_db, d.snr_db)
                tr.spur = tr.spur or d.spur
            tr.present = True
            matched.add(tr.id)
            out.append((d, tr.id))
        lo, hi = span
        for tr in list(self.tracks.values()):
            if tr.id in matched:
                pass
            elif lo <= tr.f_center_hz <= hi:            # in view but not detected: a miss
                tr.present = False
                tr.in_view += 1
                tr.recent.append(False)
            else:
                tr.present = False                     # out of view: no evidence either way
            if not tr.confirmed and sum(tr.recent) >= CONFIRM_M:
                tr.confirmed = True
            if not tr.confirmed and len(tr.recent) == CONFIRM_N and sum(tr.recent) < CONFIRM_M:
                del self.tracks[tr.id]                 # a blip that never confirmed
            elif t - tr.last_seen > self.close_after_s:
                self._close(tr)
        return out

    def _close(self, tr: Track) -> None:
        del self.tracks[tr.id]
        if tr.confirmed:
            self.closed.append(tr.summary())

    def close_all(self, t: float | None = None) -> None:
        for tr in list(self.tracks.values()):
            self._close(tr)
        self._last_t = None

    def take_closed(self) -> list[dict]:
        out, self.closed = self.closed, []
        return out

    def table(self) -> list[dict]:
        """Confirmed tracks, strongest first."""
        return [tr.to_dict() for tr in sorted(self.tracks.values(), key=lambda tr: -tr.max_snr_db) if tr.confirmed]

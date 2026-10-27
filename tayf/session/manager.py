"""Session manager: exactly one active session (recording, live or scan).

It owns the source (and, from Phase 2, the HackRF), so two parts of the app can
never fight over the receiver. Everything downstream asks the manager for frames.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from ..core.models import SpectrumFrame
from ..dsp.spectrum import overview_rows, spectrum_frame
from ..sources.recording import RecordingSource
from ..store.db import Store

MAX_SAMPLES_PER_FRAME = 262_144     # caps CPU per frame at high sample rates


@dataclass
class Playback:
    playing: bool = False
    speed: float = 1.0          # 1.0 = real time
    nfft: int = 2048
    fps: float = 20.0           # display frames per second


@dataclass
class ActiveSession:
    id: int
    mode: str
    source: RecordingSource
    recording_id: int | None = None
    playback: Playback = field(default_factory=Playback)
    overview: tuple[np.ndarray, np.ndarray] | None = None


class SessionManager:
    def __init__(self, store: Store):
        self.store = store
        self.active: ActiveSession | None = None
        self._lock = threading.RLock()
        store.end_open_sessions()

    # --- lifecycle ------------------------------------------------------
    def open_recording(self, recording_id: int) -> ActiveSession:
        row = self.store.recording(recording_id)
        if row is None:
            raise KeyError(f"No recording with id {recording_id}")
        with self._lock:
            self.close()
            source = RecordingSource(row["meta_path"])
            sid = self.store.start_session("recording", source.settings, recording_id)
            self.active = ActiveSession(sid, "recording", source, recording_id)
            return self.active

    def close(self) -> None:
        with self._lock:
            if self.active:
                self.active.source.close()
                self.store.end_session(self.active.id)
                self.active = None

    def status(self) -> dict:
        with self._lock:
            a = self.active
            if a is None:
                return {"active": False}
            s, src, pb = a.source.settings, a.source, a.playback
            return {
                "active": True, "session_id": a.id, "mode": a.mode,
                "recording_id": a.recording_id, "name": src.recording.name,
                "center_freq_hz": s.center_freq_hz, "sample_rate_hz": s.sample_rate_hz,
                "duration_s": src.duration_s, "position_s": src.position_s,
                "playing": pb.playing, "speed": pb.speed, "nfft": pb.nfft, "fps": pb.fps,
                "settings": s.to_dict(), "comparable": src.recording.comparable,
                "annotations": src.recording.meta.get("annotations", []),
            }

    # --- playback -------------------------------------------------------
    def control(self, cmd: dict) -> None:
        with self._lock:
            a = self._require()
            pb = a.playback
            if "playing" in cmd:
                if cmd["playing"] and a.source.position >= a.source.recording.n_samples:
                    a.source.seek(0)                    # play again from the start
                pb.playing = bool(cmd["playing"])
            if "speed" in cmd:
                pb.speed = float(np.clip(cmd["speed"], 0.05, 64))
            if "nfft" in cmd:
                n = int(cmd["nfft"])
                if n not in (256, 512, 1024, 2048, 4096, 8192, 16384):
                    raise ValueError("nfft must be a power of two from 256 to 16384")
                pb.nfft = n
            if "seek_s" in cmd:
                a.source.seek(float(cmd["seek_s"]))

    def _samples_per_frame(self, a: ActiveSession) -> int:
        """Samples the playhead advances per display frame."""
        fs = a.source.settings.sample_rate_hz
        return max(a.playback.nfft, int(fs * a.playback.speed / a.playback.fps))

    def next_frame(self) -> SpectrumFrame | None:
        """Advance the playhead by one display frame and return its spectrum.
        Each frame averages the samples it advanced over (up to a CPU cap), so at
        real-time speed no sample is skipped."""
        with self._lock:
            a = self._require()
            if not a.playback.playing:
                return None
            hop = self._samples_per_frame(a)
            block = a.source.read_block(hop)
            if block is None or len(block) < a.playback.nfft:
                a.playback.playing = False
                return None
            used = block.samples[:MAX_SAMPLES_PER_FRAME]
            block.samples = used
            return spectrum_frame(block, a.playback.nfft)

    def current_frame(self) -> SpectrumFrame:
        """Spectrum at the playhead without moving it (for a paused or seeked view)."""
        with self._lock:
            a = self._require()
            n = min(self._samples_per_frame(a), MAX_SAMPLES_PER_FRAME)
            block = a.source.read_at(a.source.position_s, n)
            return spectrum_frame(block, a.playback.nfft)

    def overview(self, rows: int = 300, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            a = self._require()
            if a.overview is None or a.overview[1].shape != (rows, nfft):
                src = a.source
                a.overview = overview_rows(src.read_at, src.duration_s,
                                           src.settings.sample_rate_hz, rows, nfft)
            return a.overview

    def _require(self) -> ActiveSession:
        if self.active is None:
            raise RuntimeError("No active session")
        return self.active

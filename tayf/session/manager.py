"""Session manager: exactly one active session (recording, live or scan).

It owns the source and the radio, so two parts of the app can never fight over
the HackRF. Everything downstream asks the manager for frames.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.models import IQBlock, SpectrumFrame, SweepFrame, utc_now
from ..core.sigmf_io import SigMFRecording
from ..dsp.dc import remove_dc
from ..dsp.spectrum import overview_rows, spectrum_frame
from ..sources.live import LiveSource
from ..sources.radio import LNA_STEPS, SAMPLE_RATES, VGA_STEPS, Gains, HackRFRadio, SimRadio, snap_gain
from ..sources.recording import RecordingSource
from ..sources.sweep import HackRFSweep, SimSweep, SweepSource, check_range
from ..store.db import Store

MAX_SAMPLES_PER_FRAME = 262_144     # caps CPU per frame at high sample rates
NFFT_CHOICES = (256, 512, 1024, 2048, 4096, 8192, 16384)

# ADC level guide for an 8-bit receiver (total power of the samples, dBFS)
LEVEL_LOW_DBFS = -35.0      # below this, quantization noise starts to dominate
CLIP_WARN = 0.001           # more than 0.1 % of samples at full scale


@dataclass
class Playback:
    playing: bool = False       # recording: playing; live: display running
    speed: float = 1.0          # recordings only; 1.0 = real time
    nfft: int = 2048
    fps: float = 20.0
    dc_remove: bool = True


@dataclass
class ActiveSession:
    id: int
    mode: str
    source: RecordingSource | LiveSource | SweepSource
    recording_id: int | None = None
    playback: Playback = field(default_factory=Playback)
    overview: tuple[np.ndarray, np.ndarray] | None = None
    preset_id: str | None = None        # scan: the band preset it started from


def measure_health(block: IQBlock) -> dict:
    """ADC level and clipping of one block. Clipping is when I or Q hits full scale."""
    x = block.samples
    clipped = (np.abs(x.real) >= 127 / 128) | (np.abs(x.imag) >= 127 / 128)
    level = 10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20)
    clip = float(clipped.mean())
    hint = "clipping" if clip > CLIP_WARN else "low" if level < LEVEL_LOW_DBFS else "ok"
    return {"level_dbfs": round(float(level), 1), "clip_pct": round(100 * clip, 3), "level_hint": hint}


class SessionManager:
    def __init__(self, store: Store, recordings_dir: str | Path = "data/recordings"):
        self.store = store
        self.recordings_dir = Path(recordings_dir)
        self.last_saved: dict | None = None     # the most recent recording made from live
        self.active: ActiveSession | None = None
        self.health: dict = {}
        self.rev = 0                    # bumps on every change, so paused views refresh
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
            self._changed()
            return self.active

    def open_live(self, radio: str = "hackrf", center_hz: float = 100e6, sample_rate_hz: float = 2e6,
                  amp: bool = False, lna_db: int = 16, vga_db: int = 20,
                  antenna_id: str | None = None, offset_tuning: bool = True) -> ActiveSession:
        if sample_rate_hz not in SAMPLE_RATES:
            raise ValueError(f"sample_rate_hz must be one of {SAMPLE_RATES}")
        with self._lock:
            self.close()
            device = SimRadio() if radio == "sim" else HackRFRadio()
            gains = Gains(bool(amp), snap_gain(lna_db, LNA_STEPS), snap_gain(vga_db, VGA_STEPS))
            try:
                source = LiveSource(device, center_hz, sample_rate_hz, gains, antenna_id, offset_tuning)
            except Exception:
                device.close()
                raise
            source.start()
            sid = self.store.start_session("live", source.settings)
            self.active = ActiveSession(sid, "live", source, playback=Playback(playing=True))
            self._changed()
            return self.active

    def open_scan(self, radio: str = "hackrf", f_lo_hz: float = 87e6, f_hi_hz: float = 109e6,
                  bin_hz: float = 25e3, amp: bool = False, lna_db: int = 32, vga_db: int = 30,
                  antenna_id: str | None = None, preset_id: str | None = None) -> ActiveSession:
        with self._lock:
            check_range(f_lo_hz, f_hi_hz, bin_hz)
            self.close()
            source = self._make_sweep(radio, f_lo_hz, f_hi_hz, bin_hz,
                                      Gains(bool(amp), snap_gain(lna_db, LNA_STEPS), snap_gain(vga_db, VGA_STEPS)),
                                      antenna_id)
            sid = self.store.start_session("scan", source.settings)
            self.active = ActiveSession(sid, "scan", source, playback=Playback(playing=True),
                                        preset_id=preset_id)
            self._changed()
            return self.active

    @staticmethod
    def _make_sweep(radio, f_lo, f_hi, bin_hz, gains, antenna_id) -> SweepSource:
        cls = SimSweep if radio == "sim" else HackRFSweep
        source = cls(f_lo, f_hi, bin_hz, gains, antenna_id)
        source.start()                      # raises if the HackRF is missing or busy
        return source

    def close(self) -> None:
        with self._lock:
            if self.active:
                if self.active.mode == "live":
                    self.stop_recording()
                self.active.source.close()
                self.store.end_session(self.active.id)
                self.active = None
                self.health = {}
                self._changed()

    def _changed(self) -> None:
        self.rev += 1

    # --- recording from live --------------------------------------------
    def start_recording(self) -> dict:
        """Save everything the radio delivers, raw, as a SigMF recording."""
        with self._lock:
            a = self._require()
            if a.mode != "live":
                raise RuntimeError("Recording is only available in live mode")
            src = a.source
            stamp = utc_now().strftime("%Y%m%d_%H%M%S")
            name = f"live_{stamp}_{src.tuned_hz / 1e3:.0f}kHz"     # no dots: they look like extensions
            src.start_recording(self.recordings_dir / name,
                                description=f"Recorded live, tuned to {src.tuned_hz / 1e6:.3f} MHz")
            self._changed()
            return src.recording_status()

    def stop_recording(self) -> dict | None:
        """Finish the recording and add it to the recordings list."""
        with self._lock:
            a = self.active
            if a is None or a.mode != "live" or not a.source.recording:
                return None
            meta = a.source.stop_recording()
            rec = SigMFRecording.open(meta)
            rid = self.store.add_recording(rec)
            self.last_saved = {"id": rid, "name": rec.name, "seconds": round(rec.duration_s, 1),
                               "mbytes": round(rec.n_samples * 2 / 1e6, 1),
                               "segments": len(rec.meta["captures"])}
            self._changed()
            return self.last_saved

    def status(self) -> dict:
        with self._lock:
            a = self.active
            if a is None:
                return {"active": False, "rev": self.rev}
            src, pb, s = a.source, a.playback, a.source.settings
            st = {
                "active": True, "session_id": a.id, "mode": a.mode, "rev": self.rev,
                "center_freq_hz": s.center_freq_hz, "sample_rate_hz": s.sample_rate_hz,
                "playing": pb.playing, "nfft": pb.nfft, "fps": pb.fps, "dc_remove": pb.dc_remove,
                "settings": s.to_dict(), "health": self.health,
            }
            if a.mode == "recording":
                st.update({
                    "recording_id": a.recording_id, "name": src.recording.name,
                    "duration_s": src.duration_s, "position_s": src.position_s, "speed": pb.speed,
                    "comparable": src.recording.comparable,
                    "annotations": src.recording.meta.get("annotations", []),
                })
            elif a.mode == "scan":
                st.update({
                    "name": f"Scan: {src.radio_name}", "radio": "sim" if isinstance(src, SimSweep) else "hackrf",
                    "f_lo_hz": src.f_lo, "f_hi_hz": src.f_hi, "bin_hz": src.bin_hz,
                    "preset_id": a.preset_id, "antenna_id": src.antenna_id,
                    "sweeps": src.sweeps, "sweep_rate": round(src.sweep_rate, 2), "error": src.error,
                    "gains": {"amp": src.gains.amp, "lna_db": src.gains.lna_db,
                              "vga_db": src.gains.vga_db, "total_db": src.gains.total_db},
                })
            else:
                st.update({
                    "name": f"Live: {src.radio.name}", "tuned_hz": src.tuned_hz,
                    "offset_hz": src.offset_hz, "offset_tuning": src.offset_tuning,
                    "antenna_id": src.antenna_id, "overflows": src.overflows,
                    "recording": src.recording_status(), "last_saved": self.last_saved,
                    "gains": {"amp": src.gains.amp, "lna_db": src.gains.lna_db,
                              "vga_db": src.gains.vga_db, "total_db": src.gains.total_db},
                })
            return st

    # --- control --------------------------------------------------------
    def control(self, cmd: dict) -> None:
        with self._lock:
            a = self._require()
            pb, src = a.playback, a.source
            if "nfft" in cmd:
                if int(cmd["nfft"]) not in NFFT_CHOICES:
                    raise ValueError("nfft must be a power of two from 256 to 16384")
                pb.nfft = int(cmd["nfft"])
            if "dc_remove" in cmd:
                pb.dc_remove = bool(cmd["dc_remove"])
            if "playing" in cmd:
                if a.mode == "recording" and cmd["playing"] and src.position >= src.recording.n_samples:
                    src.seek(0)                         # play again from the start
                pb.playing = bool(cmd["playing"])
            if a.mode == "scan":
                self._control_scan(a, cmd)
            elif a.mode == "recording":
                if "speed" in cmd:
                    pb.speed = float(np.clip(cmd["speed"], 0.05, 64))
                if "seek_s" in cmd:
                    src.seek(float(cmd["seek_s"]))
            else:
                if "sample_rate_hz" in cmd:
                    if float(cmd["sample_rate_hz"]) not in SAMPLE_RATES:
                        raise ValueError(f"sample_rate_hz must be one of {SAMPLE_RATES}")
                    if float(cmd["sample_rate_hz"]) != src.sample_rate_hz:
                        self.stop_recording()   # one recording file has one sample rate
                        src.set_sample_rate(float(cmd["sample_rate_hz"]))
                if "center_hz" in cmd:
                    src.tune(float(cmd["center_hz"]))
                if "offset_tuning" in cmd:
                    src.set_offset_tuning(bool(cmd["offset_tuning"]))
                if {"amp", "lna_db", "vga_db"} & cmd.keys():
                    g = src.gains
                    src.set_gains(Gains(bool(cmd.get("amp", g.amp)),
                                        snap_gain(cmd.get("lna_db", g.lna_db), LNA_STEPS),
                                        snap_gain(cmd.get("vga_db", g.vga_db), VGA_STEPS)))
                if "antenna_id" in cmd:
                    src.antenna_id = cmd["antenna_id"] or None
            self._changed()

    def _control_scan(self, a: ActiveSession, cmd: dict) -> None:
        """A new range, bin width or gain means restarting hackrf_sweep."""
        src = a.source
        if "antenna_id" in cmd:
            src.antenna_id = cmd["antenna_id"] or None
        keys = {"f_lo_hz", "f_hi_hz", "bin_hz", "amp", "lna_db", "vga_db"}
        if not keys & cmd.keys():
            return
        g = src.gains
        gains = Gains(bool(cmd.get("amp", g.amp)), snap_gain(cmd.get("lna_db", g.lna_db), LNA_STEPS),
                      snap_gain(cmd.get("vga_db", g.vga_db), VGA_STEPS))
        f_lo, f_hi = float(cmd.get("f_lo_hz", src.f_lo)), float(cmd.get("f_hi_hz", src.f_hi))
        bin_hz = float(cmd.get("bin_hz", src.bin_hz))
        check_range(f_lo, f_hi, bin_hz)
        radio = "sim" if isinstance(src, SimSweep) else "hackrf"
        src.close()
        try:
            a.source = self._make_sweep(radio, f_lo, f_hi, bin_hz, gains, src.antenna_id)
        except RuntimeError:                # go back to the scan that worked
            a.source = self._make_sweep(radio, src.f_lo, src.f_hi, src.bin_hz, src.gains, src.antenna_id)
            raise
        if "preset_id" in cmd:
            a.preset_id = cmd["preset_id"]
        elif {"f_lo_hz", "f_hi_hz"} & cmd.keys():
            a.preset_id = None              # a hand-picked range is no longer the preset

    # --- frames ---------------------------------------------------------
    @staticmethod
    def _sweep_frame(a: ActiveSession, s: SweepFrame | None) -> SpectrumFrame | None:
        if s is None:
            return None
        return SpectrumFrame(s.freqs_hz, s.power_db, s.bin_width_hz, (a.source.f_lo + a.source.f_hi) / 2,
                             s.time, offset_s=a.source.elapsed_s)

    def _samples_per_frame(self, a: ActiveSession) -> int:
        """Samples the display advances per frame (recordings: scaled by speed)."""
        fs = a.source.settings.sample_rate_hz
        speed = a.playback.speed if a.mode == "recording" else 1.0
        return max(a.playback.nfft, int(fs * speed / a.playback.fps))

    def _to_frame(self, a: ActiveSession, block: IQBlock) -> SpectrumFrame:
        block.samples = block.samples[:MAX_SAMPLES_PER_FRAME]
        self.health = measure_health(block)
        if a.playback.dc_remove:
            block.samples = remove_dc(block.samples)
        return spectrum_frame(block, a.playback.nfft)

    def next_frame(self) -> SpectrumFrame | None:
        """Advance by one display frame and return its spectrum.
        Recordings: each frame averages the samples it advanced over (up to a CPU
        cap), so at real-time speed no sample is skipped. Live: the newest samples."""
        with self._lock:
            a = self._require()
            if not a.playback.playing:
                return None
            if a.mode == "scan":
                return self._sweep_frame(a, a.source.take())
            n = self._samples_per_frame(a)
            if a.mode == "live":
                n = min(n, MAX_SAMPLES_PER_FRAME)   # live never needs more than it processes
            block = a.source.read_block(n)
            if block is None or len(block) < a.playback.nfft:
                if a.mode == "recording":
                    a.playback.playing = False
                    self._changed()
                return None
            return self._to_frame(a, block)

    def current_frame(self) -> SpectrumFrame | None:
        """Spectrum at the playhead without moving it (paused or seeked view)."""
        with self._lock:
            a = self._require()
            if a.mode == "scan":
                return self._sweep_frame(a, a.source.latest())
            n = min(self._samples_per_frame(a), MAX_SAMPLES_PER_FRAME)
            if a.mode == "recording":
                block = a.source.read_at(a.source.position_s, n)
            else:
                block = a.source.latest(n)
            if block is None or len(block) < a.playback.nfft:
                return None
            return self._to_frame(a, block)

    def overview(self, rows: int = 300, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            a = self._require()
            if a.mode != "recording":
                raise RuntimeError("The overview is only available for recordings")
            if a.overview is None or a.overview[1].shape != (rows, nfft):
                src = a.source
                a.overview = overview_rows(src.read_at, src.duration_s,
                                           src.settings.sample_rate_hz, rows, nfft)
            return a.overview

    def _require(self) -> ActiveSession:
        if self.active is None:
            raise RuntimeError("No active session")
        return self.active

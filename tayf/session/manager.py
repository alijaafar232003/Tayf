"""Session manager: exactly one active session (recording, live or scan).

It owns the source and the radio, so two parts of the app can never fight over
the HackRF. Everything downstream asks the manager for frames.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.models import IQBlock, SpectrumFrame, SweepFrame, utc_now
from ..core.models import ReceiverSettings
from ..core.sigmf_io import SigMFRecording, write_recording
from ..dsp.dc import remove_dc
from ..dsp.detect import DetectorConfig, detect, shape_for
from ..dsp.inspect import inspect as inspect_signal
from ..dsp.track import Tracker
from ..dsp.health import measure_health
from ..dsp.spectrum import overview_rows, spectrum_frame
from ..receiver.agc import AGC
from ..receiver.check import fit_for, run_check
from ..receiver.profile import load_profile
from ..sources.live import LiveSource
from ..sources.radio import LNA_STEPS, SAMPLE_RATES, VGA_STEPS, Gains, HackRFRadio, SimRadio, snap_gain
from ..sources.recording import RecordingSource
from ..sources.sweep import HackRFSweep, SimSweep, SweepSource, check_range
from ..store.db import Store
from ..store.history import origin_of

MAX_SAMPLES_PER_FRAME = 262_144     # caps CPU per frame at high sample rates
NFFT_CHOICES = (256, 512, 1024, 2048, 4096, 8192, 16384)



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


class SessionManager:
    def __init__(self, store: Store, recordings_dir: str | Path = "data/recordings"):
        self.store = store
        self.recordings_dir = Path(recordings_dir)
        self.last_saved: dict | None = None     # the most recent recording made from live
        self.checking = False                   # a Receiver Check is changing the gains
        self.agc: AGC | None = None             # automatic gain control (live), None = off
        self.profile_dir = self.recordings_dir.parent / "receiver"
        self.obs_dir = self.recordings_dir.parent / "observations"   # I/Q of inspected signals
        # Detection (Phase 3): detector + tracker run on every displayed frame
        self.detect_on = True
        self.det_cfg = DetectorConfig()
        self.tracker = Tracker()
        self.closed_tracks: list[dict] = []     # summaries of finished tracks (also saved to the database)
        self._seen_cache: dict[int, dict] = {}  # track id -> seen before? (history, for the Signals table)
        self.boxes: list[dict] = []             # the latest frame's detections, with track IDs
        self.det_rev = 0                        # bumps when boxes/tracks change
        self._det_cost_s = 0.0
        self._det_last_t = 0.0
        self._profile: dict | None = None
        self._shape_cache: dict = {}
        self.last_inspection: dict | None = None
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
            self.agc = None
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
                self._end_tracks(self.active)
                self.active = None
                self.health = {}
                self.agc = None
                self._changed()

    def _changed(self) -> None:
        self.rev += 1

    # --- detection + tracking ---------------------------------------------
    def _end_tracks(self, a: ActiveSession) -> None:
        self.tracker.close_all()
        self._save_closed(a)
        self._seen_cache = {}
        self.boxes, self._shape_cache = [], {}
        self._profile = None                    # reloaded on next use (it may have been re-measured)
        self.det_rev += 1

    def _detect(self, a: ActiveSession, frame: SpectrumFrame, track: bool = True) -> None:
        """Run the detector on a displayed frame and (for new frames) the tracker.
        Skips frames when detection is expensive (large scans) to keep the CPU free."""
        now = time.monotonic()
        if not self.detect_on or now - self._det_last_t < min(1.0, 3 * self._det_cost_s):
            return
        t0 = time.perf_counter()
        if self._profile is None:
            self._profile = load_profile(self.profile_dir) or {}
        # The receiver's floor shape only fits spectra from this HackRF (not scans:
        # hackrf_sweep stitches its own segments; not other receivers' files).
        shape = None
        hw = a.source.settings.hardware or ""
        if a.mode != "scan" and "HackRF" in hw:
            key = (len(frame.freqs_hz), float(frame.freqs_hz[1] - frame.freqs_hz[0]))
            if key not in self._shape_cache:
                self._shape_cache[key] = shape_for(self._profile, frame)
            shape = self._shape_cache[key]
        spurs = [c["freq_hz"] for c in (self._profile or {}).get("internal_carriers", [])]
        lo = None if a.mode == "scan" else frame.center_freq_hz
        dets, info = detect(frame, self.det_cfg, shape_db=shape, spurs_hz=spurs, lo_hz=lo)
        if track:
            span = (float(frame.freqs_hz[0]), float(frame.freqs_hz[-1]))
            pairs = self.tracker.update(dets, frame.time.timestamp(), span)
            self._save_closed(a)
        else:
            pairs = [(d, None) for d in dets]
        confirmed = {tr["id"] for tr in self.tracker.table()}
        self.boxes = [{**d.to_dict(), "track": tid, "confirmed": tid in confirmed} for d, tid in pairs]
        self.det_floor_dbfs = info["floor_dbfs"]
        self.det_rev += 1
        self._det_cost_s = time.perf_counter() - t0
        self._det_last_t = now

    def _origin(self, a: ActiveSession) -> str:
        return origin_of(a.source.settings.hardware)

    def _save_closed(self, a: ActiveSession) -> None:
        """Closed tracks -> light summaries in the database (D7), for history."""
        closed = self.tracker.take_closed()
        if closed:
            self.closed_tracks = (self.closed_tracks + closed)[-500:]
            self.store.add_candidate_summaries(a.id, self._origin(a), closed, a.recording_id)

    def _seen(self, a: ActiveSession, tr: dict) -> dict:
        """Seen in an earlier session? Looked up once per track (cached)."""
        if tr["id"] not in self._seen_cache:
            self._seen_cache[tr["id"]] = self.store.seen_before(
                tr["f_center_hz"], max(tr["bandwidth_hz"], tr["max_bandwidth_hz"]), self._origin(a), a.id)
        return self._seen_cache[tr["id"]]

    # --- Signal Inspector ---------------------------------------------------
    def inspect(self, track_id: int | None = None, f_hz: float | None = None, bw_hz: float | None = None,
                seconds: float = 1.0, save: bool = True) -> dict:
        """Take I/Q around a signal (live: `seconds` of fresh reception; recordings: at
        the play position) and measure it (tayf.dsp.inspect). Runs without holding the
        session lock during the capture, so the display keeps going."""
        if not 0.05 <= seconds <= 10:
            raise ValueError("seconds must be between 0.05 and 10")
        with self._lock:
            a = self._require()
            if a.mode == "scan":
                raise RuntimeError("A scan has no I/Q: use \"Open live here\" on the signal, then inspect it")
            track = None
            if track_id is not None:
                track = next((t for t in self.tracker.table() if t["id"] == track_id), None)
                if track is None:
                    raise KeyError(f"No signal #{track_id} right now")
                f_hz = track["f_center_hz"]
                bw_hz = max(track["bandwidth_hz"], track["max_bandwidth_hz"])
            if f_hz is None:
                raise ValueError("Give a track or a frequency")
            bw_hz = max(bw_hz or 10e3, 1e3)
            src, mode, settings = a.source, a.mode, a.source.settings
            if mode == "recording":
                fs = settings.sample_rate_hz
                t0 = max(0.0, min(src.position_s, src.duration_s - seconds))
                block = src.read_at(t0, int(seconds * fs))
        if mode == "live":
            block = src.capture(seconds)
        off = f_hz - block.center_freq_hz
        half = block.sample_rate_hz / 2
        if abs(off) + bw_hz / 2 > 0.9 * half:
            raise RuntimeError(f"{f_hz / 1e6:.4f} MHz is outside (or at the edge of) the current span: tune to it first")
        x = remove_dc(block.samples)
        ins = inspect_signal(x, block.sample_rate_hz, off, bw_hz, f_tuned_hz=block.center_freq_hz)
        r = ins.to_dict()
        r.update({"track_id": track_id, "requested_hz": f_hz, "mode": mode, "time": block.start_time.isoformat(),
                  "settings": settings.to_dict(), "track": track})
        if save:
            r.update(self._save_observation(a, r, ins, block.start_time))
        with self._lock:
            self.last_inspection = r
        return r

    def _save_observation(self, a: ActiveSession, r: dict, ins, start_time) -> dict:
        """Save an inspection (D7: the full observation plus its I/Q) and match it to
        history. The I/Q saved is the signal's narrow channel (complex float32, centred
        on the requested frequency), small enough to keep and what the classifier will use."""
        m = {k: v for k, v in r.items() if k not in ("views", "track", "settings")}
        settings = r["settings"]
        health = self.health if a.mode == "live" else None
        oid, hist = self.store.save_observation(a.id, self._origin(a), m, settings, health)
        s = ReceiverSettings(**{**settings, "center_freq_hz": ins.channel_center_hz,
                                "sample_rate_hz": ins.fs_channel_hz})
        base = self.obs_dir / f"obs_{oid:05d}_{ins.channel_center_hz / 1e3:.0f}kHz"
        meta = write_recording(base, ins.channel, s, "cf32_le", start_time,
                               f"Tayf observation #{oid}: {ins.f_center_hz / 1e6:.4f} MHz, "
                               f"OBW {ins.obw_hz / 1e3:.1f} kHz, SNR {ins.snr_db:.1f} dB ({a.mode})")
        self.store.set_observation_iq(oid, str(meta.resolve()))
        return {"observation_id": oid, "iq_path": str(meta), "history": hist}

    def history(self, f_hz: float, bw_hz: float) -> dict:
        """What is known about a signal (without inspecting it)."""
        with self._lock:
            a = self._require()
            origin = self._origin(a)
        return self.store.history_for(f_hz, bw_hz, origin)

    def detections(self) -> dict:
        """What the dashboard draws: this frame's boxes and the candidate table."""
        with self._lock:
            tracks = self.tracker.table()[:100]
            if self.active:
                tracks = [{**t, "seen": self._seen(self.active, t)} for t in tracks]
            return {"type": "det", "on": self.detect_on, "boxes": self.boxes, "tracks": tracks,
                    "closed": len(self.closed_tracks), "cost_ms": round(1000 * self._det_cost_s, 1)}

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
                "detector": {"on": self.detect_on, "min_snr_db": self.det_cfg.min_snr_db,
                             "tracks": len(self.tracker.table()), "closed": len(self.closed_tracks)},
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
                    "checking": self.checking,
                    "agc": {"on": self.agc is not None, "last": self.agc.last_action if self.agc else "",
                            "changes": self.agc.changes if self.agc else 0},
                    "gains": {"amp": src.gains.amp, "lna_db": src.gains.lna_db,
                              "vga_db": src.gains.vga_db, "total_db": src.gains.total_db},
                })
            return st

    # --- receiver check -------------------------------------------------
    def receiver_check(self, profile: dict | None) -> dict:
        """Try a few gain settings on the live stream and suggest one (tayf.receiver.check).
        Runs without holding the session lock, so the display keeps updating meanwhile."""
        with self._lock:
            a = self._require()
            if a.mode != "live":
                raise RuntimeError("The Receiver Check needs live mode")
            if a.source.recording:
                raise RuntimeError("Stop the recording first: every gain step would become a new segment")
            if self.checking:
                raise RuntimeError("A Receiver Check is already running")
            self.checking = True
            self._changed()
            src = a.source
        try:
            return run_check(src, profile)
        finally:
            with self._lock:
                self.checking = False
                self._changed()

    # --- control --------------------------------------------------------
    def control(self, cmd: dict) -> None:
        with self._lock:
            a = self._require()
            pb, src = a.playback, a.source
            if "detect" in cmd:
                self.detect_on = bool(cmd["detect"])
                if not self.detect_on:
                    self.boxes = []
                    self.det_rev += 1
            if "min_snr_db" in cmd:
                v = float(cmd["min_snr_db"])
                if not 2 <= v <= 40:
                    raise ValueError("min_snr_db must be between 2 and 40 dB")
                self.det_cfg.min_snr_db = v
                self.det_cfg.edge_snr_db = min(3.0, v / 2)
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
                if "agc" in cmd:
                    self._set_agc(bool(cmd["agc"]))
                if {"amp", "lna_db", "vga_db"} & cmd.keys():
                    self.agc = None             # a manual gain change takes over from the AGC
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

    # --- automatic gain control -------------------------------------------
    def _set_agc(self, on: bool) -> None:
        if not on:
            self.agc = None
        elif self.agc is None:
            self.agc = AGC(profile=load_profile(self.profile_dir))

    def _agc_step(self, a: ActiveSession) -> None:
        """Called with each live frame's health; the AGC itself decides whether to act."""
        if self.agc is None or self.checking:
            return
        src = a.source
        # The gain rule depends on frequency (front-end noise differs): use the profile
        # fit nearest to where we are tuned, refreshed after a big retune.
        if self.agc.fit_for_hz is None or abs(np.log10(self.agc.fit_for_hz / src.tuned_hz)) > 0.3:
            self.agc.fit = fit_for(self.agc.profile, src.tuned_hz)
            self.agc.fit_for_hz = src.tuned_hz
        new, _ = self.agc.update(time.monotonic(), self.health, src.gains)
        if new is not None:
            src.set_gains(new)
            self._changed()

    def _to_frame(self, a: ActiveSession, block: IQBlock) -> SpectrumFrame:
        block.samples = block.samples[:MAX_SAMPLES_PER_FRAME]
        self.health = measure_health(block)
        if a.mode == "live":
            self._agc_step(a)
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
                frame = self._sweep_frame(a, a.source.take())
                if frame is not None:
                    self._detect(a, frame)
                return frame
            n = self._samples_per_frame(a)
            if a.mode == "live":
                n = min(n, MAX_SAMPLES_PER_FRAME)   # live never needs more than it processes
            block = a.source.read_block(n)
            if block is None or len(block) < a.playback.nfft:
                if a.mode == "recording":
                    a.playback.playing = False
                    self._changed()
                return None
            frame = self._to_frame(a, block)
            self._detect(a, frame)
            return frame

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
            frame = self._to_frame(a, block)
            self._detect(a, frame, track=False)          # paused: boxes for this view, no tracking
            return frame

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

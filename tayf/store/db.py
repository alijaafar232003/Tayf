"""SQLite access: antennas, recordings, sessions, observations and history."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..core.models import ReceiverSettings, utc_now
from ..core.sigmf_io import SigMFRecording
from . import history as H

SCHEMA = Path(__file__).with_name("schema.sql")

# Columns added after a table was first created: (table, column, type).
# CREATE TABLE IF NOT EXISTS does not add them to an existing database.
MIGRATIONS = [
    ("candidate_summaries", "max_bandwidth_hz", "REAL"),
    ("candidate_summaries", "recording_id", "INTEGER"),
    ("candidate_summaries", "origin", "TEXT"),
    ("signal_profiles", "origin", "TEXT"),
    ("observations", "origin", "TEXT"),
    ("observations", "measurements_json", "TEXT"),
]

# Your antennas (handoff, decision D4). Gains are the sellers' claims.
DEFAULT_ANTENNAS = [
    ("whip-5dbi", "5 dBi magnetic whip (#4)", 40e6, 860e6, 5.0,
     "FM, airband, 433 MHz, DVB-T", "Default for low bands"),
    ("telescopic", "Telescopic (#6)", 40e6, 6e9, None,
     "General fallback", "Quarter wave = 75 / f(MHz) metres"),
    ("mag-12dbi", '"12 dBi" magnetic (#7)', 700e6, 2.7e9, 12.0,
     "Cellular downlinks, 868 MHz", "Gain claim probably optimistic"),
    ("mag-35dbi", '"35 dBi" magnetic (#8)', 700e6, 2.7e9, 35.0,
     "Cellular downlinks, 868 MHz", "35 dBi is implausible; expect a few dBi"),
    ("duck-8dbi", "8 dBi rubber duck (#5)", 2.4e9, 5.9e9, 8.0,
     "Wi-Fi, BLE, ESP32", "2.4 / 5 / 5.8 GHz bands"),
]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            for table, col, typ in MIGRATIONS:          # existing databases: add new columns first
                if self._con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                    cols = {r[1] for r in self._con.execute(f"PRAGMA table_info({table})")}
                    if col not in cols:
                        self._con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            self._con.executescript(SCHEMA.read_text())
            if not self._con.execute("SELECT 1 FROM antennas LIMIT 1").fetchone():
                self._con.executemany(
                    "INSERT INTO antennas VALUES (?, ?, ?, ?, ?, ?, ?)", DEFAULT_ANTENNAS)
            self._con.commit()

    def _q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._con.execute(sql, args)
            self._con.commit()
            return cur.lastrowid

    def close(self) -> None:
        self._con.close()

    # --- antennas -------------------------------------------------------
    def antennas(self) -> list[dict]:
        return self._q("SELECT * FROM antennas ORDER BY freq_min_hz")

    def antennas_for(self, freq_hz: float) -> list[dict]:
        """Antennas whose range covers freq_hz (the basis for recommendations)."""
        return self._q("SELECT * FROM antennas WHERE freq_min_hz <= ? AND ? <= freq_max_hz "
                       "ORDER BY freq_max_hz - freq_min_hz", (freq_hz, freq_hz))

    def upsert_antenna(self, a: dict) -> None:
        self._x("INSERT INTO antennas (id, name, freq_min_hz, freq_max_hz, gain_dbi, uses, notes) "
                "VALUES (:id, :name, :freq_min_hz, :freq_max_hz, :gain_dbi, :uses, :notes) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, freq_min_hz=excluded.freq_min_hz, "
                "freq_max_hz=excluded.freq_max_hz, gain_dbi=excluded.gain_dbi, uses=excluded.uses, "
                "notes=excluded.notes",
                {"gain_dbi": None, "uses": None, "notes": None, **a})

    def delete_antenna(self, antenna_id: str) -> None:
        self._x("DELETE FROM antennas WHERE id = ?", (antenna_id,))

    # --- recordings -----------------------------------------------------
    def add_recording(self, rec: SigMFRecording) -> int:
        s = rec.settings
        meta = str(rec.meta_path.resolve())
        self._x(
            "INSERT INTO recordings (meta_path, name, center_freq_hz, sample_rate_hz, datatype, "
            "n_samples, duration_s, start_time, antenna_id, comparable, description, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(meta_path) DO UPDATE SET center_freq_hz=excluded.center_freq_hz, "
            "sample_rate_hz=excluded.sample_rate_hz, n_samples=excluded.n_samples, "
            "duration_s=excluded.duration_s, comparable=excluded.comparable, "
            "description=excluded.description",
            (meta, rec.name, s.center_freq_hz, s.sample_rate_hz, rec.datatype, rec.n_samples,
             rec.duration_s, rec.start_time.isoformat(),
             s.antenna_id if self._antenna_exists(s.antenna_id) else None,
             int(rec.comparable), rec.description, utc_now().isoformat()))
        return self._q("SELECT id FROM recordings WHERE meta_path = ?", (meta,))[0]["id"]

    def _antenna_exists(self, antenna_id: str | None) -> bool:
        return bool(antenna_id) and bool(self._q("SELECT 1 FROM antennas WHERE id = ?", (antenna_id,)))

    def recordings(self) -> list[dict]:
        return self._q("SELECT * FROM recordings ORDER BY added_at DESC")

    def recording(self, recording_id: int) -> dict | None:
        rows = self._q("SELECT * FROM recordings WHERE id = ?", (recording_id,))
        return rows[0] if rows else None

    def scan_folder(self, folder: str | Path) -> int:
        """Register every SigMF recording in a folder. Returns how many were found."""
        n = 0
        for meta in sorted(Path(folder).glob("*.sigmf-meta")):
            try:
                self.add_recording(SigMFRecording.open(meta))
                n += 1
            except (OSError, ValueError, KeyError):
                continue
        return n

    # --- sessions -------------------------------------------------------
    def start_session(self, mode: str, settings: ReceiverSettings,
                      recording_id: int | None = None) -> int:
        antenna = settings.antenna_id if self._antenna_exists(settings.antenna_id) else None
        return self._x("INSERT INTO sessions (mode, started_at, recording_id, antenna_id, settings_json) "
                       "VALUES (?, ?, ?, ?, ?)",
                       (mode, utc_now().isoformat(), recording_id, antenna,
                        json.dumps(settings.to_dict())))

    def end_session(self, session_id: int) -> None:
        self._x("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (utc_now().isoformat(), session_id))

    def end_open_sessions(self) -> None:
        """Close sessions left open by a crash or restart."""
        self._x("UPDATE sessions SET ended_at = started_at WHERE ended_at IS NULL")

    def recent_sessions(self, limit: int = 10) -> list[dict]:
        return self._q("SELECT s.*, r.name AS recording_name FROM sessions s "
                       "LEFT JOIN recordings r ON r.id = s.recording_id "
                       "ORDER BY s.started_at DESC LIMIT ?", (limit,))

    # --- history (Phase 3 step 4) ------------------------------------------
    def add_candidate_summaries(self, session_id: int, origin: str, summaries: list[dict],
                                recording_id: int | None = None) -> int:
        """Save closed detector tracks (light summaries, D7). HackRF spurs are skipped."""
        rows = [(session_id, s["id"], s["f_center_hz"], s["bandwidth_hz"], _iso(s["first_seen"]),
                 _iso(s["last_seen"]), s.get("duty"), s.get("max_snr_db"), s.get("snr_db"),
                 s.get("max_bandwidth_hz"), recording_id, origin)
                for s in summaries if not s.get("spur")]
        if rows:
            with self._lock:
                self._con.executemany(
                    "INSERT INTO candidate_summaries (session_id, track_id, center_freq_hz, bandwidth_hz, "
                    "first_seen, last_seen, duty_cycle, peak_snr_db, mean_snr_db, max_bandwidth_hz, "
                    "recording_id, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
                self._con.commit()
        return len(rows)

    def sightings(self, f_hz: float, bw_hz: float, origin: str, exclude_session: int | None = None) -> dict:
        """Detector sightings of a signal in other sessions: how many sessions (replays of
        one recording count once), first and last seen."""
        lo, hi = H.search_window(f_hz, bw_hz)
        rows = self._q("SELECT session_id, recording_id, center_freq_hz, bandwidth_hz, max_bandwidth_hz, "
                       "first_seen, last_seen FROM candidate_summaries WHERE center_freq_hz BETWEEN ? AND ? "
                       "AND origin = ? AND session_id != ?", (lo, hi, origin, exclude_session or -1))
        rows = [r for r in rows if H.matches(f_hz, bw_hz, r["center_freq_hz"],
                                             max(r["bandwidth_hz"], r["max_bandwidth_hz"] or 0))]
        if not rows:
            return {"sessions": 0, "first_seen": None, "last_seen": None}
        keys = {("r", r["recording_id"]) if r["recording_id"] else ("s", r["session_id"]) for r in rows}
        return {"sessions": len(keys), "first_seen": min(r["first_seen"] for r in rows),
                "last_seen": max(r["last_seen"] for r in rows)}

    def find_profile(self, f_hz: float, bw_hz: float, origin: str) -> dict | None:
        lo, hi = H.search_window(f_hz, bw_hz)
        rows = self._q("SELECT * FROM signal_profiles WHERE center_freq_hz BETWEEN ? AND ? AND origin = ?",
                       (lo, hi, origin))
        rows = [r for r in rows if H.matches(f_hz, bw_hz, r["center_freq_hz"], r["bandwidth_hz"])]
        return min(rows, key=lambda r: abs(r["center_freq_hz"] - f_hz)) if rows else None

    def observations_of(self, profile_id: int, limit: int = 20) -> list[dict]:
        rows = self._q("SELECT * FROM observations WHERE profile_id = ? ORDER BY time DESC, id DESC LIMIT ?",
                       (profile_id, limit))
        return [_obs_row(r) for r in rows]

    def save_observation(self, session_id: int, origin: str, m: dict, settings: dict,
                         health: dict | None = None) -> tuple[int, dict]:
        """Save one inspection and link it to its signal profile (new or matched).
        `m` is the inspector's result (plots removed). Returns (observation id, history)."""
        f, bw, t = m["f_center_hz"], m["obw_hz"], m["time"]
        prof = self.find_profile(f, bw, origin)
        prev = self.observations_of(prof["id"], 10) if prof else []
        now = _brief(m, settings)
        if prof is None:
            status, diff = "new", []
            pid = self._x("INSERT INTO signal_profiles (center_freq_hz, bandwidth_hz, first_seen, last_seen, "
                          "n_observations, typical_snr_db, origin) VALUES (?, ?, ?, ?, 1, ?, ?)",
                          (f, bw, t, t, m["snr_db"], origin))
        else:
            pid = prof["id"]
            diff = H.changes(prev[0], now) if prev else []
            status = "changed" if diff else "known"
            n = prof["n_observations"]
            self._x("UPDATE signal_profiles SET center_freq_hz = ?, bandwidth_hz = ?, typical_snr_db = ?, "
                    "first_seen = MIN(first_seen, ?), last_seen = MAX(last_seen, ?), n_observations = ? WHERE id = ?",
                    ((prof["center_freq_hz"] * n + f) / (n + 1), (prof["bandwidth_hz"] * n + bw) / (n + 1),
                     ((prof["typical_snr_db"] or m["snr_db"]) * n + m["snr_db"]) / (n + 1), t, t, n + 1, pid))
        antenna = settings.get("antenna_id")
        oid = self._x("INSERT INTO observations (session_id, profile_id, time, center_freq_hz, bandwidth_hz, "
                      "freq_offset_hz, snr_db, power_dbfs, noise_floor_dbfs, history_status, antenna_id, "
                      "settings_json, health_json, origin, measurements_json) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (session_id, pid, t, f, bw, m["offset_hz"], m["snr_db"], m["power_dbfs"],
                       m["noise_dbfs_hz"], status, antenna if self._antenna_exists(antenna) else None,
                       json.dumps(settings), json.dumps(health) if health else None, origin, json.dumps(m)))
        h = self.history_for(f, bw, origin)
        return oid, {**h, "status": status, "changes": diff, "previous": prev, "now": now}

    def seen_before(self, f_hz: float, bw_hz: float, origin: str, session_id: int) -> dict:
        """For the Signals table: was this signal seen in an earlier session (by the
        detector or an inspection)? Returns first seen and the number of sessions."""
        seen = self.sightings(f_hz, bw_hz, origin, exclude_session=session_id)
        firsts, n = [seen["first_seen"]] if seen["first_seen"] else [], seen["sessions"]
        prof = self.find_profile(f_hz, bw_hz, origin)
        if prof:
            r = self._q("SELECT MIN(time) AS t, COUNT(DISTINCT session_id) AS n FROM observations "
                        "WHERE profile_id = ? AND session_id != ?", (prof["id"], session_id))[0]
            if r["t"]:
                firsts.append(r["t"])
                n = max(n, r["n"])
        return {"first_seen": min(firsts) if firsts else None, "sessions": n}

    def set_observation_iq(self, observation_id: int, meta_path: str) -> None:
        self._x("UPDATE observations SET iq_meta_path = ? WHERE id = ?", (meta_path, observation_id))

    def history_for(self, f_hz: float, bw_hz: float, origin: str) -> dict:
        """What is known about a signal at f_hz, without saving anything."""
        prof = self.find_profile(f_hz, bw_hz, origin)
        seen = self.sightings(f_hz, bw_hz, origin)
        firsts = [x for x in ((prof or {}).get("first_seen"), seen["first_seen"]) if x]
        return {"profile": prof, "previous": self.observations_of(prof["id"], 10) if prof else [],
                "detector": seen, "first_seen": min(firsts) if firsts else None}


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def _brief(m: dict, settings: dict) -> dict:
    """The few numbers compared between observations of the same signal."""
    b = m.get("bursts") or {}
    return {"time": m["time"], "f_center_hz": m["f_center_hz"], "obw_hz": m["obw_hz"], "snr_db": m["snr_db"],
            "power_dbfs": m["power_dbfs"], "continuous": b.get("continuous"), "period_ms": b.get("period_ms"),
            "duty": b.get("duty"), "settings": settings}


def _obs_row(r: dict) -> dict:
    m = json.loads(r["measurements_json"]) if r.get("measurements_json") else {
        "time": r["time"], "f_center_hz": r["center_freq_hz"], "obw_hz": r["bandwidth_hz"],
        "snr_db": r["snr_db"], "power_dbfs": r["power_dbfs"]}
    return {"id": r["id"], "status": r["history_status"], "iq": r["iq_meta_path"],
            **_brief(m, json.loads(r["settings_json"]))}

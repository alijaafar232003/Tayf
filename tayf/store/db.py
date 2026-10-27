"""SQLite access: antennas, recordings, sessions."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..core.models import ReceiverSettings, utc_now
from ..core.sigmf_io import SigMFRecording

SCHEMA = Path(__file__).with_name("schema.sql")

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

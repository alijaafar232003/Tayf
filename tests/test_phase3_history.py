"""Phase 3 step 4: observations, signal profiles and history."""
import sqlite3
import time

import numpy as np
import pytest

from tayf.core.models import ReceiverSettings
from tayf.store import history as H
from tayf.store.db import Store

SET = ReceiverSettings(100e6, 2e6, False, 32, 24, "whip-5dbi", hardware="HackRF One").to_dict()


def obs(f, obw, t="2026-11-04T10:00:00+00:00", cont=True, period=None, power=-30.0):
    return {"f_center_hz": f, "obw_hz": obw, "time": t, "offset_hz": 0.0, "snr_db": 30.0, "power_dbfs": power,
            "noise_dbfs_hz": -120.0, "bursts": {"continuous": cont, "period_ms": period, "duty": 1.0 if cont else 0.4}}


def test_matching_rules():
    assert H.matches(99.9e6 + 5e3, 150e3, 99.9e6, 140e3)            # same FM station, small drift
    assert not H.matches(100.0e6, 150e3, 99.9e6, 150e3)              # neighbour 100 kHz away
    assert not H.matches(433.92e6, 10e3, 433.92e6, 200e3)            # different width at the same place
    assert H.matches(2.44e9 + 15e3, 2e6, 2.44e9, 2e6)                # ppm error at 2.4 GHz
    assert H.origin_of("synthetic") == "sim" and H.origin_of("HackRF One") == "real"


def test_new_known_changed(tmp_path):
    st = Store(tmp_path / "t.db")
    sid = st.start_session("live", ReceiverSettings(100e6, 2e6))
    oid, h = st.save_observation(sid, "real", obs(99.9e6, 150e3), SET)
    assert h["status"] == "new" and h["previous"] == []
    _, h = st.save_observation(sid, "real", obs(99.902e6, 155e3, "2026-11-05T10:00:00+00:00"), SET)
    assert h["status"] == "known" and h["profile"]["n_observations"] == 2
    assert h["first_seen"].startswith("2026-11-04") and h["previous"][0]["id"] == oid
    _, h = st.save_observation(sid, "real", obs(99.9e6, 75e3, "2026-11-06T10:00:00+00:00"), SET)
    assert h["status"] == "changed" and "bandwidth" in h["changes"][0]
    # the neighbour and simulated data start their own profiles
    assert st.save_observation(sid, "real", obs(100.0e6, 150e3), SET)[1]["status"] == "new"
    assert st.save_observation(sid, "sim", obs(99.9e6, 150e3), SET)[1]["status"] == "new"


def test_burst_and_strength_changes():
    a = {**obs(314.4e6, 30e3, cont=False, period=100.0), "settings": SET}
    assert H.changes(H_brief(a), H_brief({**a, "bursts": {"continuous": False, "period_ms": 50.0}})) == \
        ["burst period 100.0 → 50.0 ms"]
    assert "strength" in H.changes(H_brief(a), H_brief({**a, "power_dbfs": -50.0}))[0]
    other_gain = {**SET, "lna_gain_db": 40}
    assert H.changes(H_brief(a), {**H_brief({**a, "power_dbfs": -50.0}), "settings": other_gain}) == []


def H_brief(m):
    from tayf.store.db import _brief
    return _brief(m, m.get("settings", SET))


def test_detector_sightings_count_sessions(tmp_path):
    st = Store(tmp_path / "t.db")
    s1 = st.start_session("live", ReceiverSettings(100e6, 2e6))
    s2 = st.start_session("live", ReceiverSettings(100e6, 2e6))
    tr = {"id": 1, "f_center_hz": 99.9e6, "bandwidth_hz": 120e3, "max_bandwidth_hz": 150e3, "first_seen": 1e9,
          "last_seen": 1e9 + 60, "duty": 1.0, "max_snr_db": 30, "snr_db": 28, "spur": False}
    spur = {**tr, "id": 2, "f_center_hz": 120e6, "spur": True}
    assert st.add_candidate_summaries(s1, "real", [tr, spur]) == 1       # spurs are not history
    st.add_candidate_summaries(s2, "real", [{**tr, "first_seen": 2e9, "last_seen": 2e9 + 5}])
    seen = st.sightings(99.9e6, 150e3, "real")
    assert seen["sessions"] == 2 and seen["first_seen"].startswith("2001")
    assert st.seen_before(99.9e6, 150e3, "real", s2)["sessions"] == 1
    assert st.seen_before(99.9e6, 150e3, "sim", s2)["sessions"] == 0


def test_old_database_is_migrated(tmp_path):
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE observations (id INTEGER PRIMARY KEY, session_id INTEGER, time TEXT, "
                      "center_freq_hz REAL, bandwidth_hz REAL, settings_json TEXT);")
    con.close()
    st = Store(p)
    cols = {r["name"] for r in st._q("PRAGMA table_info(observations)")}
    assert {"origin", "measurements_json"} <= cols


def test_inspection_is_saved_with_iq(tmp_path):
    from fastapi.testclient import TestClient
    from tayf.api.app import create_app
    from tayf.core.sigmf_io import SigMFRecording
    with TestClient(create_app(tmp_path / "data")) as c:
        rid = c.post("/api/recordings/synthetic").json()["id"]
        c.post("/api/session", json={"recording_id": rid})
        r1 = c.post("/api/session/inspect", json={"f_hz": 100.3e6, "bw_hz": 2e3}).json()
        assert r1["history"]["status"] == "new" and r1["observation_id"] >= 1
        rec = SigMFRecording.open(r1["iq_path"])
        assert rec.datatype == "cf32_le" and abs(rec.settings.center_freq_hz - 100.3e6) < 1
        x = rec.read(0, rec.n_samples).samples
        X = np.abs(np.fft.fft(x)) ** 2 / len(x) ** 2                   # the -50 dBFS tone, at the centre
        k = int(np.argmax(X))
        assert abs(np.fft.fftfreq(len(x), 1 / rec.settings.sample_rate_hz)[k]) < 200
        assert 10 * np.log10(X[max(k - 2, 0):k + 3].sum()) == pytest.approx(-50, abs=1)
        r2 = c.post("/api/session/inspect", json={"f_hz": 100.3e6, "bw_hz": 2e3}).json()
        assert r2["history"]["status"] == "known" and len(r2["history"]["previous"]) == 1
        assert c.get("/api/history", params={"f_hz": 100.3e6, "bw_hz": 2e3}).json()["profile"]["n_observations"] == 2
        # detector tracks are saved when the session ends, and the next session sees them as known
        c.post("/api/session/control", json={"playing": True})
        m = c.app.state.manager
        for _ in range(30):
            m.next_frame()
            time.sleep(0.02)
        c.delete("/api/session")
        c.post("/api/session", json={"recording_id": rid})
        c.post("/api/session/control", json={"playing": True})
        for _ in range(30):
            m.next_frame()
            time.sleep(0.02)
        tone = next(t for t in m.detections()["tracks"] if abs(t["f_center_hz"] - 100.3e6) < 5e3)
        assert tone["seen"]["sessions"] >= 1 and tone["seen"]["first_seen"]
        c.delete("/api/session")

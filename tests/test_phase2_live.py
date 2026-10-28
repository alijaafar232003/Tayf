"""Phase 2 live mode, tested with the simulated HackRF (no hardware needed)."""
import time

import numpy as np
import pytest

from tayf.dsp.dc import remove_dc
from tayf.session.manager import SessionManager, measure_health
from tayf.core.models import IQBlock, utc_now
from tayf.core.sigmf_io import SigMFRecording
from tayf.sources.live import offset_for
from tayf.store.db import Store


def wait_frame(m, timeout=3.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        f = m.next_frame()
        if f is not None:
            return f
        time.sleep(0.05)
    raise TimeoutError("no live frame")


@pytest.fixture
def manager(tmp_path):
    m = SessionManager(Store(tmp_path / "t.db"))
    yield m
    m.close()


def test_offset_tuning_moves_lo_not_view(manager):
    manager.open_live("sim", center_hz=98.5e6, sample_rate_hz=2e6)
    st = manager.status()
    assert st["tuned_hz"] == 98.5e6
    assert st["center_freq_hz"] == 98.5e6 + offset_for(2e6) == 98.7e6
    f = wait_frame(manager)
    assert f.freqs_hz[0] < 98.5e6 < f.freqs_hz[-1]            # tuned frequency is in view
    manager.control({"offset_tuning": False})
    assert manager.status()["center_freq_hz"] == 98.5e6


def test_retune_and_rate_change(manager):
    manager.open_live("sim", center_hz=98.5e6, sample_rate_hz=2e6)
    wait_frame(manager)
    manager.control({"center_hz": 433.92e6, "sample_rate_hz": 8e6})
    f = wait_frame(manager)
    assert f.freqs_hz[0] == pytest.approx(433.92e6 + offset_for(8e6) - 4e6, abs=8e6 / 2048)
    assert len(f.freqs_hz) == 2048


def test_gain_changes_adc_level_and_clipping(manager):
    manager.open_live("sim", center_hz=98.5e6, sample_rate_hz=2e6, lna_db=16, vga_db=20)
    wait_frame(manager)
    low = manager.status()["health"]["level_dbfs"]
    manager.control({"lna_db": 24, "vga_db": 30})              # +18 dB
    wait_frame(manager)
    mid = manager.status()["health"]["level_dbfs"]
    assert mid - low == pytest.approx(18, abs=2)
    manager.control({"amp": True, "lna_db": 40, "vga_db": 62})
    wait_frame(manager)
    assert manager.status()["health"]["level_hint"] == "clipping"


def test_dc_removal_flattens_center_bin():
    rng = np.random.default_rng(0)
    x = (0.01 * (rng.standard_normal(65536) + 1j * rng.standard_normal(65536)) + 0.05).astype(np.complex64)
    assert abs(x.mean()) > 0.04
    assert abs(remove_dc(x).mean()) < 1e-6


def test_health_flags():
    quiet = IQBlock(np.full(1000, 0.003 + 0j), 2e6, 0, utc_now())
    hot = IQBlock(np.full(1000, 1.0 + 1j), 2e6, 0, utc_now())
    assert measure_health(quiet)["level_hint"] == "low"
    assert measure_health(hot)["level_hint"] == "clipping"


def test_record_from_live_with_retune(tmp_path):
    m = SessionManager(Store(tmp_path / "t.db"), tmp_path / "rec")
    try:
        m.open_live("sim", center_hz=98.5e6, sample_rate_hz=2e6)
        time.sleep(0.2)
        m.start_recording()
        time.sleep(0.6)
        m.control({"center_hz": 101.3e6})
        time.sleep(0.6)
        saved = m.stop_recording()
        assert saved["segments"] == 2 and "." not in saved["name"]
        rec = SigMFRecording.open(tmp_path / "rec" / f"{saved['name']}.sigmf-meta")
        assert rec.duration_s == pytest.approx(1.2, abs=0.3)
        caps = rec.meta["captures"]
        assert caps[0]["core:frequency"] == 98.7e6 and caps[1]["core:frequency"] == 101.5e6
        assert rec.read(caps[1]["core:sample_start"] + 10, 100).center_freq_hz == 101.5e6
        assert m.store.recordings()[0]["name"] == saved["name"]
        # A sample-rate change ends the recording (one file, one rate)
        m.start_recording()
        time.sleep(0.3)
        m.control({"sample_rate_hz": 4e6})
        assert m.status()["recording"] is None and len(m.store.recordings()) == 2
    finally:
        m.close()


def test_live_api_with_sim(tmp_path):
    from fastapi.testclient import TestClient
    from tayf.api.app import create_app
    with TestClient(create_app(tmp_path / "data")) as c:
        st = c.post("/api/session/live", json={"radio": "sim", "center_hz": 101.3e6}).json()
        assert st["mode"] == "live" and st["tuned_hz"] == 101.3e6
        assert c.get("/api/devices").json()["in_use"] is True
        assert c.post("/api/session/control", json={"sample_rate_hz": 3e6}).status_code == 400
        st = c.post("/api/session/control", json={"vga_db": 31}).json()
        assert st["gains"]["vga_db"] in (30, 32)                 # snapped to 2 dB steps
        assert c.get("/api/session/overview").status_code == 409  # recordings only
        c.delete("/api/session")
        assert c.get("/api/status").json()["recent_sessions"][0]["mode"] == "live"

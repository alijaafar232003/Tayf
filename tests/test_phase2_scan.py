"""Phase 2 band scan (hackrf_sweep), tested without hardware."""
import struct
import sys
import time

import numpy as np
import pytest

from tayf.presets import BANDS
from tayf.session.manager import SessionManager
from tayf.sources.radio import Gains
from tayf.sources.sweep import HackRFSweep, SweepAssembler, check_range
from tayf.store.db import Store


def test_assembler_stitches_segments_and_crops():
    asm = SweepAssembler(88e6, 108e6)
    bin_hz = 1e6
    out = []
    for _ in range(2):                                   # two passes over 87-110 MHz in 5 MHz segments
        for lo in range(87, 110, 5):
            p = np.full(5, float(lo), dtype=np.float32)
            f = asm.add(int(lo * 1e6), int((lo + 5) * 1e6), p)
            if f is not None:
                out.append(f)
    assert len(out) == 1                                 # the second pass completed the first
    f = out[0]
    assert len(f.freqs_hz) == 20 and f.bin_width_hz == bin_hz
    assert f.freqs_hz[0] == 88.5e6 and f.freqs_hz[-1] == 107.5e6
    assert f.power_db[0] == 87 and f.power_db[-1] == 107    # bins came from the right segments
    assert not np.isnan(f.power_db).any()


def test_range_checks():
    with pytest.raises(ValueError):
        check_range(100e6, 90e6, 25e3)
    with pytest.raises(ValueError):
        check_range(1e6, 6000e6, 5e3)                    # over a million bins
    check_range(87e6, 109e6, 25e3)


FAKE_SWEEP = r'''
import struct, sys, time
out = sys.stdout.buffer
while True:
    for lo in (90_000_000, 95_000_000):
        p = [-60.0] * 200
        if lo == 95_000_000:
            p[100] = -20.0                               # a "station" at 95 + 100.5 * 25 kHz
        out.write(struct.pack("<IQQ", 16 + 4 * len(p), lo, lo + 5_000_000) + struct.pack(f"<{len(p)}f", *p))
    out.flush()
    time.sleep(0.01)
'''


def test_hackrf_sweep_binary_parser(tmp_path, monkeypatch):
    fake = tmp_path / "fake_sweep.py"
    fake.write_text(FAKE_SWEEP)
    monkeypatch.setattr(HackRFSweep, "command", lambda self: [sys.executable, str(fake)])
    s = HackRFSweep(90e6, 100e6, 25e3, Gains())
    s.exe = sys.executable
    s.start()
    try:
        time.sleep(0.3)
        f = s.take()
        assert f is not None and len(f.freqs_hz) == 400
        i = int(np.argmax(f.power_db))
        assert f.freqs_hz[i] == pytest.approx(95e6 + 100.5 * 25e3)
        assert s.sweep_rate > 10
    finally:
        s.close()
    assert s.error is None


def test_scan_session_with_sim(tmp_path):
    m = SessionManager(Store(tmp_path / "t.db"), tmp_path / "rec")
    try:
        m.open_scan("sim", 87e6, 109e6, 25e3, preset_id="fm")
        time.sleep(0.2)
        f = m.next_frame()
        assert f is not None and f.freqs_hz[0] == pytest.approx(87e6 + 12.5e3)
        assert f.freqs_hz[np.argmax(f.power_dbfs)] == pytest.approx(101.3e6, abs=25e3)   # strongest sim station
        st = m.status()
        assert st["mode"] == "scan" and st["sweeps"] > 0 and st["preset_id"] == "fm"
        m.control({"f_lo_hz": 900e6, "f_hi_hz": 1000e6, "bin_hz": 100e3})
        st = m.status()
        assert st["f_lo_hz"] == 900e6 and st["bin_hz"] == 100e3 and st["preset_id"] is None
        time.sleep(0.1)
        assert len(m.next_frame().freqs_hz) == 1000
        # Clicking a signal opens live mode there; the scan stops first.
        m.open_live("sim", center_hz=935.3e6, sample_rate_hz=2e6)
        assert m.status()["mode"] == "live"
    finally:
        m.close()


def test_scan_api_and_presets(tmp_path):
    from fastapi.testclient import TestClient
    from tayf.api.app import create_app
    with TestClient(create_app(tmp_path / "data")) as c:
        presets = c.get("/api/presets").json()
        assert {p["id"] for p in presets} == {b.id for b in BANDS}
        assert next(p for p in presets if p["id"] == "fm")["verified"] is True
        assert c.post("/api/session/scan", json={"radio": "sim", "f_lo_hz": 5e6, "f_hi_hz": 4e6}).status_code == 400
        st = c.post("/api/session/scan", json={"radio": "sim", "f_lo_hz": 430e6, "f_hi_hz": 440e6,
                                               "bin_hz": 10e3}).json()
        assert st["mode"] == "scan" and st["bin_hz"] == 10e3
        assert c.get("/api/devices").json()["in_use"] is True
        st = c.post("/api/session/control", json={"lna_db": 20}).json()
        assert st["gains"]["lna_db"] in (16, 24)            # snapped to 8 dB steps
        c.delete("/api/session")

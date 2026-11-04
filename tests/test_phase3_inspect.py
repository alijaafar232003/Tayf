"""Phase 3 Signal Inspector, on synthetic signals with known answers."""
import numpy as np
import pytest

from tayf.dsp.inspect import inspect

FS = 2e6
N = int(0.5 * FS)
T = np.arange(N) / FS


def noise(seed=0, power_db=-60.0):
    rng = np.random.default_rng(seed)
    return np.sqrt(10 ** (power_db / 10) / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))


def at(offset, baseband):
    return baseband * np.exp(2j * np.pi * offset * T)


def rrc(beta, sps_, span=8):
    t = np.arange(-span * sps_, span * sps_ + 1) / sps_
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + beta * (4 / np.pi - 1)
        elif beta and abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def test_tone_is_narrow_continuous_and_at_the_right_frequency():
    x = noise() + at(300e3 + 1.2e3, 10 ** (-30 / 20) * np.ones(N))      # 1.2 kHz off from where we look
    r = inspect(x, FS, 300e3, 2e3, f_tuned_hz=100e6)
    assert r.f_center_hz == pytest.approx(100.3012e6, abs=200)
    assert r.offset_hz == pytest.approx(1.2e3, abs=200)
    assert r.obw_hz < 2e3 and r.bursts.continuous
    assert r.power_dbfs == pytest.approx(-30, abs=0.5) and r.symbol_rate_hz is None


def test_wideband_fm_bandwidth_is_near_carson():
    fm, dev = 15e3, 75e3                                                  # Carson: 2 (75 + 15) = 180 kHz
    x = noise() + at(-400e3, 10 ** (-20 / 20) * np.exp(1j * dev / fm * np.sin(2 * np.pi * fm * T)))
    r = inspect(x, FS, -400e3, 200e3)
    assert 150e3 < r.obw_hz < 200e3
    assert r.bursts.continuous and r.snr_db > 25
    assert r.views["constellation"] is None                               # constant envelope: no constellation


def test_ook_bursts_are_timed():
    on = (T % 5e-3) < 2e-3                                                # 2 ms on, every 5 ms
    x = noise() + at(200e3, 10 ** (-35 / 20) * on)
    r = inspect(x, FS, 200e3, 5e3)
    b = r.bursts
    assert not b.continuous and b.count >= 99
    assert b.length_ms["median"] == pytest.approx(2.0, abs=0.1)
    assert b.period_ms == pytest.approx(5.0, abs=0.05) and b.duty == pytest.approx(0.4, abs=0.03)


def test_fsk_shows_two_frequency_levels():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, N // 200 + 1)                               # 10 kbit/s
    f_inst = np.repeat(np.where(bits, 20e3, -20e3), 200)[:N]              # +-20 kHz deviation
    x = noise() + at(-200e3, 10 ** (-30 / 20) * np.exp(2j * np.pi * np.cumsum(f_inst) / FS))
    r = inspect(x, FS, -200e3, 60e3)
    assert len(r.fsk_levels_hz) == 2
    assert sorted(r.fsk_levels_hz) == pytest.approx([-20e3, 20e3], abs=4e3)


def test_qpsk_symbol_rate_and_constellation():
    rs, sps_ = 50e3, 40                                                   # 50 kbaud at 2 MS/s
    rng = np.random.default_rng(4)
    sym = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, N // sps_ + 20)))
    up = np.zeros(len(sym) * sps_, complex)
    up[::sps_] = sym
    bb = np.convolve(up, rrc(0.35, sps_), mode="same")[:N]
    x = noise(power_db=-70) + at(150e3 + 500, 10 ** (-25 / 20) * bb / np.sqrt(np.mean(np.abs(bb) ** 2)))
    r = inspect(x, FS, 150e3, 70e3)
    assert r.symbol_rate_hz == pytest.approx(rs, rel=0.02)
    assert r.constellation_order == 4
    pts = np.array(r.views["constellation"]["i"]) + 1j * np.array(r.views["constellation"]["q"])
    # four tight clusters: every point near one of the 4 ideal positions (any common rotation)
    ideal = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    rot = np.angle(np.mean(pts ** 4)) / 4 - np.pi / 4
    err = np.min(np.abs(pts[:, None] * np.exp(-1j * rot) - ideal[None, :]), axis=1)
    assert np.median(err) < 0.25


def test_inspect_through_the_session(tmp_path):
    import time
    from fastapi.testclient import TestClient
    from tayf.api.app import create_app
    with TestClient(create_app(tmp_path / "data")) as c:
        rid = c.post("/api/recordings/synthetic").json()["id"]
        c.post("/api/session", json={"recording_id": rid})
        c.post("/api/session/control", json={"playing": True})
        m = c.app.state.manager
        for _ in range(40):
            m.next_frame()
            time.sleep(0.02)
        tone = next(t for t in m.tracker.table() if abs(t["f_center_hz"] - 100.3e6) < 5e3)
        r = c.post("/api/session/inspect", json={"track": tone["id"], "seconds": 1.0}).json()
        assert r["f_center_hz"] == pytest.approx(100.3e6, abs=300) and r["bursts"]["continuous"]
        ook = c.post("/api/session/inspect", json={"f_hz": 100.65e6, "bw_hz": 10e3}).json()
        assert not ook["bursts"]["continuous"] and ook["bursts"]["count"] >= 1
        assert c.post("/api/session/inspect", json={"f_hz": 105e6}).status_code == 409   # outside the span
        assert c.post("/api/session/inspect", json={"track": 999}).status_code == 404
        c.delete("/api/session")


def test_live_capture_is_fresh_and_long(tmp_path):
    from tayf.session.manager import SessionManager
    from tayf.store.db import Store
    m = SessionManager(Store(tmp_path / "t.db"), tmp_path / "rec")
    try:
        m.open_live("sim", center_hz=101.3e6, sample_rate_hz=2e6, lna_db=16, vga_db=30)
        r = m.inspect(f_hz=101.3e6, bw_hz=200e3, seconds=0.8)              # longer than the 0.5 s ring buffer
        assert r["duration_s"] == pytest.approx(0.8, abs=0.02) and r["obw_hz"] > 100e3
    finally:
        m.close()


def test_neighbours_in_the_channel_are_not_measured():
    fm = lambda off, dev: at(off, 10 ** (-25 / 20) * np.exp(1j * dev / 3e3 * np.sin(2 * np.pi * 3e3 * T)))
    x = noise() + fm(0, 40e3) + fm(250e3, 40e3) + fm(-230e3, 40e3)          # a station with close neighbours
    r = inspect(x, FS, 0, 120e3)
    assert r.obw_hz < 120e3 and abs(r.offset_hz) < 2e3
    assert any("Other signals" in n for n in r.notes)

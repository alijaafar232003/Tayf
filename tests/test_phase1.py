"""Phase 1 checks: SigMF round trip, spectrum calibration, store, session, API."""
import json

import numpy as np
import pytest

from tayf.core.models import ReceiverSettings
from tayf.core.sigmf_io import SigMFRecording, import_raw, write_recording
from tayf.dsp.spectrum import rbw_hz, welch_power
from tayf.synth import Scene, write_scene

FS = 2e6


def tone(f_hz, power_dbfs, n, fs=FS):
    return (10 ** (power_dbfs / 20) * np.exp(2j * np.pi * f_hz * np.arange(n) / fs)).astype(np.complex64)


@pytest.mark.parametrize("datatype,tol", [("ci8", 1 / 128), ("ci16_le", 1e-4), ("cf32_le", 1e-6)])
def test_sigmf_round_trip(tmp_path, datatype, tol):
    x = tone(100e3, -6, 10_000)
    settings = ReceiverSettings(433.92e6, FS, True, 16, 20, "whip-5dbi")
    write_recording(tmp_path / "t", x, settings, datatype)
    rec = SigMFRecording.open(tmp_path / "t.sigmf-meta")
    assert rec.n_samples == len(x)
    assert rec.settings == settings
    assert rec.comparable
    block = rec.read(1000, 500)
    assert block.start_index == 1000 and len(block) == 500
    assert np.max(np.abs(block.samples - x[1000:1500])) <= tol


def test_import_raw_hackrf_file(tmp_path):
    raw = tmp_path / "capture.bin"
    np.array([127, -128, 0, 64], dtype=np.int8).tofile(raw)
    meta = import_raw(raw, ReceiverSettings(100e6, FS), "ci8")
    assert json.loads(meta.read_text())["global"]["core:dataset"] == "capture.bin"
    rec = SigMFRecording.open(meta)
    assert rec.n_samples == 2
    assert np.allclose(rec.read(0, 2).samples, [127 / 128 - 1j, 0.5j])
    assert not rec.comparable


def test_tone_reads_its_power_at_its_frequency():
    f, p, _ = welch_power(tone(250e3, -20, 65_536), FS, nfft=2048)
    assert abs(f[np.argmax(p)] - 250e3) <= FS / 2048
    assert abs(p.max() - (-20)) < 1.6          # Hann scalloping loss is at most 1.4 dB


def test_noise_per_bin_is_total_noise_times_rbw_over_fs():
    """Processing gain: each bin holds only the noise inside one RBW."""
    rng = np.random.default_rng(0)
    total_dbfs = -30
    std = np.sqrt(10 ** (total_dbfs / 10) / 2)
    x = std * (rng.standard_normal(2_000_000) + 1j * rng.standard_normal(2_000_000))
    for nfft in (512, 4096):
        _, p, _ = welch_power(x, FS, nfft=nfft)
        expected = total_dbfs + 10 * np.log10(rbw_hz(FS, nfft) / FS)
        mean_db = 10 * np.log10(np.mean(10 ** (p / 10)))
        assert abs(mean_db - expected) < 0.3


def test_synthetic_scene_signals_visible(tmp_path):
    meta = write_scene(tmp_path / "scene", Scene(duration_s=1.0))
    rec = SigMFRecording.open(meta)
    block = rec.read(0, 262_144)
    f, p, _ = welch_power(block.samples, FS, nfft=4096)
    floor = np.median(p)
    for offset in (300e3, -400e3):              # tone and WBFM
        i = np.argmin(np.abs(f - offset))
        assert p[i - 3:i + 4].max() > floor + 10
    assert len(rec.meta["annotations"]) == 3


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient
    from tayf.api.app import create_app
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


def test_api_recording_flow(client):
    assert len(client.get("/api/antennas").json()) == 5
    assert [a["id"] for a in client.get("/api/antennas", params={"freq_hz": 433.92e6}).json()] \
        == ["whip-5dbi", "telescopic"]
    rid = client.post("/api/recordings/synthetic").json()["id"]
    st = client.post("/api/session", json={"recording_id": rid}).json()
    assert st["active"] and st["mode"] == "recording" and st["duration_s"] == pytest.approx(5.0)

    ov = client.get("/api/session/overview", params={"rows": 50, "nfft": 512}).json()
    assert ov["rows"] == 50 and ov["cols"] == 512

    client.post("/api/session/control", json={"nfft": 4096, "seek_s": 2.0})
    with client.websocket_connect("/ws/spectrum") as ws:
        msg = ws.receive()
        buf = msg.get("bytes")
        if buf is None:                       # status text first, then the frame
            buf = ws.receive()["bytes"]
        header = np.frombuffer(buf[:32], "<f8")
        power = np.frombuffer(buf[32:], "<f4")
        assert header[0] == pytest.approx(2.0) and len(power) == 4096

    assert client.post("/api/session/control", json={"nfft": 1000}).status_code == 400
    client.delete("/api/session")
    sessions = client.get("/api/status").json()["recent_sessions"]
    assert sessions[0]["ended_at"] is not None

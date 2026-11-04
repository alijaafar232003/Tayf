"""Phase 3 tracker: stable IDs, confirmation, gaps, closing."""
from tayf.dsp.detect import Detection
from tayf.dsp.track import CLOSE_AFTER_S, Tracker

SPAN = (90e6, 110e6)


def det(fc, bw=10e3, snr=12.0):
    return Detection(fc - bw / 2, fc + bw / 2, fc, bw, -60.0, snr, int(bw / 1e3))


def test_same_signal_keeps_its_id_and_blips_never_confirm():
    tr = Tracker()
    ids = set()
    for k in range(10):
        dets = [det(98.5e6 + (k % 3) * 1e3)]          # wobbles by a few kHz
        if k == 4:
            dets.append(det(101.0e6))                  # a one-frame blip
        ids |= {i for _, i in tr.update(dets, k * 0.05, SPAN) if _.f_center_hz < 100e6}
    assert len(ids) == 1
    table = tr.table()
    assert [t["f_center_hz"] // 1e5 for t in table] == [985]   # the blip is not listed
    assert table[0]["duty"] == 1.0


def test_bursts_keep_their_id_and_duty_cycle():
    tr = Tracker()
    ids = set()
    for k in range(40):                                # on 5 frames, off 5 frames
        dets = [det(314.4e6)] if (k // 5) % 2 == 0 else []
        ids |= {i for _, i in tr.update(dets, k * 0.05, (300e6, 330e6))}
    t = tr.table()[0]
    assert len(ids) == 1 and abs(t["duty"] - 0.5) < 0.05


def test_out_of_view_is_not_a_miss_and_silence_closes():
    tr = Tracker()
    for k in range(5):
        tr.update([det(98.5e6)], k * 0.05, SPAN)
    for k in range(5, 20):                             # retuned away: 98.5 MHz is out of view
        tr.update([], k * 0.05, (400e6, 410e6))
    t = tr.table()[0]
    assert t["duty"] == 1.0 and t["confirmed"]
    tr.update([], 0.25 + CLOSE_AFTER_S + 0.1, SPAN)
    assert tr.table() == []
    closed = tr.take_closed()
    assert len(closed) == 1 and closed[0]["hits"] == 5


def test_fragments_join_their_signal_and_neighbours_stay_apart():
    tr = Tracker()
    for k in range(10):
        dets = [det(100.70e6, 160e3, 30), det(100.80e6, 100e3, 30)]     # two stations 100 kHz apart
        if k % 2:
            dets += [det(100.635e6, 20e3, 7)]                           # an edge piece of the first
        out = tr.update(dets, k * 0.05, SPAN)
    table = tr.table()
    assert len(table) == 2                              # no fragment track, no merged station
    assert all(t["bandwidth_hz"] < 200e3 for t in table)


def test_two_stations_two_ids_and_seek_back_starts_over():
    tr = Tracker()
    for k in range(5):
        out = tr.update([det(97.1e6, 150e3), det(97.5e6, 150e3)], k * 0.05, SPAN)
    assert len({i for _, i in out}) == 2
    tr.update([det(97.1e6, 150e3)], 0.0, SPAN)        # time jumped back: tracks closed and saved
    assert len(tr.take_closed()) == 2


def test_session_tracks_the_synthetic_scene(tmp_path):
    import time
    from fastapi.testclient import TestClient
    from tayf.api.app import create_app
    with TestClient(create_app(tmp_path / "data")) as c:
        rid = c.post("/api/recordings/synthetic").json()["id"]
        c.post("/api/session", json={"recording_id": rid})
        c.post("/api/session/control", json={"playing": True, "speed": 1})
        m = c.app.state.manager
        for _ in range(60):                            # ~3 s of the scene
            m.next_frame()
            time.sleep(0.02)
        d = c.get("/api/session/detections").json()
        found = sorted(round(t["f_center_hz"] / 1e3) for t in d["tracks"])
        assert found[:1] == [99600] and 100300 in found  # WBFM (-400 kHz) and the tone (+300 kHz)
        assert c.post("/api/session/control", json={"min_snr_db": 1}).status_code == 400
        assert c.post("/api/session/control", json={"detect": False}).json()["detector"]["on"] is False
        c.delete("/api/session")
        assert m.status()["active"] is False and len(m.closed_tracks) >= 2

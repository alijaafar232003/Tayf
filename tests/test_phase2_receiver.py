"""Phase 2 receiver characterization: the parts that need no hardware."""
import numpy as np
import pytest

from tayf.receiver.characterize import fit_floor, floor_model
from tayf.receiver.profile import load_profile, save_profile


def test_fit_floor_recovers_the_three_noise_sources():
    q, n_vga, n_front = -121.0, -144.5, -173.0
    rng = np.random.default_rng(1)
    rows = [{"lna_db": l, "vga_db": v,
             "floor_dbfs_hz": float(floor_model(l, v, q, n_vga, n_front) + rng.normal(0, 0.3))}
            for l in range(0, 41, 8) for v in range(0, 63, 10)]
    f = fit_floor(rows)
    assert f["adc_floor_dbfs_hz"] == pytest.approx(q, abs=0.5)
    assert f["n_vga_dbfs_hz"] == pytest.approx(n_vga, abs=0.7)
    assert f["n_front_dbfs_hz"] == pytest.approx(n_front, abs=1.5)
    assert f["lna_db_front_dominates"] == pytest.approx(n_vga - n_front, abs=1.5)


def test_profile_save_and_load(tmp_path):
    with pytest.raises(ValueError):
        save_profile({}, "antenna on", tmp_path)
    path = save_profile({"spurs": {"spurs": [], "_full": (1, 2)}, "x": 1}, "antenna removed", tmp_path)
    assert path.exists() and "antenna_removed" in path.name
    p = load_profile(tmp_path)
    assert p["condition"] == "antenna removed" and p["x"] == 1
    assert "_full" not in p["spurs"]                     # plotting data is not saved


# --- Receiver Check ------------------------------------------------------------

from tayf.receiver.check import candidate_gains, choose, vga_to_clear_adc   # noqa: E402

FIT = {"adc_floor_dbfs_hz": -121.0, "n_vga_dbfs_hz": -144.5, "n_front_dbfs_hz": -173.0, "fc_hz": 1e8}


def _row(lna, vga, floor, snrs=(), clip=0.0, freqs=(99.9e6, 100.1e6, 100.7e6)):
    """A measured setting; its spectrum has the given SNRs at fixed frequencies."""
    fa = np.linspace(95e6, 105e6, 1001)
    p = np.zeros_like(fa)
    for f, s in zip(freqs, snrs):
        p[np.argmin(abs(fa - f))] = s
    return {"amp": False, "lna_db": lna, "vga_db": vga, "total_db": lna + vga, "level_dbfs": -20.0,
            "clip_pct": clip, "floor_dbfs_hz": floor,
            "peaks": [{"freq_hz": f, "snr_db": s} for f, s in zip(freqs, snrs)], "_spec": (fa, p, 0.0)}


def test_gain_rule_lifts_noise_above_the_adc():
    v = vga_to_clear_adc(FIT, 32)
    rx = 10 * np.log10(10 ** (FIT["n_vga_dbfs_hz"] / 10) + 10 ** ((FIT["n_front_dbfs_hz"] + 32) / 10)) + v
    assert rx >= FIT["adc_floor_dbfs_hz"] + 6 > rx - 2                  # just enough, in 2 dB steps
    assert len(candidate_gains(FIT)) == 8 and all(not g.amp for g in candidate_gains(FIT))


def test_check_prefers_least_gain_among_equals_and_rules_out_clipping():
    rows = [_row(16, 30, -112, (38, 28, 26)), _row(32, 24, -108, (45, 36, 34)),
            _row(40, 16, -109, (45.5, 35.8, 34)), _row(40, 26, -99, (46, 37, 34), clip=2.0)]
    best, why = choose(rows, FIT)
    assert (best["lna_db"], best["vga_db"]) == (32, 24)                 # 40/16 is no better, same gain
    assert "clipping" in rows[3]["problems"] and "Ruled out" in why[-1]


def test_check_flags_overload():
    # +8 dB of LNA raised the floor by 14 dB: the receiver is making its own distortion
    rows = [_row(24, 30, -110, (40,)), _row(32, 30, -96, (41,))]
    choose(rows, FIT)
    assert rows[1]["problems"] == ["overload"]


def test_quiet_band_goes_for_sensitivity():
    # No signals. Input-referred floor (floor - gain): 16/30 -> -158, 40/30 -> -173.6
    rows = [_row(16, 30, -112.0), _row(32, 26, -112.1), _row(40, 20, -112.3), _row(40, 30, -103.6)]
    best, why = choose(rows, FIT)
    assert (best["lna_db"], best["vga_db"]) == (40, 30) and "sensitivity" in why[0]


def test_check_session_rules(tmp_path):
    import time
    from tayf.session.manager import SessionManager
    from tayf.store.db import Store
    m = SessionManager(Store(tmp_path / "t.db"), tmp_path / "rec")
    try:
        with pytest.raises(RuntimeError):
            m.receiver_check(None)                                        # no session
        m.open_live("sim", center_hz=98.5e6, sample_rate_hz=2e6, lna_db=16, vga_db=20)
        time.sleep(0.3)
        m.start_recording()
        with pytest.raises(RuntimeError, match="recording"):
            m.receiver_check(None)                                        # would split the recording
        m.stop_recording()
        r = m.receiver_check(None)
        assert len(r["rows"]) == 8 and "_spec" not in r["rows"][0]
        assert m.status()["gains"]["lna_db"] == 16 and m.status()["gains"]["vga_db"] == 20   # put back
        assert not m.status()["checking"]
    finally:
        m.close()

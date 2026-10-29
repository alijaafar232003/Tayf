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

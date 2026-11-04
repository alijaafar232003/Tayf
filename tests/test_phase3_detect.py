"""Phase 3 FCME detector, on synthetic spectra with known truth."""
import numpy as np
import pytest

from tayf.core.models import SpectrumFrame, utc_now
from tayf.dsp.detect import DetectorConfig, detect, fcme, threshold_factor

N, DF, FC, K = 2048, 1e3, 100e6, 200


def frame(signals=(), tilt_db=0.0, dc_db=0.0, seed=0, k=K):
    """Averaged noise (Gamma(K) around -80 dBFS/bin, optionally tilted) plus signals:
    (offset_hz, width_hz, snr_db)."""
    rng = np.random.default_rng(seed)
    f = FC + (np.arange(N) - N // 2) * DF
    floor = -80 + tilt_db * (np.arange(N) / N - 0.5)
    x = 10 ** (floor / 10) * rng.gamma(k, 1 / k, N)
    for off, width, snr in signals:
        m = np.abs(f - (FC + off)) <= max(width / 2, DF / 2)
        x[m] += 10 ** ((floor[m] + snr) / 10)
    x[N // 2] += 10 ** ((-80 + dc_db) / 10) if dc_db else 0
    return SpectrumFrame(f, (10 * np.log10(x)).astype(np.float32), DF, FC, utc_now(), n_averages=k)


def test_threshold_factor_matches_the_false_alarm_rate():
    rng = np.random.default_rng(1)
    t = threshold_factor(1e-3, 50)
    x = rng.gamma(50, 1 / 50, 2_000_000)
    assert np.mean(x > t) == pytest.approx(1e-3, rel=0.15)


def test_fcme_finds_the_noise_mean_despite_strong_signals():
    rng = np.random.default_rng(2)
    x = rng.gamma(K, 1 / K, N)
    x[200:600] *= 100                                  # a fifth of the band filled with a strong signal
    mu, clean = fcme(x, threshold_factor(1e-3, K))
    assert mu == pytest.approx(1.0, abs=0.02)          # the plain mean would be ~20
    assert not clean[200:600].any() and clean[:200].mean() > 0.95


def test_detects_the_right_signals():
    sig = [(-300e3, 0, 10.0), (200e3, 0, 4.0), (500e3, 200e3, 8.0)]
    dets, info = detect(frame(sig, tilt_db=6.0, dc_db=15.0), DetectorConfig(), lo_hz=FC, spurs_hz=[FC + 800e3])
    fcs = sorted(d.f_center_hz - FC for d in dets)
    assert len(dets) == 2                              # the 4 dB tone is below min_snr_db; DC is excluded
    assert fcs[0] == pytest.approx(-300e3, abs=DF) and fcs[1] == pytest.approx(500e3, abs=5 * DF)
    block = max(dets, key=lambda d: d.bandwidth_hz)
    assert block.bandwidth_hz == pytest.approx(200e3, rel=0.1)
    assert abs(info["floor_dbfs"] + 80) < 1             # the tilted floor is followed


def test_no_false_alarms_in_noise_and_spurs_are_flagged():
    total = sum(len(detect(frame(seed=s, tilt_db=3.0), lo_hz=FC)[0]) for s in range(20))
    assert total == 0                                  # 20 x 2048 noise bins, nothing found
    dets, _ = detect(frame([(800e3, 0, 20.0)]), lo_hz=FC, spurs_hz=[FC + 800e3])
    assert len(dets) == 1 and dets[0].spur


def test_flattening_with_the_receiver_shape():
    # A receiver floor that falls 6 dB towards both edges: without the shape the
    # edges look quiet and the middle hump can turn into detections.
    f = frame()
    shape = -6 * ((f.freqs_hz - FC) / (N * DF / 2)) ** 2
    hump = SpectrumFrame(f.freqs_hz, (f.power_dbfs + shape).astype(np.float32), DF, FC, utc_now(), n_averages=K)
    with_shape, _ = detect(hump, DetectorConfig(floor_window_frac=1 / 4), shape_db=shape, lo_hz=FC)
    assert with_shape == []

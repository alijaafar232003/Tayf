"""Synthetic test scene with known signals, written as a SigMF recording.

Because every signal's frequency, bandwidth and power is known exactly, the scene
is the ground truth for checking the spectrum now and the detector later.
The truth is stored as SigMF annotations in the meta file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core.models import ReceiverSettings, utc_now
from .core.sigmf_io import write_recording


def db_to_amp(power_dbfs: float) -> float:
    return 10 ** (power_dbfs / 20)


@dataclass
class Scene:
    sample_rate_hz: float = 2e6
    center_freq_hz: float = 100e6
    duration_s: float = 5.0
    noise_dbfs: float = -30.0           # total noise power across the whole span
    seed: int = 1


def default_signals(scene: Scene) -> list[dict]:
    """The signals in the default scene: offset from center, power, bandwidth."""
    return [
        # Tone 20 dB BELOW the total noise power: invisible in the time domain,
        # clearly visible in the spectrum thanks to FFT processing gain.
        {"label": "Tone", "offset_hz": 300e3, "power_dbfs": -50.0, "bw_hz": 0.0},
        # Broadcast-FM-like: 75 kHz deviation, about 200 kHz occupied (Carson's rule).
        {"label": "WBFM", "offset_hz": -400e3, "power_dbfs": -35.0, "bw_hz": 200e3,
         "deviation_hz": 75e3},
        # On-off keyed bursts (key-fob-like): 0.2 s long, once per second.
        {"label": "OOK", "offset_hz": 650e3, "power_dbfs": -40.0, "bw_hz": 10e3,
         "bit_rate": 2e3, "burst_s": 0.2, "period_s": 1.0},
    ]


def generate(scene: Scene, signals: list[dict], chunk_s: float = 0.25):
    """Yield complex64 chunks of the scene, keeping every signal phase-continuous."""
    rng = np.random.default_rng(scene.seed)
    fs = scene.sample_rate_hz
    n_total = int(scene.duration_s * fs)
    n_chunk = int(chunk_s * fs)
    noise_std = np.sqrt(db_to_amp(scene.noise_dbfs) ** 2 / 2)
    fm_phase = 0.0
    # Slowly varying "audio": a few tones, normalised to +-1.
    audio_tones = [(1.0e3, 0.5), (2.7e3, 0.3), (400.0, 0.2)]
    ook_bits = rng.integers(0, 2, size=10_000)

    for start in range(0, n_total, n_chunk):
        n = min(n_chunk, n_total - start)
        t = (start + np.arange(n)) / fs
        x = noise_std * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        for sig in signals:
            a = db_to_amp(sig["power_dbfs"])
            carrier = np.exp(2j * np.pi * sig["offset_hz"] * t)
            if sig["label"] == "Tone":
                x += a * carrier
            elif sig["label"] == "WBFM":
                audio = sum(w * np.sin(2 * np.pi * f * t) for f, w in audio_tones)
                inst = 2 * np.pi * sig["deviation_hz"] * audio / fs
                phase = fm_phase + np.cumsum(inst)
                fm_phase = phase[-1]
                x += a * carrier * np.exp(1j * phase)
            elif sig["label"] == "OOK":
                in_burst = (t % sig["period_s"]) < sig["burst_s"]
                bit = ook_bits[(t * sig["bit_rate"]).astype(int) % len(ook_bits)]
                x += a * carrier * (in_burst & (bit == 1))
        yield x.astype(np.complex64)


def annotations(scene: Scene, signals: list[dict]) -> list[dict]:
    fc, fs = scene.center_freq_hz, scene.sample_rate_hz
    out = []
    for sig in signals:
        half = max(sig["bw_hz"], 1.0) / 2
        out.append({
            "core:sample_start": 0,
            "core:sample_count": int(scene.duration_s * fs),
            "core:freq_lower_edge": fc + sig["offset_hz"] - half,
            "core:freq_upper_edge": fc + sig["offset_hz"] + half,
            "core:label": sig["label"],
            "core:comment": f"truth: {sig['power_dbfs']} dBFS",
            "core:generator": "tayf.synth",
        })
    return out


def write_scene(path: str | Path, scene: Scene | None = None, datatype: str = "ci8") -> Path:
    scene = scene or Scene()
    signals = default_signals(scene)
    settings = ReceiverSettings(scene.center_freq_hz, scene.sample_rate_hz,
                                hardware="synthetic")
    return write_recording(path, generate(scene, signals), settings, datatype, utc_now(),
                           description="Synthetic scene: tone, WBFM, OOK bursts in noise",
                           annotations=annotations(scene, signals))

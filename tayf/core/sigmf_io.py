"""Read and write SigMF recordings.

A SigMF recording is two files with the same base name:
  <name>.sigmf-data  raw interleaved I/Q samples
  <name>.sigmf-meta  JSON: sample rate, datatype, center frequency, time, hardware

Tayf-specific fields (gains, antenna) live in the "tayf:" extension namespace.
A raw file from hackrf_transfer can be imported without copying it: the meta file
points at it through "core:dataset" (a SigMF non-conforming dataset).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .models import IQBlock, ReceiverSettings, utc_now

SIGMF_VERSION = "1.0.0"
TAYF_EXT = {"name": "tayf", "version": "0.1.0", "optional": True}

# datatype -> (numpy component dtype, scale to full scale 1.0, offset)
DATATYPES = {
    "ci8": (np.int8, 1 / 128, 0.0),             # HackRF native
    "cu8": (np.uint8, 1 / 128, 127.5),          # RTL-SDR native
    "ci16_le": (np.dtype("<i2"), 1 / 32768, 0.0),
    "cf32_le": (np.dtype("<f4"), 1.0, 0.0),
}


def _bytes_per_sample(datatype: str) -> int:
    return 2 * np.dtype(DATATYPES[datatype][0]).itemsize


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def meta_path_for(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix in (".sigmf-data", ".sigmf-meta"):
        p = p.with_suffix("")
    return p.with_suffix(".sigmf-meta")


def to_complex(raw: np.ndarray, datatype: str) -> np.ndarray:
    """Interleaved I,Q components -> complex64 with full scale 1.0."""
    _, scale, offset = DATATYPES[datatype]
    x = raw.astype(np.float32)
    if offset:
        x -= offset
    x *= scale
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64)


def from_complex(samples: np.ndarray, datatype: str) -> np.ndarray:
    """complex (full scale 1.0) -> interleaved components; clips at full scale."""
    dtype, scale, offset = DATATYPES[datatype]
    inter = np.empty(2 * len(samples), dtype=np.float32)
    inter[0::2] = samples.real
    inter[1::2] = samples.imag
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        inter = np.clip(np.round(inter / scale + offset), info.min, info.max)
    return inter.astype(dtype)


def build_meta(settings: ReceiverSettings, datatype: str, start_time: datetime,
               description: str = "", dataset: str | None = None,
               annotations: list[dict] | None = None) -> dict:
    glob = {
        "core:datatype": datatype,
        "core:sample_rate": settings.sample_rate_hz,
        "core:version": SIGMF_VERSION,
        "core:description": description,
        "core:recorder": "Tayf",
        "core:extensions": [TAYF_EXT],
    }
    if settings.hardware:
        glob["core:hw"] = settings.hardware
    if dataset:
        glob["core:dataset"] = dataset
    for key in ("amp_on", "lna_gain_db", "vga_gain_db", "antenna_id", "bias_tee"):
        value = getattr(settings, key)
        if value is not None:
            glob[f"tayf:{key}"] = value
    return {
        "global": glob,
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": settings.center_freq_hz,
            "core:datetime": _iso(start_time),
        }],
        "annotations": annotations or [],
    }


def write_recording(base_path: str | Path, samples: np.ndarray | Iterable[np.ndarray],
                    settings: ReceiverSettings, datatype: str = "ci8",
                    start_time: datetime | None = None, description: str = "",
                    annotations: list[dict] | None = None) -> Path:
    """Write samples (one array or an iterable of chunks) as a SigMF recording.
    Returns the meta path."""
    meta = meta_path_for(base_path)
    data = meta.with_suffix(".sigmf-data")
    meta.parent.mkdir(parents=True, exist_ok=True)
    chunks = [samples] if isinstance(samples, np.ndarray) else samples
    with open(data, "wb") as f:
        for chunk in chunks:
            from_complex(np.asarray(chunk), datatype).tofile(f)
    m = build_meta(settings, datatype, start_time or utc_now(), description,
                   annotations=annotations)
    meta.write_text(json.dumps(m, indent=2))
    return meta


class SigMFWriter:
    """Write a recording while it happens: raw chunks go straight to the data file,
    and the meta file is (re)written at each new capture segment and at close.

    A capture segment starts whenever the frequency or gains change mid-recording,
    so one file can hold a whole tuning session with correct metadata for every part."""

    def __init__(self, base_path: str | Path, settings: ReceiverSettings,
                 datatype: str = "ci8", start_time: datetime | None = None,
                 description: str = ""):
        self.meta_path = meta_path_for(base_path)
        self.data_path = self.meta_path.with_suffix(".sigmf-data")
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.datatype = datatype
        self.settings = settings
        self.start_time = start_time or utc_now()
        self.description = description
        self.n_samples = 0
        self.extra_global: dict = {}
        self.captures: list[dict] = []
        self._f = open(self.data_path, "wb")
        self.new_capture(0, settings)

    def new_capture(self, sample_start: int, settings: ReceiverSettings) -> None:
        cap = {
            "core:sample_start": int(sample_start),
            "core:frequency": settings.center_freq_hz,
            "core:datetime": _iso(self.start_time + timedelta(
                seconds=sample_start / self.settings.sample_rate_hz)),
        }
        for key in ("amp_on", "lna_gain_db", "vga_gain_db", "antenna_id"):
            value = getattr(settings, key)
            if value is not None:
                cap[f"tayf:{key}"] = value
        if self.captures and self.captures[-1]["core:sample_start"] == cap["core:sample_start"]:
            self.captures[-1] = cap                 # two changes at once: keep the last
        else:
            self.captures.append(cap)
        self._write_meta()

    def write_raw(self, raw: np.ndarray) -> None:
        """Interleaved components in this recording's datatype (e.g. int8 for ci8)."""
        raw.tofile(self._f)
        self.n_samples += len(raw) // 2

    @property
    def n_bytes(self) -> int:
        return self.n_samples * _bytes_per_sample(self.datatype)

    def _write_meta(self) -> None:
        m = build_meta(self.settings, self.datatype, self.start_time, self.description)
        m["global"].update(self.extra_global)
        m["captures"] = self.captures
        self.meta_path.write_text(json.dumps(m, indent=2))

    def close(self) -> Path:
        self._f.close()
        self._write_meta()
        return self.meta_path


def import_raw(data_path: str | Path, settings: ReceiverSettings, datatype: str = "ci8",
               start_time: datetime | None = None, description: str = "") -> Path:
    """Describe an existing raw I/Q file (e.g. from hackrf_transfer) with a SigMF
    meta file next to it. The data file is left where it is, untouched."""
    data = Path(data_path)
    if not data.exists():
        raise FileNotFoundError(data)
    meta = data.with_name(data.name + ".sigmf-meta") if data.suffix != ".sigmf-data" \
        else data.with_suffix(".sigmf-meta")
    if start_time is None:
        start_time = datetime.fromtimestamp(data.stat().st_mtime, timezone.utc)
    dataset = None if data.suffix == ".sigmf-data" else data.name
    m = build_meta(settings, datatype, start_time, description, dataset=dataset)
    meta.write_text(json.dumps(m, indent=2))
    return meta


@dataclass
class SigMFRecording:
    """An opened recording. Samples are read on demand (memory-mapped), so
    multi-gigabyte files open instantly."""

    meta_path: Path
    data_path: Path
    meta: dict
    datatype: str
    settings: ReceiverSettings
    start_time: datetime
    n_samples: int

    @classmethod
    def open(cls, path: str | Path) -> "SigMFRecording":
        meta_path = meta_path_for(path) if not str(path).endswith(".sigmf-meta") else Path(path)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text())
        g = meta["global"]
        datatype = g["core:datatype"]
        if datatype not in DATATYPES:
            raise ValueError(f"Unsupported SigMF datatype {datatype!r}; "
                             f"supported: {', '.join(DATATYPES)}")
        dataset = g.get("core:dataset")
        data_path = meta_path.parent / dataset if dataset else meta_path.with_suffix(".sigmf-data")
        captures = meta.get("captures") or [{}]
        cap = captures[0]
        settings = ReceiverSettings(
            center_freq_hz=float(cap.get("core:frequency", 0.0)),
            sample_rate_hz=float(g["core:sample_rate"]),
            amp_on=g.get("tayf:amp_on"),
            lna_gain_db=g.get("tayf:lna_gain_db"),
            vga_gain_db=g.get("tayf:vga_gain_db"),
            antenna_id=g.get("tayf:antenna_id"),
            bias_tee=g.get("tayf:bias_tee"),
            hardware=g.get("core:hw"),
        )
        start = _parse_iso(cap["core:datetime"]) if "core:datetime" in cap \
            else datetime.fromtimestamp(meta_path.stat().st_mtime, timezone.utc)
        n = data_path.stat().st_size // _bytes_per_sample(datatype)
        return cls(meta_path, data_path, meta, datatype, settings, start, n)

    @property
    def name(self) -> str:
        return self.meta_path.name.removesuffix(".sigmf-meta")

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.settings.sample_rate_hz

    @property
    def description(self) -> str:
        return self.meta["global"].get("core:description", "")

    @property
    def comparable(self) -> bool:
        """Can this recording be compared with history? Needs gains and antenna."""
        return self.settings.gains_known and self.settings.antenna_id is not None

    def frequency_at(self, sample: int) -> float:
        """Center frequency of the capture segment holding `sample` (a recording made
        while retuning has several segments)."""
        freq = self.settings.center_freq_hz
        for cap in self.meta.get("captures", []):
            if cap.get("core:sample_start", 0) <= sample and "core:frequency" in cap:
                freq = float(cap["core:frequency"])
        return freq

    def read(self, start: int, count: int) -> IQBlock:
        """Read `count` samples from sample index `start` (clipped to the file)."""
        start = max(0, min(int(start), self.n_samples))
        count = max(0, min(int(count), self.n_samples - start))
        comp = DATATYPES[self.datatype][0]
        raw = np.memmap(self.data_path, dtype=comp, mode="r",
                        offset=start * _bytes_per_sample(self.datatype), shape=(2 * count,)) \
            if count else np.empty(0, dtype=comp)
        s = self.settings
        return IQBlock(to_complex(np.asarray(raw), self.datatype), s.sample_rate_hz,
                       self.frequency_at(start),
                       self.start_time + timedelta(seconds=start / s.sample_rate_hz), start)

    def blocks(self, block_size: int, start: int = 0) -> Iterator[IQBlock]:
        pos = start
        while pos < self.n_samples:
            block = self.read(pos, block_size)
            yield block
            pos += len(block)

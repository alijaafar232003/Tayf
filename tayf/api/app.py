"""FastAPI backend: REST for actions and data, WebSocket for the live spectrum stream.

Spectrum frames go over the WebSocket as binary messages:
  float64[4] header = [position_s, first_bin_freq_hz, bin_spacing_hz, rbw_hz]
  float32[nfft]     = power per bin in dBFS
Status updates go over the same socket as JSON text messages.
"""
from __future__ import annotations

import asyncio
import base64
import struct
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.models import ReceiverSettings, SpectrumFrame
from ..core.sigmf_io import DATATYPES, SigMFRecording, import_raw
from ..session.manager import SessionManager
from ..store.db import Store
from .. import synth

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class AntennaIn(BaseModel):
    id: str
    name: str
    freq_min_hz: float
    freq_max_hz: float
    gain_dbi: float | None = None
    uses: str | None = None
    notes: str | None = None


class ImportIn(BaseModel):
    path: str
    # Needed only for raw files (e.g. hackrf_transfer output) that have no SigMF meta yet.
    center_freq_hz: float | None = None
    sample_rate_hz: float | None = None
    datatype: str = "ci8"
    antenna_id: str | None = None
    amp_on: bool | None = None
    lna_gain_db: float | None = None
    vga_gain_db: float | None = None
    description: str = ""


class OpenIn(BaseModel):
    recording_id: int


class ControlIn(BaseModel):
    playing: bool | None = None
    speed: float | None = None
    nfft: int | None = None
    seek_s: float | None = None


def frame_bytes(f: SpectrumFrame) -> bytes:
    df = float(f.freqs_hz[1] - f.freqs_hz[0]) if len(f.freqs_hz) > 1 else 0.0
    header = struct.pack("<4d", f.offset_s, float(f.freqs_hz[0]), df, f.rbw_hz)
    return header + np.asarray(f.power_dbfs, dtype="<f4").tobytes()


def create_app(data_dir: str | Path = "data") -> FastAPI:
    data_dir = Path(data_dir).resolve()
    rec_dir = data_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "tayf.db")
    store.scan_folder(rec_dir)
    manager = SessionManager(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.close()
        store.close()

    app = FastAPI(title="Tayf", lifespan=lifespan)
    app.state.store, app.state.manager = store, manager

    # --- pages ----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/api/status")
    def status():
        return {"session": manager.status(), "recent_sessions": store.recent_sessions(),
                "data_dir": str(data_dir)}

    # --- antennas -------------------------------------------------------
    @app.get("/api/antennas")
    def antennas(freq_hz: float | None = None):
        return store.antennas_for(freq_hz) if freq_hz is not None else store.antennas()

    @app.put("/api/antennas")
    def put_antenna(a: AntennaIn):
        if a.freq_min_hz >= a.freq_max_hz:
            raise HTTPException(400, "freq_min_hz must be below freq_max_hz")
        store.upsert_antenna(a.model_dump())
        return store.antennas()

    @app.delete("/api/antennas/{antenna_id}")
    def delete_antenna(antenna_id: str):
        store.delete_antenna(antenna_id)
        return store.antennas()

    # --- recordings -----------------------------------------------------
    @app.get("/api/recordings")
    def recordings():
        store.scan_folder(rec_dir)
        return store.recordings()

    @app.post("/api/recordings/import")
    def import_recording(body: ImportIn):
        path = Path(body.path.strip().strip('"'))
        if not path.exists():
            raise HTTPException(404, f"File not found: {path}")
        if path.name.endswith((".sigmf-meta", ".sigmf-data")):
            rec = SigMFRecording.open(path)
        else:
            if body.center_freq_hz is None or body.sample_rate_hz is None:
                raise HTTPException(400, "A raw file needs center_freq_hz and sample_rate_hz")
            if body.datatype not in DATATYPES:
                raise HTTPException(400, f"datatype must be one of {list(DATATYPES)}")
            settings = ReceiverSettings(body.center_freq_hz, body.sample_rate_hz,
                                        body.amp_on, body.lna_gain_db, body.vga_gain_db,
                                        body.antenna_id, hardware="HackRF One")
            rec = SigMFRecording.open(import_raw(path, settings, body.datatype,
                                                 description=body.description))
        return {"id": store.add_recording(rec)}

    @app.post("/api/recordings/synthetic")
    def make_synthetic():
        meta = synth.write_scene(rec_dir / "synthetic_scene")
        return {"id": store.add_recording(SigMFRecording.open(meta))}

    # --- session --------------------------------------------------------
    @app.get("/api/session")
    def session():
        return manager.status()

    @app.post("/api/session")
    def open_session(body: OpenIn):
        try:
            manager.open_recording(body.recording_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return manager.status()

    @app.delete("/api/session")
    def close_session():
        manager.close()
        return manager.status()

    @app.post("/api/session/control")
    def control(body: ControlIn):
        try:
            manager.control(body.model_dump(exclude_none=True))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return manager.status()

    @app.get("/api/session/overview")
    def overview(rows: int = 300, nfft: int = 1024):
        try:
            times, p = manager.overview(min(rows, 1000), nfft)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        lo, hi = float(np.percentile(p, 5)) - 3, float(p.max())
        q = np.clip((p - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        st = manager.status()
        return {"rows": int(p.shape[0]), "cols": int(p.shape[1]),
                "times_s": times.tolist(), "min_db": lo, "max_db": hi,
                "freq_lo_hz": st["center_freq_hz"] - st["sample_rate_hz"] / 2,
                "freq_hi_hz": st["center_freq_hz"] + st["sample_rate_hz"] / 2,
                "data_b64": base64.b64encode(q.tobytes()).decode()}

    # --- spectrum stream ------------------------------------------------
    @app.websocket("/ws/spectrum")
    async def spectrum_ws(ws: WebSocket):
        await ws.accept()
        last_status = None
        sent_paused_at = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                t0 = loop.time()
                st = manager.status()
                if not st.get("active"):
                    await asyncio.sleep(0.2)
                    continue
                if st["playing"]:
                    frame = await asyncio.to_thread(manager.next_frame)
                    sent_paused_at = None
                    if frame is not None:
                        await ws.send_bytes(frame_bytes(frame))
                else:
                    # Paused: send the spectrum at the playhead once, and again after a seek.
                    key = (st["session_id"], st["position_s"], st["nfft"], st["speed"])
                    if key != sent_paused_at:
                        frame = await asyncio.to_thread(manager.current_frame)
                        await ws.send_bytes(frame_bytes(frame))
                        sent_paused_at = key
                st = manager.status()
                brief = {k: st.get(k) for k in ("session_id", "playing", "speed", "nfft")}
                if brief != last_status:
                    await ws.send_json({"type": "status", **st})
                    last_status = brief
                # Keep the frame rate steady: subtract the time spent computing.
                await asyncio.sleep(max(0.0, 1 / st.get("fps", 20) - (loop.time() - t0)))
        except WebSocketDisconnect:
            pass

    return app

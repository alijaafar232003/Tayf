"""Command line: python -m tayf <serve | synth | import-raw>."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="tayf")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data", default="data", help="Folder for the database and recordings")

    g = sub.add_parser("synth", help="Write the synthetic test scene")
    g.add_argument("--out", default="data/recordings/synthetic_scene")
    g.add_argument("--datatype", default="ci8", choices=["ci8", "ci16_le", "cf32_le"])

    r = sub.add_parser("import-raw", help="Add SigMF metadata to a raw I/Q file (e.g. hackrf_transfer)")
    r.add_argument("path")
    r.add_argument("--fc", type=float, required=True, help="Center frequency in Hz")
    r.add_argument("--fs", type=float, required=True, help="Sample rate in Hz")
    r.add_argument("--datatype", default="ci8")
    r.add_argument("--antenna")
    r.add_argument("--amp", type=int, choices=[0, 1])
    r.add_argument("--lna", type=float, help="HackRF LNA (IF) gain, dB")
    r.add_argument("--vga", type=float, help="HackRF VGA (baseband) gain, dB")
    r.add_argument("--description", default="")

    a = p.parse_args()
    if a.cmd == "serve":
        import uvicorn
        from .api.app import create_app
        uvicorn.run(create_app(a.data), host=a.host, port=a.port)
    elif a.cmd == "synth":
        from .synth import write_scene
        print(write_scene(Path(a.out), datatype=a.datatype))
    elif a.cmd == "import-raw":
        from .core.models import ReceiverSettings
        from .core.sigmf_io import import_raw
        settings = ReceiverSettings(a.fc, a.fs, None if a.amp is None else bool(a.amp),
                                    a.lna, a.vga, a.antenna, hardware="HackRF One")
        print(import_raw(a.path, settings, a.datatype, description=a.description))


if __name__ == "__main__":
    main()

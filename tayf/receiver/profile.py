"""The receiver profile: the HackRF's own fingerprint, saved as JSON.

Written by notebook 02 (receiver characterization). `condition` says how it was
measured: "antenna removed" (a preview: stray signals can leak into the open
port) or "50 ohm load" (the real thing). The newest profile is profile.json;
every run is also kept with its date.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.models import utc_now

PROFILE_DIR = Path("data") / "receiver"
CONDITIONS = ("antenna removed", "50 ohm load")


def _clean(obj):
    """Drop plotting-only keys (starting with _) before saving."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def save_profile(profile: dict, condition: str, folder: str | Path = PROFILE_DIR) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    data = {"created": now.isoformat(timespec="seconds"), "condition": condition, **_clean(profile)}
    text = json.dumps(data, indent=1)
    tag = condition.replace(" ", "_")
    dated = folder / f"profile_{now:%Y%m%d_%H%M%S}_{tag}.json"
    dated.write_text(text)
    (folder / "profile.json").write_text(text)
    return dated


def load_profile(folder: str | Path = PROFILE_DIR) -> dict | None:
    p = Path(folder) / "profile.json"
    return json.loads(p.read_text()) if p.exists() else None

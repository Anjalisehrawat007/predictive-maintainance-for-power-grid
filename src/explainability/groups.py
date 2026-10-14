"""Feature → sensor-group mapping (dependency-free; used by the serving engine)."""
from __future__ import annotations

SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def feature_group(name: str) -> str:
    if name.startswith("OT_resid"):
        return "thermal residual (OT vs load)"
    for s in SENSORS:
        if name == s or name.startswith(s + "_"):
            return s
    if name.startswith("load_total"):
        return "total load"
    return "calendar (hour/day)"

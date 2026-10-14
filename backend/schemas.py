"""Validated request bodies (bounds keep the demo safe from oversized or malformed requests)."""
from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

SENSORS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


class SessionCreate(BaseModel):
    start_index: int | None = Field(None, ge=0)
    stop_index: int | None = Field(None, ge=0)
    assets: list[str] | None = Field(None, min_length=1, max_length=8)


class StepBody(BaseModel):
    n: int = Field(1, ge=1, le=64)


class ReplayBody(BaseModel):
    hours_before: float = Field(6, gt=0, le=24)
    hours_after: float = Field(1, ge=0, le=6)


class AckBody(BaseModel):
    by: str = Field("operator", min_length=1, max_length=64)
    note: str = Field("", max_length=500)


class SensorRow(BaseModel):
    timestamp: datetime
    HUFL: float
    HULL: float
    MUFL: float
    MULL: float
    LUFL: float
    LULL: float
    OT: float

    @field_validator(*SENSORS)
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("sensor values must be finite numbers")
        return v


class PredictBody(BaseModel):
    asset: str = Field(..., min_length=1, max_length=32)
    rows: list[SensorRow] = Field(..., min_length=1, max_length=5000)

    @field_validator("rows")
    @classmethod
    def increasing(cls, rows: list[SensorRow]) -> list[SensorRow]:
        ts = [r.timestamp for r in rows]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("timestamps must be strictly increasing")
        return rows

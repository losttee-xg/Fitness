from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class UnitSpec:
    canonical: str
    factor: float
    offset: float = 0.0


UNIT_ALIASES: dict[str, UnitSpec] = {
    "bpm": UnitSpec("bpm", 1),
    "beats/min": UnitSpec("bpm", 1),
    "beats per minute": UnitSpec("bpm", 1),
    "count/min": UnitSpec("bpm", 1),
    "count/minute": UnitSpec("bpm", 1),
    "count/mins": UnitSpec("bpm", 1),
    "ms": UnitSpec("ms", 1),
    "s_ms": UnitSpec("ms", 1000),
    "millisecond": UnitSpec("ms", 1),
    "milliseconds": UnitSpec("ms", 1),
    "s": UnitSpec("s", 1),
    "sec": UnitSpec("s", 1),
    "second": UnitSpec("s", 1),
    "seconds": UnitSpec("s", 1),
    "min": UnitSpec("min", 1),
    "minute": UnitSpec("min", 1),
    "minutes": UnitSpec("min", 1),
    "h": UnitSpec("min", 60),
    "hr": UnitSpec("min", 60),
    "hour": UnitSpec("min", 60),
    "hours": UnitSpec("min", 60),
    "kg": UnitSpec("kg", 1),
    "g": UnitSpec("kg", 0.001),
    "lb": UnitSpec("kg", 0.45359237),
    "lbs": UnitSpec("kg", 0.45359237),
    "steps": UnitSpec("steps", 1),
    "%": UnitSpec("%", 1),
    "percent": UnitSpec("%", 1),
    "breaths/min": UnitSpec("breaths/min", 1),
    "rpm": UnitSpec("breaths/min", 1),
}

METRIC_UNITS: dict[str, set[str]] = {
    "SL-02": {"min"},
    "SL-03": {"min"}, "SL-04": {"min"}, "SL-05": {"min"}, "SL-06": {"min"}, "SL-07": {"min"},
    "SL-08": {"min"}, "SL-09": {"%"}, "SL-10": {"%"}, "SL-11": {"min"}, "SL-12": {"min"},
    "SL-14": {"min"}, "SL-15": {"%"}, "SL-16": {"%"}, "SL-19": {"min"}, "SL-20": {"min"},
    "RS-01": {"bpm"}, "RS-02": {"bpm"}, "RS-03": {"bpm"}, "RS-04": {"bpm"},
    "RS-05": {"ms"},
    "RS-06": {"breaths/min"},
    "RS-10": {"%"},
    "HS-05": {"kg"},
    "AC-09": {"steps"},
}


def normalize_unit(metric_code: str, value: float, unit: str) -> tuple[float, str] | None:
    spec = UNIT_ALIASES.get(unit.strip().casefold())
    if spec is None or spec.canonical not in METRIC_UNITS.get(metric_code, set()):
        return None
    normalized_value = (float(value) + spec.offset) * spec.factor
    return normalized_value, spec.canonical

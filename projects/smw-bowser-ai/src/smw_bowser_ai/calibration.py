from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memory import MemoryMap
from .telemetry import read_jsonl


@dataclass
class SignalCalibration:
    name: str
    seen: bool = False
    changed: bool = False
    minimum: int | None = None
    maximum: int | None = None
    samples: int = 0
    first: Any = None
    last: Any = None


@dataclass
class CalibrationReport:
    ok: bool
    signals: dict[str, SignalCalibration] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)


def calibrate_trace(path: str | Path, memory_map: MemoryMap) -> CalibrationReport:
    signals = {name: SignalCalibration(name=name) for name in memory_map.signals}
    for event in read_jsonl(path):
        ram = event.get("ram")
        if not isinstance(ram, dict):
            continue
        for name, calibration in signals.items():
            if name not in ram:
                continue
            value = ram[name]
            calibration.samples += 1
            calibration.seen = True
            if calibration.first is None:
                calibration.first = value
            if calibration.last is not None and value != calibration.last:
                calibration.changed = True
            calibration.last = value
            if isinstance(value, int):
                calibration.minimum = value if calibration.minimum is None else min(calibration.minimum, value)
                calibration.maximum = value if calibration.maximum is None else max(calibration.maximum, value)
    missing_required = [
        name
        for name in memory_map.required_signal_names()
        if not signals.get(name) or not signals[name].seen
    ]
    return CalibrationReport(ok=not missing_required, signals=signals, missing_required=missing_required)


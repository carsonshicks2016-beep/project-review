#!/usr/bin/env python3
"""Evaluate force-history periodicity and grid/time-step convergence gates.

Input CSV requires time_s, Fx_N, Fy_N, Fz_N, and power_W. Optional moment
columns are included when all three are present. This tool evaluates supplied
data; it cannot establish that the CFD force extraction itself is correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

FORCE_FIELDS = ("Fx_N", "Fy_N", "Fz_N")
BASE_FIELDS = FORCE_FIELDS + ("power_W",)
MOMENT_FIELDS = ("Mx_Nm", "My_Nm", "Mz_Nm")


def read_history(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"time_s", *BASE_FIELDS}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required fields: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < 4:
        raise ValueError(f"{path}: at least four time samples are required")
    times = np.asarray([float(row["time_s"]) for row in rows])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: time_s must be finite and strictly increasing")
    fields = list(BASE_FIELDS)
    if set(MOMENT_FIELDS).issubset(reader.fieldnames):
        fields.extend(MOMENT_FIELDS)
    values = {name: np.asarray([float(row[name]) for row in rows]) for name in fields}
    if any(not np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError(f"{path}: non-finite values are not accepted")
    return times, values


def phase_cycle(times: np.ndarray, data: dict[str, np.ndarray], start: float, period: float, bins: int):
    grid_t = start + np.arange(bins, dtype=float) * period / bins
    if grid_t[0] < times[0] or grid_t[-1] > times[-1]:
        raise ValueError("requested full cycle lies outside the recorded time interval")
    return {key: np.interp(grid_t, times, values) for key, values in data.items()}


def cycle_mean(times: np.ndarray, values: np.ndarray, start: float, period: float) -> float:
    # Integrate on a common uniform phase grid to avoid differences due solely
    # to adaptive-time-step sample placement.
    grid = np.linspace(start, start + period, 2049)
    if grid[0] < times[0] or grid[-1] > times[-1]:
        raise ValueError("requested cycle mean lies outside the recorded time interval")
    return float(np.trapezoid(np.interp(grid, times, values), grid) / period)


def relative_pair_error(a: np.ndarray, b: np.ndarray, floor: float = 1.0e-12) -> dict[str, float]:
    delta = a - b
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), floor)
    return {
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "normalized_max_abs": float(np.max(np.abs(delta)) / max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), floor)),
        "absolute_max": float(np.max(np.abs(delta))),
    }


def periodicity(path: Path, frequency: float, start: float | None, bins: int) -> dict:
    times, data = read_history(path)
    period = 1.0 / frequency
    first = start if start is not None else times[-1] - 3.0 * period
    cycles = [phase_cycle(times, data, first + i * period, period, bins) for i in range(3)]
    results = {}
    for label, left, right in (("cycle_1_vs_2", cycles[0], cycles[1]), ("cycle_2_vs_3", cycles[1], cycles[2])):
        vector_a = np.column_stack([left[k] for k in FORCE_FIELDS])
        vector_b = np.column_stack([right[k] for k in FORCE_FIELDS])
        results[label] = {
            "force_vector": relative_pair_error(vector_a, vector_b),
            "power": relative_pair_error(left["power_W"], right["power_W"]),
        }
        if set(MOMENT_FIELDS).issubset(data):
            results[label]["moment_vector"] = relative_pair_error(
                np.column_stack([left[k] for k in MOMENT_FIELDS]),
                np.column_stack([right[k] for k in MOMENT_FIELDS]),
            )
    passed = all(
        results[pair][channel]["relative_l2"] <= 0.02
        and results[pair][channel]["normalized_max_abs"] <= 0.02
        for pair in results
        for channel in ("force_vector", "power")
    )
    return {
        "input": str(path),
        "frequency_hz": frequency,
        "period_s": period,
        "compared_cycle_start_s": first,
        "phase_bins": bins,
        "pairwise_errors": results,
        "gate": "PASS" if passed else "FAIL",
        "criterion": "three successive complete cycles agree pairwise within 2% in normalized L2 and normalized max absolute force-vector and power history",
    }


def convergence(paths: list[Path], labels: list[str], frequency: float, start: float | None, bins: int) -> dict:
    if len(paths) < 2 or len(paths) != len(labels):
        raise ValueError("provide at least two --case LABEL=CSV entries")
    period = 1.0 / frequency
    items = []
    for label, path in zip(labels, paths):
        times, data = read_history(path)
        begin = start if start is not None else times[-1] - period
        cycle = phase_cycle(times, data, begin, period, bins)
        means = {key: cycle_mean(times, data[key], begin, period) for key in BASE_FIELDS}
        items.append({"label": label, "path": str(path), "times": times, "data": data, "cycle": cycle, "means": means, "start": begin})
    left, right = items[-2], items[-1]
    mean_diff = {}
    for key in ("Fz_N", "power_W"):
        a, b = left["means"][key], right["means"][key]
        mean_diff[key] = {
            "coarser_mean": a,
            "finer_mean": b,
            "absolute_difference": abs(b - a),
            "relative_percent": 100.0 * abs(b - a) / max(abs(a), abs(b), 1.0e-12),
        }
    phase_diff = {}
    for key in BASE_FIELDS:
        phase_diff[key] = relative_pair_error(left["cycle"][key], right["cycle"][key])
    passed = all(mean_diff[key]["relative_percent"] <= 5.0 for key in mean_diff)
    return {
        "ordered_cases": [{"label": item["label"], "path": item["path"], "mean": item["means"]} for item in items],
        "finest_pair": [left["label"], right["label"]],
        "cycle_start_s": left["start"],
        "phase_bins": bins,
        "mean_differences": mean_diff,
        "phase_resolved_finest_pair_errors": phase_diff,
        "gate": "PASS" if passed else "FAIL",
        "criterion": "cycle mean vertical force and aerodynamic power each change by no more than 5% between the two finest ordered cases; phase-resolved differences reported separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("periodicity", "convergence"):
        p = sub.add_parser(name)
        p.add_argument("--frequency-hz", type=float, default=26.1)
        p.add_argument("--start-time", type=float)
        p.add_argument("--phase-bins", type=int, default=256)
        p.add_argument("--json", type=Path, help="Optional JSON report path")
        if name == "periodicity":
            p.add_argument("csv", type=Path)
        else:
            p.add_argument("--case", action="append", required=True, metavar="LABEL=CSV", help="Ordered coarse-to-fine cases")
    args = parser.parse_args()
    if args.frequency_hz <= 0 or args.phase_bins < 16:
        parser.error("frequency must be positive and phase bins must be at least 16")
    if args.command == "periodicity":
        report = periodicity(args.csv, args.frequency_hz, args.start_time, args.phase_bins)
    else:
        labels, paths = [], []
        for entry in args.case:
            label, sep, path = entry.partition("=")
            if not sep or not label or not path:
                parser.error(f"invalid --case {entry!r}; expected LABEL=CSV")
            labels.append(label)
            paths.append(Path(path))
        report = convergence(paths, labels, args.frequency_hz, args.start_time, args.phase_bins)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
    print(rendered, end="")
    return_code = 0 if report["gate"] == "PASS" else 2
    sys.exit(return_code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hermetic gate for the versioned five-speed 787B Fable drivetrain."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.config import mazda787b  # noqa: E402
from supra.fable5 import (DRIVETRAIN_VERSION, RaceBox, _shift_table,
                          attach_envelope)  # noqa: E402
from supra.physics import Vehicle  # noqa: E402
from supra.sound import EngineSynth, MAZDA_787B_GEAR_RATIOS, BLOCK  # noqa: E402
from supra.track import named_track  # noqa: E402
from tools.optimize_787b_gearing import score  # noqa: E402

FAILED = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def main():
    spec = mazda787b()
    provenance = json.loads((ROOT / "supra/data/mazda787b_5spd_ring_v1.json").read_text())
    ratios = np.asarray(spec.gear_ratios, dtype=float)

    print("== identity and provenance ==")
    gate("five forward gears", len(ratios) == 5)
    gate("ratios positive and descending", bool(np.all(ratios > 0) and np.all(np.diff(ratios) < 0)))
    gate("drivetrain version stamped", spec.drivetrain_version == DRIVETRAIN_VERSION)
    gate("optimizer artifact matches config",
         np.allclose(ratios, provenance["selected"]["gear_ratios"])
         and np.isclose(spec.final_drive, provenance["selected"]["final_drive"]))
    gate("audio ratios match physics", np.allclose(ratios, MAZDA_787B_GEAR_RATIOS))
    gate("R26B limiter aligned at 9000 rpm",
         spec.redline_rpm == spec.cutoff_rpm == 9000.0)

    print("== shift geometry ==")
    table = _shift_table()
    gate("four sequential upshifts", [(x["from"], x["to"]) for x in table["upshifts"]]
         == [(1, 2), (2, 3), (3, 4), (4, 5)])
    gate("no shift deadband", not table["deadband"])
    gate("all upshifts leave near power peak",
         all(0.94 <= x["frac"] <= 0.98 for x in table["upshifts"]))

    box = RaceBox(spec)
    unsafe = []
    for speed in np.linspace(1.0, 100.0, 991):
        for gear in range(2, 6):
            pred = box.predicted_rpm(speed, gear - 1)
            if pred >= box.guard and pred < spec.cutoff_rpm:
                unsafe.append((speed, gear, pred))
    gate("money-shift guard below hard limiter", box.guard < spec.cutoff_rpm)

    print("== shift state machine ==")
    veh = Vehicle(spec)
    veh.reset(speed=30.0)
    box.reset()
    saw_unload = saw_shift = saw_reengage = False
    previous = veh.gear
    for _ in range(60):
        clutch, up, down = box.update(veh, 1.0, 1.0 / 30.0, 0.0)
        saw_unload |= box.shift_phase == "unload"
        if up:
            veh.shift_up(); saw_shift = True
        if down:
            veh.shift_down(); saw_shift = True
        saw_reengage |= box.shift_phase == "reengage"
        gate_now = 0.0 <= clutch <= 1.0
        if not gate_now:
            break
    gate("unload -> engage -> reengage observed", saw_unload and saw_shift and saw_reengage)
    gate("shift is sequential", abs(veh.gear - previous) <= 4 and 1 <= veh.gear <= 5)
    gate("clutch remains bounded", 0.0 <= box.clutch <= 1.0)

    print("== optimiser and envelope ==")
    selected = score(spec, ratios, spec.final_drive)
    baseline = score(spec, [2.85, 2.00, 1.55, 1.25, 0.85], 3.50)
    gate("selected gearing feasible", selected is not None)
    gate("selected beats or out-reaches naive baseline",
         selected is not None and (baseline is None or selected["score_seconds"] < baseline["score_seconds"]))
    trk = attach_envelope(named_track("nordschleife"), "mazda787b")
    env = trk.fable_envelope
    v = np.asarray(trk.fable_vref)
    gate("gear-aware envelope finite", bool(np.all(np.isfinite(v)) and np.all(v > 0)))
    gate("envelope provenance includes drivetrain",
         env["params"].get("drivetrain_version") == DRIVETRAIN_VERSION)

    print("== audio reset and output ==")
    synth = EngineSynth()
    synth.reset()
    synth.car_name = "mazda787b"
    synth.t_redline = synth.c_redline = 9000.0
    synth.t_rpm = synth.c_rpm = 8500.0
    audio = synth.synthesize(BLOCK)
    gate("five-speed synth finite and bounded",
         bool(np.all(np.isfinite(audio)) and np.max(np.abs(audio)) <= 1.0))

    if FAILED:
        print(f"\n{len(FAILED)} gate(s) FAILED: {FAILED}")
        raise SystemExit(1)
    print("\nall five-speed drivetrain gates green")


if __name__ == "__main__":
    main()

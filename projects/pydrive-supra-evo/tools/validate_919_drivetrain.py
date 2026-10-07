#!/usr/bin/env python3
"""Hermetic gate for the versioned seven-speed Porsche 919 Evo Fable drivetrain.

Mirrors tools/validate_787b_drivetrain.py for the 919: identity/provenance,
shift geometry, the shift state machine, optimiser feasibility, the gear-aware
Nordschleife envelope, and finite/bounded audio.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.config import porsche_919evo  # noqa: E402
from supra.fable5 import RaceBox, _shift_table, attach_envelope  # noqa: E402
from supra.physics import Vehicle  # noqa: E402
from supra.sound import EngineSynth, BLOCK  # noqa: E402
from supra.track import named_track  # noqa: E402
from tools.optimize_919_gearing import score  # noqa: E402

DRIVETRAIN_VERSION = "porsche919evo-7spd-ring-v2"
FAILED = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def main():
    spec = porsche_919evo()
    provenance = json.loads((ROOT / "supra/data/porsche_919evo_7spd_ring_v2.json").read_text())
    ratios = np.asarray(spec.gear_ratios, dtype=float)

    print("== identity and provenance ==")
    gate("seven forward gears", len(ratios) == 7)
    gate("ratios positive and descending",
         bool(np.all(ratios > 0) and np.all(np.diff(ratios) < 0)))
    gate("drivetrain version stamped", spec.drivetrain_version == DRIVETRAIN_VERSION)
    gate("optimizer artifact matches config",
         np.allclose(ratios, provenance["selected"]["ratios"])
         and np.isclose(spec.final_drive, provenance["selected"]["final_drive"]))
    gate("hybrid limiter aligned at 9000 rpm",
         spec.redline_rpm == spec.cutoff_rpm == 9000.0)
    retention = ratios[1:] / ratios[:-1]
    gate("retention within race band 0.74-0.90",
         bool(np.all(retention >= 0.74) and np.all(retention <= 0.90)),
         f"{np.round(retention, 3).tolist()}")

    print("== shift geometry ==")
    table = _shift_table(car="porsche_919evo", v_max=110.0)
    gate("six sequential upshifts",
         [(x["from"], x["to"]) for x in table["upshifts"]]
         == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    gate("no shift deadband", not table["deadband"])
    gate("all upshifts leave near power peak",
         all(0.94 <= x["frac"] <= 0.98 for x in table["upshifts"]),
         f"fracs={[x['frac'] for x in table['upshifts']]}")

    box = RaceBox(spec)
    gate("money-shift guard below hard limiter", box.guard < spec.cutoff_rpm)

    print("== shift state machine ==")
    veh = Vehicle(spec)
    veh.reset(speed=40.0)
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
        if not (0.0 <= clutch <= 1.0):
            break
    gate("unload -> engage -> reengage observed", saw_unload and saw_shift and saw_reengage)
    gate("shift is sequential", abs(veh.gear - previous) <= 6 and 1 <= veh.gear <= 7)
    gate("clutch remains bounded", 0.0 <= box.clutch <= 1.0)

    print("== optimiser and envelope ==")
    selected = score(spec, ratios, spec.final_drive)
    gate("selected gearing feasible", selected is not None)
    trk = attach_envelope(named_track("nordschleife"), "porsche_919evo", use_raceline=True)
    env = trk.fable_envelope
    v = np.asarray(trk.fable_vref)
    gate("gear-aware envelope finite", bool(np.all(np.isfinite(v)) and np.all(v > 0)))
    gate("envelope provenance includes drivetrain",
         env["params"].get("drivetrain_version") == DRIVETRAIN_VERSION)
    lap = env["lap_time"]
    gate("envelope brackets the 319.55 s record (raceline reachable)",
         315.0 <= lap <= 330.0, f"raceline={lap:.1f}s")

    print("== audio reset and output ==")
    synth = EngineSynth()
    synth.reset()
    synth.car_name = "porsche_919evo"
    synth.t_redline = synth.c_redline = 9000.0
    synth.t_rpm = synth.c_rpm = 8000.0
    audio = synth.synthesize(BLOCK)
    gate("seven-speed synth finite and bounded",
         bool(np.all(np.isfinite(audio)) and np.max(np.abs(audio)) <= 1.0))

    if FAILED:
        print(f"\n{len(FAILED)} gate(s) FAILED: {FAILED}")
        raise SystemExit(1)
    print("\nall seven-speed drivetrain gates green")


if __name__ == "__main__":
    main()

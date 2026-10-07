"""One-off: extract real assets for the Fable Five presentation deck.

Pulls the crown-jewel data straight from the repo — the Nordschleife spline
(with a computed physics-true speed envelope for heat colouring), one full clean
lap of telemetry (with per-wheel Fz reconstructed from the frozen obs layout),
and the training/curriculum history parsed from the run logs. Writes tidy,
downsampled CSV/JSON into presentation_assets/.

Hermetic-ish: reads only committed data + logs, writes only under
presentation_assets/. Run:  python3 tools/extract_presentation_assets.py
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "presentation_assets")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)


def _landmark_at(arc, landmarks):
    for lm in landmarks:
        if lm["start_arc"] <= arc < lm["end_arc"]:
            return lm["name"]
    return ""


# --------------------------------------------------------------------------- #
# 1. Track centerline + physics-true speed envelope (crown jewel)
# --------------------------------------------------------------------------- #
def track_asset(target_points=720):
    trk_path = os.path.join(ROOT, "supra/data/tracks/nordschleife.json")
    d = json.load(open(trk_path))
    center = np.asarray(d["center"], dtype=float)     # (N,2)
    elev = np.asarray(d["elevation"], dtype=float)    # (N,)
    n = len(center)
    spacing = d["generated_spacing_m"]
    arc = np.arange(n) * spacing
    length = d["official_length_m"]

    # physics-true speed envelope (m/s per centerline point), computed from the
    # actual 787B model + real geometry — this is what feeds the pace obs block.
    vref = None
    try:
        from supra.track import nordschleife
        from supra import fable5
        trk = nordschleife()
        fable5.attach_envelope(trk, fable5.FABLE_CAR)
        raw = getattr(trk, "fable_vref", None)
        if raw is not None:
            # resample onto our arc grid
            vref = np.interp(arc, trk.arc, np.asarray(raw), period=trk.length)
    except Exception as e:  # pragma: no cover - best effort
        print(f"  [envelope] could not compute ({e.__class__.__name__}: {e}); "
              f"writing spline without vref")

    # local grade (dz/ds) for the flight-zone story
    grade = np.gradient(elev, arc)

    stride = max(1, n // target_points)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    rows = []
    for i in idx:
        rows.append({
            "arc_m": round(float(arc[i]), 2),
            "x": round(float(center[i, 0]), 3),
            "y": round(float(center[i, 1]), 3),
            "elevation_m": round(float(elev[i]), 3),
            "grade": round(float(grade[i]), 5),
            "envelope_speed_ms": (round(float(vref[i]), 3) if vref is not None else ""),
            "envelope_speed_kmh": (round(float(vref[i]) * 3.6, 2) if vref is not None else ""),
            "landmark": _landmark_at(arc[i], d["landmarks"]),
        })
    _write_csv(os.path.join(OUT, "nordschleife_centerline.csv"), rows)

    # landmarks (all 45) as a lean JSON, arc-tagged, jump zones flagged
    JUMP = {"Flugplatz", "Pflanzgarten", "Schwalbenschwanz"}
    lms = [{
        "name": lm["name"],
        "start_arc_m": round(lm["start_arc"], 1),
        "end_arc_m": round(lm["end_arc"], 1),
        "flight_zone": lm["name"] in JUMP,
    } for lm in d["landmarks"]]
    json.dump({
        "track": d["display_name"],
        "length_m": length,
        "width_m": d["width_m"],
        "elevation_min_m": round(float(elev.min()), 2),
        "elevation_max_m": round(float(elev.max()), 2),
        "elevation_range_m": round(float(elev.max() - elev.min()), 2),
        "point_count_full": n,
        "point_count_downsampled": len(idx),
        "sectors": d["sectors"],
        "landmarks": lms,
    }, open(os.path.join(OUT, "nordschleife_landmarks.json"), "w"), indent=1)

    print(f"  track: {len(idx)} pts (from {n}), "
          f"envelope={'yes' if vref is not None else 'no'}, "
          f"elev {elev.min():.0f}-{elev.max():.0f} m")


# --------------------------------------------------------------------------- #
# 2. One full clean lap of telemetry (with reconstructed per-wheel Fz)
# --------------------------------------------------------------------------- #
def telemetry_asset(target_rows=600):
    diag = os.path.join(
        ROOT, "diagnostics/fable5_ring_best_nordschleife_20260703_151106")
    z = np.load(os.path.join(diag, "raw_trace.npz"))
    n = len(z["t"])
    length = 20832.0
    # 787B mass*g — the obs normalises each wheel's Fz by this (see sensors.py)
    MG = 830.0 * 9.81
    obs = z["obs"]
    # obs layout (fable-v1): 9 beams + 25 proprio + 6 lookahead + 18 hill + 8 pace
    # per-wheel Fz normalised by mg live at proprio idx 17..20 -> global 26..29
    fz = obs[:, 26:30] * MG

    stride = max(1, n // target_rows)
    idx = list(range(0, n, stride))
    rows = []
    for i in idx:
        rows.append({
            "t_s": round(float(z["t"][i]), 3),
            "distance_m": round(float(z["progress"][i]) * length, 1),
            "progress_frac": round(float(z["progress"][i]), 5),
            "speed_ms": round(float(z["speed"][i]), 3),
            "speed_kmh": round(float(z["speed"][i]) * 3.6, 2),
            "slip_deg": round(float(z["slip_deg"][i]), 3),
            "yaw_rate": round(float(z["yaw_rate"][i]), 4),
            "curvature": round(float(z["curvature"][i]), 5),
            "steer": round(float(z["steer"][i]), 4),
            "throttle": round(float(z["throttle"][i]), 4),
            "brake": round(float(z["brake"][i]), 4),
            "Fz_FL_N": round(float(fz[i, 0]), 1),
            "Fz_FR_N": round(float(fz[i, 1]), 1),
            "Fz_RL_N": round(float(fz[i, 2]), 1),
            "Fz_RR_N": round(float(fz[i, 3]), 1),
            "reward": round(float(z["reward"][i]), 5),
            "off_track": int(bool(z["off_track"][i])),
        })
    _write_csv(os.path.join(OUT, "lap_telemetry.csv"), rows)
    print(f"  telemetry: {len(idx)} rows (from {n}), "
          f"speed {z['speed'].min():.1f}-{z['speed'].max():.1f} m/s, "
          f"lap 571.2s, Fz reconstructed from obs")


# --------------------------------------------------------------------------- #
# 3. Training history + curriculum staircase (parsed from the run logs)
# --------------------------------------------------------------------------- #
IT_RE = re.compile(
    r"it\s+(\d+)\s+ret\s+([-\d.]+)\s+laps\s+([-\d.]+)\s+diff\s+([-\d.]+)\s+"
    r"pi\s+([-\d.+]+)\s+vf\s+([-\d.]+)\s+ent\s+([-\d.]+)\s+kl\s+([-\d.]+)")
EVAL_RE = re.compile(
    r"\[eval-fable\] stage=(\S+) clean=(\d+)/(\d+) chain=(\d+) pace=([-\d.]+) "
    r"lap=([-\d.]+) theo=([-\d.]+) progress=(\d+)m terminal=([-\d.]+) "
    r"offtrack=([-\d.]+) worst=\S+ metric=([-\d.]+)")
PIT_RE = re.compile(
    r"\[pit\]\s+(\d\d:\d\d:\d\d)\s+\S+:\s+metric=([-\d.]+)\s+best=([-\d.]+)\s+->\s+(\w+)")


def training_asset():
    log = os.path.join(ROOT, "KLGUARD_run.log")
    it_rows, eval_rows, pit_rows = [], [], []
    for line in open(log, errors="ignore"):
        m = IT_RE.search(line)
        if m:
            it_rows.append({
                "iteration": int(m.group(1)), "return": float(m.group(2)),
                "laps": float(m.group(3)), "difficulty": float(m.group(4)),
                "pi_loss": float(m.group(5)), "value_loss": float(m.group(6)),
                "entropy": float(m.group(7)), "kl": float(m.group(8)),
            })
            continue
        m = EVAL_RE.search(line)
        if m:
            eval_rows.append({
                "stage": m.group(1), "clean_sectors": int(m.group(2)),
                "sector_count": int(m.group(3)), "clean_chain": int(m.group(4)),
                "pace_ratio": float(m.group(5)), "lap_frac": float(m.group(6)),
                "theoretical_lap_s": float(m.group(7)),
                "progress_m": int(m.group(8)), "terminal_rate": float(m.group(9)),
                "offtrack_s": float(m.group(10)), "metric": float(m.group(11)),
            })
            continue
        m = PIT_RE.search(line)
        if m:
            pit_rows.append({
                "time": m.group(1), "metric": float(m.group(2)),
                "lap_best_bar": float(m.group(3)), "decision": m.group(4),
            })
    for i, r in enumerate(eval_rows):
        r_ordered = {"eval": i + 1, **r}
        eval_rows[i] = r_ordered
    _write_csv(os.path.join(OUT, "training_iterations.csv"), it_rows)
    _write_csv(os.path.join(OUT, "training_evals.csv"), eval_rows)
    _write_csv(os.path.join(OUT, "pitwall_decisions.csv"), pit_rows)
    print(f"  training: {len(it_rows)} PPO iters, {len(eval_rows)} evals, "
          f"{len(pit_rows)} pit decisions")


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    print("extracting presentation assets ->", OUT)
    track_asset()
    telemetry_asset()
    training_asset()
    print("done.")

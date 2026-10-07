"""Fable Five brain diagnostics — run a checkpoint through a deep,
envelope-aware test battery and write a JSON report dense enough that the
car's behavior can be read off it without watching a single frame.

The battery (all deterministic, act_mean — exactly how the eval drives):
  * two FLYING line laps (race drop-in + cautious drop-in) with a full
    per-10m telemetry trace: speed vs envelope, pedals, steering, gear,
    rpm, slip, lateral position, off-track;
  * the 16 standard sector starts (the eval battery) with per-sector stats;
  * lap-CLOSURE starts at 70/80/85/90/95% — can it carry a lap home?
  * every death is dissected: where, why, how fast vs the envelope on
    entry, slip/steering in the final seconds, and a one-word verdict tag;
  * whole-run behavior stats: gear usage, shift counts, gear-offset action,
    throttle/brake/coast split, steering roughness, pace histogram.

Reports land in diagnostics/fable5/<checkpoint>_<timestamp>.json and are
served by the command center's Brain Lab page.
"""
from __future__ import annotations

from pathlib import Path
import datetime
import json
import os
import time

import numpy as np
import torch

from .config import PPOSpec, SimSpec
from .track import named_track
from .fable5 import (FABLE_CAR, FABLE_BEST, HUMAN_BENCHMARKS, RING_TRACK,
                     SUPERHUMAN_LAP, FableEnv, attach_envelope,
                     stage_defaults, vref_at)

DIAG_DIR = os.path.join("diagnostics", "fable5")
N_SECTORS = 16
CLOSURE_FRACS = (0.70, 0.80, 0.85, 0.90, 0.95)

# approximate landmark names for the 16 equal-arc eval sectors of the full
# 20.832 km Nordschleife (start/finish on Döttinger Höhe side of T13)
SECTOR_NAMES = (
    "T13-Hatzenbach", "Hocheichen-Quiddelbacher", "Flugplatz-Schwedenkreuz",
    "Aremberg-Fuchsroehre", "Adenauer Forst-Metzgesfeld", "Kallenhard-Wehrseifen",
    "Breidscheid-Ex-Muehle", "Bergwerk-Kesselchen", "Klostertal",
    "Karussell-Hohe Acht", "Hedwigshoehe-Wippermann", "Eschbach-Bruennchen",
    "Eiskurve-Pflanzgarten", "Bellof-S-Schwalbenschwanz", "Galgenkopf",
    "Doettinger Hoehe-Tiergarten",
)


def _sector_of(arc_m: float, length: float) -> int:
    return min(N_SECTORS - 1, int(max(0.0, arc_m) / max(1.0, length) * N_SECTORS))


class _Brain:
    """act_mean wrapper around a loaded fable checkpoint."""

    def __init__(self, path: str):
        from .ppo import PPO
        self.net, self.norm, self.meta = PPO.load_policy(path)
        if not self.meta.get("sensor_pace_block"):
            raise SystemExit(
                f"'{path}' is not a fable-v1 checkpoint (no pace block) — "
                f"use the generic Diagnostics exports for legacy brains.")
        self.sdim = int(self.meta["sdim"])
        self.act_dim = int(self.meta.get("act_dim", 2))

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        s = self.norm.normalize(obs[None, : self.sdim])
        nobs = np.concatenate([s, obs[None, self.sdim:]], axis=-1)
        t = torch.as_tensor(nobs, dtype=torch.float32)
        return self.net.act_mean(t).squeeze(0).cpu().numpy()


def _make_env(brain: _Brain, spec, trk, car: str) -> FableEnv:
    cfg = PPOSpec()
    cfg.random_start = False
    cfg.episode_seconds = 4000.0
    cfg.sensor_pace_block = True
    if brain.meta.get("sensor_pace_distances"):
        cfg.sensor_pace_distances = tuple(brain.meta["sensor_pace_distances"])
    cfg.sensor_hybrid_block = bool(brain.meta.get("sensor_hybrid_block"))
    if brain.meta.get("sensor_lookahead_distances"):
        cfg.sensor_lookahead_distances = tuple(
            brain.meta["sensor_lookahead_distances"])
    return FableEnv(mode="race", car=car, ppo=cfg, sim=SimSpec(),
                    fixed_track=trk, fable_spec=spec, rng_seed=880,
                    diagnostics=True)


def _run_traced(brain: _Brain, env: FableEnv, trk, idx: int, speed: float,
                max_steps: int, keep_trace: bool = False) -> dict:
    """One deterministic run; returns summary + (optionally) a full trace and
    always a fine-grained tail for death forensics."""
    obs = env.reset_at(idx, speed=speed)
    L = float(trk.length)
    start_arc = float(trk.arc[idx])
    hz = float(env.ppo.control_hz)
    rec: dict[str, list] = {k: [] for k in
                            ("t", "arc", "v", "vref", "thr", "brk", "steer",
                             "gear", "rpm", "slip", "lat", "off", "goff")}
    shifts_up = shifts_down = 0
    prev_gear = env.veh.gear
    reversal = 0
    prev_steer_sign = 0
    info: dict = {}
    lap_time = None
    clean_lap = False
    steps = 0
    for _ in range(max(1, max_steps)):
        a = brain.act(obs)
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        fr = trk.frame(env.veh.x, env.veh.y)
        v = float(env.veh.speed)
        vr = float(vref_at(trk, fr["arc"]))
        rec["t"].append(env.t)
        rec["arc"].append(float(fr["arc"]))
        rec["v"].append(v)
        rec["vref"].append(vr)
        rec["thr"].append(max(float(a[1]), 0.0))
        rec["brk"].append(max(-float(a[1]), 0.0))
        rec["steer"].append(float(a[0]))
        rec["gear"].append(int(env.veh.gear))
        rec["rpm"].append(float(env.veh.rpm))
        rec["slip"].append(float(np.degrees(env.veh.slip_angle)))
        half = float(fr.get("half_width", trk.half)) or 1.0
        rec["lat"].append(float(fr["lateral"]) / half)
        rec["off"].append(1.0 if fr["off_track"] else 0.0)
        # applied gear offset: the env clips a[2] to [-1,1] then scales to
        # +-2 gears, so report what the gearbox ACTUALLY saw (act_mean is
        # unbounded — the raw value overstates the override)
        rec["goff"].append(float(np.clip(a[2], -1.0, 1.0)) * 2.0
                           if len(a) > 2 else 0.0)
        if env.veh.gear > prev_gear:
            shifts_up += env.veh.gear - prev_gear
        elif env.veh.gear < prev_gear:
            shifts_down += prev_gear - env.veh.gear
        prev_gear = env.veh.gear
        ss = 1 if a[0] > 0.06 else (-1 if a[0] < -0.06 else 0)
        if ss and prev_steer_sign and ss != prev_steer_sign:
            reversal += 1
        if ss:
            prev_steer_sign = ss
        if info.get("laps", 0.0) >= 1.0 and not info.get("fable_invalid"):
            lap_time = float(info.get("t", 0.0))
            clean_lap = True
            break
        if term or trunc:
            break

    arr = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    reason = info.get("termination_reason")
    progress_m = float(info.get("fable_progress_m", 0.0))
    dist_km = max(0.001, float(np.sum(arr["v"])) / hz / 1000.0)
    on = arr["off"] < 0.5
    pace = np.divide(arr["v"], np.maximum(arr["vref"], 1e-6))
    out = {
        "start_frac": round(start_arc / L, 4),
        "start_speed": round(float(speed), 1),
        "steps": steps,
        "sim_seconds": round(steps / hz, 1),
        "clean_lap": clean_lap,
        "lap_time": round(lap_time, 2) if lap_time else None,
        "termination_reason": reason,
        "clean": bool(reason in (None, "time_limit")
                      and not info.get("fable_invalid", False)),
        "progress_m": round(progress_m, 1),
        "progress_frac": round(progress_m / L, 4),
        "offtrack_seconds": round(float(info.get("fable_offtrack_seconds",
                                                 0.0)), 2),
        "mean_pace_on_track": (round(float(np.mean(pace[on])), 4)
                               if on.any() else 0.0),
        "mean_speed": round(float(np.mean(arr["v"])), 2) if steps else 0.0,
        "max_speed": round(float(np.max(arr["v"])), 2) if steps else 0.0,
        "shifts_up": shifts_up,
        "shifts_down": shifts_down,
        "steer_reversals_per_km": round(reversal / dist_km, 2),
        "slip_p95": (round(float(np.percentile(np.abs(arr["slip"]), 95)), 2)
                     if steps else 0.0),
        "edge_p95": (round(float(np.percentile(np.abs(arr["lat"]), 95)), 3)
                     if steps else 0.0),
    }
    # death forensics: the final ~3 s at 0.5 s stride, plus entry pace
    if reason not in (None, "time_limit"):
        n3 = min(steps, int(3.0 * hz))
        stride = max(1, int(0.5 * hz))
        tail = slice(steps - n3, steps)
        death_arc = float(arr["arc"][-1])
        e0, e1 = max(0, steps - int(3.0 * hz)), max(1, steps - int(1.0 * hz))
        entry_pace = float(np.mean(pace[e0:e1])) if e1 > e0 else 0.0
        sec = _sector_of(death_arc, L)
        out["death"] = {
            "arc_m": round(death_arc, 1),
            "sector": sec,
            "sector_name": SECTOR_NAMES[sec],
            "reason": reason,
            "speed": round(float(arr["v"][-1]), 2),
            "vref_here": round(float(arr["vref"][-1]), 2),
            "entry_pace": round(entry_pace, 3),
            "max_slip_last3s": round(float(np.max(np.abs(arr["slip"][tail]))), 2),
            "max_lat_last3s": round(float(np.max(np.abs(arr["lat"][tail]))), 3),
            "final": {k: [round(float(x), 3) for x in arr[k][tail][::stride]]
                      for k in ("v", "vref", "slip", "steer", "thr", "brk",
                                "lat")},
            "tag": _death_tag(reason, entry_pace,
                              float(np.max(np.abs(arr["slip"][tail]))),
                              float(np.max(np.abs(arr["lat"][tail]))),
                              float(arr["v"][-1])),
        }
    if keep_trace and steps:
        out["_arrays"] = arr        # caller downsamples / aggregates, then drops
    return out


def _death_tag(reason: str, entry_pace: float, max_slip: float,
               max_lat: float, v_final: float) -> str:
    if reason == "spin" or max_slip > 26.0:
        return "traction-break"
    if entry_pace > 1.03:
        return "overspeed-entry"
    if max_lat > 0.92 and max_slip < 14.0:
        return "ran-wide"
    if v_final < 10.0:
        return "bogged-down"
    return "line-error"


def _downsample_trace(arr: dict, L: float, ds_m: float = 10.0) -> dict:
    """Compact per-distance trace for charts: one sample every ~ds_m along
    the covered arc (last write wins per bin, which is fine for plotting)."""
    arc = arr["arc"]
    if not len(arc):
        return {}
    keys = ("v", "vref", "thr", "brk", "steer", "gear", "rpm", "slip",
            "lat", "off")
    bins: dict[int, list] = {}
    for j in range(len(arc)):
        bins[int(arc[j] / ds_m)] = j
    idx = [bins[b] for b in sorted(bins)]
    out = {"s_m": [round(float(arc[j]), 1) for j in idx]}
    rounding = {"gear": 0, "rpm": 0, "off": 0}
    for k in keys:
        r = rounding.get(k, 3)
        out[k] = [round(float(arr[k][j]), r) if r else int(arr[k][j])
                  for j in idx]
    return out


def _sector_stats(runs: list[dict], line_arrays: list[dict], L: float) -> list[dict]:
    """Per-eval-sector aggregates from the sector battery + line-lap traces."""
    rows = []
    for s in range(N_SECTORS):
        r = runs[s] if s < len(runs) else {}
        row = {
            "sector": s,
            "name": SECTOR_NAMES[s],
            "start_m": round(s * L / N_SECTORS, 0),
            "clean": bool(r.get("clean")),
            "reason": r.get("termination_reason"),
            "pace": r.get("mean_pace_on_track", 0.0),
            "offtrack_s": r.get("offtrack_seconds", 0.0),
            "slip_p95": r.get("slip_p95", 0.0),
            "edge_p95": r.get("edge_p95", 0.0),
            "steer_reversals_per_km": r.get("steer_reversals_per_km", 0.0),
        }
        # what the LINE LAP did through this sector (the truer racing context)
        vs, offs = [], 0.0
        for arr in line_arrays:
            m = (arr["arc"] >= s * L / N_SECTORS) & \
                (arr["arc"] < (s + 1) * L / N_SECTORS)
            if m.any():
                vs.append(float(np.mean(np.divide(
                    arr["v"][m], np.maximum(arr["vref"][m], 1e-6)))))
                offs += float(np.sum(arr["off"][m]))
        if vs:
            row["line_pace"] = round(float(np.mean(vs)), 3)
        rows.append(row)
    return rows


def _behavior(all_arrays: list[dict], runs: list[dict], hz: float) -> dict:
    v = np.concatenate([a["v"] for a in all_arrays]) if all_arrays else np.zeros(1)
    vref = np.concatenate([a["vref"] for a in all_arrays]) if all_arrays else np.ones(1)
    thr = np.concatenate([a["thr"] for a in all_arrays]) if all_arrays else np.zeros(1)
    brk = np.concatenate([a["brk"] for a in all_arrays]) if all_arrays else np.zeros(1)
    steer = np.concatenate([a["steer"] for a in all_arrays]) if all_arrays else np.zeros(1)
    gear = np.concatenate([a["gear"] for a in all_arrays]) if all_arrays else np.zeros(1)
    goff = np.concatenate([a["goff"] for a in all_arrays]) if all_arrays else np.zeros(1)
    off = np.concatenate([a["off"] for a in all_arrays]) if all_arrays else np.zeros(1)
    on = off < 0.5
    pace = np.divide(v, np.maximum(vref, 1e-6))[on]
    edges = (0.0, 0.5, 0.7, 0.85, 0.95, 1.05, 9.0)
    hist = np.histogram(pace, bins=edges)[0] / max(1, len(pace))
    gears, counts = np.unique(gear.astype(int), return_counts=True)
    km = max(0.001, float(np.sum(v)) / hz / 1000.0)
    return {
        "sim_km": round(km, 1),
        "mean_speed": round(float(np.mean(v)), 2),
        "max_speed": round(float(np.max(v)), 2),
        "mean_pace_on_track": round(float(np.mean(pace)), 4) if len(pace) else 0.0,
        "time_over_envelope": round(float(np.mean(pace > 1.0)), 4) if len(pace) else 0.0,
        "pace_histogram": {f"{edges[i]:.2f}-{edges[i+1]:.2f}":
                           round(float(hist[i]), 4) for i in range(len(hist))},
        "offtrack_time_frac": round(float(np.mean(off)), 4),
        "throttle_mean": round(float(np.mean(thr)), 3),
        "full_throttle_frac": round(float(np.mean(thr > 0.9)), 4),
        "braking_frac": round(float(np.mean(brk > 0.1)), 4),
        "coasting_frac": round(float(np.mean((thr < 0.1) & (brk < 0.1))), 4),
        "steer_abs_mean": round(float(np.mean(np.abs(steer))), 3),
        "steer_saturation_frac": round(float(np.mean(np.abs(steer) > 0.95)), 4),
        "steer_reversals_per_km": round(float(np.mean(
            [r.get("steer_reversals_per_km", 0.0) for r in runs])), 2),
        "gear_histogram": {int(g): round(float(c / max(1, len(gear))), 4)
                           for g, c in zip(gears, counts)},
        "shifts_up": int(sum(r.get("shifts_up", 0) for r in runs)),
        "shifts_down": int(sum(r.get("shifts_down", 0) for r in runs)),
        "gear_offset_mean": round(float(np.mean(goff)), 3),
        "gear_offset_std": round(float(np.std(goff)), 3),
        "gear_offset_active_frac": round(float(np.mean(np.abs(goff) > 0.5)), 4),
    }


def _findings(report: dict) -> tuple[str, list[str]]:
    """Rule-generated plain-English findings + a one-line verdict."""
    f: list[str] = []
    laps = report["line_laps"]
    beh = report["behavior"]
    deaths = report["deaths"]
    theo = report["envelope"]["theoretical_lap"]
    best_lap = min((l["lap_time"] for l in laps if l.get("lap_time")),
                   default=None)
    best_prog = max((l["progress_frac"] for l in laps), default=0.0)
    clean_secs = sum(1 for s in report["sectors"] if s["clean"])

    if best_lap:
        vs_b = best_lap / SUPERHUMAN_LAP
        f.append(f"Banks a clean flying lap of {best_lap:.2f}s "
                 f"({vs_b:.2f}x Bellof's {SUPERHUMAN_LAP:.2f}s; theoretical "
                 f"centerline {theo:.1f}s).")
        verdict = (f"FINISHER — clean {best_lap:.1f}s flying lap "
                   f"({vs_b:.2f}x Bellof)")
        if vs_b < 1.0:
            verdict = f"SUPERHUMAN — {best_lap:.1f}s beats Bellof"
    else:
        f.append(f"No clean line lap: farthest run dies at "
                 f"{best_prog * 100:.1f}% of the lap.")
        verdict = f"NON-FINISHER — best {best_prog * 100:.0f}% of a lap"

    if deaths:
        by_sec: dict[int, list[dict]] = {}
        for d in deaths:
            by_sec.setdefault(d["sector"], []).append(d)
        worst = sorted(by_sec.items(), key=lambda kv: -len(kv[1]))[:3]
        for sec, ds in worst:
            tags = sorted({d["tag"] for d in ds})
            reasons = sorted({d["reason"] for d in ds})
            ep = float(np.mean([d["entry_pace"] for d in ds]))
            f.append(f"Dies {len(ds)}x in sector {sec} ({SECTOR_NAMES[sec]}): "
                     f"{'/'.join(reasons)} [{'/'.join(tags)}], entry pace "
                     f"{ep:.2f}x envelope.")
        top = worst[0]
        if not best_lap:
            verdict += (f"; killer = sector {top[0]} "
                        f"({SECTOR_NAMES[top[0]]}, {top[1][0]['tag']})")
    else:
        f.append("Zero terminations across the whole battery.")

    f.append(f"Sector battery: {clean_secs}/{N_SECTORS} clean; on-track pace "
             f"{beh['mean_pace_on_track'] * 100:.1f}% of the physics envelope; "
             f"over the envelope {beh['time_over_envelope'] * 100:.1f}% of the "
             f"time.")

    closures = report.get("closure") or []
    if closures:
        ok = sum(1 for c in closures if c.get("crossed_line") and c.get("clean"))
        f.append(f"Lap closure: carries it home clean from {ok}/{len(closures)} "
                 f"late-lap drop-ins "
                 f"({', '.join(str(int(c['start_frac'] * 100)) + '%' for c in closures)}).")

    top_gear = max((int(g) for g in beh["gear_histogram"]), default=0)
    top_gear_frac = float(beh["gear_histogram"].get(top_gear, 0.0)) if top_gear else 0.0
    if abs(beh["gear_offset_mean"]) > 1.6 and beh["gear_offset_std"] < 1.1:
        direction = "upshift (+2, short-shifting)" if beh["gear_offset_mean"] > 0 \
            else "downshift (-2)"
        pinned = (f"Gear head is PINNED at max {direction}: mean offset "
                  f"{beh['gear_offset_mean']:+.2f}, active "
                  f"{beh['gear_offset_active_frac'] * 100:.0f}% of the time.")
        if beh["gear_offset_mean"] > 0 and top_gear_frac > 0.4 \
                and beh["mean_pace_on_track"] < 0.75:
            pinned += (f" It sits in gear {top_gear} "
                       f"{top_gear_frac * 100:.0f}% of the lap while only "
                       f"holding {beh['mean_pace_on_track'] * 100:.0f}% pace — "
                       f"lugging the engine off the powerband is a prime "
                       f"suspect for the missing speed.")
        f.append(pinned)
    elif beh["gear_offset_active_frac"] < 0.02 and beh["shifts_down"] < 3:
        f.append("Gear head is passive: it almost never overrides the RaceBox "
                 "and barely downshifts — braking-zone gears come from the box "
                 "alone.")
    elif beh["gear_offset_active_frac"] > 0.4:
        f.append(f"Gear head is very active "
                 f"({beh['gear_offset_active_frac'] * 100:.0f}% of steps "
                 f"override by >=1 gear, mean {beh['gear_offset_mean']:+.2f}).")

    if beh["steer_reversals_per_km"] > 22:
        f.append(f"Steering is rough: {beh['steer_reversals_per_km']:.0f} "
                 f"reversals/km (sawing).")
    if beh["steer_saturation_frac"] > 0.08:
        f.append(f"Steering saturates {beh['steer_saturation_frac'] * 100:.0f}% "
                 f"of the time — running out of steering authority.")
    if beh["coasting_frac"] > 0.25:
        f.append(f"Coasts {beh['coasting_frac'] * 100:.0f}% of the time — "
                 f"neither throttle nor brakes (indecision or fear).")

    slow = sorted((s for s in report["sectors"] if s.get("line_pace")),
                  key=lambda s: s["line_pace"])[:3]
    if slow:
        f.append("Slowest line-lap sectors: " + "; ".join(
            f"{s['sector']} ({s['name']}) {s['line_pace'] * 100:.0f}%"
            for s in slow) + ".")
    return verdict, f


def run_brain_diagnostics(checkpoint: str = FABLE_BEST, car: str | None = None,
                          out_dir: str = DIAG_DIR) -> str:
    t0 = time.time()
    if not os.path.exists(checkpoint):
        raise SystemExit(f"no checkpoint at '{checkpoint}'")
    brain = _Brain(checkpoint)
    meta = brain.meta
    checkpoint_car = str(meta.get("car") or FABLE_CAR)
    if car is not None and car != checkpoint_car:
        raise ValueError(
            f"checkpoint vehicle identity is {checkpoint_car!r}, but diagnostics "
            f"requested {car!r}; cross-car evaluation is forbidden"
        )
    car = checkpoint_car
    stage = meta.get("fable_stage") or "finish"
    scale = float(meta.get("fable_envelope_scale") or
                  stage_defaults(stage).envelope_scale)
    print(f"[diag] brain: {os.path.basename(checkpoint)} stage={stage} "
          f"scale={scale:.2f} updates={meta.get('updates')} "
          f"act_dim={brain.act_dim}", flush=True)

    trk = attach_envelope(named_track(RING_TRACK), car)
    L = float(trk.length)
    theo = float(trk.fable_envelope["lap_time"])
    spec = stage_defaults(stage)
    spec.envelope_scale = scale
    env = _make_env(brain, spec, trk, car)
    hz = float(env.ppo.control_hz)
    lap_budget = max(900.0, 2.4 * theo)

    # 1) flying line laps: race drop-in (the eval's 0.90x) + cautious (0.80x)
    line_laps, line_arrays, deaths, all_runs = [], [], [], []
    for tag, drop in (("race-dropin", 0.90), ("cautious-dropin", 0.80)):
        v0 = min(60.0, vref_at(trk, float(trk.arc[0])) * scale * drop)
        r = _run_traced(brain, env, trk, 0, v0, int(lap_budget * hz),
                        keep_trace=True)
        r["name"] = tag
        arr = r.pop("_arrays", None)
        if arr is not None:
            line_arrays.append(arr)
        if r["clean_lap"]:
            status = f"CLEAN {r['lap_time']}s"
        else:
            status = (f"{r['progress_frac'] * 100:.1f}% then "
                      f"{r['termination_reason']}")
        print(f"[diag] line lap ({tag}): {status}", flush=True)
        line_laps.append(r)
        all_runs.append(r)
        if r.get("death"):
            deaths.append(dict(r["death"], run=f"line:{tag}"))

    # 2) the 16 standard eval sector starts
    sector_runs = []
    steps_sector = int(70.0 * hz)
    for s in range(N_SECTORS):
        idx = int(len(trk.center) * s / N_SECTORS)
        v0 = min(60.0, vref_at(trk, float(trk.arc[idx])) * scale * 0.90)
        r = _run_traced(brain, env, trk, idx, v0, steps_sector)
        r["name"] = f"sector:{s}"
        sector_runs.append(r)
        all_runs.append(r)
        if r.get("death"):
            deaths.append(dict(r["death"], run=f"sector:{s}"))
    clean_secs = sum(1 for r in sector_runs if r["clean"])
    print(f"[diag] sector battery: {clean_secs}/{N_SECTORS} clean", flush=True)

    # 3) lap-closure drop-ins: can it carry the last 5-30% home?
    closure = []
    for frac in CLOSURE_FRACS:
        idx = int(len(trk.center) * frac)
        v0 = min(60.0, vref_at(trk, float(trk.arc[idx])) * scale * 0.90)
        need = (1.0 - frac) * L
        budget = int((need / max(20.0, 0.5 * float(np.mean(trk.fable_vref)))
                      + 40.0) * hz)
        r = _run_traced(brain, env, trk, idx, v0, budget)
        r["name"] = f"closure:{int(frac * 100)}"
        r["crossed_line"] = bool(r["progress_m"] >= need - 25.0)
        closure.append(r)
        all_runs.append(r)
        if r.get("death"):
            deaths.append(dict(r["death"], run=r["name"]))
    crossed = sum(1 for c in closure if c["crossed_line"] and c["clean"])
    print(f"[diag] closure drills: {crossed}/{len(closure)} carried home clean",
          flush=True)

    report = {
        "kind": "fable5_brain_diag",
        "version": 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": {
            "path": checkpoint,
            "name": os.path.basename(checkpoint),
            "car": car,
            "stage": stage,
            "envelope_scale": scale,
            "updates": meta.get("updates"),
            "act_dim": brain.act_dim,
            "obs_layout": meta.get("obs_layout"),
            "resumed_from": meta.get("fable_resumed_from"),
            "stored_eval": meta.get("fable_eval") or {},
        },
        "envelope": {"theoretical_lap": theo,
                     "v_max": float(trk.fable_envelope.get("v_max", 0.0)),
                     "track_length_m": L},
        "benchmarks": dict(HUMAN_BENCHMARKS),
        "line_laps": line_laps,
        "sectors": _sector_stats(sector_runs, line_arrays, L),
        "closure": closure,
        "deaths": deaths,
        "behavior": _behavior(line_arrays, all_runs, hz),
        "trace": (_downsample_trace(line_arrays[0], L) if line_arrays else {}),
        "wall_seconds": None,
    }
    verdict, findings = _findings(report)
    report["verdict"] = verdict
    report["findings"] = findings
    report["wall_seconds"] = round(time.time() - t0, 1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(checkpoint)[:-3] if checkpoint.endswith(".pt") \
        else os.path.basename(checkpoint)
    out_path = os.path.join(out_dir, f"{base}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"[diag] VERDICT: {verdict}", flush=True)
    for line in findings:
        print(f"[diag]   - {line}", flush=True)
    print(f"[diag] report -> {out_path} ({report['wall_seconds']}s)", flush=True)
    return out_path

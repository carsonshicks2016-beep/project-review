"""
Supra Command Center — a dedicated, self-contained dashboard for the whole
Supra AI program. All-Python (Flask), purpose-built: every control maps to a
real run.py flag, so there's no generic-template wiring to break.

Run:  python3 command-center/server.py     (or double-click SUPRA.command)
Then open http://localhost:8770

Endpoints
  POST /api/launch            {action, params}  -> spawn run.py, return pid
  GET  /api/stream/<pid>      SSE: live log lines + parsed training metrics
  POST /api/stop/<pid>        SIGINT (clean checkpoint) then SIGKILL fallback
  GET  /api/status            running tasks + system load
  GET  /api/checkpoints       saved policies w/ metadata
  GET  /api/fable-ga          GA champion + eval snapshot status
  GET  /api/diagnostics       recent exported telemetry bundles
  POST /api/checkpoint        {op, name, ...}  backup / rename / delete / activate
  GET  /api/tracks            track catalogue
  GET  /api/thumb/<name>      rendered PNG thumbnail of a track (cached)
  GET  /3d/                   standalone Three.js driving viewer
  WS   /api/3d/ws             isolated 3D viewer physics bridge
"""
from __future__ import annotations

import resource
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 8192), _hard))
except Exception:
    pass

import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Supra Ai 2/
HERE = os.path.dirname(os.path.abspath(__file__))
THUMBS = os.path.join(HERE, "thumbs")
PORT = int(os.environ.get("PORT", 8770))   # env override -> second instance / preview
sys.path.insert(0, ROOT)

# render thumbnails headlessly in THIS process (spawned sims get a clean env)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder=None)

from viewer3d.routes import register_viewer3d
register_viewer3d(app)

from observatory_api import register_observatory
register_observatory(app, ROOT)

# --------------------------------------------------------------------------- #
# command registry — purpose-built mapping of UI action -> run.py args
# --------------------------------------------------------------------------- #
# Single source of truth: the cars the dashboard offers ARE the real presets.
# Adding a car in supra/config.py auto-registers it here (and via /api/cars,
# in the frontend) — no hand-maintained list to drift out of sync.
from supra.config import PRESETS as _PRESETS
from supra.fable_editions import (
    CheckpointRejected,
    FableEdition,
    get_edition,
    load_edition_manifest,
    require_playable_checkpoint,
    resolve_checkpoint,
)
CARS = list(_PRESETS.keys())
TRACK_NAMES = ["club", "national", "coast", "sprint", "tech", "oval2", "akina",
               "pass", "nordschleife", "ridge", "speedbowl", "superspeed",
               "circuit", "crescent", "esses", "hook", "oval", "random",
               "touge", "endless"]

# current sensor layout (PHYSICS_3D_PLAN Stage 5+): checkpoints with another
# obs size are pre-hills museum pieces — badge them, don't silently activate
from supra.config import SensorSpec as _SensorSpec
from supra.sensors import SensorSuite as _SensorSuite
CUR_OBS = _SensorSuite(_SensorSpec()).obs_size          # 58 in hills-v1
GEN_STYLES = ["gp", "technical", "speedway", "touge"]
RING_TRACK = "nordschleife"
RING_CAR = "mazda787b" if "mazda787b" in CARS else "supra"
RING_DEFAULT_OUT = "nordschleife_787b_ppo.pt"
RING_RACE_BEST = "ring_787b_best.pt"
RING_RACE_MANIFEST = "ring_787b_pipeline.json"
FABLE_BEST = "fable5_ring_best.pt"
FABLE_MANIFEST = "fable5_ring_pipeline.json"
FABLE_GA_CHECKPOINT = "ga_787b_ring.npz"
FABLE_GA_SNAPSHOT = "fable_ga_eval_latest.json"
FABLE_STAGES = ("foundation", "flow", "finish", "fast", "frontier")
FABLE_ACTIONS = frozenset({
    "fable_train", "fable_auto", "fable_scratch", "fable_live",
    "fable_continue", "fable_watch", "fable_diagnose",
})


def _i(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _trk_flag(p):
    """Specialist track selector: 'pool'/'' -> generalist; else --track NAME."""
    t = (p.get("track") or "").strip()
    return ["--track", t] if t and t != "pool" else []


def _out_flag(p):
    """Custom run name -> --out NAME.pt (so runs stay separate)."""
    name = (p.get("out") or "").strip()
    if not name:
        return []
    name = os.path.basename(name)
    if not name.endswith((".pt", ".npz")):
        name += ".pt"
    return ["--out", name]


def _pt_name(name, default=RING_DEFAULT_OUT):
    name = os.path.basename((name or default).strip() or default)
    return name if name.endswith(".pt") else name + ".pt"


def _best_pt_name(name, default=RING_DEFAULT_OUT):
    name = _pt_name(name, default)
    return name[:-3] + "_best.pt"


def _ppo_extra(p):
    """Shared PPO/drift training options: LR+entropy anneal, parallel workers, and
    the plateau auto-reseed pipeline (patience + reseed tries)."""
    x = []
    if p.get("anneal"):
        x += ["--anneal"]
    w = _i(p.get("workers"), 1)
    if w and w > 1:
        x += ["--workers", str(w)]
    envs = p.get("pop", p.get("envs"))
    if envs not in (None, ""):
        x += ["--pop", str(max(1, _i(envs, 8)))]
    pat = p.get("patience")
    if pat not in (None, ""):
        x += ["--patience", str(_i(pat, 500))]
    mr = p.get("max_restarts")
    if mr not in (None, ""):
        x += ["--max-restarts", str(_i(mr, 3))]
    return x


def _target_flag(p):
    """Drift generalist: graduate onto this named track after the curriculum maxes."""
    t = (p.get("target") or "").strip()
    return ["--target", t] if t and t not in ("", "none") else []


def _real_elevation_track(p):
    return (p.get("track") or "").strip().lower() in (
        "nordschleife", "nurburgring", "nuerburgring"
    )


def _hill_flags(p):
    """Terrain control. Procedural tracks default flat; Nordschleife keeps real DEM elevation."""
    if p.get("flat"):
        return ["--flat"]
    if not p.get("hills") and not _real_elevation_track(p):
        return ["--flat"]
    if not p.get("hills") and _real_elevation_track(p):
        return []
    out = ["--hills"]
    hs = _f(p.get("hill_scale"), 1.0)
    if abs(hs - 1.0) > 1e-6:
        out += ["--hill-scale", str(round(hs, 2))]
    return out


def _style_flag(p):
    """Hybrid: drift-style strength override."""
    if p.get("style") in (None, ""):
        return []
    try:
        return ["--style", str(float(p["style"]))]
    except Exception:
        return []


def _fable_edition_from_params(params) -> FableEdition:
    """Resolve one edition and reject a forged/mismatched car selection."""
    payload = dict(params or {})
    requested = payload.get("edition") or payload.get("car") or "787b"
    try:
        edition = get_edition(str(requested))
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    supplied_car = str(payload.get("car") or "").strip().lower()
    if supplied_car and supplied_car != edition.car_id:
        raise ValueError(
            f"Fable edition {edition.id} requires car {edition.car_id}, "
            f"not {supplied_car}"
        )
    return edition


def _fable_checkpoint_for_command(
    edition: FableEdition, checkpoint: str | None
) -> str:
    """Return a real, same-edition checkpoint name for a launch command."""
    selection = os.path.basename(str(checkpoint or "").strip())
    try:
        if selection and selection != "active-best":
            return require_playable_checkpoint(ROOT, edition, selection).name
        return fable_watch_default(edition)
    except CheckpointRejected as exc:
        raise ValueError(str(exc)) from exc


def fable_watch_default(edition: FableEdition) -> str:
    """Resolve the active best from this edition's authoritative registry.

    This deliberately does not fall back to the old global 787B manifest: a
    requested Porsche watch/diagnostic must either resolve a playable 919
    checkpoint or fail before a cross-car subprocess is launched.
    """
    try:
        return resolve_checkpoint(ROOT, edition).checkpoint.name
    except CheckpointRejected as exc:
        raise ValueError(str(exc)) from exc


def build_command(action, p):
    if action in FABLE_ACTIONS:
        fb = dict(p or {})
        fb["track"] = RING_TRACK
        edition = _fable_edition_from_params(fb)
        fb["edition"] = edition.id
        fb["car"] = edition.car_id
        stage = (fb.get("stage") or "foundation").strip().lower()
        if action in {"fable_auto", "fable_scratch"}:
            stage = "auto"
        if action == "fable_scratch":
            # scratch ladder: no seed lineage, own run prefix so its files
            # never collide with (or clobber) the main pipeline's banks
            fb.pop("resume", None)
            out = str(fb.get("out") or edition.checkpoint_prefix)
            if out.endswith(".pt"):
                out = out[:-3]
            if not out.endswith("_scratch"):
                out += "_scratch"
            fb["out"] = out
        if stage not in (*FABLE_STAGES, "auto"):
            stage = "foundation"
        if action == "fable_watch":
            cp = _fable_checkpoint_for_command(edition, fb.get("checkpoint"))
            return [sys.executable, "run.py", "--watch-fable",
                    "--checkpoint", os.path.basename(cp), "--car", fb["car"]]
        if action == "fable_diagnose":
            # deep envelope-aware brain diagnostics (supra/fable5_diag.py)
            cp = _fable_checkpoint_for_command(edition, fb.get("checkpoint"))
            cmd = [sys.executable, "run.py", "--fable-diag",
                   "--checkpoint", os.path.basename(cp), "--car", fb["car"]]
            return cmd

        cmd = [sys.executable, "run.py", "--fable",
               str(_i(fb.get("iters"), 6000)), "--fable-stage", stage,
               "--car", fb["car"]]
        if action == "fable_live" or fb.get("live"):
            cmd += ["--live"]
        if action == "fable_continue":
            cp = _fable_checkpoint_for_command(
                edition, fb.get("checkpoint") or fb.get("resume")
            )
            cmd += ["--resume", os.path.basename(cp)]
        elif fb.get("resume"):
            cp = _fable_checkpoint_for_command(edition, fb.get("resume"))
            cmd += ["--resume", os.path.basename(cp)]
        if fb.get("anneal"):
            cmd += ["--anneal"]
        if fb.get("workers") not in (None, ""):
            cmd += ["--workers", str(max(1, _i(fb.get("workers"), 8)))]
        if fb.get("pop", fb.get("envs")) not in (None, ""):
            cmd += ["--pop", str(max(1, _i(fb.get("pop", fb.get("envs")), 8)))]
        if fb.get("patience") not in (None, ""):
            cmd += ["--patience", str(max(0, _i(fb.get("patience"), 700)))]
        if fb.get("max_restarts") not in (None, ""):
            cmd += ["--max-restarts", str(max(0, _i(fb.get("max_restarts"), 5)))]
        if fb.get("lr") not in (None, ""):
            cmd += ["--lr", str(_f(fb.get("lr"), 0.0))]
        if fb.get("envelope_scale") not in (None, ""):
            cmd += ["--fable-scale", str(_f(fb.get("envelope_scale"), 1.0))]
        if fb.get("no_promote"):
            cmd += ["--fable-no-promote"]
        if fb.get("open_best_floor"):
            cmd += ["--fable-open-best-floor"]
        if action == "fable_scratch":
            cmd += ["--fable-fresh"]
        if fb.get("out") not in (None, ""):
            # auto mode: --out is a run PREFIX (multiple ladder runs coexist)
            cmd += ["--out", os.path.basename(_pt_name(fb.get("out")))]
        return cmd

    if action in {"ring_race_survive", "ring_race_fast", "ring_race_attack",
                  "ring_race_auto", "ring_race_live", "ring_race_continue",
                  "ring_race_watch", "ring_race_diagnose"}:
        ring = dict(p or {})
        ring["track"] = RING_TRACK
        ring.setdefault("car", RING_CAR)
        stage = (ring.get("stage") or "survive").strip().lower()
        if action == "ring_race_survive":
            stage = "survive"
        elif action == "ring_race_fast":
            stage = "fast"
        elif action == "ring_race_attack":
            stage = "attack"
        elif action == "ring_race_auto":
            stage = "auto"
        if action == "ring_race_watch":
            cp = ring.get("checkpoint") or RING_RACE_BEST
            return [sys.executable, "run.py", "--watch-ring-race",
                    "--checkpoint", os.path.basename(cp), "--car", ring["car"]]
        if action == "ring_race_diagnose":
            ring.setdefault("checkpoint", RING_RACE_BEST)
            ring.pop("out", None)
            ring.setdefault("scenario", "solo")
            ring.setdefault("episodes", 1)
            ring.setdefault("max_steps", "")
            return build_command("diagnose_checkpoint", ring)

        cmd = [sys.executable, "run.py", "--ring-race",
               str(_i(ring.get("iters"), 8000)), "--ring-stage", stage,
               "--car", ring["car"]]
        if action == "ring_race_live" or ring.get("live"):
            cmd += ["--live"]
        if action == "ring_race_continue":
            cp = ring.get("checkpoint") or ring.get("resume") or RING_RACE_BEST
            cmd += ["--resume", os.path.basename(cp)]
        elif action == "ring_race_live" and ring.get("checkpoint"):
            cmd += ["--resume", os.path.basename(ring.get("checkpoint"))]
        if ring.get("anneal"):
            cmd += ["--anneal"]
        if ring.get("workers") not in (None, ""):
            cmd += ["--workers", str(max(1, _i(ring.get("workers"), 8)))]
        if ring.get("pop", ring.get("envs")) not in (None, ""):
            cmd += ["--pop", str(max(1, _i(ring.get("pop", ring.get("envs")), 8)))]
        if ring.get("patience") not in (None, ""):
            cmd += ["--patience", str(max(0, _i(ring.get("patience"), 850)))]
        if ring.get("max_restarts") not in (None, ""):
            cmd += ["--max-restarts", str(max(0, _i(ring.get("max_restarts"), 6)))]
        if ring.get("lr") not in (None, ""):
            cmd += ["--lr", str(_f(ring.get("lr"), 0.0))]
        if ring.get("critical_starts") not in (None, ""):
            cmd += ["--ring-critical-starts", str(ring.get("critical_starts"))]
        if ring.get("critical_prob") not in (None, ""):
            cmd += ["--ring-critical-prob", str(_f(ring.get("critical_prob"), 0.0))]
        if ring.get("ent_coef") not in (None, ""):
            cmd += ["--ring-ent-coef", str(_f(ring.get("ent_coef"), 0.0))]
        if ring.get("init_log_std") not in (None, ""):
            cmd += ["--ring-init-log-std", str(_f(ring.get("init_log_std"), -0.55))]
        if ring.get("force_log_std") not in (None, ""):
            cmd += ["--ring-force-log-std", str(_f(ring.get("force_log_std"), -0.55))]
        if ring.get("reset_optimizer"):
            cmd += ["--ring-reset-optimizer"]
        if ring.get("no_promote"):
            cmd += ["--ring-no-promote"]
        if ring.get("open_best_floor"):
            cmd += ["--ring-open-best-floor"]
        if ring.get("out") not in (None, "") and stage != "auto":
            cmd += ["--out", os.path.basename(_pt_name(ring.get("out")))]
        if ring.get("flat"):
            cmd += ["--flat"]
        elif ring.get("hills"):
            cmd += ["--hills"]
        return cmd

    if action in {"ring_ppo_train", "ring_ppo_live", "ring_ppo_continue",
                  "ring_ppo_watch", "ring_ppo_diagnose"}:
        ring = dict(p or {})
        ring["track"] = RING_TRACK
        ring.setdefault("car", RING_CAR)
        ring.setdefault("out", RING_DEFAULT_OUT)
        ring.setdefault("anneal", True)
        ring.setdefault("workers", 8)
        ring.setdefault("patience", 850)
        ring.setdefault("max_restarts", 6)
        if action == "ring_ppo_watch":
            ring.setdefault("checkpoint", _best_pt_name(ring.get("out")))
            return build_command("ppo_watch", ring)
        if action == "ring_ppo_diagnose":
            ring.setdefault("checkpoint", _best_pt_name(ring.get("out")))
            ring.pop("out", None)
            ring.setdefault("scenario", "solo")
            ring.setdefault("episodes", 1)
            ring.setdefault("max_steps", "")
            return build_command("diagnose_checkpoint", ring)
        mapped = {
            "ring_ppo_train": "ppo_train",
            "ring_ppo_live": "ppo_live",
            "ring_ppo_continue": "ppo_continue",
        }[action]
        if action == "ring_ppo_continue":
            ring.setdefault("checkpoint", ring.get("resume") or _best_pt_name(ring.get("out")))
        return build_command(mapped, ring)

    if action == "diagnose_checkpoint":
        cp = (p.get("checkpoint") or "").strip()
        if not cp:
            raise ValueError("diagnose requires a checkpoint")
        cmd = [sys.executable, "tools/diagnose_checkpoint.py", os.path.basename(cp),
               "--track", p.get("track", "club"),
               "--scenario", p.get("scenario", "auto"),
               "--episodes", str(_i(p.get("episodes"), 4))]
        if p.get("max_steps") not in (None, ""):
            cmd += ["--max-steps", str(_i(p.get("max_steps"), 0))]
        if p.get("flat"):
            cmd += ["--flat"]
        elif p.get("hills"):
            cmd += ["--hills"]
        elif not _real_elevation_track(p):
            cmd += ["--flat"]
        if p.get("hill_scale") not in (None, ""):
            cmd += ["--hill-scale", str(_f(p.get("hill_scale"), 1.0))]
        if p.get("opponents"):
            cmd += ["--opponents", p.get("opponents")]
        if p.get("out"):
            cmd += ["--out", p.get("out")]
        return cmd

    car = p.get("car", "supra")
    if car not in CARS:
        car = "supra"
    b = ["--car", car]
    t = _trk_flag(p)
    o = _out_flag(p)
    # explicit checkpoint to watch (else the viewers fall back to their default
    # slot: ppo_race.pt / ppo_drift.pt / ga_champion.npz)
    ck = []
    cp = (p.get("checkpoint") or "").strip()
    if cp:
        ck = ["--checkpoint", os.path.basename(cp)]
    xa = _ppo_extra(p)            # --anneal / --workers (race + drift)
    tg = _target_flag(p)          # --target NAME (drift / hybrid generalist)
    sf = _style_flag(p)           # --style W (hybrid)
    hf = _hill_flags(p)           # --flat / --hill-scale X (terrain, all modes)

    def _opp_flag(p):
        opp = p.get("opponents")
        if not opp:
            return []
        return ["--opponents", opp]

    A = {
        "drive":          lambda: ["--drive", "--track", p.get("track", "club"), *hf, *b]
                                  + (["--no-audio"] if p.get("noaudio") else []),
        "drive_gen":      lambda: ["--drive", "--gen", p.get("style", "gp"),
                                   "--difficulty", str(_f(p.get("difficulty"), 0.5)),
                                   "--length", str(_i(p.get("length"), 1200)), *hf, *b],
        "ga_train":       lambda: ["--train", str(_i(p.get("iters"), 120)),
                                   "--pop", str(_i(p.get("pop"), 60)),
                                   "--seed", str(_i(p.get("seed"), 7)), *t, *o, *hf, *b],
        "ga_live":        lambda: ["--train", str(_i(p.get("iters"), 60)), "--live",
                                   "--pop", str(_i(p.get("pop"), 60)),
                                   "--seed", str(_i(p.get("seed"), 7)), *t, *o, *hf, *b],
        "ga_continue":    lambda: ["--train", str(_i(p.get("iters"), 120)),
                                   "--resume", p.get("checkpoint", "ga_champion.npz"),
                                   "--pop", str(_i(p.get("pop"), 60)),
                                   "--seed", str(_i(p.get("seed"), 7)), *t, *o, *hf, *b],
        "ga_watch":       lambda: ["--watch", "--seed", str(_i(p.get("seed"), 7)), *ck, *hf, *b],
        "ppo_train":      lambda: ["--ppo", str(_i(p.get("iters"), 600)), *t, *o, *xa, *hf, *b],
        "ppo_live":       lambda: ["--ppo", str(_i(p.get("iters"), 100000)), "--live", *t, *o, *xa, *hf, *b],
        "ppo_continue":   lambda: ["--ppo", str(_i(p.get("iters"), 1000)),
                                   "--resume", p.get("checkpoint", "ppo_race.pt"), *t, *o, *xa, *hf, *b],
        "ppo_watch":      lambda: ["--watch-ppo", "--track", p.get("track", "national"), *ck, *hf, *b],
        "drift_train":    lambda: ["--drift", str(_i(p.get("iters"), 1000)), *t, *o, *xa, *tg, *hf, *b],
        "drift_live":     lambda: ["--drift", str(_i(p.get("iters"), 100000)), "--live", *t, *o, *xa, *tg, *hf, *b],
        "drift_continue": lambda: ["--drift", str(_i(p.get("iters"), 1500)),
                                   "--resume", p.get("checkpoint", "ppo_drift.pt"), *t, *o, *xa, *tg, *hf, *b],
        "hybrid_train":   lambda: ["--hybrid", str(_i(p.get("iters"), 2000)), *t, *o, *xa, *tg, *sf, *hf, *b],
        "hybrid_live":    lambda: ["--hybrid", str(_i(p.get("iters"), 100000)), "--live", *t, *o, *xa, *tg, *sf, *hf, *b],
        "hybrid_continue": lambda: ["--hybrid", str(_i(p.get("iters"), 2000)),
                                    "--resume", p.get("checkpoint", "ppo_hybrid.pt"), *t, *o, *xa, *tg, *sf, *hf, *b],
        "hybrid_watch":   lambda: ["--watch-hybrid", "--track", p.get("track", "national"), *ck, *hf, *b],
        "drift_watch":    lambda: ["--watch-drift", "--track", p.get("track", "club"), *ck, *hf, *b],
        "multiagent_train":      lambda: ["--ppo", str(_i(p.get("iters"), 600)), *t, *o, *xa, *hf, *b, *_opp_flag(p)],
        "multiagent_live":       lambda: ["--ppo", str(_i(p.get("iters"), 100000)), "--live", *t, *o, *xa, *hf, *b, *_opp_flag(p)],
        "multiagent_continue":   lambda: ["--ppo", str(_i(p.get("iters"), 1000)),
                                   "--resume", p.get("checkpoint", "ppo_race.pt"), *t, *o, *xa, *hf, *b, *_opp_flag(p)],
        "multiagent_watch":      lambda: ["--watch-ppo", "--track", p.get("track", "national"), *ck, *hf, *b, *_opp_flag(p)],
        "watch_race":            lambda: ["--watch-race", "--track", p.get("track", "national"), *hf, *_opp_flag(p)],
        "selfplay_train":        lambda: ["--ppo", str(_i(p.get("iters"), 600)), *t, *o, *xa, *hf, *b, "--multi-self-play"],
        "selfplay_live":         lambda: ["--ppo", str(_i(p.get("iters"), 100000)), "--live", *t, *o, *xa, *hf, *b, "--multi-self-play"],
        "selfplay_continue":     lambda: ["--ppo", str(_i(p.get("iters"), 1000)),
                                          "--resume", p.get("checkpoint", "ppo_race.pt"), *t, *o, *xa, *hf, *b, "--multi-self-play"],
    }
    if action not in A:
        raise ValueError(f"unknown action '{action}'")
    return [sys.executable, "run.py"] + A[action]()


# --------------------------------------------------------------------------- #
# process management
# --------------------------------------------------------------------------- #
PROCS = {}        # pid -> dict(proc, cmd, label, action, logs, metrics, subs, start, gui)
PLOCK = threading.Lock()

PPO_RE = re.compile(
    r"it\s+(\d+)\s+ret\s+([+-]?[\d.]+)\s+(laps|drift)\s+([+-]?[\d.]+)\s+diff\s+"
    r"([+-]?[\d.]+)\s+pi\s+([+-]?[\d.]+)\s+vf\s+([+-]?[\d.]+)\s+ent\s+([+-]?[\d.]+)"
    r"(?:\s+kl\s+([\d.]+)(\*?))?")   # kl prints when the trust region is armed
                                     # (fable stages); * = the update early-stopped
GA_RE = re.compile(r"gen\s+(\d+)\s+best\s+([\d.]+)\s+laps\s+mean\s+([\d.]+)")
# the HONEST deterministic-eval line (the number to trust — see HANDOFF §6.7).
# It carries no iteration of its own; _reader stamps it onto the current iter.
EVAL_DRIFT_RE = re.compile(r"\[eval\]\s+eval drift\s+([\d.]+)\s+x lap\s+([\d.]+)\s+=\s+([\d.]+)")
EVAL_HYBRID_RE = re.compile(r"\[eval\]\s+eval lap\s+([\d.]+)\s+x style\s+([\d.]+)\s+=\s+([\d.]+)")
EVAL_RACE_RE = re.compile(r"\[eval\]\s+eval laps\s+([\d.]+)")
EVAL_RING_RE = re.compile(r"\[eval-ring\]\s+(.*)")
EVAL_FABLE_RE = re.compile(r"\[eval-fable\]\s+(.*)")
# plateau auto-restart pipeline: reseed-from-best kicks and final convergence
RESTART_RE = re.compile(r"\[restart\].*?reseed\s+(\d+)/(\d+)")
CONVERGED_RE = re.compile(r"\[converged\].*?\(best\s+([\d.]+)")
# hills-era eval suffix: "... jumps=3 air=2.4s" — the trace that shows the
# moment a generation discovers jumping (PHYSICS_3D_PLAN Stage 6)
AIR_RE = re.compile(r"jumps=(\d+)\s+air=([\d.]+)s")


def parse_metrics(line):
    m = EVAL_FABLE_RE.search(line)
    if m:
        body = m.group(1)
        out = {"fable_eval": True}
        # canonical chart keys so the fable eval rides the existing series
        remap = {"chain": "clean_chain", "progress": "clean_progress_m",
                 "terminal": "terminal_rate", "offtrack": "offtrack_seconds",
                 "pace": "pace_ratio", "metric": "fable_metric",
                 "theo": "theoretical_lap"}
        for key, val in re.findall(r"([a-zA-Z_]+)=([^\s]+)", body):
            if "/" in val and key == "clean":
                a, b = val.split("/", 1)
                out["clean_sectors"] = _f(a, 0.0)
                out["sector_count"] = _f(b, 0.0)
                continue
            if key == "stage":
                out["fable_stage"] = val
                continue
            if key == "worst":
                out["worst_sectors"] = val      # "6,9,13" — keep as text
                continue
            out[remap.get(key, key)] = _f(val.rstrip("m"), 0.0)
        if out.get("lap", 0) > 0:
            out["best_clean_lap"] = out["lap"]
            out["eval_lap_time"] = out["lap"]
        return out

    m = EVAL_RING_RE.search(line)
    if m:
        body = m.group(1)
        out = {"ring_eval": True}
        for key, val in re.findall(r"([a-zA-Z_]+)=([^\s]+)", body):
            if "/" in val and key == "clean_sectors":
                a, b = val.split("/", 1)
                out["clean_sectors"] = _f(a, 0.0)
                out["sector_count"] = _f(b, 0.0)
                continue
            if key == "stage":
                out["ring_stage"] = val
                continue
            out[key] = _f(val.rstrip("m"), 0.0)
            if key == "clean_lap":
                out["best_clean_lap"] = out[key]
        if "metric" in out:
            out["ring_metric"] = out["metric"]
        if "clean_progress" in out:
            out["clean_progress_m"] = out["clean_progress"]
        if "best_clean_lap" in out and out["best_clean_lap"] > 0:
            out["eval_lap_time"] = out["best_clean_lap"]
        return out

    m = PPO_RE.search(line)
    if m:
        out = {"x": int(m.group(1)), "return": float(m.group(2)),
               m.group(3): float(m.group(4)), "difficulty": float(m.group(5)),
               "policy_loss": float(m.group(6)), "value_loss": float(m.group(7)),
               "entropy": float(m.group(8))}
        if m.group(9):
            out["kl"] = float(m.group(9))
            out["kl_stop"] = bool(m.group(10))
        return out
    m = GA_RE.search(line)
    if m:
        return {"x": int(m.group(1)), "best_lap": float(m.group(2)),
                "mean_lap": float(m.group(3))}
    def _with_air(d):
        a = AIR_RE.search(line)
        if a:
            d["jumps"] = int(a.group(1))
            d["air"] = float(a.group(2))
        return d

    m = EVAL_DRIFT_RE.search(line)
    if m:        # no "x" -> stamped onto the latest iteration by _reader
        return _with_air({"eval_drift": float(m.group(1)),
                          "eval_lap": float(m.group(2)),
                          "eval_score": float(m.group(3))})
    m = EVAL_HYBRID_RE.search(line)
    if m:        # hybrid: lap x style = score (style shown on the eval_drift series)
        return _with_air({"eval_lap": float(m.group(1)),
                          "eval_drift": float(m.group(2)),
                          "eval_score": float(m.group(3))})
    m = EVAL_RACE_RE.search(line)
    if m:
        return _with_air({"eval_lap": float(m.group(1))})
    m = RESTART_RE.search(line)
    if m:        # plateau reseed-from-best kick (no "x" -> rides the latest iter)
        return {"restart": int(m.group(1)), "max_restarts": int(m.group(2))}
    m = CONVERGED_RE.search(line)
    if m:        # trainer gave up after exhausting the reseeds
        return {"converged": True, "converged_best": float(m.group(1))}
    return None


def _reader(pid):
    info = PROCS[pid]
    last_x = 0
    for raw in iter(info["proc"].stdout.readline, b""):
        line = raw.decode("utf-8", "replace")
        if "pkg_resources" in line:
            continue
        info["logs"].append(line)
        if len(info["logs"]) > 4000:
            del info["logs"][:1000]
        met = parse_metrics(line)
        if met is not None:
            if "x" in met:
                last_x = met["x"]
                info["metrics"].append(met)
            else:
                # eval line: ride on the current iteration's point (it's printed
                # right after that iter's log line) so the honest [eval] metric
                # charts on the same x as the rolling numbers.
                met["x"] = last_x
                if info["metrics"] and info["metrics"][-1].get("x") == last_x:
                    info["metrics"][-1].update(met)
                else:
                    info["metrics"].append(met)
        payload = {"text": line, "metric": met}
        for q in list(info["subs"]):
            try:
                q.put_nowait(payload)
            except Exception:
                pass
    info["proc"].wait()
    try:
        info["proc"].stdout.close()
    except Exception:
        pass
    end = {"text": f"\n--- finished (exit {info['proc'].returncode}) ---\n",
           "metric": None, "done": True}
    info["logs"].append(end["text"])
    info["running"] = False
    for q in list(info["subs"]):
        try:
            q.put_nowait(end)
        except Exception:
            pass


@app.route("/api/launch", methods=["POST"])
def launch():
    data = request.get_json(force=True)
    action = data.get("action")
    params = data.get("params", {})
    try:
        cmd = build_command(action, params)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    fable_edition = None
    if action in FABLE_ACTIONS:
        # build_command has already validated this relationship; retain it on
        # the task so edition-specific dashboard state never scans another car.
        fable_edition = _fable_edition_from_params(params).id

    env = os.environ.copy()
    env.pop("SDL_VIDEODRIVER", None)      # spawned sims use the real display
    env.pop("SDL_AUDIODRIVER", None)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    pid = proc.pid
    gui = any(f in cmd for f in ("--drive", "--live", "--watch", "--watch-ppo",
                                 "--watch-drift", "--watch-ring-race",
                                 "--watch-fable"))
    PROCS[pid] = {"proc": proc, "cmd": " ".join(cmd[1:]), "label": action,
                  "logs": [], "metrics": [], "subs": [], "start": time.time(),
                  "running": True, "gui": gui, "fable_edition": fable_edition}
    threading.Thread(target=_reader, args=(pid,), daemon=True).start()
    return jsonify({"pid": pid, "command": " ".join(cmd), "gui": gui})


@app.route("/api/stream/<int:pid>")
def stream(pid):
    info = PROCS.get(pid)
    if not info:
        return "no such task", 404
    q = queue.Queue(maxsize=2000)
    info["subs"].append(q)

    def gen():
        yield "data: " + json.dumps({"text": "".join(info["logs"]),
                                     "history": info["metrics"],
                                     "snapshot": True,
                                     "running": info["running"]}) + "\n\n"
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                    yield "data: " + json.dumps(item) + "\n\n"
                    if item.get("done"):
                        break
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            if q in info["subs"]:
                info["subs"].remove(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/stop/<int:pid>", methods=["POST"])
def stop(pid):
    info = PROCS.get(pid)
    if not info or not info.get("running"):
        return jsonify({"ok": True, "msg": "not running"})
    try:
        os.killpg(os.getpgid(pid), signal.SIGINT)   # clean: training checkpoints

        def hardkill():
            time.sleep(6)
            if info.get("running"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
        threading.Thread(target=hardkill, daemon=True).start()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    procs = [{"pid": k, "label": v["label"], "cmd": v["cmd"],
              "running": v["running"], "gui": v["gui"],
              "edition": v.get("fable_edition"),
              "elapsed": int(time.time() - v["start"])}
             for k, v in PROCS.items()]
    procs.sort(key=lambda p: -p["pid"])
    try:
        load = os.getloadavg()[0]
        cpu = min(100, int(load / (os.cpu_count() or 4) * 100))
    except Exception:
        cpu = 0
    return jsonify({"tasks": procs, "cpu": cpu})


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
_ck_cache = {}


def _ckpt_meta(path):
    st = os.stat(path)
    key = (path, st.st_mtime)
    if key in _ck_cache:
        return _ck_cache[key]
    meta = {"size_kb": round(st.st_size / 1024), "mtime": st.st_mtime}
    try:
        if path.endswith(".pt"):
            import torch
            d = torch.load(path, map_location="cpu", weights_only=False)
            obs = int(d.get("obs_dim", 0))
            layout = (d.get("obs_layout")
                      if d.get("obs_layout") in ("hills-v1", "fable-v1")
                      else "hills-v1" if obs in (CUR_OBS + 2, CUR_OBS + 20)
                      else "pre-hills")
            meta.update(kind=d.get("mode", "ppo"), updates=int(d.get("updates", 0)),
                        difficulty=round(float(d.get("difficulty", 0)), 2),
                        car=str(d.get("car", "supra")),
                        track=d.get("track") or None,
                        metric=round(float(d.get("metric", 0)), 3),
                        drift=round(float(d.get("drift", 0)), 2),
                        laps=round(float(d.get("laps", 0)), 2),
                        obs=obs,
                        layout=layout,
                        train_arena=d.get("train_arena"),
                        n_agents=d.get("n_agents"),
                        track_profile=d.get("track_profile"),
                        ring_pipeline=bool(d.get("ring_pipeline")),
                        ring_stage=d.get("ring_stage"),
                        ring_reward_version=d.get("ring_reward_version"),
                        ring_eval=d.get("ring_eval"),
                        fable_pipeline=bool(d.get("fable_pipeline")),
                        fable_stage=d.get("fable_stage"),
                        fable_eval=d.get("fable_eval"),
                        fable_theoretical_lap=d.get("fable_theoretical_lap"))
        else:
            import numpy as np
            d = np.load(path, allow_pickle=True)
            obs = int(d["obs_size"]) if "obs_size" in d.files else 0
            meta.update(kind="ga", generation=int(d["generation"]),
                        fitness=round(float(d["fitness"]), 1),
                        car=str(d["car"]),
                        obs=obs,
                        layout=("hills-v1" if obs == CUR_OBS else "pre-hills"))
    except Exception as e:
        meta["error"] = str(e)
    _ck_cache[key] = meta
    return meta


@app.route("/api/checkpoints")
def checkpoints():
    out = []
    for f in sorted(os.listdir(ROOT)):
        if f.endswith((".pt", ".npz")):
            m = _ckpt_meta(os.path.join(ROOT, f))
            m["name"] = f
            m["active"] = f in ("ppo_race.pt", "ppo_drift.pt", "ga_champion.npz")
            out.append(m)
    return jsonify(out)


@app.route("/api/fable-ga")
def fable_ga_state():
    """Return the GA's own persisted status without touching Fable PPO state.

    The Evolution dashboard polls this independently from the Fable Five
    edition APIs.  A missing or partially-written artifact is normal before a
    GA run has completed an evaluation, so expose an empty state instead of a
    failing request.
    """
    snapshot = {}
    snapshot_path = os.path.join(ROOT, FABLE_GA_SNAPSHOT)
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                snapshot = loaded
        except (OSError, json.JSONDecodeError):
            # The trainer writes snapshots atomically.  Still, a manually
            # edited or stale file should not make the dashboard unusable.
            pass

    champion = {}
    checkpoint_path = os.path.join(ROOT, FABLE_GA_CHECKPOINT)
    champion_exists = os.path.exists(checkpoint_path)
    if champion_exists:
        try:
            import numpy as np
            with np.load(checkpoint_path, allow_pickle=False) as data:
                fields = ("generation", "fitness", "metric", "lap_time",
                          "car", "track", "stage", "obs_layout", "pop_size")
                for field in fields:
                    if field not in data.files:
                        continue
                    value = data[field]
                    champion[field] = value.item() if value.ndim == 0 else value.tolist()
        except (OSError, ValueError, KeyError):
            # Keep the status endpoint available while a trainer atomically
            # replaces its champion or a manual artifact is malformed.
            champion = {}

    return jsonify({
        "checkpoint": FABLE_GA_CHECKPOINT,
        "champion_exists": champion_exists,
        "champion": champion,
        "snapshot": snapshot,
        "superhuman_lap": 371.13,
    })


@app.route("/api/ring-race")
def ring_race_state():
    path = os.path.join(ROOT, RING_RACE_MANIFEST)
    manifest = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as e:
            manifest = {"error": str(e)}
    latest_diag = None
    diag_base = os.path.join(ROOT, "diagnostics")
    if os.path.isdir(diag_base):
        candidates = []
        for name in os.listdir(diag_base):
            folder = os.path.join(diag_base, name)
            man = os.path.join(folder, "manifest.json")
            if not os.path.isdir(folder) or not os.path.exists(man):
                continue
            try:
                with open(man, "r", encoding="utf-8") as fh:
                    dm = json.load(fh)
                cp = os.path.basename(dm.get("checkpoint") or "")
                if "ring_787b" in cp or dm.get("track") == RING_TRACK:
                    candidates.append((os.stat(folder).st_mtime, name, dm))
            except Exception:
                pass
        if candidates:
            candidates.sort(reverse=True)
            latest_diag = {
                "name": candidates[0][1],
                "path": os.path.join("diagnostics", candidates[0][1]),
                "checkpoint": os.path.basename(candidates[0][2].get("checkpoint") or ""),
                "track": candidates[0][2].get("track"),
            }
    return jsonify({
        "manifest": manifest,
        "latest_diagnostic": latest_diag,
        "best_exists": os.path.exists(os.path.join(ROOT, RING_RACE_BEST)),
    })


def _requested_fable_edition():
    """Resolve the mandatory edition query parameter for Fable dashboard data."""
    try:
        return get_edition(request.args.get("edition", "")), None
    except KeyError as exc:
        return None, (jsonify({"error": str(exc)}), 400)


def _event_matches_edition(event, edition: FableEdition) -> bool:
    """Fail closed unless an artifact positively identifies the selected car."""
    if not isinstance(event, dict):
        return False
    checkpoint = event.get("checkpoint")
    source = checkpoint if isinstance(checkpoint, dict) else event
    stored_eval = source.get("stored_eval")
    stored_eval = stored_eval if isinstance(stored_eval, dict) else {}
    car = str(source.get("car") or event.get("car") or "").lower()
    drivetrain = str(
        source.get("drivetrain_version")
        or event.get("drivetrain_version")
        or stored_eval.get("drivetrain_version")
        or ""
    )
    layout = str(source.get("obs_layout") or event.get("obs_layout") or "")
    return (
        car == edition.car_id
        and (not drivetrain or drivetrain == edition.drivetrain)
        and (not layout or layout == edition.observation_layout)
    )


def _fable_diag_records(edition: FableEdition):
    """Return only deep Brain Lab reports whose embedded identity is valid."""
    base = os.path.join(ROOT, "diagnostics", "fable5")
    if not os.path.isdir(base):
        return []
    records = []
    for name in os.listdir(base):
        if not name.endswith(".json"):
            continue
        path = os.path.join(base, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                report = json.load(fh)
            if not _event_matches_edition(report, edition):
                continue
            records.append((os.stat(path).st_mtime, name, report))
        except Exception:
            # An unreadable or unstamped report must never be assigned to a car.
            continue
    return sorted(records, reverse=True)


def _fable_diag_summary(mtime, name: str, report: dict, edition: FableEdition):
    checkpoint = report.get("checkpoint") or {}
    laps = report.get("line_laps") or []
    best_lap = min((lap.get("lap_time") for lap in laps if lap.get("lap_time")),
                   default=None)
    best_prog = max((lap.get("progress_frac") or 0.0 for lap in laps), default=0.0)
    behavior = report.get("behavior") or {}
    stored_eval = checkpoint.get("stored_eval") or {}
    return {
        "name": name[:-5],
        "mtime": mtime,
        "created": report.get("created"),
        "edition": edition.id,
        "car": edition.car_id,
        "checkpoint": checkpoint.get("name"),
        "policy_sha256": stored_eval.get("evaluated_policy_sha256"),
        "stage": checkpoint.get("stage"),
        "updates": checkpoint.get("updates"),
        "verdict": report.get("verdict"),
        "best_lap": best_lap,
        "best_progress": round(best_prog, 4),
        "clean_sectors": sum(1 for sector in (report.get("sectors") or [])
                             if sector.get("clean")),
        "mean_pace": behavior.get("mean_pace_on_track"),
        "n_deaths": len(report.get("deaths") or []),
        "wall_seconds": report.get("wall_seconds"),
    }


def _fable_active_tasks(edition: FableEdition):
    return [
        {"pid": pid, "label": info.get("label"), "edition": edition.id,
         "running": bool(info.get("running")), "start": info.get("start")}
        for pid, info in PROCS.items()
        if info.get("fable_edition") == edition.id
    ]


@app.route("/api/fable-five")
def fable_five_state():
    edition, error = _requested_fable_edition()
    if error:
        return error
    manifest = load_edition_manifest(ROOT, edition)
    try:
        resolved = resolve_checkpoint(ROOT, edition, manifest=manifest).as_dict()
        resolution_error = None
    except CheckpointRejected as exc:
        resolved = None
        resolution_error = str(exc)
    diagnostics = _fable_diag_records(edition)
    latest_diag = None
    if diagnostics:
        mtime, name, report = diagnostics[0]
        latest_diag = {
            "name": name[:-5],
            "path": os.path.join("diagnostics", "fable5", name),
            "checkpoint": (report.get("checkpoint") or {}).get("name"),
            "edition": edition.id,
            "car": edition.car_id,
            "mtime": mtime,
        }
    return jsonify({
        "edition": edition.as_dict(),
        "manifest": manifest,
        "resolved_checkpoint": resolved,
        "resolution_error": resolution_error,
        "active_tasks": _fable_active_tasks(edition),
        "latest_diagnostic": latest_diag,
        "best_exists": os.path.exists(os.path.join(ROOT, edition.canonical_champion)),
    })


@app.route("/api/fable-diag")
def fable_diag_list():
    """List only Brain Lab reports positively stamped for this edition."""
    edition, error = _requested_fable_edition()
    if error:
        return error
    return jsonify([
        _fable_diag_summary(mtime, name, report, edition)
        for mtime, name, report in _fable_diag_records(edition)[:60]
    ])


@app.route("/api/fable-diag/<name>")
def fable_diag_report(name):
    """Return a full, same-edition Brain Lab report by public name."""
    edition, error = _requested_fable_edition()
    if error:
        return error
    safe = os.path.basename(name)
    if not safe.endswith(".json"):
        safe += ".json"
    path = os.path.join(ROOT, "diagnostics", "fable5", safe)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if not _event_matches_edition(report, edition):
        return jsonify({"error": "not found"}), 404
    return jsonify(report)


_TRACK_CACHE = os.path.join(HERE, "fable_track_cache.json")


@app.route("/api/fable-track")
def fable_track():
    """Nordschleife geometry for the Pit Wall circuit map: a downsampled
    normalised centreline, elevation, landmarks, and the 16 eval-sector names.
    Built once (supra.track is numpy-only — no torch/SDL) and cached beside
    the server so restarts stay instant."""
    if os.path.exists(_TRACK_CACHE):
        try:
            return send_file(_TRACK_CACHE, mimetype="application/json")
        except Exception:
            pass
    try:
        import numpy as np
        from supra.track import named_track
        trk = named_track("nordschleife")
        c = np.asarray(trk.center)
        step = max(1, len(c) // 420)
        pts = c[::step]
        mn = pts.min(axis=0)
        span = float((pts.max(axis=0) - mn).max())
        pn = (pts - mn) / span * 1000.0
        z = np.asarray(getattr(trk, "z", np.zeros(len(c))))[::step]
        lms = [{"name": l.get("name"), "arc": round(float(l.get("start_arc", 0)), 1)}
               for l in (getattr(trk, "landmarks", None) or []) if l.get("name")]
        # sector names, diag-style: first–last landmark inside each 1/16th
        n_sec, L = 16, float(trk.length)
        names = []
        for s in range(n_sec):
            a0, a1 = s * L / n_sec, (s + 1) * L / n_sec
            inside = [l["name"] for l in lms if a0 <= l["arc"] < a1]
            if not inside:      # sector starts mid-landmark: take the last before
                before = [l["name"] for l in lms if l["arc"] < a0]
                inside = [before[-1]] if before else [f"S{s+1}"]
            names.append(inside[0] if len(inside) == 1 or inside[0] == inside[-1]
                         else f"{inside[0]}-{inside[-1]}")
        payload = {
            "pts": [[round(float(x), 1), round(float(y), 1)] for x, y in pn],
            "z": [round(float(v), 1) for v in z],
            "len_m": float(trk.length),
            "landmarks": lms,
            "sector_names": names,
            "half_width": float(getattr(trk, "half", 5.0)),
        }
        with open(_TRACK_CACHE + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(_TRACK_CACHE + ".tmp", _TRACK_CACHE)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# [pit] 01:59:33 KLGUARD: metric=5.275 best=167.736 lap=650.00s -> ROLLBACK (reason)
PIT_LINE_RE = re.compile(
    r"\[pit\]\s+(\d\d:\d\d:\d\d)\s+(.+?):\s+metric=([+-]?[\d.]+)\s+"
    r"best=([+-]?[\d.]+)(?:\s+lap=([\d.]+)s)?\s+->\s+(\w+)\s+\((.*)\)\s*$")
PIT_LR_RE = re.compile(r"lr -> ([\d.eE+-]+)")
PIT_STD_RE = re.compile(r"log-std -([\d.]+)")


@app.route("/api/fable-pit-log")
def fable_pit_log():
    """Return the selected edition's persisted PitWall decisions.

    The old fable5_pit_log.txt is shared by every car and contains no reliable
    edition stamp.  The edition manifest persists the same decisions together
    with car/drivetrain identity, so it is the only safe dashboard timeline.
    """
    edition, error = _requested_fable_edition()
    if error:
        return error
    n = max(20, min(2000, int(request.args.get("n", 400))))
    manifest = load_edition_manifest(ROOT, edition)
    pit = manifest.get("pit") if isinstance(manifest, dict) else None
    decisions = pit.get("decisions") if isinstance(pit, dict) else None
    decisions = decisions if isinstance(decisions, list) else []
    try:
        resolved = resolve_checkpoint(ROOT, edition, manifest=manifest).checkpoint
    except CheckpointRejected:
        resolved = None
    rows = []
    for decision in decisions[-n:]:
        if not isinstance(decision, dict):
            continue
        row = dict(decision)
        row.update({
            "edition": edition.id,
            "car": edition.car_id,
            "drivetrain_version": edition.drivetrain,
            "checkpoint": resolved.name if resolved is not None else None,
            "policy_sha256": resolved.policy_sha256 if resolved is not None else None,
        })
        rows.append(row)
    return jsonify({
        "edition": edition.id,
        "car": edition.car_id,
        "drivetrain_version": edition.drivetrain,
        "source": "edition-manifest-pit-state",
        "rows": rows,
    })


@app.route("/api/diagnostics")
def diagnostics():
    base = os.path.join(ROOT, "diagnostics")
    out = []
    if not os.path.isdir(base):
        return jsonify(out)
    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder):
            continue
        manifest_p = os.path.join(folder, "manifest.json")
        summary_p = os.path.join(folder, "summary.json")
        events_p = os.path.join(folder, "events.json")
        if not os.path.exists(summary_p):
            continue
        try:
            with open(summary_p, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            manifest = {}
            if os.path.exists(manifest_p):
                with open(manifest_p, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            events = []
            if os.path.exists(events_p):
                with open(events_p, "r", encoding="utf-8") as fh:
                    events = json.load(fh)
            runs = summary.get("runs") or []
            event_counts = summary.get("event_counts") or {}
            bad_terms = sum(1 for r in runs if r.get("termination_reason"))
            best_clean = summary.get("best_clean_lap_time")
            verdict = "developing"
            if best_clean:
                verdict = "clean-lap"
            elif bad_terms == len(runs) and runs:
                verdict = "failure"
            elif summary.get("offtrack_steps", 0) > 0:
                verdict = "unstable"
            out.append({
                "name": name,
                "mtime": os.stat(folder).st_mtime,
                "created": manifest.get("created"),
                "checkpoint": os.path.basename(manifest.get("checkpoint") or ""),
                "track": manifest.get("track"),
                "scenario": manifest.get("scenario"),
                "episodes": manifest.get("episodes"),
                "rows": summary.get("rows"),
                "mean_speed": round(float(summary.get("mean_speed") or 0), 2),
                "max_speed": round(float(summary.get("max_speed") or 0), 2),
                "best_clean_lap_time": best_clean,
                "offtrack_steps": int(summary.get("offtrack_steps") or 0),
                "collision_steps": int(summary.get("collision_steps") or 0),
                "event_counts": event_counts,
                "event_total": sum(int(v) for v in event_counts.values()),
                "termination_count": bad_terms,
                "run_count": len(runs),
                "verdict": verdict,
            })
        except Exception as e:
            out.append({"name": name, "error": str(e),
                        "mtime": os.stat(folder).st_mtime})
    out.sort(key=lambda d: d.get("mtime", 0), reverse=True)
    return jsonify(out[:50])


@app.route("/api/checkpoint", methods=["POST"])
def checkpoint_op():
    d = request.get_json(force=True)
    op, name = d.get("op"), d.get("name", "")
    src = os.path.join(ROOT, os.path.basename(name))
    if not os.path.exists(src) and op != "rename":
        return jsonify({"error": "not found"}), 404
    try:
        if op == "delete":
            os.remove(src)
        elif op == "backup":
            shutil.copy2(src, os.path.join(ROOT, os.path.basename(d["dest"])))
        elif op == "rename":
            os.rename(src, os.path.join(ROOT, os.path.basename(d["dest"])))
        elif op == "activate":      # copy onto the slot a watch/continue loads
            shutil.copy2(src, os.path.join(ROOT, os.path.basename(d["dest"])))
        else:
            return jsonify({"error": "bad op"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# tracks + thumbnails
# --------------------------------------------------------------------------- #
TRACK_INFO = {
    "club": "flowing · easy", "national": "flowing · mid", "coast": "flowing · long",
    "sprint": "technical · short", "tech": "technical · hard", "oval2": "speedway",
    "akina": "touge · hard", "pass": "touge · tightest",
    "nordschleife": "real 20.832 km · elevation",
    "ridge": "touge · JUMPS ⤴", "speedbowl": "top speed",
    "superspeed": "top speed · long",
    "circuit": "road course · long straights", "crescent": "sweeping C-curve",
    "esses": "snaking S-curves", "hook": "hook + inner notch",
    "oval": "legacy oval", "random": "legacy random", "touge": "legacy touge",
}
_thumb_lock = threading.Lock()
_car_sprite_lock = threading.Lock()
CAR_SPRITE_FRAMES = 360


import os
import signal

@app.route("/api/quit", methods=["POST"])
def quit_server():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"status": "shutting down"})

@app.route("/api/cars")
def cars():
    return jsonify(CARS)


@app.route("/api/tracks")
def tracks():
    return jsonify([{"name": n, "desc": TRACK_INFO.get(n, "")} for n in TRACK_NAMES])


def _render_thumb(name, path):
    import numpy as np
    import pygame
    from supra import track as tk
    from supra.carart import draw_car
    from supra.config import get_car
    from supra.physics import Vehicle
    if not pygame.get_init():
        pygame.init()
    if name in tk.NAMED or name in tk.SPECIAL:
        trk = tk.named_track(name)
    elif name == "oval":
        trk = tk.oval()
    elif name == "touge":
        trk = tk.touge(seed=7)
    else:
        trk = tk.random_circuit(seed=7)
    W, H = 300, 200
    surf = pygame.Surface((W, H))
    surf.fill((26, 28, 33))
    c = trk.center
    mn, mx = c.min(0), c.max(0)
    span = np.maximum(mx - mn, 1e-3)
    sc = min(W / span[0], H / span[1]) * 0.82
    cx, cy = (mn + mx) / 2

    def to(p):
        return (int((p[0] - cx) * sc + W / 2), int((p[1] - cy) * sc + H / 2))
    left = [to(p) for p in trk.left]
    right = [to(p) for p in trk.right]
    pygame.draw.polygon(surf, (48, 50, 57), left + right[::-1])
    pygame.draw.lines(surf, (120, 70, 70), True, left, 2)
    pygame.draw.lines(surf, (120, 70, 70), True, right, 2)
    pygame.draw.circle(surf, (90, 220, 120), to(trk.center[0]), 4)
    v = Vehicle(get_car("supra"))
    sx, sy, sa = trk.start_pose()
    v.reset(sx, sy, sa)
    draw_car(surf, lambda x, y: to((x, y)), max(sc, 4), v, v.spec)
    try:
        pygame.image.save(surf, path)
    except (pygame.error, NotImplementedError) as exc:
        # Some macOS pygame builds can render a surface but omit extended-image
        # encoders, making every dashboard sprite request return HTTP 500.
        # Pillow is already a project dependency for thumbnail-safe image IO;
        # use the same RGBA pixels rather than degrading the dashboard surface.
        if "extended format" not in str(exc).lower():
            raise
        from PIL import Image
        pixels = pygame.image.tostring(surf, "RGBA")
        Image.frombytes("RGBA", (W, H), pixels).save(path, format="PNG")


@app.route("/api/thumb/<name>")
def thumb(name):
    name = os.path.basename(name)
    path = os.path.join(THUMBS, f"{name}.png")
    if not os.path.exists(path):
        with _thumb_lock:
            if not os.path.exists(path):
                try:
                    _render_thumb(name, path)
                except Exception as e:
                    return str(e), 500
    return send_file(path, mimetype="image/png")


def _render_car_sprite(car, frame, path):
    import pygame
    import sys
    import importlib
    import supra.carart
    importlib.reload(sys.modules['supra.carart'])
    from supra.carart import draw_car
    from supra.config import get_car
    from supra.physics import Vehicle
    if not pygame.get_init():
        pygame.init()

    spec = get_car(car)
    veh = Vehicle(spec)
    veh.reset(0.0, 0.0, (frame % CAR_SPRITE_FRAMES) / CAR_SPRITE_FRAMES * math.tau)

    W, H = 340, 220
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    scale = 36.0

    def to_screen(x, y):
        return (int(W / 2 + x * scale), int(H * 0.64 - y * scale))

    draw_car(surf, to_screen, scale, veh, spec, brake=0.35, headlights=False)
    try:
        pygame.image.save(surf, path)
    except (pygame.error, NotImplementedError) as exc:
        # Some macOS pygame builds can render a surface but omit extended-image
        # encoders, making every dashboard sprite request return HTTP 500.
        # Pillow is already a project dependency for thumbnail-safe image IO;
        # use the same RGBA pixels rather than degrading the dashboard surface.
        if "extended format" not in str(exc).lower():
            raise
        from PIL import Image
        pixels = pygame.image.tostring(surf, "RGBA")
        Image.frombytes("RGBA", (W, H), pixels).save(path, format="PNG")


@app.route("/api/car-sprite/<car>/<int:frame>.png")
def car_sprite(car, frame):
    car = re.sub(r"[^a-z0-9_ -]", "", car.lower())
    if car not in CARS:
        car = "supra" if "supra" in CARS else CARS[0]
    frame = frame % CAR_SPRITE_FRAMES
    os.makedirs(THUMBS, exist_ok=True)
    path = os.path.join(THUMBS, f"car_{car}_{frame:02d}.png")
    sources = [os.path.join(ROOT, "supra", "carart.py"), __file__]
    source_mtime = max(os.path.getmtime(src) for src in sources)
    stale = not os.path.exists(path) or os.path.getmtime(path) < source_mtime
    if stale:
        with _car_sprite_lock:
            stale = not os.path.exists(path) or os.path.getmtime(path) < source_mtime
            if stale:
                try:
                    _render_car_sprite(car, frame, path)
                except Exception as e:
                    return str(e), 500
    resp = send_file(path, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


# --------------------------------------------------------------------------- #
# static frontend
# --------------------------------------------------------------------------- #
def _nocache(resp):
    # the frontend changes often during dev — never let the browser serve a
    # stale index.html / app.js / style.css (that's why new controls 'don't appear')
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/hologram")
def serve_hologram():
    return _nocache(send_from_directory("../", "hologram_demo.html"))

@app.route("/")
def index():
    return _nocache(send_from_directory(os.path.join(HERE, "static"), "index.html"))


@app.route("/static/<path:f>")
def static_files(f):
    return _nocache(send_from_directory(os.path.join(HERE, "static"), f))


if __name__ == "__main__":
    os.makedirs(THUMBS, exist_ok=True)
    print(f"\n  Supra Command Center  ->  http://localhost:{PORT}\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)

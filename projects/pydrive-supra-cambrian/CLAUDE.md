# Supra Ai 2 — orientation for agents

From-scratch RL driving simulator (custom 4-wheel physics, procedural + real
tracks, PyTorch PPO, live synthesized engine audio, a Flask dashboard). Full
overview: [README.md](README.md). **Read [HANDOFF.md](HANDOFF.md) top-to-just-enough**
before doing anything nontrivial — it's a living, newest-first project log and
is the actual source of truth for "what's going on right now." This file is
just the fast map to get you there.

## Active goal

Train a Mazda 787B to lap the real-elevation Nürburgring Nordschleife replica
at superhuman pace (beat Bellof's 371.13s qualifying lap; outright record
319.55s). The pipeline for this is **Fable Five**: `supra/fable5.py`, design
doc [FABLE5_PLAN.md](FABLE5_PLAN.md), CLI `run.py --fable / --fable-stage /
--watch-fable`, dashboard tab "Fable Five", gates `tools/validate_fable5.py`
+ `tools/validate_fable5_auto.py` (hermetic — run these, not ad hoc scripts,
before claiming a Fable change works). HANDOFF.md's newest entries are almost
always about this pipeline.

## Hard invariants — do not break these without being asked

- **`python3`/`pip3` only**, never bare `python`/`pip`. Headless sim needs
  `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.
- **Fable Five obs/action layout is frozen per car**: obs `fable-v1` = 58
  sensors + 8 pace + 2 mode = 68 dims; hybrid-ledger cars (919 Evo) ride
  `fable-v2` = fable-v1 + battery SOC + signed MGU power = 70 dims
  (`supra.fable5.obs_layout_for`); 3 actions (steer, long, gear-offset).
  Checkpoint compatibility depends on this — `PPO.load_policy` reconstructs
  the layout from checkpoint meta, don't hand-edit dims.
- **2D Viewer v1** (`supra/app.py`, `background.py`, `carart.py`) is frozen —
  it's the "saved" viewer, selected via `--classic` / `SUPRA_CLASSIC_VIEWER=1`.
  **Viewer v2** (`supra/viewer2.py`) is the default for drive/watch; all new
  viewer work goes there, not in v1.
- **Audio V5 Doppler is required** — world sources use phase-continuous
  propagation delay at 343 m/s. Never add independent pitch shifting after the
  delay. Camera cuts crossfade for 100 ms; listener-local wind is excluded.
- **No envelope heat ribbon** on the road surface (was explicitly removed) —
  `trk.fable_vref` feeds the HUD pace readout only, not road paint.
- Trust only deterministic `[eval]` / `[eval-ring]` / `[eval-fable]` lines for
  judging a run — rolling training-log numbers overstate reality. Always
  resume/watch from a `_best.pt`, and prefer `--fable-diag` (Brain Lab) over
  the eval summary alone when you need to know *why* something is slow/dying.
- Command Center (Flask, port 8770): edits to `command-center/server.py` need
  a dashboard restart; edits to `supra/*.py` / `run.py` apply to new runs
  automatically; static JS/HTML/CSS are no-cache (just refresh the browser).
- Never run `tools/validate_fable5_auto.py` (or similar pipeline tests) from
  the repo root without checking it's hermetic first — it and its manifest
  writes can clobber real files like `fable5_ring_eval_latest.json` if run
  carelessly. `validate_fable5_auto.py` runs in a scratch dir; trust that one.

## Layout

- `supra/` — the actual simulator: physics, tracks, sensors, PPO (`ppo.py`),
  Fable Five (`fable5.py`), viewers (`app.py`=v1, `viewer2.py`=v2), audio
  (`sound.py`), diagnostics (`diagnostics.py`, `fable5_diag.py`).
- `run.py` — CLI entry point for everything (drive/train/watch/diag flags).
- `command-center/` — the Flask web dashboard (`server.py` + `static/`).
- `tools/` — validators/gates (`validate_*.py`) and one-off diagnostic
  scripts (`diagnose_checkpoint.py`).
- `observatory/` — clean-room Fable Five browser Observatory (served via
  Command Center's `/observatory/`); Python physics and policy inference live
  in `supra/observatory.py`, with transport in `command-center/observatory_api.py`.
- Repo root is also where trained checkpoints (`*.pt`), pipeline manifests
  (`*_pipeline.json`), and eval-latest snapshots (`*_eval_latest.json`) live —
  it's cluttered with dozens of experiment checkpoints; don't assume a name
  without checking `HANDOFF.md` or the relevant manifest for which one is
  current.
- `.agents/` — scratch working notes from prior sub-agent runs
  (`BRIEFING.md`/`progress.md`/`handoff.md` per agent name). Not a maintained
  map of the repo — skip unless you're specifically resuming that agent's task.
- `project-dashboard/` (Node, port 8000) is a **separate, unrelated** general
  homepage — not part of this project.

## Where to look next

1. [HANDOFF.md](HANDOFF.md) — newest entries first; this is the real changelog
   and design-decision record.
2. [FABLE5_PLAN.md](FABLE5_PLAN.md) — Fable Five design doc, if the task
   touches training.
3. [README.md](README.md) — broader project status table and how to run the
   dashboard / drive / train / watch.

# Supra Ai 2 — orientation for agents

From-scratch RL driving simulator (custom 4-wheel physics, procedural + real
tracks, PyTorch PPO, live synthesized engine audio, a Flask dashboard). Full
overview: [README.md](README.md). **Read [docs/legacy/HANDOFF.md](docs/legacy/HANDOFF.md)
top-to-just-enough** before doing anything nontrivial — it's a living,
newest-first project log. For the currently active 919 auto-ladder, the newer
[FABLE5_919_LIVE_JOURNAL.md](FABLE5_919_LIVE_JOURNAL.md) supersedes it. This
file is just the fast map to get you there.

## Active goal

**The main car is the Porsche 919 Evo.** Train it to lap the real-elevation
Nürburgring Nordschleife replica at superhuman pace (beat Bellof's 371.13s
qualifying lap; outright record 319.55s). When a task says "the car" without
naming one, it means the 919 — check/verify/demo against that unless told
otherwise.

- Prefix `fable5_919_ring`, champion `fable5_919_ring_best.pt`, manifest
  `fable5_919_ring_pipeline.json`, journal
  [FABLE5_919_LIVE_JOURNAL.md](FABLE5_919_LIVE_JOURNAL.md).
- Champion evidence as of 2026-07-23: **437.73s clean flying lap, 16/16
  sectors, terminal 0.0**. Stage `frontier`.
- Rides `fable-v2` (70 dims), drivetrain `porsche919evo-7spd-ring-v2`,
  compatibility class `fable-v2-70x3-porsche919evo-7spd-ring-v2`.

The **Mazda 787B** (`fable-v1`, prefix `fable5_ring`, champion
`fable5_ring_best.pt`) is the older, secondary edition — still trainable and
playable, but not the live program. Its `frontier` checkpoints are broken
(`terminal_rate 1.0`, dies ~1.7% into the lap), so don't take a 787B frontier
run as evidence of anything.

The pipeline for both is **Fable Five**: `supra/fable5.py`, design doc
[docs/04_Fable_Five_Pipeline.md](docs/04_Fable_Five_Pipeline.md) (original
plan archived at [docs/legacy/FABLE5_PLAN.md](docs/legacy/FABLE5_PLAN.md)),
CLI `run.py --fable / --fable-stage / --watch-fable` (pass
`--car porsche_919evo` for the 919), dashboard tab "Fable Five", gates
`tools/validate_fable5.py` + `tools/validate_fable5_auto.py` (hermetic — run
these, not ad hoc scripts, before claiming a Fable change works). The live
journal's newest entries are almost always about this pipeline.

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
- `observatory/` — from-scratch Fable Five Observatory remake (served via
  Command Center's `/observatory/`). The frozen pre-remake client is archived
  in `legacy/observatory/`. Python physics and policy inference live in
  `supra/observatory.py`, with transport in `command-center/observatory_api.py`.
  Defaults to the **919** edition; the 787B is `?edition=787b`. It is a built
  client — `npm run build` in `observatory/` after any `src/` edit or the
  served `dist/` stays stale, and `npm run validate` (assets:check + tests +
  build) is the gate. Vehicle/world GLBs are generated, not hand-authored:
  `assets/vehicles/<car>/build_asset.py` then
  `tools/build_observatory_assets.py` to publish into `observatory/public/`;
  both have a read-only `--check` that asserts byte-for-byte reproducibility.
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

1. [README.md](README.md) — the consolidated project map: every subsystem,
   how to run everything, and a documentation index with currency labels.
2. [FABLE5_919_LIVE_JOURNAL.md](FABLE5_919_LIVE_JOURNAL.md) +
   [docs/legacy/HANDOFF.md](docs/legacy/HANDOFF.md) — newest entries first;
   the real changelog and design-decision record.
3. [docs/04_Fable_Five_Pipeline.md](docs/04_Fable_Five_Pipeline.md) — Fable
   Five design doc, if the task touches training.

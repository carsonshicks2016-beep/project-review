# Melee Bot v2: Puff Bot Lab

A Jigglypuff specialist for Super Smash Bros. Melee, rebuilt from scratch using what
version one taught us. Version one lives untouched in `../melee bot v1`, and an
unfinished Codex attempt at v2 is kept in `archive/`.

## Quick start

1. Double-click **`Launch Puff Bot.command`**. The dashboard opens at http://127.0.0.1:8777.
2. Press **New run**. Six emulators is a good default on this Mac; eight is the fastest.
3. Watch the ladder climb. Stop any time: the run saves a checkpoint and can be resumed.

From a terminal (inside this folder):

```sh
.venv/bin/python -m puffbot train --workers 6            # new run
.venv/bin/python -m puffbot train --resume RUN_NAME       # continue a run
.venv/bin/python -m puffbot evaluate runs/RUN/checkpoints/policy-....pt --level 9 --games 20
.venv/bin/python -m puffbot watch runs/RUN/policy.pt --opponent FOX --level 9   # real-time window
.venv/bin/python -m puffbot bench --counts 1,4,6,8                             # emulator throughput
.venv/bin/python -m pytest                                                      # 108 tests, no emulator needed
```

## Why it trains faster

| | v1 | v2 |
|---|---|---|
| Dolphin | Slippi 3.6.4 (Ishiiruka), **locked to 1× speed**, must render a window | Mainline Slippi 4.0 beta 19, **unlimited speed**, Null video (no window) |
| One emulator | 58 fps | 257 fps |
| Whole machine | ~190 frames/s trained (6–8 emulators, CPU-saturated) | ~1,150–1,250 frames of game per second with 6 emulators (~20× real time) |
| Stepping | Lockstep: every emulator waited for the slowest one, so one emulator in a menu froze all of them | Independent workers. Nobody waits on anybody |
| Between games | ~6–9 s of menus, often wedging at character select | **Instant Match**: the next game starts with no menus. Opponents re-roll every 4 games via a fresh ~4 s launch |
| Self-play | Opponent's frames discarded | Mirror games train on **both** sides, doubling that emulator's data |

Measured emulator throughput on this M2 Pro (`puffbot bench`): 1 → 257 fps, 4 → 831,
6 → 1,082, 8 → 1,207, 12 → 1,259. Past 8 the machine is saturated. At 6 emulators,
one hour of training plays roughly 20 hours of Melee. Mirror self-play games give two
players' experience per frame of game, so the dashboard shows experience separately
from game time (the first days' reports of "31× real time" counted both players).

## What v1 got wrong, and what v2 does instead

| v1 finding (from its measurements) | v2 |
|---|---|
| Reward paid a LOSS +226 and a WIN +143, so the bot learned to hold shield and stall | Every term is bounded; stocks ±1, damage 0.01/%, result ±1. The one shaping term (distance offstage) can't be farmed (tested); it is not refunded on death, so it is also a deliberate extra penalty for dying deep offstage |
| Puff's deaths were 68–87% self-destructs: a factored controller pressed shield in the air (an air-dodge) ~1 frame in 8 | A 43-move Puff vocabulary. Shield actions are only legal on the ground or ledge, so no air-dodge is reachable (tested). No Rollout or Sing |
| Policy was "state-blind": behavioural-cloning anchor held it to the average tournament input | Pure reinforcement learning from scratch. Imitation can be added later as a warm start, never as a permanent anchor |
| LSTM measured 0.0 influence | Single-frame MLP. The game already reports velocities, hitstun and animation frame |
| One emulator's menu froze the farm; stuck CSS slider cost minutes | Async workers; a stuck worker relaunches its own Dolphin in ~4 s |
| CPU level silently wrong on ~10% of games | Every game is credited to the level the game *reports*, never the one requested |
| A worker exception wedged the whole run for hours (atexit join on a child) | Every wait has a deadline; workers hard-exit; the learner restarts dead workers with backoff; shutdown steps are independent |
| Orphaned Dolphins squatted on Slippi ports | Each Dolphin's pid is recorded and reaped on relaunch and at shutdown |
| Assists applied to one player only made self-play comparisons meaningless | The action vocabulary and its legality rules are identical for every policy-controlled player |
| 10-game win rates quoted without error bars | Evaluations report 95% Wilson intervals |
| Self-destructs and launches were conflated | v1's corrected classifier: a stock is a self-destruct only if you were in control and unhit |

## How training works

```
            policy.pt (weights, rewritten after every update)
        ┌──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
   worker 0                    worker 1        …          worker N-1
   Dolphin (Null video)        Dolphin                    Dolphin
   local policy copy           local policy copy          local policy copy
   plays at its own pace       plays at its own pace      plays at its own pace
        │  128-decision fragments  │                          │
        └──────────────┬───────────┴──────────────────────────┘
                       ▼
                 learner (main process, Apple GPU)
                 V-trace + PPO update every ~4k decisions
                 CPU ladder, self-play league, checkpoints, status.json
```

* **Decisions** every 3 frames (20 per second), each one a macro: a short hop, a bair
  drifting in, Rest, a roll. Macros that need longer (a full hop, a short-hop aerial)
  run to completion.
* **Observation**: both players' position, percent, stocks, speeds, hitstun, hitlag,
  shield, jumps, ledge distance, animation state (learned embedding) and character;
  distance between them; the nearest two enemy projectiles; the last macro chosen.
* **Learning**: workers can be a version or two behind the learner, so targets use
  V-trace and updates use the PPO clip. Fragments more than 4 versions old are dropped
  and counted on the dashboard.
* **Opponents**: 80% CPUs drawn from a Jigglypuff-relevant roster (Fox, Falco, Marth,
  Falcon, Peach, Puff, …) at levels around the **ladder frontier**. The frontier rises
  when the bot wins 65% of 40 games there and falls below 20%. 20% self-play: half
  mirror (both sides learn), half frozen snapshots saved every 10M frames.

## Me vs AI (Play tab)

Plug in or pair a controller, open **Play**, pick a model, your character, the button
layout and how fast the bot reacts, and press **Start match**. The harness picks both
characters itself (hands off for those few seconds), then your controller has port 1
and the bot plays Puff on port 2 on Final Destination. Rematches start by themselves;
**End match** closes it.

* Your port listens to both the harness's pipe and your controller (Dolphin's `|` takes
  whichever is pressed), which is how the harness can drive the character select.
* Layouts: *Switch Pro positions* (GameCube A on the right face button, B bottom,
  X top, Y left) or *Xbox letters*. Bumpers are Z, triggers are analog L/R, sticks are
  the control stick and C-stick. Mapping lives in `puffbot/dolphin.py` (`FACE_LAYOUTS`).
* The controller is found through the SDL library inside Dolphin (an Xbox One pad over
  Bluetooth is `SDL/0/Xbox One S Controller`). Background input is on, so the pad works
  while the browser has focus.
* Reaction time: the bot normally acts on the frame it sees; 10/15/21-frame delays make
  it see the game late, closer to a human (Phillip uses 21). It never trained with one.
* From a terminal: `puffbot versus CHECKPOINT --character FOX --layout switch --delay 15`.

## Phillip opponents (Slippi-AI)

The game's CPUs top out at level 9, and beating them does not mean much against a
learned opponent: the 50M-frame bot that won 72% against level-9 CPUs lost **0 of 4**
to Phillip's Fox and Falco (1.5 stocks taken per game, 4 lost). Phillip is
[Slippi-AI](https://github.com/vladfi1/slippi-ai)'s released `medium-v2` model: learned
from human replays, then tuned with reinforcement learning, 21-frame (350 ms) reaction
delay, playing FOX, FALCO, MARTH, CPTFALCON, PEACH, JIGGLYPUFF, SAMUS, PIKACHU, YOSHI
and LUIGI here (Sheik and Ice Climbers are left out: our menus cannot pick them for a
human-controlled port).

* **Train against it:** `phillip_workers` (dashboard: *Phillip opponents*) is how many
  emulators always play Phillip. Each keeps one model loaded in a child process and
  costs about one CPU core; those emulators run near 110-120 fps instead of ~200.
  Character mix: `phillip_characters`.
* **Evaluate against it:** `puffbot evaluate CHECKPOINT --kind phillip --opponents FOX,FALCO`
  or *Opponent: Phillip* in the dashboard's evaluation dialog.
* **Setup (not in git):** `vendor/slippi-ai` (commit 275c072) with its own `.venv`
  (`uv venv --python 3.12 && uv pip install -e ".[jax,tf]"`, about 1.9 GB) and
  `vendor/slippi-ai-models/medium-v2` (95.6 MB, sha256 48dcfd87...) from the Dropbox link
  in Slippi-AI's README. The model is TensorFlow-only (the upstream JAX converter does not
  support it). Loading needs no `wandb`; `puffbot/phillip_server.py` supplies stand-ins
  for the two training-only imports.

## The dashboard

Four tabs. **Live** is everything below. **Report** is a morning summary: what happened
over the last 8/12/24 hours or the whole run in plain sentences, an hour-by-hour table
(CPU-9 win rate, Phillip record, tech rate, time offstage, game length, ...) colored
green/red, and the highlights (ladder moves, alarms, scorecards, every Phillip win).
**Phillip** is the record against Slippi-AI by character, stocks taken per game over
time, and every win. **Progress** charts every 60-game scorecard ever run, with its
likely range, and how Puff's move mix changed over the run (which moves rose or fell
most). A run started before 2026-09-26 has no event log, so its report judges stalling
from the games themselves.

* **Emulators**: one tile each with fps, opponent and health. Red means it's being restarted.
* **CPU ladder**: win rate at each level, and the current frontier.
* **Charts**: win rate vs CPU with the level played; damage ratio; self-destruct share;
  Rest hits; learner entropy and value loss; training speed.
* **Stalling watch**: share of game time spent offstage and game length over the last
  100 games. On 2026-09-26 a run learned to hide offstage to put off losing stocks (win
  rate vs CPU 9 fell from ~60% to 10% in an hour) while every learning metric looked
  normal. Offstage time was the tell: 25-48% healthy, 65-78% stalling. At 50% the
  dashboard shows a warning banner, at 60% an alarm, both in Events too, naming the last
  checkpoint saved before it started (kept safe from pruning) with a button to evaluate it.
* **What Puff chooses**: how often each macro is picked. A healthy policy concentrates
  on aerials, movement and Rest.
* **Checkpoints**: *Evaluate* (fixed opponents, 95% intervals), *Watch* (opens a
  real-time game vs level-9 Fox), *Crown* (copies it to `champions/`).
* **Evaluations** say *complete* only when every opponent got its full number of
  finished games at the requested level; otherwise *incomplete* with what is missing.
  Each report stores the checkpoint's SHA-256, the code commit and the Dolphin build.
* **New run** can start from any champion or checkpoint.

## Settings worth knowing (`config.local.json`)

Anything in `puffbot/config.py` can be overridden here. The ones you'd actually change:

| key | default | meaning |
|---|---|---|
| `workers` | 6 | emulators; 8 is the throughput peak on this Mac |
| `visible_workers` | 0 | emulators rendered in a window (the rest use no video) |
| `visible_speed` | 0 | 0 = unlimited; 1.0 = real time. Only affects visible ones |
| `selfplay_fraction` | 0.2 | share of games against itself |
| `cpu_start_level` | 3 | starting ladder frontier |
| `opponents` | roster | CPU characters and their weights |
| `act_every` | 3 | frames per decision |
| `gamma` | 0.999 | per-frame discount: an 11.5 s planning half-life. 0.997 (3.85 s) made hiding to postpone a stock loss pay |
| `critic_warmup_updates` | 150 | after starting from weights with a different or unrecorded `gamma`, train only the value side this many updates first |
| `stall_offstage_warn` / `_alarm` | 0.50 / 0.60 | stalling-watch thresholds (share of game time offstage, last `stall_window` = 100 games) |
| `scorecard_hours` | 3 | every this many hours the newest checkpoint plays the 60-game scorecard vs CPU 9 on `scorecard_workers` (2) extra emulators; 0 = off |
| `dolphin` | mainline | set to `vendor/Slippi Dolphin.app/...` to fall back to 3.6.4 |

## Layout

```
puffbot/
  dolphin.py      launch/configure Dolphin, frames with deadlines, menus, cleanup
  actions.py      the 43 Jigglypuff macros, legality rules, frame-by-frame executor
  observation.py  what the policy sees
  reward.py       bounded objective + per-game stats (SD classifier, Rest hits)
  model.py        policy/value network, checkpoint files
  vtrace.py       off-policy targets and PPO loss
  actor.py        a worker: one emulator, never waits
  learner.py      training loop, supervision, ladder, status
  league.py       ladder, opponent sampling, Wilson intervals
  evaluate.py     fixed-protocol evaluation (also powers "watch")
  bench.py        emulator throughput benchmark
  server.py, web/ dashboard
runs/RUN/         config.json, status.json, games.jsonl, metrics.jsonl, policy.pt,
                  checkpoints/, league/ (self-play snapshots), workers/ (Dolphin homes + logs)
champions/        crowned checkpoints
vendor/           mainline Slippi Dolphin (used), Slippi 3.6.4 (fallback)
```

## Troubleshooting

* **First run from the dashboard: every emulator fails once with "never opened its
  controller pipes"**: seen once on 2026-09-24, most likely macOS asking whether the
  emulator may use the Desktop/Downloads folders (the ISO lives in Downloads). If a
  permission prompt appears, click Allow. Workers retry by themselves either way; it
  did not recur.
* **A worker tile keeps going red**: open `runs/RUN/workers/wNN/dolphin.log`. The
  learner restarts it automatically with backoff; the rest keep training.
* **Run shows "crashed"**: the learner's traceback is in the Events panel and
  `runs/RUN/train.log`. Resume from the dashboard; the last checkpoint is at most 10
  minutes old.
* **Leftover Dolphins after a hard kill**: `pkill -f Slippi_Dolphin`.
* **Mainline Dolphin misbehaves after an update**: point `dolphin` at the 3.6.4 build;
  everything works, just at 1× speed per emulator with visible windows.

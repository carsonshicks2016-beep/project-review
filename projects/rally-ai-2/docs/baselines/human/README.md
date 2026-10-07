# Human baseline (Phase D3 — offline path)

The North Star claim is *faster than a competent human on a keyboard*. Until F4
live drive exists in WATCH, baselines are recorded here with the terminal
keyboard recorder.

**No times in this directory are claimable until Carson has actually driven the
held-out seeds.** `times.json` is a placeholder schema instance with an empty
`seeds` list.

## Invariants (non-negotiable)

| Rule | Value |
|---|---|
| Physics | Same `RallyEnv` / vendored Supra path as training |
| Stages | Same `generate(seed, tier)` as eval |
| Control rate | **30 Hz** (`CONTROL_DT = 1/30`) — wall-clock paced |
| Replay contract | `contracts.write_json(..., kind="replay")`, `source: "human"` |
| Held-out seeds | `seed % 10 == 7` (train draws the complement) |

A human at 60 Hz against an agent at 30 Hz is **not** a comparison. The
recorder sleeps to `CONTROL_DT` each step on purpose.

## Record one attempt

From the sim package (real terminal — curses needs a TTY):

```bash
cd /Users/REVIEW_USER/RallyAI2/packages/sim
.venv/bin/python -m rallyai.env.drive --seed 7 --tier 0 --attempt 1
```

Controls: `W/↑` throttle · `S/↓` brake · `A/D` steer · `Space` handbrake ·
`R` reset · `Enter` save · `Q` quit.

Replay path (default):

```
replays/human/human_seed{seed}_tier{tier}_a{attempt}.json
```

List the default held-out seed set:

```bash
.venv/bin/python -m rallyai.env.drive --list-held-out 12
```

Headless smoke (CI only — **not** a baseline):

```bash
.venv/bin/python -m rallyai.env.drive --seed 7 --tier 0 --script throttle \
  --max-steps 90 --out /tmp/human_smoke.json
```

## N attempts per seed workflow

Recommended: **N = 5** finished attempts per `(seed, tier)` before summarizing.

1. Pick a held-out seed (`seed % 10 == 7`) and tier matching the eval harness.
2. For `attempt` in `1..N`:
   - Run the drive command with `--attempt k`.
   - Drive to a termination (finish preferred). The recorder auto-saves on
     episode end; `Enter` also saves mid-run if needed.
   - If the run is a discard (wrong stage feel, distraction), delete that
     replay file and re-record the same `--attempt`.
3. Only **finished** runs (`meta.termination == "finish"`) enter the baseline.
4. After N finished times for a seed, fill `times.json`:
   - `best_time_s` = minimum finished time
   - `median_time_s` = median of the N finished times
   - `attempts` = list of `{attempt, time_s, replay, clean}`
5. Do **not** edit replay files by hand — re-record. Hashes will catch edits.

Suggested first pass (easy tier):

| seed | tier | attempts |
|---|---|---|
| 7 | 0 | 5 |
| 17 | 0 | 5 |
| 27 | 0 | 5 |
| 37 | 0 | 5 |

Harder tiers come after the easy-tier bar exists.

## Files

| Path | Role |
|---|---|
| `times.schema.json` | JSON Schema for the aggregate baseline table |
| `times.json` | Aggregate best/median times — empty until real drives land |
| `../../../replays/human/*.json` | Per-attempt replays (`source: human`) |

## Blockers for a claimable baseline

- Real keyboard drives have not been recorded yet.
- F4 live human drive (viewer) is nicer UX but not required for validity of
  this offline path.
- Aggregate `times.json` must be filled only from finished held-out replays.
- Eval harness (D1) reads this file via `load_human_baseline` / `evaluate(..., human_baseline_path=)` and scores `time_vs_human` when rows exist. Empty `seeds` → `time_vs_human: null`.

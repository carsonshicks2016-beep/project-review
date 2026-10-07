# SMW-RL — a PPO agent that speedruns Super Mario World

Trains one PPO policy per level of Super Mario World (SNES), then chains them
into a route you can watch play back-to-back with a live timer and splits.

Runs on Apple Silicon. Measured on an M2 Pro: **~930 agent-steps/s** (≈3,700
emulated frames/s) with 12 parallel emulators and the policy on MPS.

## Where this actually stands

| level | unaided clear | note |
|---|---|---|
| Yoshi's Island 2 | **~50%** | first level solved |
| Yoshi's Island 1 | **~45%** | solved after the curriculum-coverage fix |
| everything else attempted | 0% | skipped: no usable curriculum |

Two levels clear from a standing start about half the time. The boot and
overworld harness works from power-on, and `smwrl.playthrough` can drive a
continuous session.

**The binding constraint is curriculum coverage, not compute.** Random
exploration cannot find a route from start to goal on the harder levels, so
there is nothing to build a curriculum from and training on them is wasted.
Levels where naive play nearly succeeds (YoshiIsland2) work cleanly; levels
needing a precise trick (YoshiIsland3's vine room, YoshiIsland4) do not. More
GPU hours will not change that -- a better exploration method would.

Numbers here come from `smwrl.evaluate` over 60 fresh episodes. Treat any pass
measured on 30 episodes with suspicion: sampling noise at these rates is about
±9%, which is how a 45% policy once reported 57%.

---

## Quick start

```bash
python -m smwrl.setup_core                    # verify/repair the emulator core
python -m smwrl.probe_ram --state YoshiIsland1  # sanity-check the RAM map
python -m smwrl.train --level YoshiIsland1 --steps 5_000_000
python -m smwrl.watch --level YoshiIsland1
```

Everything below assumes the venv is active (`source .venv/bin/activate`) or
that you prefix commands with `./.venv/bin/python`.

## The ROM

Not included — supply your own legally-obtained dump at
`integration/SuperMarioWorld-Snes-v0/rom.sfc`.

It must be the headerless US ROM, 524,288 bytes,
SHA-1 `6b47bb75d16514b6a476aa0c73a683a2a4c18765`. If yours is a `.smc` of
524,800 bytes it has a 512-byte copier header; strip it:

```bash
tail -c +513 "Super Mario World.smc" > integration/SuperMarioWorld-Snes-v0/rom.sfc
```

## Read this before you touch the emulator core

The snes9x core in stable-retro's macOS arm64 wheel **is broken** — it renders a
black screen forever and the game never boots.

`cores/snes/port.h` picks endianness like this:

```c
#if defined(__i386__) || ... || defined(__x86_64__) || defined(ARM) || defined(ANDROID)
#define LSB_FIRST
#define FAST_LSB_WORD_ACCESS
#else
#define MSB_FIRST      /* <-- Apple Silicon lands here */
#endif
```

Apple Silicon defines `__aarch64__`/`__arm64__`, none of which are in that list,
and the Makefile's macOS branch hardcodes `arch = intel` so `ARM` is never
defined either. The core is compiled **big-endian on a little-endian CPU**: it
byte-swaps every word access, so the CPU runs but the game derails and the PPU
never draws anything.

`vendor/snes9x_libretro.dylib` is the same source rebuilt with `-DARM`.
`python -m smwrl.setup_core` detects the broken core and installs it.

**Anything that reinstalls or upgrades `stable-retro` restores the broken core.**
Re-run `python -m smwrl.setup_core` afterwards, or `bash tools/build_snes9x.sh`
to rebuild from scratch.

## Training a level

```bash
python -m smwrl.train --level YoshiIsland1 --steps 5_000_000
```

At ~930 steps/s that is roughly **1.5 hours per 5M steps**. `--resume` continues
from the last checkpoint; Ctrl-C saves before exiting. Watch progress live with
`tensorboard --logdir runs`.

The console line is the thing to watch:

```
[YoshiIsland1] 1,200,000 steps | 931 steps/s | mean_x 1840 | best_x 3220 | clear 12.0% | eps 8,412
```

`mean_x` is average distance into the level, `clear` is the fraction of recent
episodes that reached the goal.

### How fast can this go? (measured, M2 Pro 12-core)

The workers are **not** CPU-bound. At 12 envs each worker sits at ~30% CPU and
the whole run uses ~370% of 1200% available. Every vectorised step is a sync
barrier: the workers block while the main process does its GPU forward and pipe
round-trips, so throughput is limited by per-step *latency* in a single Python
thread, not by cores.

That makes widening one run a weak lever, and running several a strong one:

| Config | Aggregate steps/s |
|---|---|
| 1 run × 12 envs | 936 |
| 1 run × 24 envs | 1,113 |
| 1 run × 32 envs | 1,179 |
| 1 run × 48 envs | 1,230 |
| 3 runs × 12 envs | 1,840 |
| **4 runs × 12 envs** | **2,092** ← peak |
| 6 runs × 12 envs | 2,064 (and each run is much slower) |

Quadrupling the workers in one run buys only +31%. Running four separate
trainings buys +123%.

So:

- **One level, as fast as possible:** `--n-envs 32` (~1.2h per 5M, +26% over 12).
- **Working through the route:** four levels at once with `--n-envs 12` each.
  Each takes ~2.7h but you finish four, so ~0.67h per level — 24 levels in
  ~16h instead of ~36h.

Reproduce any of this with `python tools/bench_scaling.py --envs 12,24,32`.

**Training is always headless** — the emulators run with `render_mode=None` and
never draw. The only rendering cost is the optional spectator, measured at 32
envs with `tools/measure_rate.py`:

| | steps/s | h per 5M |
|---|---|---|
| headless | 1,188 | 1.17h |
| `smwrl.live --brain` running | 1,146 | 1.21h |

So watching costs about **3.5%** — roughly 2.5 extra minutes per 5M steps. Not
worth turning off. Measure it yourself with `python tools/measure_rate.py`;
don't parse the console log for this, it only prints every 20k steps and the
quantisation error swamps the effect.

### Exploration comes first (this is the important part)

PPO alone gets stuck. Each level has an obstacle it cannot stumble past, and
once stuck it stays stuck: YoshiIsland3 sat at x≈310 for 600k steps across
6,230 episodes and never moved.

The fix is a **Go-Explore frontier archive** (`smwrl/archive.py`,
`smwrl/explore.py`): return to a promising save state, take *sticky random*
actions from there, and archive anything that reaches somewhere new. It needs no
policy at all, so it cannot be blocked by a bad one.

```bash
python -m smwrl.explore --level YoshiIsland3 --iters 12000 --rollout 200
python -m smwrl.train   --level YoshiIsland3 --curriculum --steps 4_000_000
```

On the level PPO could not move in, ~32 minutes of exploration produced:

```
cells: 475
frontier: room 1 x 4610 (reached in 937 steps)
clears: 3379
```

It found a route through a second room and reached the goal 3,379 times.

Three details carry the method, and each was necessary:

* **Sticky actions.** One action held for 4–16 steps, not re-rolled every step.
  Per-step random jitters on the spot and clears nothing.
* **A rightward prior.** Uniform over 14 actions wastes most rollouts walking
  left.
* **Delayed commit.** A state is archived only once the player is *still alive*
  12 steps later. Without this the archive fills with states captured mid-fall
  into a pit — they have the highest x, frontier-weighted selection picks them
  forever, and every excursion dies instantly. That bug capped the first run at
  x=437; fixing it reached x=960 immediately, and then the whole level.

Cells are also scored by a Laplace-smoothed survival rate, so a cell that
*sometimes* works keeps its priority (YoshiIsland3 only broke through after
~1,400 retries of one hard cell) while a cell that has never once survived is
abandoned.

### The reverse curriculum

A level is 4,000–5,000 pixels long, so from a standing start the agent never
sees the later sections and training plateaus. Fix it by seeding some episodes
from save states part-way through:

Curricula are persisted by `smwrl/curriculum_store.py`, which is **fail-closed**:
a curriculum is only loaded if it still matches the archive that produced it, so
a stale or hand-edited `curriculum.pkl` is rejected rather than silently trained
on.

`smwrl.explore` writes `curriculum.pkl` alongside the archive, so normally you
just pass `--curriculum`. Two things about how it is *sampled* matter:

**It anneals backwards.** States are ordered start → goal, and training begins
by drawing only from the states nearest the goal, widening the window back
toward the level start over `--curriculum-anneal` resets (default 4000). Each
segment is therefore learned with the segment after it already solved. Sampling
the whole curriculum uniformly instead asks the agent to learn every section at
once — with uniform sampling YoshiIsland3's curriculum episodes reached x=5083
while its *unaided* score stayed at 367.

**It should be dense.** `--curriculum-size` defaults to 48. The agent has to
practise the exact obstacle it is stuck on, not one 400 px away.

`smwrl.curriculum` (harvesting states from the policy's own rollouts) still
exists and is useful for collecting *faster* routes once a policy is good, but
it cannot break through a wall — every state it produces is somewhere the policy
already reaches. Use `smwrl.explore` for that.

## Watching training live

`train.py` publishes the in-progress policy to `latest.zip` every
`--save-every` steps (default 100k, about 2 minutes). Run this in a second
terminal and it hot-reloads each new policy as it appears, so you watch the
agent actually improve:

```bash
python -m smwrl.live --level YoshiIsland1 --brain
```

Both files are written to a temp path and `os.replace`d into place, so the
spectator never reads a half-written checkpoint.

The `--brain` engineer panel shows:

| Readout | What it tells you |
|---|---|
| **Agent view** | the 84×84 grayscale the policy actually sees — not the pretty frame |
| **Saliency** | which pixels the chosen action was most sensitive to (∂log π(a)/∂pixels). Diffuse early on; it sharpens onto Mario and nearby hazards as the policy learns |
| **Action distribution** | the full 14-way policy, not just the sampled action. Watch it collapse from near-uniform onto `→+Y+B` |
| **Value** | the critic's estimate, tracked across the episode |
| **Entropy** | starts near ln(14)=2.64 and falls as the policy commits |
| **Training status** | steps, steps/s, mean_x, clear rate, and how old the loaded policy is |

The spectator samples from the policy by default, because that is what training
actually does — taking the argmax of a half-trained policy just replays the same
degenerate episode. Pass `--deterministic` for the argmax.

It runs its own emulator and keeps the policy on CPU, so it costs a core but
never touches the GPU. If training slows noticeably, use `--fps 30` or train
with `--n-envs 11`.

For curves rather than gameplay: `tensorboard --logdir runs`.

## Watching a finished run

```bash
python -m smwrl.watch --level YoshiIsland1     # one level
python -m smwrl.watch --route                  # every trained level, with splits
python -m smwrl.watch --route --record run.mp4 # also write a video
```

`watch.py` is the showcase runner and defaults to deterministic (argmax) play —
use it once a level is actually trained.

Route mode skips levels that have no policy yet and tells you which, so it stays
useful while you are still working through the list. Esc or closing the window
stops it.

## Why the observation is colour, and the HUD is cropped

YoshiIsland1 kept dying in one pit at x≈3420 (deaths clustered within 10 px).
That gap is spanned by SMW's **dotted-line blocks** — they look like a floor and
are intangible until a P-switch is hit. Rendering the agent's 84×84 view beside
the real frame showed the dashed outline was *completely erased* by grayscale
downsampling: the whole region read as uniform gray.

A controlled A/B settled it — identical curriculum, crop and entropy, differing
only in colour vs grayscale:

| curriculum stage reached | colour | grayscale |
|---|---|---|
| 26 | 40k steps | 40k steps |
| 24 | **80k** | 440k |
| 22 | **140k** | 700k |
| 20 | 820k | 840k |

| | colour | grayscale |
|---|---|---|
| unaided mean_x | **206** | 17 |

Colour reached the early curriculum stages ~5× faster and its unaided progress
was ~12× better; grayscale stayed pinned at x≈16, dying at the level start. The
two converge by stage 20, so the advantage is largest early — but the unaided
gap is decisive. Caveat: one seed, one level.

The HUD crop (`crop_top=32`) applied to both arms and is untested in isolation,
but it costs nothing: it removes ~32 of 224 rows of score/timer pixels the
policy cannot act on and hands them back to the playfield.

`ObsConfig(color=False, crop_top=0)` restores the old behaviour. **Changing the
observation changes the network's input shape, so it invalidates existing
checkpoints for that level.**

## Running many levels unattended

```bash
python -m smwrl.autopilot --worker a --levels YoshiIsland1,DonutPlains1 --bar 0.5
```

Works through a queue: explore -> check coverage -> verify -> train in chunks ->
evaluate -> next level. Every job is a subprocess in its own process group, so a
wedged trainer is killed with its emulator children rather than orphaning them,
and a crash cannot take the queue down. Decisions land in
`checkpoints/autopilot_<worker>.json`.

Three guards, each of which exists because its absence cost real hours:

* **Coverage gate.** A curriculum reaching back under 50% of the level is
  skipped, not trained. `verify` alone passes such a curriculum because it only
  checks the stages nearest the goal.
* **Budget extension.** A level still improving at its step budget gets extended
  rather than abandoned mid-climb.
* **Promotion on evidence.** `best.zip` is only replaced when a frozen candidate
  beats the incumbent on the same fixed seed suite, compared by the Wilson lower
  bound rather than the raw rate.

## Playing the whole game

```bash
python -m smwrl.playthrough
```

One continuous session from power-on: scripted menus and overworld, then each
gameplay segment dispatched by live translevel ID to its specialist. Nothing is
reset to a save state, so lives, powerups and map progress carry forward exactly
as they do for a human. Deliberately fail-closed -- an unknown translevel,
missing policy, exhausted lives or transition timeout stops the run, and mode
`0x29` (THE END) is the only success signal.

This is the end goal's skeleton. It is limited by how many levels have policies,
which is currently two.

## Measuring whether a level is actually solved

```bash
python -m smwrl.evaluate --all --episodes 50
```

Every episode starts fresh, so this is the honest number. Training's own
`clear` column mixes in curriculum-seeded episodes that begin next to the goal
and clear trivially — YoshiIsland2 once reported **25% while its unaided rate
was 0%**. The training log now prints a separate `SOLO` column computed only
from fresh starts; trust that one.

Reliability is the target, not a personal best: a route of N levels completes at
roughly (clear rate)^N, so 24 levels at 90% each finishes only 8% of the time.
`evaluate --all` prints that product for you.

## Tests

```bash
python -m pytest tests/ -q          # 81 tests, ~5s
```

**81 tests.** `tests/test_core.py` needs no ROM. `tests/test_integration.py` drives the real
emulator and skips itself if the ROM is missing. They encode the failures that
actually cost time here — a black-screen core, idling beating attempting, the
archive fixating on doomed states, progress accounting breaking across rooms —
so a regression in any of them fails loudly instead of quietly wasting hours of
GPU time.

Two tests are deliberately in tension and both must hold: `test_failing_cells_lose_weight`
(abandon true dead ends) and `test_hard_but_passable_cells_keep_priority`
(keep retrying hard ones). Tuning the archive to satisfy only one breaks the
method.

## How it works

| Piece | Choice | Why |
|---|---|---|
| Observation | 84×84 grayscale, 4-frame stack | Standard, and enough to see momentum |
| Actions | 14 curated combos | Raw SNES is `MultiBinary(12)` = 4,096 mostly-meaningless combinations |
| Frameskip | 4, max-pooled over last 2 | SMW needs buttons *held*; max-pool defuses sprite flicker |
| Reward | new-ground progress − time − death + clear&nbsp;bonus | Time penalty is what buys speed |
| Algorithm | PPO (SB3), 12 envs, MPS | Emulator is single-threaded C, so throughput scales with processes |

The reward only pays for **new** maximum x. Rewarding raw `dx` lets the agent
farm reward forever by oscillating left and right.

One constraint in `RewardConfig` is load-bearing: `death_penalty` (50) must
exceed the total time penalty an episode can accrue (1,200 steps × 0.05 = 60 —
capped well below by the stuck-detector at 220 steps). Otherwise dying
immediately becomes the optimal policy.

## The RAM map is verified, not assumed

`integration/SuperMarioWorld-Snes-v0/data.json` taps SNES bus addresses
directly (`rambase` is `0x7E0000`). Every one was confirmed against a live game
with `python -m smwrl.probe_ram`:

| Variable | Address | Confirmed behaviour |
|---|---|---|
| `x_pos` | `$7E0094` | 16 → 764 holding right |
| `y_pos` | `$7E0096` | signed — negative above the level |
| `game_mode` | `$7E0100` | `0x14` in play; 11→13→18 on death |
| `player_anim` | `$7E0071` | hits `9` one frame before `lives` drops |
| `lives` / `coins` / `score` | `$7E0DBE/0DBF/0F34` | counting correctly |
| `end_level_timer` | `$7E1493` | non-zero once the goal is triggered |

`end_level_timer` is the one still unconfirmed in practice — nothing has cleared
a level yet. Verify it the first time an agent finishes one.

## Scope — what this can and cannot do

**It can** learn individual levels well and play a route of them quickly. That
is a well-trodden result for 2D Mario and the hardware here is sufficient.

**It cannot** reproduce a real any% world record. The ~41-second record uses the
credits warp — arbitrary code execution driven by frame-perfect sprite
manipulation. No amount of PPO finds that; it is a fundamentally different kind
of search.

Also out of scope right now: the overworld map, castles, ghost houses and Star
Road, because stable-retro ships no save states for them. `ROUTE` in
`smwrl/levels.py` covers the 24 levels that do have states. Adding the others
means generating those states first.

## Layout

```
smwrl/
  retro_env.py    emulator construction, custom integration registration
  ram.py          RAM decoding (verified addresses)
  actions.py      the 14-action discrete space
  wrappers.py     frameskip, reward shaping, termination, checkpoint pool
  env.py          vectorised env construction
  levels.py       level registry + route
  archive.py      Go-Explore frontier archive (cells, selection, persistence)
  explore.py      random frontier exploration -- run this before training
  train.py        PPO training entry point
  curriculum.py   harvest states from a trained policy (cannot break walls)
  evaluate.py     unaided clear rate; the number that actually matters
  diagnose.py     find where a policy stalls, and screenshot it
  live.py         single-level viewer + engineer panel (saliency, policy, value)
  live_grid.py    watch several levels at once
  watch.py        route runner with splits, optional mp4
  probe_ram.py    verify the RAM map against a live game
  setup_core.py   detect/repair the big-endian core
tools/
  pipeline.sh     explore -> train -> evaluate for one or more levels
  train_batch.sh  several levels concurrently (the throughput peak)
  bench_scaling.py  measure throughput vs worker count
  measure_rate.py   exact steps/s from status.json
  build_snes9x.sh   rebuild the core from source
tests/            81 tests; integration ones skip without a ROM
integration/      data.json, scenario.json, 24 level save states, your ROM
vendor/           the fixed core binary
```

## Known rough edges

- `stable-retro` game names need the `-v0` suffix here (`SuperMarioWorld-Snes-v0`).
- `retro.make` defaults to opening a render window, which costs throughput and
  crashes on close under macOS pyglet. `make_raw_env` always passes
  `render_mode` explicitly.
- `reset()` returns an empty info dict; the wrapper takes a no-op step to
  populate it.
- stable-retro 0.9.x arm64 macOS wheels contain x86_64 binaries and will not
  import at all. Stay on 1.0.1.
- `smwrl.watch` prints an `objc[...] Class METAL_TextureData is implemented in
  both ...` warning on startup, because pygame and OpenCV each bundle their own
  copy of SDL2. It is noisy but harmless here.

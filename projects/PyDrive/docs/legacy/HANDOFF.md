# Supra Drift — Project Handoff & Reference

> **2026-07-23 — Frontier pipeline doctor (2026-07-22) triage: probe
> hermeticity + remaining P1 fixes landed**
> Acting on `docs/FABLE5_919_FRONTIER_PIPELINE_DOCTOR_2026-07-22.md` (the
> Phase 1–3 commits of 07-22 had already landed its P0 machinery):
> 1. **Probe isolation (real bug fixed):** the 07-22 18:07 probe run wrote
>    through the REAL `fable5_ring_pipeline.json` +
>    `fable5_ring_eval_latest.json` and hard-coded the 787B champion
>    regardless of `--car`. `run_frontier_probe` now loads
>    `champion_path_for(car)` and confines every artifact (manifest,
>    eval-latest, checkpoints, events.jsonl, metrics.jsonl) to
>    `runtime/fable5/probe/<prefix>_frontier_probe/`. The clobbered real
>    manifests were restored from the last authentic event in
>    `runtime/fable5/fable5_ring_frontier/events.jsonl` (the pre-probe 787B
>    frontier state was itself already collapsed, metric -3.89; only
>    provenance was repaired). `FableEvaluator` grew a `run_name` override so
>    pit-less runs stop dumping events into a bare stage-name dir.
> 2. **Probe correctness:** critic-only calibration now freezes everything
>    except the value head — previously `log_std` stayed trainable, so the
>    "action-preserving" control window still widened action noise. Fixed the
>    final-eval off-by-N, skipped the actor phase after a calibration
>    numerical error, and the probe now actually WRITES per-update
>    metrics.jsonl (phase-tagged; the Phase 3 code accepted the flag but never
>    logged). Metrics entry construction refactored into
>    `PPO._append_update_metrics`, shared with `train()`.
> 3. **Frontier failure-weighted starts (doctor P1):** frontier spec now
>    carries `sector_fail_gain=2.5, seed_bias_cap=5.0` (was generic 1.0/3.0)
>    so dying sectors (919 heat: 1/3/7) get majority-weight rehearsal.
> 4. **Goal semantics (doctor P1):** the frontier gate string no longer says
>    "beat 319.55s, then chase the theoretical 415.5s" (backwards — the
>    centerline estimate is slower than the record); it now orders: clean lap
>    -> improve champion -> reach envelope estimate -> external real-world
>    benchmark. Same fix for the SUPERHUMAN recommendation string.
> 5. **Reward decomposition (doctor P1):** `--fable-diag` reports now include
>    per-run cumulative `reward_parts`, a per-sector split on the line laps,
>    and a `reward_decomposition.by_termination` aggregate — the A/B lens for
>    the clean-vs-failed probe pair before anyone retunes coefficients.
> GATES: `validate_fable5.py`, `validate_fable5_auto.py`,
> `validate_fable5_safeguards.py` all pass; end-to-end probe smoke verified
> byte-identical real manifests. NOTE: the 787B champion
> `fable5_ring_best.pt` currently evals as a collapsed NON-FINISHER (dies at
> ~2% of the lap) — the intact 437.73s finisher is the 919's
> `fable5_919_ring_best.pt`. Disk headroom is down to ~17 GiB (99%).

> **2026-07-12 — Command Center: the Fable Five tab is now the PIT WALL
> (quantum-particle-swarm training screen)**
> Carson approved the concept mockup (artifact "Fable Five — Pit Wall",
> particle-swarm v2) — this lands it for real. The old Fable tab's
> status/eval widgets are replaced by a race-engineering board rendered from
> live data; the LAUNCH controls survived untouched (same `fb-*` element ids,
> re-housed in a collapsible "Pit board" drawer + dock, so `initFableFive`'s
> existing wiring in app.js still drives every run.py action).
> NEW FILES: `command-center/static/pitwall.js` (particle engine + panel
> renderers, ~1.1k lines) and `pitwall.css` (all styles scoped under
> `#tab-fable`, `pw-` prefixed — zero bleed into other tabs).
> SERVER (needs the usual dashboard restart, done):
> 1. `/api/fable-track` — downsampled Nordschleife centreline + elevation +
>    landmarks + the 16 diag-style sector names, built once from
>    `supra.track` (numpy-only import) and cached in
>    `command-center/fable_track_cache.json`;
> 2. `/api/fable-pit-log` — parsed tail of `fable5_pit_log.txt` (t/run/
>    metric/best/lap/decision/reason + lr and log-std cuts regexed out of
>    rollback reasons) so the eval timeline + calm ladder survive dashboard
>    restarts instead of living only in SSE memory;
> 3. `PPO_RE` grew an optional `kl 0.0123*` capture -> SSE metrics now carry
>    `kl`/`kl_stop` (the trust-region lever is watchable live).
> THE BOARD: race-control header (stage rail from auto.history, run identity,
> TRAINING/IDLE led, ladder budget) · "the chase" hero (gap to Bellof as a
> SWARM-FORMED numeral, lap axis with 919/Bellof/physics-floor/banked-best
> marks) · circuit map (real centreline; sector failure heat colours a
> particle river that literally AGITATES in death zones; worst-sector
> callouts; landmark whispers) · sector tower (names + pace bars + heat +
> clean/reason chips, hover mirrors onto the map) · eval timeline (pit-log
> evals, banked-best staircase, the 0.25x collapse line, rollback/best
> flags) · optimiser channels (live SSE: KL w/ early-stop count, entropy,
> value loss, return) · calm ladder (lr staircase per pit call) + envelope
> scale/segment · gates panel (healthy 12/16 · 0.25 · 0.90 and mastery 0.88
> as bullet gauges w/ verdict — thresholds MIRRORED from fable5.py, update
> both if retuned) · pit levers (rollback/reseed budgets) · pit feed (race-
> control messages w/ flag colours) · garage (hall-of-fame brains w/ real
> Watch/Diagnose launches) · Brain Lab strip (newest fable5 diag trace:
> v vs v_ref over the elevation ghost, throttle/brake, gear, lateral;
> scrub-synced to the map cursor) · behaviour tiles + pace histogram.
> THE SWARM: one canvas glued to the tab; panel borders are flowing particle
> streams, cursor drags a 140px gravity well w/ swirl, every click emits a
> shockwave (clicked buttons scatter 3.4x and reassemble), the circuit is a
> river of light. Engineering: pre-rendered glow sprites (no per-particle
> blur), DPR budget-capped, FRAME-RATE INDEPENDENT physics (unit substeps —
> throttled/hidden tabs don't turn into slow motion), rAF with setInterval
> fallback for hosts that never fire rAF, entrance scatter plays once per
> tab activation (20s data polls rebuild QUIETLY near-home), engine stops
> when the tab deactivates, prefers-reduced-motion = static board with
> dotted borders.
> APP.JS SURGERY (minimal): `loadFableState()` is now a one-line shim
> delegating to `PitWall.refresh()` (all old call sites intact);
> `activateTab` calls `PitWall.setActive(name === "fable")`. Old `fable-*`
> CSS in style.css is now dead weight — left in place deliberately (zero
> risk); prune whenever style.css is next touched.
> Verified in-browser against LIVE data (a GWAGWA3 run banked a 547.79s lap
> at flow 0.86 — the board showed it: gap +176.66s, foundation gated in 25
> it under the new ladder): all panels render, drawer selects fill (87
> ckpts), other tabs untouched, zero console errors; hero swarm forms and
> converges (1px drift); endpoints smoke-tested. Caveat: the embedded
> verification pane throttles hidden-tab timers, so full-speed motion +
> the track river were visually proven in the artifact mockup (identical
> engine); logic verified here.
> EARNS 0.96 (second external audit, triaged against the live KLGUARD state)**
> The audit's core thesis held up against the pit log: KLGUARD's fast stage
> banked a 596s lap (metric 167.7) and then logged **18 ROLLBACKs in one hour
> (01:02–01:59), ~3 minutes apart** — restore the banked brain, one
> eval-window of training at scale 0.96 kills it, repeat, while the calm
> ladder ground lr to the 2e-5 floor. The healthy gate wouldn't have helped
> (those evals were genuinely unhealthy: clean 10/16, terminal 0.25, progress
> 0.63, worst sectors 13/12/2 offtrack). Diagnosis: this is a **DATA problem,
> not an update-guard problem** — finish banks its lap at ~0.90 and the old
> attempt branch dropped that brain straight into a fixed 0.96 world (~6.6%
> more entry speed everywhere at once; spawns scale with envelope_scale too,
> so rehearsals were just as hot). Rollouts turn crash-dominated and every
> update points away from the lap; rollback can't fix what the batch no
> longer contains. Fixes (ladder only; rewards, spawns, PitWall untouched):
> 1. **fast now runs adaptive segments like frontier** (`run_auto`): seeded
>    from the scale the incoming brain BANKED (own resume, else finish's
>    stored `fable_envelope_scale`, else the 0.88 floor — never the 0.96
>    default), climbing +0.03 toward its 0.96 target only on mastery, easing
>    −0.04 on a lost lap or a stalled segment. Soft-advances to frontier once
>    its share is spent (unlike finish it never holds the ladder — frontier's
>    own ladder keeps adapting). Fresh segments also re-open the lr/entropy
>    the pit wall's calm ladder closed, killing the "frozen at 2e-5" end
>    state, and each segment resumes from the banked best.
> 2. **Two scale-policy flaws fixed in `_frontier_scale_next` (now shared by
>    fast + frontier, defaults = legacy behaviour):** (a) *stale mastery* —
>    the climb judged the banked best's eval, which may have been earned at
>    an easier scale; it now only counts if `proven_scale` (the scale stored
>    in the best checkpoint) >= the current scale, else the ladder climbs on
>    old evidence until the car dies; (b) *forever-hold* — once a lap is
>    banked, the eval keeps it for the rest of the run, so "hold while
>    consolidating" could pin a fragile brain at an unlearnable scale
>    indefinitely (KLGUARD's exact frozen state); a segment that banks NO new
>    best (`progressed=False`, metric snapshot before vs after) now eases
>    instead. Both flags are threaded through the fast and frontier loops.
> 3. New constant `FAST_SCALE_MIN = 0.88` (just under finish's 0.90 so a
>    struggling fast can ease slightly below its inheritance without
>    re-running finish); the ceiling is the stage default 0.96.
> Rejected from the audit: FableReward tweaks for fast (the target pace is
> the root cause, not the weights; reward edits break cross-run
> comparability and fast's reward was retuned recently); start_speed_lo/hi
> changes (spawns already scale with envelope_scale, so the ladder eases
> them for free); extra intermediate STAGES at 0.92/0.94 (the segment ladder
> subsumes them without new gates/checkpoints/dashboard rows).
> Expected KLGUARD rerun behaviour: fast resumes at its banked 0.96 scale
> (proven there), but the first stalled segment eases it to 0.92 where the
> restored brain's rollouts survive; it refines laps at survivable pace
> (metric 100000/lap is scale-independent, so faster laps still bank), then
> re-earns 0.93/0.96 on proven mastery.
> Gates: `validate_fable5_auto.py` grew section 2b (fast runs as segments;
> seeds from the finish-banked 0.90, not 0.96; ungated fast soft-advances)
> plus scale-policy unit gates (stale mastery does NOT climb; stalled
> segment eases despite the banked lap; fast floor holds) — all green in
> 19.5s, plus `validate_fable5.py`. Manifest/pit log confirmed untouched by
> the gate runs.

> **2026-07-11 — External-audit triage: pit-wall crash fix, race-rank wrap bug,
> atomic manifests, vf_clip lever**
> (Carson relayed a second-model audit of the training pipeline; each claim was
> verified against the code before acting. Four accepted, five rejected.)
> Accepted + fixed:
> 1. **PitWall healthy-gate crash (would have killed the next run's FIRST
>    eval)** — an uncommitted work-in-progress in `PitWall.note()` added the
>    HEALTHY gate (don't retire a car that drives 1.7 laps of distance but
>    misses the binary flying lap — the KLGUARD night burned 10/12 rollbacks
>    to lr 2e-5 / log-std −1.0 exactly this way) but referenced
>    `self.healthy_clean/healthy_terminal/healthy_progress` that were never
>    defined -> AttributeError from the bare `pit.note()` call in the eval
>    callback. Kept the gate, defined the thresholds in `__init__`
>    (clean ≥12/16 sectors, terminal ≤0.25, flying-line progress ≥0.90 of lap
>    distance; all ctor kwargs). While lap-armed, a HEALTHY lapless eval now
>    neither rolls back nor reseeds — refinement can accumulate; an UNHEALTHY
>    collapse still rolls back immediately.
> 2. **Race position ranked by instantaneous track fraction** (`multi_env.py`
>    + `race_env.py`): the fraction wraps 0.99 -> 0.01 at the lap line, so the
>    leader dropped to last for a step and every other car banked a phantom
>    2.0/rank overtake bonus once per lap. Multi ranks by the existing
>    wrap-corrected `self.cums`; RaceEnv now accumulates per-vehicle
>    wrap-corrected progress (`_pos_cums`, re-anchored on reset/reset_at).
>    Fable is untouched (FableEnv has no ranking).
> 3. **Manifest/eval-latest JSON writes were not atomic** — the dashboard
>    polls them, `write_text` truncates first, so a poll mid-write read torn
>    JSON. `_write_json_atomic` (tmp + `os.replace`, same pattern as
>    `PPO.save`) now backs the pipeline manifest, `LATEST_EVAL_JSON`, and
>    `_update_manifest_auto`. PIT_LOG stays a plain append (single writer,
>    O_APPEND).
> 4. **`PPOSpec.vf_clip` — PPO2 value-loss clip, opt-in, default None =
>    legacy** and deliberately UNARMED in `_configure_ppo`: the clip is in raw
>    return units, and fable returns are O(100) (lap bonuses), so the textbook
>    0.2 would freeze value learning. Arm it at ~10% of typical return scale
>    only if the train log's `vf` column starts spiking.
> Rejected, with reasons: `box.__init__(spec)` reset idiom (unusual but a
> complete reset — every field is set in `__init__`); `diagnostics.py`
> and/or precedence "bug" (Python precedence already gives the intended
> grouping); PIT_LOG file locking (single-writer append); **separate
> actor/critic trunks** (changes the `state_dict` layout — bricks every banked
> checkpoint, a hard invariant); **return/reward normalisation** (changes V's
> semantics mid-run and across resumes; advantages are already normalised per
> batch); **"hard reset to a wider log-std when calm rollbacks fail"** (that
> is verbatim the MANPLEASEWORK failure loop — re-widened noise at race pace
> makes rollouts crash-dominated and the next update re-destroys the restored
> brain).
> Gates: `validate_fable5_auto.py` grew healthy-gate coverage (healthy lapless
> -> continue; healthy streak -> no reseed; unhealthy collapse -> rollback)
> and vf_clip gates (tiny clip raises vf loss vs identically-seeded legacy;
> fable leaves it unarmed) — all green, plus `validate_fable5.py`, plus live
> rollout smokes of MultiAgentRaceEnv/RaceEnv (ranks stay a permutation over
> 120 random steps; reset_at re-anchors).

> **2026-07-10 — Fast-stage collapse: root cause + trainer stability levers**
> (the MANPLEASEWORK night: fast climbed to metric 176.6 — a banked 565s lap —
> then collapsed to 0–6 and burned 24 reseeds + 2 consolidations without
> recovering; same signature as AlleyHouse.) Root cause, from the pit log +
> code: **the collapse happens *between* evals, and every pit lever made it
> worse.** (a) `fast` evals every 45 iterations; one eval-window of unguarded
> PPO updates (4 epochs x 8 minibatches each, ratio-clip only — which bounds
> the objective, not policy drift) repeatedly took the policy from best-ever
> to dying at ~20% of the lap (04:47 metric 176.56 -> 04:48 metric −3.39, one
> iteration apart). Mechanism: exploration std ≥0.55 at 0.96x envelope makes
> rollouts crash-dominated, so the batch stops visiting the lap-line states
> and updates freely destroy behaviour there. (b) RESEED restored good weights
> (eval right after: 166.5) but *re-widened the noise and reset the LR/entropy
> anneal to max* — guaranteeing the next window re-destroyed them; it also
> never restored the obs normaliser the banked brain was evaluated under, and
> needed a 3-eval streak + 4-eval gap while a dead policy kept training.
> Fixes (obs/action layout untouched; all opt-in so non-fable pipelines are
> unchanged):
> 1. **KL trust region** — `PPOSpec.target_kl` (default None = legacy);
>    `PPO.update` early-stops its epoch loop past 1.5x target. Fable arms it
>    per stage (0.08/0.06/0.04/0.02/0.015, foundation->frontier); the train
>    log now prints `kl` (with `*` when it stopped an update).
> 2. **ROLLBACK, a new pit lever distinct from RESEED** — once the run banks a
>    LAP-SCALE best (metric ≥60 *and* a lap; lap metrics are 100000/lap_time
>    ≈130+ vs ≤35 for no-lap, so the floor separates cleanly), any eval below
>    0.25x best restores the best IMMEDIATELY (no streak, no gap) and CALMLY:
>    `reseed_from_best(calm=True)` skips the noise re-widening and anneal
>    reset, and cfg.lr + cfg.ent_coef decay x0.85 (floors 2e-5 / 1e-4) per
>    rollback. Budget 8, refills on a new best; when spent, the old
>    reseed->consolidate->plateau ladder takes over. The trainer is now never
>    more than one eval-window from its best brain instead of grinding a dead
>    policy for hours.
>    **Arm-from-resume (found live on the first confirmation run):** a
>    176.56/565s fast resume degraded within its FIRST eval window, the pit
>    banked the wreck (0.063) as the run's "best", and rollback never armed —
>    `PitWall.seed_from_resume()` now arms best_metric/best_lap from the
>    resumed checkpoint's stored eval (5% discount so matching the resumed
>    quality still refills budgets; cross-stage resumes seed the shared
>    100000/lap core), and the resumed checkpoint is the rollback/reseed
>    source of last resort until the run banks its own _best.
> 3. **Reseeds/rollbacks restore the obs normaliser** from the checkpoint
>    (weights alone don't reproduce the banked behaviour once the running
>    norm has drifted).
> 4. **Consolidate's entropy halving actually sticks now** — it halved
>    `_ent_coef`, which `set_schedule` recomputes from `cfg.ent_coef` every
>    iteration when anneal is on (finish/fast/frontier), i.e. it was a no-op
>    exactly where it mattered; now `cfg.ent_coef` halves too.
> 5. **SubprocVecEnv is crash-tolerant** — a worker env exception becomes a
>    terminal transition + env reset; a hard-dead worker sends its traceback
>    (`__error__` sentinel) and is respawned in place (difficulty/track
>    replayed, capped 5/worker) with its shard surfacing as terminal
>    transitions — a crash costs one episode boundary, not the overnight run
>    (the old behaviour was ConnectionResetError -> whole trainer dead).
> 6. v1 viewer `to_screen()` NaN/inf guard (parks off-screen instead of
>    crashing pygame; huge-finite clamped to int16 range).
> Gates: `validate_fable5_auto.py` grew **section 9** (rollback: immediate,
> calm, lr-decay, budget+refill, lap-armed only, falls back to reseed;
> KL: blow-past stops, None = legacy full pass, stages armed tightest at
> frontier) — all 59 gates green, plus `validate_fable5.py`. Judged by code +
> hermetic gates; the real proof is the next overnight: expect `[pit]`
> ROLLBACK lines instead of hour-long "levers spent" stretches, and evals
> that stay lap-scale. Shift-point verdict (0.63 box) still open — needs
> `--fable-diag` on a fast/finish brain retrained under these levers.

> **2026-07-03 — Audit fixes: reporting/liveness/storage footguns**
> (implemented the valid findings from a full-system audit Carson relayed.)
> Six code fixes, each verified:
> 1. **Eval-latest clobber** — `FableEvaluator._write_manifest` wrote the global
>    `fable5_ring_eval_latest.json` via a cwd-relative path, so the smoke
>    validator (`validate_fable5.py`, which redirects `manifest_path` to /tmp)
>    silently overwrote the REAL repo-root eval the dashboard reads. Now written
>    next to its manifest (`manifest_path.parent / LATEST_EVAL_JSON`). Proven:
>    repo-root file byte-identical before/after a smoke run; the smoke's copy
>    lands in /tmp.
> 2. **Stage-blind "best"** — manifest now carries `active_best_checkpoint`
>    (the CURRENT run's `<ckpt>_best.pt`), so a collapsed/interrupted run whose
>    global champion `fable5_ring_best.pt` is a stale foundation checkpoint no
>    longer hides the real fast-stage best. Champion-promotion semantics
>    untouched (a collapse still can't demote the banked champion).
> 3. **Dashboard watch/diagnose default** — `fable_watch`/`fable_diagnose` with
>    no checkpoint now resolve `active_best_checkpoint → best_checkpoint →
>    FABLE_BEST` from the manifest (server `fable_watch_default()`), instead of
>    hardcoding the stale champion. Unit-tested across all four resolution cases.
>    CLIENT half (found when Carson asked if the buttons were still live): the
>    "Watch Best"/"Diagnose Best" buttons in app.js hardcoded `checkpoint:
>    FABLE_BEST`, bypassing the resolver — now send no checkpoint so the server
>    picks. Stage Board also gained an "active best" row (was surfacing only the
>    stale champion). Verified in-browser: row renders, both buttons send no
>    checkpoint.
> 4. **Diagnostics disk bloat** — `run_diagnostics` wrote a 200–270 MB
>    `raw_trace.jsonl` per run that nothing reads programmatically (the `.npz`
>    is the machine form). New `raw_trace` arg defaults to **gzip**
>    (`raw_trace.jsonl.gz`, ~15× smaller); `plain`/`none` escape hatches on the
>    CLI. Existing 25 uncompressed traces (~3.6 GB) are untouched — offered to
>    compress-in-place separately.
> 5. **`[best-ring]` mislabel** — PPO's keep-best line is now tagged from the
>    eval result (`result.get("best_tag")`); Fable emits `[best-fable/<stage>]`
>    so overnight logs read as the right pipeline. Ring path unchanged.
> 6. **Silent optimizer-load failure** + **2D-viewer validator import path**:
>    `PPO.load_state` now prints when optimizer state can't be restored (was a
>    bare `except: pass` hiding real mismatches); `validate_ring_2d_viewer.py`
>    inserts the repo root into `sys.path` so it runs without `PYTHONPATH=$PWD`.
> All green: `validate_fable5_auto.py` (hermetic, §1–8), `validate_fable5.py`.
> FLAGGED, not silently changed (need Carson's call): finish→fast gate strictness
> (contradicts the deliberate "finish = can it lap; fast = make it fast" design),
> killing the duplicate ntfy watchers (a screen session + a login-shell one, both
> heartbeating the same topic/pid-file — a system action), and dashboard
> discovery of external/`screen` trainers (a larger feature; `PROCS` is in-memory
> and only tracks dashboard-launched jobs). No trainer was running at audit time.
>
> **2026-07-03 — Fable Five RaceBox shift point out of the lug zone**
> (the Brain Lab's smoking gun made physical: the 787B was upshifting at only
> 50–60% of redline). `RaceBox` grabbed the tallest gear that kept rpm above
> `rpm_lo_frac`, and that fraction was the stock ECONOMY 0.42 — so the box
> upshifted the instant the next gear cleared 42%, landing at ~50% in the gear
> it left, and the R26B never saw the top half of its powerband (peak power
> ~94% redline). On top of that the policy had LEARNED to short-shift (+2 gear
> offset) to dodge the wheelspin penalty. Fix (curriculum/gearbox only, zero
> physics/obs/action changes): new module constant `RACE_SHIFT_LO_FRAC = 0.63`
> is the single source of truth — it is the default for both `RaceBox` and a
> new per-stage `FableSpec.shift_lo_frac` field, and it rides in checkpoint
> metadata (`fable_shift_lo_frac`) so the viewer/diag drive the box the brain
> trained against (old checkpoints fall back to it). Upshifts now land 75–90%
> of redline. HARD CEILING: the 1→2 ratio step is 1.425 and rpm_hi=0.97·redline,
> so the fraction must stay ≤ 0.681 or a dead-band opens (1st overrevs before
> 2nd clears rpm_lo); 0.63 keeps margin. New `_shift_table()` helper sweeps the
> box for the upshift table + dead-band check; validate_fable5_auto.py grew
> section 8 (powerband/no-dead-band/ceiling-is-real/per-stage-tunable, all
> green), and validate_fable5.py's downshift test still passes (now walks 6→1
> under braking, no money-shift). rpm_hi ceiling and the money-shift guard
> (cutoff·0.96) are untouched. **JUDGE BY RETRAIN, NOT THE OLD BRAIN**: the
> current champion trained against the 0.42 box will drive worse under 0.63
> until retrained — that's expected, not a regression. Success = retrain finish
> from its banked best, then `--fable-diag` shows the gear histogram shifted to
> higher rpm and mean on-track pace risen above the ~0.65 baseline WITHOUT a
> terminal-rate regression (if terminal spikes, ease `FableReward.spin`, don't
> revert the box). Rollback = set the constant back to 0.42.

> **2026-07-03 — Fable Five FINISH rework + Brain Lab diagnostics**
> (from Carson's critique after an overnight run stalled in `finish`:
> `flyfish_1_finish_best.pt` hit 96.4% of a lap, 14/16 clean, 11 clean chain,
> 12.5% terminal — close, but never banked a clean lap). Curriculum +
> checkpoint-selection fixes, none of which touch the physics/obs:
> (1) **FINISH is now a segment stage** like frontier: it starts at the scale
> FLOW actually banked (min of that and 0.90), and a clean LAP is the gate —
> `fast`/`frontier` train speed on top of a finisher, so while the lap is
> missing finish KEEPS the budget (eating the fast/frontier share) and eases
> the target toward a 0.70 floor. The only way past without a lap is a
> QUALITY BAR (≥98% progress, ≥15/16 clean, terminal ≤0.125). "Finish the
> lap first, then go fast."
> (2) **Lap-closure drills**: finish spawns 30% of episodes in the last
> 70–98% of the lap at race pace (`closure_start_prob`/`closure_window`) so
> the car rehearses carrying a lap HOME, not just starting slices.
> (3) **Closure milestones**: one-time reward bumps at 50/75/90% of a lap
> survived (`FableReward.milestone_bonus`) — the gradient a near-lap brain
> needs to close the final sectors.
> (4) **Hall of fame**: the evaluator banks `<ckpt>_hof_{progress,clean,lap}.pt`
> beside the metric best; the pit wall reseeds from the RIGHT one by failure
> mode (dying a lot → cleanest brain; progress collapsed → farthest brain).
> (5) **Weak-sector memory**: eval failures persist as a decayed per-stage
> heatmap in the manifest (`sector_heat`); the next run seeds its
> exploring-start weights to rehearse just BEFORE those zones
> (`sector_seed_bias`).
> (6) **Own-stage-best-first resume**: `_chain_resume` now tries the stage's
> own `_best` before dropping to the previous stage (keep-best floor protects
> it), so a rerun never throws away same-stage learning.
> (7) light edge + slip_guard shaping in finish (line discipline vs traction).
> **Brain Lab** (`supra/fable5_diag.py`, `run.py --fable-diag`, dashboard
> Diagnostics tab): runs a checkpoint through an envelope-aware battery — 2
> flying line laps with full per-10 m traces, 16 sector starts, 5 lap-closure
> drop-ins — and writes a dense JSON report to `diagnostics/fable5/`
> (`/api/fable-diag[/<name>]`). Rule-generated verdict + findings, a
> death-forensics map (where/why/entry-pace/slip, tagged
> traction-break/overspeed/ran-wide/line-error), a speed-vs-envelope trace,
> sector table, and behavior stats (gear/pace histograms, steering roughness,
> gear-offset usage). First run on the finish brain independently reproduced
> the 96.4%/14-clean picture AND surfaced the smoking gun: the gear head is
> pinned at +2 (short-shifting into 6th ~65% of the lap while holding only
> ~65% envelope pace = lugging off the powerband). Gates:
> validate_fable5_auto.py grew section 7 (finish bar / HOF keys / sector-heat
> bias / failure-aware reseed).

> **2026-07-03 — Fable Five SELF-CORRECTING ladder** (full pipeline audit).
> Audit found three overnight-staleness faults: the auto ladder HALTED on an
> unmet gate (idle rest of night); envelope scale was FROZEN per stage (and
> the theoretical centerline lap 400.4s is SLOWER than Bellof's 371.13s —
> superhuman requires pushing the reward target past 1.0, where racing-line
> gains live); the pit wall had one lever and a non-refilling budget. Fixes:
> (1) gated stages now RETRY once at an eased envelope scale (−0.05) then
> SOFT-ADVANCE — the ladder never idles; (2) FRONTIER runs in adaptive-scale
> SEGMENTS (~150+ iters): mastery (clean lap, pace ≥0.88, terminal ≤0.25)
> pushes scale +0.03 up to 1.15×, a lost lap eases −0.04 down to 0.90×, each
> segment resumes from the banked best (= built-in rollback), scale persists
> in checkpoints (fable_envelope_scale) and resumes across reruns;
> (3) PitWall v2: reseed budget REFILLS on every new best, and a new
> CONSOLIDATE lever halves lr+entropy once reseeds stop working (once per
> dry spell); (4) the eval line lap is now FLYING (drops in at 0.9× pace —
> Bellof's benchmark was a flying lap; `lap_style: "flying"` in evals);
> (5) the post-train TAIL is always evaluated so final weights can't be
> silently lost; (6) overspeed penalty margin tracks scale (target past the
> envelope no longer fights its own penalty); frontier patience 1400→700,
> restarts 8→3 (segments handle the long horizon). Dashboard: mini-metrics
> show frontier scale / ladder state / budget / lap style / pit calls
> (reseeds+consolidations); pit log renders CONSOLIDATE (blue). Gates:
> validate_fable5.py all green; validate_fable5_auto.py REWRITTEN (31 gates,
> now hermetic in a scratch dir — the old one could clobber the real
> manifest; the real manifest was rebuilt from the champion's stored eval).

> **2026-07-03 — viewer2 BROADCAST pass** (training logic untouched).
> CAMERA DIRECTOR (`Director` in viewer2, V toggles; auto-ON in watch mode):
> cuts between chase / low rear-quarter / locked side pan / wide drone /
> fixed trackside shots on 6–11 s timers (trackside ends when the car
> passes); View grew director hooks (cam/rot/anchor overrides, zoom_mult,
> lead_mult, `cut` = hard broadcast cut, also fired on AI auto-reset so the
> camera never swoops home). Camera drama: high-rpm buzz, kerb-strike
> nibble, off-track jolt. HUD is now a BROADCAST OVERLAY: top-centre chip
> (FABLE FIVE · course · pulsing LIVE · CAM name) over a lap-progress bar
> with sector ticks + CLEAN/INVALID pill; horizontal RPM bar w/ red zone +
> limiter strobe; lap board gained a live Δ BEST row (per-progress
> interpolation against the best clean lap's trace); broadcast MOMENTS:
> sector split flashes, gold NEW BEST LAP banner, checkered finish strip.
> FX moved to supra/fx.py (V2 section appended — v1's FX/SkidTrail classes
> untouched) + new SpeedTension (speed vignette + subtle edge streaks,
> re-added at Carson's request in this broadcast spec). SOUND: downshift
> rev-blip pitch flare, rev-limiter ignition-cut chop + snaps, rear
> wheelspin "chew" AM, and kerb-strike thwack rattle — kerb needed new
> telemetry, so the shared audio array stride went 10→11 floats/car
> (viewer sets veh._snd_kerb; v1 never sets it → reads 0, stays silent).
> carart.py deliberately untouched (mesh contract is load-bearing; the 787B
> identity pass already reads well on camera). Gates: validate_sound.py all
> green (0.83 ms/block), mixer subprocess smoke OK, headless V2_SHOT runs.

> **2026-07-02 — viewer2 CINEMATIC pass; speed streaks REMOVED** (both by
> request). New `Cinematic` class (prerendered, per-frame cost = a few
> blits): drifting soft cloud shadows anchored in world space, a per-mood
> colour grade (additive sun-corner glow + corner vignette, built lazily at
> quarter-res per mood), and 3-plate cycling film grain. draw_world now lays
> hashed tonal grass patches (±6%, ~24 m cells) under the speckle so the
> ground reads as meadow, not flat fill. The Streaks class and its wiring
> are gone.

> **2026-07-02 — viewer2 "cooler" part 2 wired in; heat ribbon REMOVED** (by
> request). New run-loop FX: landing dust bursts + camera impulse kick scaled
> by landing g, kerb-strike sparks (Puffs.spark fired when riding a kerb near
> the road edge at speed), downshift backfire flames out of the tail pipes
> (draw_exhaust_flame, matches the audio's downshift bang), screen-space speed
> HUD: gear read-out POPS on every shift, over-redline arc segments
> STROBE with a SHIFT cue, and a PACE ±km/h vs the Fable envelope limit
> read-out (uses trk.fable_vref at the nearest sample — the ribbon's data,
> now text-only). The envelope heat ribbon on the road surface was removed
> from RoadGeom/draw_world; the plain dashed centreline is back.

> **2026-07-02 — engine sound recreated** (supra/sound.py, DSP only — the
> battle-hardened mixer subprocess, shared-array layout, BLOCK=2048, DC
> blocker, and NO-doppler rule are all untouched). 787B R26B upgrades:
> load model with a real OVERRUN voice (open pipes stay ~84% as loud off
> throttle, spectrum tilts to exhaust thump, intake formant dies, gargle
> chop), inter-rotor imbalance beating so the note breathes, a second
> body/collector resonance (~640 Hz, snapped to a firing harmonic),
> coast gear whine, and a low driveline THUMP under the dogbox shift
> chirp. Piston cars: asymmetric exhaust clipping + V8 idle lope
> (lr4/f150). NEW environment layers from existing telemetry (no shared
> layout change): speed²-scaled gusting wind that brightens with speed,
> and road rumble that roughens with tyre slide. Validated by
> tools/validate_sound.py (finite/bounded/DC-free per block, 787B
> dominant tone tracks rpm·4/60 firing harmonics, overrun loudness, env
> noise vs speed, 0.8 ms/block ≪ 46 ms budget) — all green; renders
> listenable WAV sweeps.

> **2026-07-02 — the training dock was redesigned** ("Pit Wall / Training
> Telemetry", Fable styling): six mono telemetry tiles (incl. NEW "Best this
> run" and "Pit Call" — the supervisor's latest decision parsed from [pit]
> log lines), gradient-hairline chart panels with amber-mono Chart.js theme,
> the rotating holo 787B in a bracketed HOLO BAY, and a colorized Session
> Feed (PIT amber · EVAL teal · BEST green · GATE gold · ALERT red; last 400
> lines rendered). All dock element IDs kept — only markup/CSS/render fns
> changed (renderConsole now innerHTML-colorizes; charts get td-best wiring).

> **2026-07-02 — PIT WALL supervisor added** (after an autonomous-agent
> critique). `fable5.PitWall` sits above PPO: judges every eval against the
> RUN'S TREND (rolling windows), reseeds from best after a regression streak
> (3 far-below-best/terminating evals, rate-limited, budget 6) long before
> PPO's patience fires, and logs every decision with a reason to
> `fable5_pit_log.txt` + the manifest (`pit` section; dashboard renders it).
> Evals now name the 3 WORST SECTORS (`worst=` in the eval line, dicts in the
> manifest). Checkpoints/manifest carry `envelope_scale` + `resumed_from`
> provenance. Auto mode now HONORS `--resume` (seeds the first trained stage,
> loudly) instead of silently ignoring it. Diagnostics never crash on an
> output-path collision. Deferred from the critique: file RENAMES (provenance
> metadata disambiguates without breaking existing files) and BRANCH-style
> parallel reseeding (worthy, its own project). Gates:
> tools/validate_fable5_auto.py (§5 = pit wall).

> **2026-07-02 — the AUTO ladder is overnight-ready.**
> `--fable N --fable-stage auto [--out PREFIX]` runs foundation→…→frontier
> unattended: stage gates are judged on the BANKED BEST (not the final eval),
> a stage stops the moment its gate is met (`ppo.request_stop`) and the
> leftover budget rolls into frontier (weights .15/.15/.20/.20/.30+rest);
> reruns SKIP already-gated stages (crash-resumable); `--out` is a run PREFIX
> (`night1` → `night1_<stage>.pt`), so multiple ladder runs coexist; promotion
> to `fable5_ring_best.pt` is metric-compared (clean lap > no lap, faster lap
> wins) so a weak run never clobbers the champion. Ladder progress is stamped
> into the manifest under `auto`. Gates: `tools/validate_fable5_auto.py`.

> **2026-07-02 — Fable Five v2: the AGENT shifts gears** (reward version
> `fable5-v2-gears`). Root cause: AutoBox only downshifts below 1500 rpm —
> under the 787B's 2000 idle — so it NEVER downshifted at speed; every slow
> corner exited in 6th off the powerband. Fix: `fable5.RaceBox` (speed-matched
> recommendation that walks down through braking zones, over-rev guard at 96%
> cutoff, 0.28 s cooldown) + a THIRD action = gear offset (-2..+2) from that
> recommendation. Offset-not-pulse is the learnable form; zero-action == sane
> box, so transplants stay sane (`PPOSpec.action_bias=(0,.6,0)` — a[2] is NOT
> a handbrake). Transplant now maps 2->3 actions AND 60->68 obs; watch uses a
> gear-aware agent + `viewer2.run(shift_controller=...)`; legacy 2-action
> checkpoints keep AutoBox when watched. First eval after transplant: pace
> 0.89 -> 0.95 from the box alone. Old `ring_race` pipeline untouched.

> **2026-07-01 — the 787B car model was rebuilt from reference** (`carart.py`
> `get_car_mesh` "mazda787b" branch): cab-forward bubble canopy, long kamm
> tail, Renown QUADRANT livery (orange/green swap at the cockpit, white spine
> stripe), wide low wing on twin pylons, deck intake scoops, and — Group C —
> wheels tucked INSIDE the enclosed arches (787B-specific branch in
> `add_wheels_to_mesh`). Viewer V2 adds an identity pass (covered headlights,
> #55 roundel with drawn digits, tail-light bar); V1's name-keyed overlays
> still fire (face names preserved). The perimeter-index contract
> (0,8,18,19,17,7 / 0,7,29,31,30,20) is load-bearing for shadows + LOD — keep
> it if you touch the mesh. Dashboard car thumbs cache was cleared to regen.

> **2026-07-01 — 2D Viewer V2 is the default drive/watch viewer.**
> `supra/viewer2.py` — a from-scratch projected 2.5D renderer: ONE projector
> (rotate → zoom → z-lift) for road, kerbs, props, skids, and the v1 car
> meshes; the road carries real DEM elevation and banking; heading-up chase
> cam (C toggles north-up); road-glued shadows that detach on jumps; G cycles
> day/dusk/night with headlight cones; Fable-styled HUD + clean-lap board.
> V1 (`supra/app.py` + background.py) is UNTOUCHED — `--classic` or
> `SUPRA_CLASSIC_VIEWER=1` selects it; racers/opponents modes auto-fallback
> to it. Entry points routed in run.py, fable5.py, ring_race.py.

> **2026-07-01 — FABLE FIVE is the active Nordschleife program.** The 787B
> superhuman-lap effort moved to a new pipeline: `supra/fable5.py` +
> `FABLE5_PLAN.md` (design/rationale), CLI `--fable/--fable-stage/--watch-fable`,
> dashboard tab "Fable Five", gates `tools/validate_fable5.py`. It computes a
> physics-true speed envelope from the car model (obs pace block → layout
> `fable-v1`, 68-dim; envelope-relative reward; benchmarked eval), runs five
> gated stages (foundation→flow→finish→fast→frontier), and warm-starts by
> TRANSPLANTING `ring_787b_best.pt`. First result: the transplanted policy
> immediately banked the project's **first full clean lap — 11:56.97**
> (theoretical envelope lap 6:40, superhuman bar = Bellof 6:11.13). The old
> `ring_race` pipeline (§ below) is retired but intact.

> A from-scratch 2D neuro-evolution + reinforcement-learning driving simulator in
> Python. Trains AI to **grip-race** and **drift** a Toyota Supra (with RX-7 and
> R34 Skyline presets) around procedurally-generated tracks. Top-down 2D, real
> vehicle physics, a GA path and a PPO path, a drift-scoring system, an in-app
> watch viewer, and a self-contained web dashboard ("Command Center").

**Project root:** `~/Desktop/Supra Ai 2/`
**Last updated:** 2026-06-06

---

## 0.0 ⚠️ CORRECTION (2026-06-06) — the eval window was lying to us

The deterministic eval (§6.7) ran a **fixed 14 s window** (`steps_per=420` at
30 Hz). On any track over ~1 km that caps the *lap-completion* reading at
**~0.2–0.4 for ANY policy, however good** — there simply isn't time to get
around the loop. So the headline "drift slides but only completes ~0.2–0.4 of
the lap → it craps out" was **largely a measurement artifact, not a policy
failure.** Proof, re-scoring the SAME checkpoints under a full-episode eval:

| checkpoint | OLD 14 s lap | NEW full-episode lap |
|---|---|---|
| `ppo_race.pt` (race) | 0.40 | **1.00** (completes a full lap) |
| `akina_drift_specialist` | 0.24 | **0.55** (drifts 0.20 + over halfway) |
| `ppo_drift_refined` (generalist) | (under-read) | **drift 0.65 / lap 0.79** |
| `pass` specialist (hardest touge) | 0.12 | 0.12 (a *genuine* spin-out) |

**Fix shipped:** `_deterministic_eval` now runs a **full-episode budget**
(`episode_seconds`), so lap-completion is actually reachable. Re-judge all drift
work against the corrected `[eval]` line. The real remaining failure is narrow:
the **hardest touges** (`pass`, top-difficulty `akina`) genuinely spin out — the
medium tracks were fine all along.

**Also shipped this pass (see §0.1):** anti-spin reward shaping, a
completion-gated drift curriculum, a fixed default `ppo_drift.pt` slot, a live-
view specialist-track fix, working `requirements.txt`, and doc updates.

## 0. TL;DR — current status (read this first)

- **Race PPO works.** `ppo_race.pt` completes laps competently (full-episode eval lap **1.00**).
- **Drift mostly works on easy/medium tracks; the hardest touges still spin out.**
  Honest (full-episode) numbers: generalist `ppo_drift_refined` ≈ **drift 0.65 /
  lap 0.79**; `akina` specialist ≈ drift 0.20 / lap 0.55. Only `pass` (and top-diff
  touges) genuinely fail to complete. The old "craps out everywhere at 0.2–0.4
  lap" picture was the 14 s eval window (see §0.0).
- **The training→watching pipeline is provably faithful** (identical actions);
  the honest metric is the `[eval]` line, NOT the rolling `drift`/`laps`.
- **Drift reward now has anti-spin shaping** (yaw-rate gate + steady-angle hold
  bonus) and the curriculum is **completion-gated** (won't advance while spinning).
  Plus the earlier **progress reward**. All structurally sound; the anti-spin
  defaults are conservative and want a tuning pass + fresh full run to dial in.
- **Immediate next step:** ~~fresh full drift specialist on a hard touge~~ —
  SUPERSEDED by hills (§0.05): tracks gained real elevation + jumps and the obs
  grew 42→60, so **every existing checkpoint is a museum piece**. The next
  training act is the hills retraining program (PHYSICS_3D_PLAN Stage 7):
  race generalist first, then drift, then the touge specialists.

## 0.05 Hills + jumps are LIVE (2026-06-10 → 06-12) — the world is 3D now

The whole program runs on real elevation. Authoritative plan + per-stage status:
**`PHYSICS_3D_PLAN.md`** (rev 2 — read it before touching track/physics/
sensors/training). The short version:

- **Tracks** carry `z / grade / vcurv / bank` + per-point `launch_speed`
  (style-shaped, seed-deterministic; `Track.set_elevation()` is the single
  entry point). New named track **`ridge`** = the canonical jump touge (two
  authored crests, 28.7 / 31.9 m/s launch).
- **Physics is 2.5D + a vertical DOF.** Every driving loop feeds
  `veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])`
  right after the `surface_grip` line (the same caller contract). Slope
  gravity is a real force in `ax/ay`; loads scale by slope + v²·crest;
  **takeoff is emergent** when the crest term hits zero — ballistic flight,
  suspension-impact landing, grip cost via load-sensitive μ. No jump zones.
- **Obs went 40→58 (+2 mode = 60), append-only**: old 40 dims bit-identical,
  then grade/bank/pitch/vz/height/airborne + look-ahead grade & vcurv ×6.
  **ALL pre-hills checkpoints refuse to load, loudly** (PPO `load_state`/
  `load_policy` + GA `load_champion` carry the "pre-hills checkpoint —
  retrain" message; Command Center badges them ⚠ pre-hills).
- **Levers:** `--flat` = byte-identical pre-hills physics everywhere (proven,
  not vibes: `tools/regression_baseline.py` compares against pre-edit
  checksums and asserts the airborne canary — run it after ANY physics edit).
  `--hill-scale 0..1.5` dials amplitude. Command Center exposes both per run.
- **Gates if you touch this:** `tools/validate_track_elevation.py` (caps,
  launch floors, determinism), `validate_hills.py` (slope forces, analytic
  loads, energy audit), `validate_jumps.py` (takeoff threshold, ballistics,
  landing). All must stay green; baseline must stay byte-identical.
- **Observatory** renders server-authoritative Fable playback pose and brain
  telemetry; the retired browser driving viewer is no longer shipped. The
  unchanged 2D dashboard (Tab) retains the ROAD line + look-ahead grade strip.
- **Remaining:** Stage 7 = the retraining program (race generalist → drift →
  specialists → GA refresh, `*_hills` naming). Expect early curves worse than
  flat-era at the same iteration; judge by `[eval]` only. The curriculum
  hill_scale ramp (race `0.4+0.6·d`, drift `0.25+0.5·d`) lands with Stage 7.

## 0.1 Changelog — 2026-06-06 maintenance pass

All verified headless; existing checkpoints' *watch* behaviour is unchanged (the
drift reward is only used during training, never when watching).

- **`ppo.py` `_deterministic_eval`:** fixed 14 s window → **full-episode budget**
  (`episode_seconds`). The honest metric was structurally capped; see §0.0. Bad
  policies still terminate early so eval stays cheap.
- **`drift.py` + `config.py` `DriftReward`:** added **anti-spin shaping** — a
  yaw-RATE gate (`spin_rate_deg`/`spin_rate_span`) that scales the drift reward
  down when the car is rotating out of control (the old angle-only falloff
  reacted too late), and a **controlled-hold bonus** (`hold_bonus`/`hold_rate_deg`)
  for keeping a steady in-band angle through a corner. Defaults are conservative
  (only bite on genuine spins) → tune + revalidate against `[eval]`.
- **`ppo.py` `maybe_promote` + `PPOSpec.drift_promote_lap`:** the drift curriculum
  now requires rolling lap-completion AND drift fraction before advancing, so it
  can't climb to a harder track while the policy just slides and spins out.
- **Default `ppo_drift.pt` slot:** was a mislabeled touge specialist (lap 0.03 as
  a generalist). Backed up to `ppo_drift_touge_backup.pt`, replaced with the real
  generalist champion `ppo_drift_refined.pt` (drift 0.65 / lap 0.79).
- **`app_ppo.py`:** the live training view now builds its display car on the
  **specialist's fixed track** (was always a generalist curriculum track) and
  starts at the line; removed dead `_car`/`mode_vec`.
- **`requirements.txt`:** now actually lists `torch` and `flask` (were missing /
  commented out, so a fresh install left PPO + the dashboard broken).
- **Docs:** `README.md` rewritten to match reality (Command Center port 8770,
  correct install, current status); stale docstrings in `run.py`/`__init__.py`
  fixed; removed dead `_draw_car` in `app.py`.

### 🔴 Named tracks were a DIFFERENT layout every process (specialist watch was broken)
- **Root cause:** `track.named_track()` seeded the generator from `abs(hash(name)) %
  9999`. Python salts `hash(str)` per process (PYTHONHASHSEED), so `named_track('akina')`
  produced a **different track every launch**. A specialist trained on one 'akina'
  was then *evaluated and watched on a different 'akina' it had never seen* → it
  spun out and completed almost none of the lap, and looked like it "couldn't
  drift." Proven: 3 processes → 3 different akina geometries (minR 14.3 / 15.6 /
  12.7). It also made saved checkpoint metrics non-reproducible (a `_best` stored
  lap 0.73 re-scored to 0.36 — two different akinas).
- **Fix:** `named_track` now seeds from `zlib.crc32(name)` (`_stable_seed`),
  process-independent. Verified: train-process and a separate watch/eval-process
  now produce byte-identical tracks and identical eval scores. Cleared the
  `command-center/thumbs/` cache so thumbnails regenerate from the stable tracks.
- **Consequence for existing specialists:** any pre-fix named-track specialist
  (`akina_drift_new1`, the `tech_*`/`pass_*`/`touge_*` runs) was trained on a
  now-unrecoverable random layout, so it still won't watch well on the stable
  track. **Retrain named-track specialists** — they'll now train/eval/watch on the
  SAME track. (Generalists are unaffected: they use seeded curriculum tracks.)

### Command Center dashboard (so you can actually *watch* training in the UI)
- **The honest `[eval]` metric is now charted.** `server.py` parses the
  `[eval] eval drift D x lap L = S` / `eval laps L` lines and stamps them onto the
  current iteration; `static/app.js` draws **`eval lap ★` / `eval drift ★` /
  `eval score ★`** bright + thick (rolling drift/laps muted), so the chart leads
  with the number to trust. Before, only the misleading rolling numbers showed.
- **`--live` runs now stream metrics + keep-best too.** The live-window trainer
  (`app_ppo._Trainer`) prints the same `it … ` log line and runs the deterministic
  eval + keep-best every 20 iters (shared `PPO.eval_and_save`). So watching a run
  in the pygame window no longer means flying blind in the browser OR losing the
  `_best.pt` snapshot. (No auto early-stop on live runs — you control the stop.)
- **Headless "New run (charts)" auto-opens the live chart view**; GUI runs
  (drive/live/watch) show the log (they have their own window).
- **Continue ↻ now surfaces the honest metric + the ★ peak.** Each "Continue from"
  option shows `Nu · drift D · lap L · 🎯track`, ★-flags `_best` files, and
  defaults to the slot's `_best` when one exists (continue from the peak, not the
  latest). The Checkpoints tab shows drift/lap too. Drive/Watch tabs list the
  in-window keys (F effects · 1-0 toggles · G grade · V all · B beams · M mute).
  Static-only edits → just refresh the browser. NOTE: checkpoints saved before the
  metric was stored show `0.0` (legacy metadata, not a real score); any new run
  stores the honest number.
- ⚠️ **`server.py` changes need a one-time Command Center restart** (kill the
  server / re-run `python3 command-center/server.py`, or double-click the
  launcher). Training-code changes (`supra/*.py`, `run.py`) are picked up on the
  next launch with no restart. This pass restarted it for you (now on pid-of-day).

### Keep-best is monotonic across continuations now
- **`<run>_best.pt` never regresses within a run** (only written on a new eval high);
  **`<run>.pt` (latest) CAN over-train and collapse** — so always watch/continue
  from `_best`, and specialist auto early-stop banks the peak before collapse.
- **Fix:** continuing under a NEW `--out` name used to reset the keep-best floor to
  -inf, so the first eval could bank a WORSE "best." `load_state` now stashes the
  resumed checkpoint's metric and `init_best_metric` uses it as a floor — **but
  only when the track matches** (a generalist's score isn't comparable once you
  warm-start it onto a harder specialist track). So "continue from best, repeat"
  is now truly monotonic, while cross-track warm-starts still bank their own bests.
- **Best single lever for a hard specialist:** `--resume ppo_drift.pt` (the
  generalist already drifts-AND-completes, 0.65/0.79) instead of learning the
  brutal track from scratch.

### Car art — made it actually look like an A90 Supra (`carart.py`)
Redesigned the top-down silhouette from a generic red egg into the A90's signature
**hourglass**: pointed nose → wide front fender arches → pinched waist → big rear
haunches (widest point) → squared **ducktail**. Added a long power-dome hood with
creases, front-fender gills, a **double-bubble dark canopy** with a centre spine,
and swept headlights; replaced the floating GT wing with an integrated ducktail
(stock A90). Reads at chase-cam scale; LOD still drops to the silhouette when
zoomed out (GA swarm / live view). All three presets keep their colours (Supra
red / RX-7 white / Skyline blue). Auto-applies on the next viewer launch.

### New distinctive tracks (the ring generator only made blobs)
`make_track`'s ring generator places control points radially-by-angle, so every
procedural named track is a convex blob — they all look the same. Added 4
**hand-crafted, deterministic** tracks with real character (explicit waypoints via
`_handcrafted`, registered in `SPECIAL`): **`circuit`** (road course, genuine long
straights + wide hairpin), **`crescent`** (sweeping C), **`esses`** (snaking
gourd), **`hook`** (boot + inner notch). All drivable (minR 12–26 m), non-self-
intersecting, and wired everywhere (drive/train/watch/specialist/graduate +
thumbnails) via `server.py` `TRACK_NAMES`. Existing tracks' geometry is UNCHANGED
(specialists stay valid). Prototyped by rendering to PNG and checking minR.

### 🏁💨 Race+Drift HYBRID (slice 6 — the fusion goal)
A *fused* policy: race the straights for speed, drift the corners for style — one
behaviour, not a mode switch.
- **Mode `hybrid`** (`ppo_env.py`): action space = drift's (steer/long/handbrake),
  obs mode one-hot = **[1,1]** (both goals), so a hybrid can WARM-START from a drift
  policy (same arch). Reward = **race backbone** (progress/align/speed/lap_bonus +
  race termination, so it completes FAST) **plus** `_hybrid_style`: a controlled-
  drift bonus (soft-entry, anti-spin, over-rotation falloff) **gated to corners**
  (`corner_curv`) so it grips straights and drifts turns. `HybridReward.style` is
  the master dial (bigger = driftier/slower).
- **Eval/keep-best** = `lap × (0.6 + 0.4·style)` — completion-dominant with a style
  multiplier; prints `[eval] eval lap L x style S = M`.
- **CLI:** `--hybrid ITERS` / `--watch-hybrid`, `--style W` to tune driftiness.
  Plus all the shared flags (`--workers --anneal --target --lr --pop --out --resume`).
- **Dashboard:** a "🏁💨 PPO Hybrid" card (generalist/specialist, graduate-to-target,
  anneal, workers, **Drift style slider**, continue ↻). Watch routes by kind.
- **Recipe:** train a generalist hybrid (curriculum), optionally graduate onto a
  target; OR warm-start from a drift policy (`--resume <drift>` — auto-uses the
  reduced fine-tune LR). Tune `--style` until the `[eval]` shows high `lap` AND a
  healthy `style`. (cross-mode warm-start doesn't carry an incomparable keep-best
  floor — load_state requires same mode+track to carry it.)

### Drift watch starts ROLLING (drifting needs entry speed)
The watch viewer used to start the car at a **standstill**. A drift policy can't
initiate a slide from 0 km/h, so on a hard track it washes off the first corner
and looks far worse than it is (e.g. `overnight2_best` on akina: **lap 0.17 from a
standstill vs 1.05 from a rolling start — a full lap**). `_watch_drift` now starts
at a curvature-aware rolling pace (same as the eval); `run(start_speed=...)` carries
it (init + every respawn). Also note the eval's lap is a **4-start AVERAGE** around
the loop — a single bad corner (e.g. akina ~25%: lap 0.03 while the others are
0.65–1.05) drags the average down, so judge per-section too, not just the mean.

### Toward a superhuman drifter (4 training upgrades)
All opt-in / off-by-default where they'd change behaviour, and all wired into the
Command Center (PPO Race + Drift cards). The intended recipe: train a **drift
generalist** (wide→tight curriculum) → **graduate**/warm-start onto the hard
target → chain continues from `★best`, with `--anneal` on for the long runs.
- **Slide-recovery reward (`drift.py`/`DriftReward`):** rewards bringing the yaw
  rate back DOWN while over-rotating (the countersteer "save"), plus a bit more
  off-track/spin grace (`offtrack_grace 0.9`, `spin_grace 1.2`) so a wide slide can
  be RECOVERED instead of insta-terminating. Targets the core "drifts then spins
  out". `recover_bonus`/`recover_yaw_deg` tune it.
- **Drift curriculum wide→tight (`track.drift_curriculum_track`):** drift now
  trains on wider/gentler tracks early (room to hold a slide AND complete),
  narrowing to touge. Used by drift envs + drift eval. Plus **graduate-to-target**:
  `--target NAME` (or the dashboard "Graduate to" select) converts a generalist
  into a specialist on that track once the curriculum is mastered (`PPO._graduate`).
- **LR + entropy schedules (`PPOSpec.anneal`, `--anneal`):** cosine-decay LR→0.2×
  and ent_coef→0.15× over a run (explore early, sharpen/refine late). Used by both
  headless `train()` and the live trainer via `PPO.set_schedule`.
- **Parallel rollouts (`PPOSpec.n_workers`, `--workers N`):** `SubprocVecEnv`
  (in `ppo_env.py`, numpy-only so workers stay light) shards the envs across N
  spawn subprocesses. ~1.7× at 2 workers on 4 envs; scales with more. Safe
  fallback to in-process if spawn fails; `n_envs` must be divisible by `workers`;
  workers torn down in `train()`/on exit.

### Audio — the "loud crack at the start of a level"
- **Two causes, both fixed.** (1) The stream opened *before* the level build in
  `app.run`, so the synth callback starved during track gen → buffer-underrun
  crackle. Now the stream opens AFTER setup, buffer is 512→1024, latency "high",
  and the master **fades in over ~80 ms** (`sound.py`). (2) A reset drops the
  gearbox to 1st → read as a *downshift* → backfire crack on every respawn. Added
  `EngineAudio.reset()` (called on every `veh.reset`) + a `speed > 5` gate on the
  downshift pop. Real downshifts-at-speed still pop. Verified offline.

### Visuals — drift FX + cinematic (bundles A + C)
- **`fx.py`**: new `SkidTrail` (persistent rear-tyre rubber marks, dark→hot by
  slide, fade ~6 s, stroke-break on grip return); re-enabled smoke/dust/sparks;
  `heat_color` cold end darkened so marks read on tarmac.
- **`app.py`**: tyre smoke (rear, on-track slides), skid marks, off-track dust,
  curb sparks, a live **drift meter** + floating score popups (angle · combo);
  **dynamic camera** (speed zoom-out + slide shake), **speed streaks**, and a
  **day/dusk/night colour grade** (`G` cycles). All toggle in the `F` menu
  (number keys 1-0), `V` = all on/off; particle-capped + amortised. Removed dead
  `COL_*` constants. Verified by rendering frames to PNG (skids + smoke + HUD).
- **Not built (yet):** bundle B (curbs/grass/start-line/vignette) and the `--3d`
  viewer — left as future work.

---

## 1. Environment & dependencies

- **OS:** macOS (developed on Carson's Mac). Uses `python3` / `pip3` (NOT `python`/`pip`).
- **Python:** 3.14.3
- **Packages (all installed):**
  - `numpy` 2.4.4
  - `torch` 2.12.0 (CPU; `DEVICE = cpu` in `ppo.py`)
  - `pygame` 2.6.1 (SDL 2.32.10)
  - `sounddevice` 0.5.5 (real-time engine audio)
  - `flask` (Command Center web backend)
- **Headless rendering:** set `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy` to run
  pygame code without a window/audio (used for thumbnails, tests, eval).
- No GPU required. Training is CPU, single-process (in-process vectorized envs).

---

## 2. Quick start

```bash
cd "~/Desktop/Supra Ai 2"

# --- DRIVE yourself ---
python3 run.py --drive --track club            # named track
python3 run.py --drive --gen touge --difficulty 0.8   # generate one

# --- PPO RACE ---
python3 run.py --ppo 600                        # train generalist (curriculum)
python3 run.py --ppo 800 --track akina --out akina_race.pt   # specialist
python3 run.py --watch-ppo --checkpoint ppo_race.pt          # watch

# --- PPO DRIFT ---
python3 run.py --drift 1500                      # train generalist drift
python3 run.py --drift 2000 --track national --out national_drift.pt  # specialist
python3 run.py --watch-drift --checkpoint national_drift.pt           # watch (auto track)
python3 run.py --drift 1000 --resume X.pt --out X_v2.pt               # continue

# --- GA ---
python3 run.py --train 200 --track akina --out akina_ga.npz   # specialist
python3 run.py --train 100 --resume akina_ga.npz --out akina_ga.npz  # continue
python3 run.py --watch --checkpoint akina_ga.npz              # watch (auto track)

# --- DASHBOARD (Command Center) ---
python3 command-center/server.py     # then open http://localhost:8770
# or double-click  ~/Desktop/Supra Command Center.command
```

Flags: `--car {supra,rx7,skyline}`, `--track NAME` (specialist) or default `random`
(generalist), `--out NAME` (named save), `--resume PATH` (warm-start/continue),
`--live` (watch training in a window), `--no-audio`, `--seed N`, `--pop N`.

---

## 3. Repository map

```
run.py                     CLI entry point — arg parsing + dispatch to all modes
command-center/            Self-contained web dashboard (Flask + vanilla JS)
  server.py                Backend: launches run.py subprocesses, SSE log/metric stream,
                           checkpoint manager, track thumbnails. Port 8770.
  SUPRA.command            In-repo launcher
  static/index.html        Dashboard UI (tabs: Drive/Train/Watch/Checkpoints/Tracks)
  static/app.js            Frontend logic (launch/stop, charts, pickers, toggles)
  static/style.css         Dark theme
  thumbs/                  Cached track thumbnail PNGs
supra/
  config.py                ALL dataclasses: CarSpec(+presets), SimSpec, SensorSpec,
                           EvoSpec, PPOSpec, RaceReward, DriftReward
  physics.py               Vehicle — 4-wheel dynamics (Pacejka, weight transfer, etc.)
  track.py                 Catmull-Rom tracks, raycasting, generators, named tracks
  sensors.py               SensorSuite — 40-dim observation
  brain.py                 MLP (NumPy) for the GA
  evolution.py             GA — population, elitism, tournament, crossover, mutation
  agent.py                 CarAgent — wraps a brain to drive a vehicle (GA eval)
  ppo.py                   PPO trainer + ActorCritic (PyTorch) + deterministic eval
  ppo_env.py               SupraEnv — Gym-style env, RunningNorm, drift integration
  drift.py                 DriftScorer — the drift reward logic
  app.py                   PyGame watch/drive loop (the viewer) + FX + AI-viz hooks
  app_ga.py                Live GA evolution viewer (--train --live)
  app_ppo.py               Live PPO training viewer (--ppo/--drift --live, S/L save/load)
  carart.py                Procedural top-down car art (Supra/RX-7/Skyline)
  fx.py                    Particle FX engine (smoke/dust/sparks/backfire flames)
  aiviz.py                 AI-brain viz: PolicyAgent adapter + predicted-path drawing
  dashboard.py             In-viewer telemetry panel + raycast beam drawing
  sound.py                 EngineAudio — real-time engine synthesis
```

~5,700 lines total.

---

## 4. Core concepts at a glance

- **Observation (42-dim):** 40 sensor values + a 2-way **mode one-hot** `[race, drift]`.
  The mode flag makes the policy *mode-conditioned* — same architecture trains race,
  drift, and (future) a goal-conditioned hybrid. The 40 sensors =
  9 raycast wall-distance beams + proprioception (speed, slips, yaw rate, wheel loads,
  gear/rpm, etc.) + look-ahead curvature samples.
- **Action:** race = 2 continuous `[-1,1]` (steer, longitudinal=throttle/brake);
  drift adds a 3rd (handbrake). `long>0`→throttle, `long<0`→brake. Handbrake = `(a[2]+1)/2`.
- **Control rate:** policy decides at **30 Hz**, physics runs at **120 Hz**
  (`control_period = round(1/(30·dt)) = 4` substeps per decision). **Critical:** the
  watch viewer queries the policy at this same 30 Hz — running a drift policy at the
  render framerate (60 Hz) makes it over-correct and spin (this was a real bug, fixed).
- **Two AI paths:** GA (NumPy MLP, evolution, race-only) and PPO (PyTorch
  actor-critic, race + drift). PPO is the main path for drift.

---

## 5. The simulator

### 5.1 Vehicle physics (`physics.py`)
4-wheel model with: Pacejka magic-formula tyres, friction-ellipse combined slip,
tyre relaxation length (lagged slip), longitudinal + lateral weight transfer with
roll-stiffness distribution, downforce, full-velocity-vector aero drag, sub-stepped
drivetrain (`drivetrain_substeps=6`) with clutch (tanh), idle governor, LSD, and
rate-limited steering. `Vehicle.reset(x,y,yaw, speed=0.0)` — the optional `speed`
spins wheels+engine+gear up to match (jolt-free rolling start, used by exploring
starts and the deterministic eval). Key state: `x,y,yaw,vx,vy,r` (body vel + yaw
rate), `ax,ay` (body accel, used for the viewer's weight-transfer lean), `wheel_w`,
`engine_w`, `gear`, `wheel_grip[4]`, `slip_angle`.

### 5.2 Tracks (`track.py`)
Catmull-Rom centerlines → left/right walls, arc length, signed curvature.
- `raycast()` — vectorized beam casting against walls (windowed for speed).
- Generators: `make_track(difficulty, style, length, seed)`, `random_circuit`,
  `oval`, `touge`, `stadium`.
- **Styles** (`STYLES`): gp / technical / speedway / touge (spacing, jitter, width,
  elongation, min-radius).
- **Named tracks** (`NAMED`): `club`(gp .25), `national`(gp .55), `coast`(gp .45),
  `sprint`(technical .65), `tech`(technical .90), `oval2`(speedway .30),
  `akina`(touge .80), `pass`(touge 1.00). Plus `SPECIAL`: `speedbowl`, `superspeed`.
- `curriculum_track(difficulty, seed)` — used by generalist training; archetype mix
  shifts harder as difficulty rises.
- `named_track(name)`, `named_list()`, `pose_at(index)` (arbitrary start pose).

### 5.3 Sensors (`sensors.py`)
`SensorSuite.observe(veh, trk) -> Observation` with a 40-value `.vector`
(`obs_size=40`). The PPO env appends the 2-way mode one-hot → 42.

---

## 6. Training systems

### 6.1 GA (`evolution.py`, `brain.py`, `agent.py`)
NumPy MLP (`EvoSpec.hidden=(32,24)`, `n_actions=3`). Elitism (`elite_frac=0.15`),
tournament selection (`k=3`), crossover, Gaussian mutation (`rate=0.18, sigma=0.3`).
`GA(evo, car, seeds, tracks=None, track_name=None)` — pass `tracks=[fixed]` for a
**specialist**, else seeded random circuits (generalist). `warm_start(genome, fit)`
resumes from a champion. **GA has inherent keep-best via elitism** — the saved
champion is always the best genome ever seen, so longer = never worse (no separate
`_best` file needed). GA is **race-only** (drives the track fast; no drift mode).

### 6.2 PPO (`ppo.py`)
Standard PPO: GAE (`gamma=0.997, lambda=0.95`), clipped surrogate (`clip=0.2`),
value loss, entropy bonus (`ent_coef=0.003`), `epochs=4`, `minibatches=8`,
`lr=3e-4`. Continuous Gaussian policy, state-independent `log_std` **clamped to
[-2.2, 0.0]** (this clamp is essential — see §9). In-process `SyncVecEnv`
(`n_envs=8`, `rollout=256`). Truncation bootstrapping. Actor init has a
forward-throttle bias (`mean.bias[1]=0.6`) so the untrained policy moves (otherwise
it stalls and never discovers reward). `hidden=(128,128)`.

### 6.3 SupraEnv (`ppo_env.py`)
Gym-style. `RunningNorm` (Welford) normalizes the 40 sensor dims; the 2 mode dims
stay raw. Track pool for generalists (`track_pool=16`, regenerated on difficulty
bump); `fixed_track` for specialists.

### 6.4 Curriculum
Generalists raise track difficulty once the rolling task metric clears a threshold
(`promote_at=0.72` lap-completion for race; `drift_promote_at=0.30` drift-fraction
for drift; `difficulty_step=0.08`, `start_difficulty=0.15`, `max_difficulty=1.0`).
**Specialists disable the curriculum** (fixed track) and so plateau — hence keep-best
+ early-stop matter most for them.

### 6.5 Specialist vs generalist + named saves + resume
- `--track NAME` → **specialist** on one fixed track. `--track random` (default) →
  **generalist** (curriculum pool for PPO, seeded circuit for GA).
- `--out NAME` → save to your own file (`.pt` for PPO, `.npz` for GA) so runs stay
  separate. Default slots: `ppo_race.pt`, `ppo_drift.pt`, `ga_champion.npz`.
- `--resume PATH` → continue/warm-start from a checkpoint (PPO loads weights+optimizer
  +normalizer+curriculum; GA seeds the population from the champion + mutants).
- Checkpoints store `mode`, `track`, `car`, `updates`, `difficulty`, and the
  honest `metric`/`drift`/`laps` (see §6.7). Atomic writes (tmp + os.replace), so a
  concurrent watch can never read a half-written file.

### 6.6 Exploring starts + randomized speed (`ppo_env.reset`)
`PPOSpec.random_start=True` (default): each reset drops the car at a **random point
along the track** (so it practices the *whole* loop, not just the opening section it
can reach before terminating), at a **curvature-aware randomized speed**
(0.55–1.20× the natural cornering pace, 15% standstill), plus mild lateral-offset /
heading jitter. The rolling start is jolt-free (`Vehicle.reset(speed=)` syncs
wheels/engine/gear). **Watching always starts at the line, deterministic** — only
*training's* distribution is widened.

### 6.7 Deterministic eval + keep-best + early-stop (the trustworthy metric)
**This is the key reliability fix.** During training:
- The **rolling `drift`/`laps` in the log** is measured under *stochastic* actions +
  *exploring starts* — it OVERSTATES real drift (exploration noise and at-speed
  spawns produce transient slides). **Do not judge by this number.**
- Every `save_every` (20) iters, the trainer runs a **deterministic eval**
  (`PPO._deterministic_eval`): `act_mean` (no noise), from 4 fixed points around the
  track at racing pace, over a **full episode** (`episode_seconds`) per start so a
  whole lap is actually reachable. This **matches what you see when watching**. It
  prints `[eval] eval drift D x lap L = SCORE` and is **fully reproducible**.
  ⚠️ This budget was a fixed 14 s before 2026-06-06, which capped `lap` at ~0.2–0.4
  on 1+ km tracks for any policy — see §0.0. Old stored `metric`/`laps` in
  pre-fix checkpoints reflect that capped number; re-score with the new eval.
- **Keep-best** saves a `<run>_best.pt` whenever the eval **composite score** hits a
  new high. Composite (drift mode) = `drift × (0.3 + 0.7·lap_completion)` — so a
  "suicide drifter" (big angle, spins out, no progress) and a "grippy racer"
  (drift≈0) both score ~0; only *drift-while-completing* wins.
- **Auto early-stop** (specialists only): if the eval best hasn't improved in
  `specialist_patience=500` iters, stop (peak already banked). Generalists run the
  full curriculum. So you can set a specialist to `--drift 5000`, walk away, and it
  trains to plateau, banks the best, and quits before over-training collapse.
- **Watch picker** offers **Latest / Best ★** per run; specialists auto-watch on
  their trained track.

---

## 7. The viewer (`app.py`) + dashboard

### 7.1 Watch/drive loop
Chase camera, procedural car art, telemetry dashboard, raycast beam overlay, engine
audio. Controller (a trained policy) queried at 30 Hz; auto-resets on crash.

**Framerate decouple (fixed):** physics now consumes the *actual wall-clock time*
each frame (catch-up clamp 0.20 s, substep cap ~26), so heavy rendering lowers FPS
but **never slows the car** (real-time held down to ~5 FPS). Proven: at 10 FPS the
sim ran 0.50× before, 1.00× after.

### 7.2 Visual effects (current, deliberately minimal)
After a big visual build-out then a deliberate cull, **only three** effects remain:
- **Daytime track view** (the clean original look)
- **Predicted path** — dotted ghost-line of where the policy thinks it's going,
  rolled forward via the policy on a cloned car. **Amortized** across frames (a
  small step budget per frame) so it never spikes — was a 30 ms/6-frame stutter,
  now ~3 ms/frame.
- **Backfire flames** — exhaust pops on downshift/overrun, synced to BOV audio.

All effects are toggleable: press **`F`** for the effects menu, number keys to flip,
**`V`** = all on/off. (`fx.py` still contains smoke/dust/sparks and `aiviz.py` still
has the brain-panel/gauge drawing code, but they're disabled in `app.py`. A richer
"world/atmosphere" system was built and then **removed** — backups at
`/tmp/world_heavy.py`, but `supra/world.py` was deleted.)

### 7.3 Command Center dashboard (`command-center/`)
Flask app on **port 8770**, all-Python (no Node). Tabs: Drive / Train / Watch /
Checkpoints / Tracks. Launches `run.py` subprocesses, streams logs + parsed metrics
over SSE (live charts), manages checkpoints (backup/rename/delete/activate),
renders track thumbnails, and re-attaches to running tasks on refresh. Static files
served with **no-cache** headers (so UI edits always show on refresh).
- Train cards (GA / PPO Race / PPO Drift) each have: iters, chassis, **Train on**
  (generalist vs specialist track), **Run name** (named save), **Continue ↻**
  (resume), New-run / Live buttons.
- Watch tab: **checkpoint picker** (any saved checkpoint, auto-routed to the right
  viewer by kind) + **Latest/Best ★** toggle + track grid.
- Stop button sends SIGINT (clean checkpoint save) with SIGKILL fallback.
- **Note:** training-code changes apply to new runs automatically (it spawns
  `run.py` fresh); only `server.py` changes need a dashboard restart.

---

## 8. Config reference (current values)

```
SimSpec:    dt=1/120, fps=60, drivetrain_substeps=6
PPOSpec:    hidden=(128,128) lr=3e-4 gamma=0.997 gae_lambda=0.95 clip=0.2
            epochs=4 minibatches=8 ent_coef=0.003 vf_coef=0.5 max_grad_norm=0.5
            init_log_std=-0.5  n_envs=8 rollout=256 control_hz=30 episode_seconds=60
            start_difficulty=0.15 promote_at=0.72 difficulty_step=0.08 max_difficulty=1.0
            drift_promote_at=0.30 track_pool=16
            random_start=True  specialist_patience=500
EvoSpec:    pop_size=60 hidden=(32,24) n_actions=3 elite_frac=0.15 tournament_k=3
            mutation_rate=0.18 mutation_sigma=0.3 episode_seconds=45 control_hz=30
            bias_throttle=0.73 bias_brake=0.18 bias_clutch=0.88
RaceReward: progress=60 align=0.06 speed=0.005 offtrack=0.15 smooth=0.002
            crash=3.0 lap_bonus=5.0
DriftReward:scale=0.015  progress=30 (NEW)  drive=0.05 (trimmed from 0.12)
            angle_speed=3.5 entry_lo_deg=2.3 entry_hi_deg=11.5 peak_deg=55 spin_deg=92
            speed_gate=12.5 initiation=2.0 donut_speed=4.0 donut_penalty=3.0
            transition_bonus=4.0 chain_window=1.8 sustain_rate=0.5 sustain_max=3.0
            gutter_bonus=1.5 drift_min_deg=6.0 offtrack=0.1 crash=2.0
```

### Drift reward structure (`drift.py` DriftScorer + `ppo_env.py` drift branch)
Per step, on-track: `drift_score (angle×speed, soft-entry, sustain multiplier,
transition/chain bonuses, gutter bonus, over-rotation falloff peak→spin) × scale`
+ `progress × Δlap` (NEW) + `drive × speed` (small bootstrap) − `offtrack penalty`.
Off-track earns ~nothing and terminates fast. `drift_min_deg=6°` defines "drifting"
for the `drift_frac` metric.

---

## 9. The development journey — struggles, fixes, what worked / didn't

### Fixes that WORKED
| Problem | Fix | Result |
|---|---|---|
| PPO wouldn't learn (stalled) | forward-throttle init bias, length-independent reward, track pool, heading bootstrap | learns now |
| Entropy explosion (`log_std`→4.4 over 16k iters degraded policy) | clamp `log_std∈[-2.2,0]`, `ent_coef` 0.006→0.003 | stable |
| Drift watch spun out | query policy at **30 Hz** (control_period), not render rate | major fix |
| macOS tkinter+pygame crash | `osascript` file dialogs instead of tkinter | fixed |
| Loud audio crackle on akina | cooldown + low probability + amplitude 0.32 | fixed |
| Not all tracks watchable | fixed `_resolve_track` (legacy oval/random/touge) | all 13 work |
| Generic Node dashboard kept breaking | built all-Python **Command Center** | solid |
| Slow-motion under heavy render | **framerate decouple** (consume real wall-time) | real-time to ~5 FPS |
| Predicted-path stutter (30ms spike) | **amortize** rollout across frames | ~3ms/frame |
| Browser showing stale UI | **no-cache headers** on static files | edits show on refresh |
| Specialist over-training collapse (touge 4780: drift 0.16→0.04) | keep-best + auto early-stop | peak banked |
| Watching specialist on wrong track | auto-default watch to the checkpoint's trained track | fixed |
| **"Trains 0.5 drift but doesn't drift watched"** | proved pipeline byte-identical; root cause = metric measured under stochastic+exploring-starts. Added **deterministic eval** as the honest metric/keep-best signal | metric now trustworthy |

### Fixes that DID NOT work / were reverted
- **Drift "slides off then drives back on":** tried on-track gating + tightened
  termination + combo-break-on-grass. *Partially* helped (off-track dropped) but the
  deeper "doesn't really drift" problem remained.
- **Grace-window drift metric de-bias:** hypothesized spawn transients inflated the
  metric; adding a 0.75 s grace **backfired** (it removed the no-drift accel phase
  and *raised* the number). **Reverted.**
- **DriftReward retune via short resume probes:** added progress + lowered `peak_deg`
  to 44° and bumped progress to 40. Short (~80–100 iter) resume tests were **too
  noisy to validate** (lap bounced 0.15–0.28, no clean trend); **reverted `peak_deg`
  to 55**, kept progress at a moderate 30 and `drive` at 0.05. *The progress term is
  structurally justified but UNVALIDATED — needs a fresh full run.*
- **Phase-3 world/atmosphere (night/weather/forest/etc.):** built, deemed too busy /
  framerate-heavy, then **removed entirely** per user direction.

### THE core open problem
**Drift policies slide but don't complete the track.** Even measured honestly
(deterministic, whole-loop), the best drift specialists hit ~0.4–0.5 drift fraction
but only ~0.2–0.4 lap completion — they over-rotate/spin and terminate. Race PPO
completes laps fine; drift is the hard part. Leading hypothesis (now addressed but
unvalidated): drift mode had **no progress reward**, so completing the lap was never
incentivized — only sliding + raw speed. The just-added `progress=30` term targets
exactly this.

---

## 10. Checkpoint inventory (current)

> **Honest-number caveat:** `drift`/`laps` only populate for checkpoints saved *after*
> the deterministic-eval change. Older ones show `0.0` (no stored metric) — that does
> NOT mean they're bad, just unmeasured by the new system.

```
PPO RACE
  ppo_race.pt              race, 3069 upd, generalist     — works, completes laps

PPO DRIFT (generalists)
  ppo_drift_refined.pt     drift, 2806 upd, generalist    — the good generalist drifter
  ppo_drift.pt             drift, 2164 upd, track=touge    — DEFAULT SLOT was overwritten
                            by a touge specialist! Good generalist is *_refined above.
  ppo_drift_prev.pt        drift, 600 upd

PPO DRIFT (specialists)
  tech_drift_2_best_resume1_best.pt  drift, 2601 upd, tech — best tech: eval drift~0.42 lap~0.23
  tech_drift_2_best_resume1.pt       drift, 2716 upd, tech
  tech_drift_2_best.pt / tech_drift_2.pt   tech specialists
  touge_drift_specialist_Mk1.pt      drift, 2164 upd, touge — best legacy-touge (~0.16 drift)
  touge_drift_2.pt                   drift, 4780 upd, touge — OVER-TRAINED/collapsed (0.04)
  akina_drift_specialist.pt          drift, 700 upd, akina  — old-code, incomplete run
  touge_drift.pt                     drift, 5 upd           — stub

GA
  ga_champion.npz          gen 60, generalist
  tech_ga.npz              gen 135, tech specialist
```
`*_best.pt` = keep-best peak snapshots (paired with their run; reached via the Watch
"Best ★" toggle).

---

## 11. Recommended next steps

1. **Validate the progress reward (highest priority).** Fresh full drift specialist
   on a **medium** track (`national` 0.55 or `coast`) — completion is achievable
   there, so you'll *see* whether the new reward produces drift-AND-complete. Watch
   the `[eval]` line: success = `lap` climbing while `drift` stays ~0.4+. Use
   `--drift 3000 --track national --out national_drift.pt` (auto eval-keep-best +
   early-stop). THEN attempt the hard touges (`akina`, `pass`).
2. **If completion still stalls,** deeper drift-reward work:
   - A "controlled angle held *through a corner*" bonus (reward sustaining a target
     slip band, not chasing max angle).
   - Stronger anti-spin shaping (penalize approaching `spin_deg` / rapid yaw past
     control), or lower `peak_deg` toward ~45 *and validate properly this time*.
   - Consider a **drift curriculum** (start on easy/wide tracks where sustained
     drift is feasible, then ramp), instead of straight to hard specialists.
3. **LR decay** for long generalist runs (anneal `lr`) — stabilizes late training.
   (Specialists already auto-stop, so this matters most for generalists.)
4. **Re-establish a clean default `ppo_drift.pt`** (it's currently a touge specialist;
   copy `ppo_drift_refined.pt` over it, or retrain a generalist) so the default
   Watch slot is your best generalist.
5. **Slice 6 — race/drift hybrid.** Infra is ready: the 42-dim obs already carries
   the `[race,drift]` mode one-hot, so one mode-conditioned policy can do both.
6. **Speed:** multiprocessing for `SyncVecEnv` (currently single-process, ~1100 sps).
7. **Possible:** LSTM/recurrent policy (`--recurrent` is stubbed), a 19-track
   "showcase exam" eval, and re-validate `drift_min_deg`/`peak_deg` definitions of
   "a drift."

---

## 12. Gotchas / things a new maintainer must know

- **`python3`/`pip3`**, not `python`/`pip` (this Mac).
- **Judge drift by the `[eval]` line, not the rolling `drift`** — the rolling number
  is measured under exploration noise + exploring starts and overstates reality.
- **Watch queries the policy at 30 Hz** (sim-time), decoupled from render FPS. Both
  facts are load-bearing; don't "simplify" them.
- **Dashboard:** spawns `run.py` fresh → training-code changes are picked up on the
  next launch with no restart. Only `server.py` changes need a server restart.
  Static files are no-cache → just refresh the browser for UI changes.
- **Headless:** export `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.
- **A running training process uses the code as it was at spawn time** — editing
  source does NOT affect an already-running run (Python loads modules once). To apply
  a change to a live run, Stop → Continue.
- **GA = race only.** "GA specialist" = champion bred to nail one circuit fast.
- **Checkpoints are atomic-write safe** to watch mid-train.
- **`specialist_patience=0`** disables auto early-stop if you want a fixed-length run.
- Backups of the removed heavy world FX live at `/tmp/world_heavy.py` (ephemeral).

---

*End of handoff. The single most valuable thing to internalize: the infrastructure
(training, eval, dashboard, viewer) is solid and trustworthy; the remaining work is
almost entirely **drift reward shaping** to turn a slides-then-spins policy into one
that drifts AND completes — validated against the deterministic `[eval]` metric, on a
medium track first.*

---

## Removed browser viewer (historical, 2026-06-10)

This section described a browser viewer that was later removed in full. It has
no route or compatibility fallback. The current clean-room Fable Five
Observatory is documented in `docs/05_UI_and_Visualization.md`; the 2D viewer
remains the supported second option.

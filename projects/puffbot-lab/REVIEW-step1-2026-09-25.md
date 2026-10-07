# Step 1 review — September 25, 2026

**Verdict: meaningful fixes, but step 1 is not ready to be treated as a reliable evaluation/training foundation.** The jump transition, match cap, rejection of incomplete evaluation results, stale-fragment recheck, and legacy checkpoint checks work in the cases tested. Two new failure paths need repair: an evaluation can hang after a worker dies, and a crash immediately after a genuine game ending can discard that completed game's terminal training data. Legitimate timeout-heavy evaluations also exhaust their batch budget before reaching the requested game count.

Reviewed `main` (`ab98448`) against `step1-fixes` at `f070e5b5c30e3d442ada7a2c0fa40f2ac73f73a1`, including `08aaec4`. This is a review only. No source/configuration/run/checkpoint edits, training, promotion, merging, or deletion were performed. The only project artifact created is this report. Scratch scripts, emulator homes, checkpoint copies and reports are in `/tmp/puff-step1-review.cxTjI3`.

## Verified results

- **56 tests passed in 1.65 seconds**, using bytecode suppression and disabling pytest's cache provider. These are the repository's emulator-free tests.
- **Native four-game evaluation passed:** final 49.91M/v3429 policy, Fox and Marth CPU 1, two games each, two workers; `complete`, 4/4 counted, four wins, no reported match errors, process exit 0. SHA-256: `77d337a31b7588ffd2986156b76472dc942c327b6a93d3ed47219c3ac8731098`. This validates the successful evaluation path, not competitive strength.
- **Native cap/CLI failure path passed:** cap 30, one requested Fox CPU-1 game; three attempts were rejected as `capped`, report `incomplete`, 0/1 counted, process exit **2**. No cap termination became a loss or normal terminal result.
- **Native consecutive jumps reproduced the claimed improvement:** six macros consume only the ground jump with the baseline executor; the new executor consumes the ground jump and two aerial jumps, leaving three. Further timing details are below.
- **Native defensive-input counterexample:** digital L sent while observing neutral-air frame 49 produced `AIRDODGE` on the next frame. A current-state aerial-attack check alone cannot establish the proposed safety guarantee.
- Read-only restore of the existing full checkpoint retained version 3429, CPU frontier 9 and `mastered=true`, reported the executor change, and estimated **178.81 hours** of game time versus **231.08 hours** of player experience.

The claimed 28-game, two-worker training smoke was **not rerun**, because training was explicitly prohibited. Native evaluations emitted resource-tracker semaphore warnings at shutdown; they are not evidence of a sustained live leak by themselves. No six-emulator soak or Slippi-AI inference benchmark was performed.

## Findings requiring follow-up

### R1 — P1: A dead worker can leave evaluation waiting forever

**New coordinator failure.** [evaluate.py:177](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/evaluate.py:177>) exits on an empty queue only when *all* workers have died. Fatal messages are logged but do not release work. `Quota` tracks outstanding counts by opponent, without a batch owner or lease. If one worker dies after taking a batch, its reservation remains outstanding; surviving workers wait for work and emit status messages. Neither `complete` nor `exhausted` becomes true.

A real multiprocessing reproduction, using synthetic workers rather than Dolphin, requested ten Fox games. One worker completed five; the other took five and died. No replacement batch was issued. The harness forcibly interrupted the coordinator after five seconds: report `cancelled`, 5/10 counted, two batches issued. The forced interruption is what ended the test; the implementation has no subsequent deadline that would resolve the abandoned reservation. See [coordinator-result.json](/tmp/puff-step1-review.cxTjI3/coordinator-result.json) and [coordinator_probe.py](/tmp/puff-step1-review.cxTjI3/coordinator_probe.py).

The same bookkeeping failure can follow a lost `setup_done`: [actor.py:367](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:367>) uses a fallible, two-second queue send for the completion token. This second trigger is code-derived, not a separately forced queue-saturation reproduction.

**Repair:** assign batch/game/attempt IDs, acknowledge ownership, reclaim dead or expired workers' leases, add job/progress deadlines and bounded retries, and make result acceptance idempotent. Persist a running report before launching workers. A control message that releases work needs reliable acknowledgement or recoverable state.

Normal sentinel shutdown worked in both native evaluations. Within one producer, delivered game messages precede its delivered completion token; cross-worker interleaving does not itself explain this hang. The missing recovery of abandoned work does.

### R2 — P2: Normal Sudden Death games consume the retry budget too quickly

[evaluate.py:57](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/evaluate.py:57>) allows three times the initially planned number of **batches**. [actor.py:254](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:254>) deliberately ends a setup after its first Sudden Death game, even when that game completed legitimately.

For ten requested games with five per setup, a deterministic quota simulation in which every batch produces one valid timeout draw yields batch sizes `5, 5, 4, 4, 3, 3`. All six games are accepted at the correct CPU level, there are no failed games, but the budget is exhausted at **6/10**. The final state is correctly incomplete; the defect is failure to fulfill a valid timeout-heavy protocol. A single short batch can be topped up successfully, as the existing test establishes.

**Repair:** give each requested game its own attempt allowance, or make legitimate completed-game progress replenish the setup allowance. Retain a separate absolute deadline/failure budget. Scheduling one game per batch is simpler but incurs more emulator setup overhead.

Evidence: [synthetic-results.json](/tmp/puff-step1-review.cxTjI3/synthetic-results.json), keys `sudden_batch` and `all_legal_timeouts`; `test_quota_tops_up_batches_that_end_short` does not cover persistent legitimate short batches.

### R3 — P2: An emulator error just after a win discards completed-game data

**New truncation regression.** [actor.py:307](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:307>) closes the final decision as terminal and reports the result, but leaves `game_row0` pointing at the completed game's start. An emulator error before the next match reset calls `cut(None, ...)` at [actor.py:295](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:295>). Despite `last=None`, `cut` still truncates both buffers back to that boundary at line 322.

A synthetic match won normally and failed on the very next emulator frame. With fragment size 8, 24 earlier rows had been sent, but **zero terminal rows survived**. With fragment size 128, **all the completed game's buffered data vanished**. The game remained reported as a win. This biases the retained experience toward the pre-ending portion of affected matches.

**Repair:** cut only an active, unfinished game; protect the completed boundary immediately after `finish`, or explicitly skip truncation when the preceding game already ended. Add coverage for failure after a win but before frame reset, for both player buffers and for fragment sizes on either side of game length.

The existing `test_finished_games_before_a_failure_keep_their_data` covers a failure *during the next game*, after the boundary has advanced; that path passed additional probes. `emit()` resets the boundary when a fragment leaves, which is correct. Already-emitted nonterminal fragments need not be retroactively discarded or turned into terminal transitions.

Mirror port 2 still unconditionally loses its leftover partial fragment at setup exit ([actor.py:299](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:299>)). A completed first mirror game followed by an interrupted second retained the first game's port-1 terminal; with fragment size 128 the port-2 tail was discarded entirely. That behavior already existed on `main`, so it is a remaining data-efficiency/asymmetry issue, not a new step-1 regression.

### R4 — P2: Historical dashboard data still labels experience as game time

The new counters correctly separate the two sides of mirror training, but [server.py:111](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/server.py:111>) and [web/app.js:64](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/web/app.js:64>) fall back from missing `game_hours_played` to old `game_hours_trained` under the new label. The old saved status therefore still displays **231.08 hours as game time**, without the estimate marker. The speed display also reuses the old experience-derived `realtime_multiple`. Merely opening a stopped old run does not invoke the corrected resume calculation.

**Repair:** translate legacy statuses read-only using an explicitly marked estimate, or label their numbers as legacy player experience and world speed as unavailable. Do not relabel the old counter silently or require training to repair the display.

There are two further accounting limitations:

1. `_logged_game_frames` sums the entire log, without the restored checkpoint's time/version boundary ([learner.py:129](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/learner.py:129>), [learner.py:531](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/learner.py:531>)). The latest checkpoint produced the plausible 178.81-hour estimate, but a synthetic older checkpoint with 100 experience frames was assigned 38,622,876 world frames from the later log. Estimate only through the checkpoint's boundary, or identify the number separately as whole-run logged gameplay.
2. New cumulative world time counts **port-1 frames accepted into updates**, whereas world throughput counts **received port-1 fragments before stale rejection**. Neither is total emulator gameplay, which includes dropped tails and discarded fragments. Preserve distinct labels for simulated, received, accepted-world and player-experience frames; use accepted throughput for scaling decisions. The six-worker “about 20×” claim was not independently benchmarked here.

### R5 — P2: A compatible checkpoint with a different hidden size cannot be evaluated under the default config

A valid current-contract policy with `hidden=32` passes `load_policy`, which returns the saved hidden size and no notes. The evaluation worker nevertheless constructs its actor from `cfg.hidden` (default 512) and then loads the 32-wide weights ([actor.py:74](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:74>), [actor.py:344](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/actor.py:344>)). A temporary compatible policy reproduced a `load_state_dict` shape error.

This is inherited behavior, not introduced by the contract patch, and does not affect the tested 512-wide checkpoint. It means contract acceptance is not yet sufficient for successful evaluation. Instantiate from the checkpoint architecture, as the frozen-opponent path already does, or reject incompatible runtime configuration early with a specific explanation. Test this through the worker, not only through the file loader.

### R6 — P3: Protocol capture and watchdog guarantees remain narrower than advertised

- The native evaluation records the checkpoint hash, source commit, dirty flag, Dolphin version, seed and selected settings. Its `libmelee` version is **null**: the installed module lacks the queried `__version__`. Use package-distribution metadata. The report contains only a config subset, no Gecko hash, no dirty-source content hash, and no declared/observed stocks, timer or port validation. ISO hashing is intentionally deferred; recording size does not identify contents. See [evaluate.py:117](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/evaluate.py:117>) and the native report.
- Timing is checked before the evaluator replaces `cfg.act_every` with the checkpoint value ([evaluate.py:146](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/evaluate.py:146>)). For a mismatch, the saved warning can describe the old runtime timing although execution then uses the saved timing. Generate notes after resolving the effective config and validate that config again.
- The learner's new watchdog measures **message silence**, not frame/game progress. Any status refreshes `worker_seen` ([learner.py:236](</Users/REVIEW_USER/Desktop/melee bot v2/puffbot/learner.py:236>)). A synthetic worker emitting unchanged progress at 500 seconds was not killed. Detect repeated nonprogress separately, with phase-aware deadlines and worker-generation IDs. The evaluation coordinator has neither this silence watchdog nor a progress deadline.
- Normal boot/menu/backoff durations did not demonstrate a false positive: synthetic silence at 45, 90, 120 and 239 seconds was tolerated; 241 seconds triggered a kill. These are logical boundary checks, not a slow-machine soak. Host sleep or a blocked learner can delay message processing; any future progress watchdog must distinguish those conditions from worker failure.

## Original audit disposition, including sub-items

“Not addressed” below includes deliberate deferrals and is not itself a regression. “Fixed” describes the stated defect, not proof of broad playing strength.

| Original item | Disposition | Evidence and remaining scope |
|---|---|---|
| 1: Learned opponents all Puff; automatic 60% Puff self-play after mastery | **Not addressed** | `league.py:80,108`; curriculum remains unchanged. |
| 1: Pooled 26/40 mastery without per-matchup coverage; sticky mastered flag | **Not addressed** | `league.py:46`; restored checkpoint remains mastered. |
| 1: Recency-based history; no payoff matrix, independent character policies, exploiters or weakness-driven sampling | **Not addressed** | `learner.py:189`, `league.py:108`; retain as step-2 priorities. |
| 2: No fresh airborne shield/tech/L-cancel/air-dodge actions | **Not addressed** | `actions.py:118`; intentional deferral. Proposed safety gate needs revision, below. |
| 2: Consecutive jump macros hold Y without a fresh press | **Fixed** | `actions.py:167`; `test_consecutive_jumps_each_press_the_button`; synthetic and native probes agree. |
| 2: Every further Puff multijump requires a new press | **Disagree with that generalization** | The initial jump transition needs release/repress; native held-input probe triggered later multijumps at their animation gates without new edges. The original two-macro defect remains valid. |
| 2: Up-to-16-frame noninterruptible macros, coarse angles, blanket offstage Rest restriction | **Not addressed** | `actions.py:63,118,183`; no interruptible control or precision expansion. |
| 2: Verified skill/controller architecture and jump-budget-aware recovery | **Partial** | Edge executor improved; no general recovery/defense scenario suite or flexible controller policy. |
| 3: Finished evaluation with zero of ten games | **Fixed** | Rerun now reports `failed`, 0/10; native cap evaluation is `incomplete`, exit 2. |
| 3: Sudden Death abandons remaining games in a setup | **Partial** | Top-ups added, but persistent valid short batches exhaust budget: R2. |
| 3: Retries repeat results; no durable game/attempt identities | **Partial** | Workers no longer independently replay whole failed batches, but IDs/deduplication remain absent. Re-submitting the same synthetic result twice counts twice before quota is full. R1 covers abandoned reservations. |
| 3: CPU-level mismatch accepted | **Fixed for actual CPU worker records** | `Quota.accept`, `test_quota_only_counts_real_results_at_the_requested_level`. Full observed character/stage/port/stocks/timer validation remains absent. |
| 3: Complete/incomplete/failed/cancelled states and exact requested evidence | **Partial** | States and count gates work; R1 can prevent termination and R2 prevents sufficient valid samples. |
| 3: Freeze/hash checkpoint and capture protocol | **Partial** | Frozen copy and SHA verified natively; source/emulator metadata present, remaining protocol gaps in R6. |
| 3: Champion requires complete qualifying evaluation | **Not addressed** | `server.py:211`; explicitly deferred. Needed before the next promotion, not necessarily before unrelated mechanics work. |
| 4: FD-only geometry and no platforms/other stages | **Not addressed** | `config.py:101`, `observation.py:68`; intentional stage deferral. |
| 4: Sheik/ICs/missing characters and Nana representation | **Not addressed** | `config.py:15`, `observation.py:76`; partial roster and two-main-player encoding unchanged. |
| 4: Both ports/spawn sides and coverage reporting | **Not addressed** | Fixed evaluation setup; mirror port-2 experience is not a balanced evaluation protocol. |
| 5: Projectile direction/type aliasing and extra entities | **Not addressed** | `observation.py:94`; still position/presence for two projectiles. Own projectiles were already filtered; remaining ownership/entity detail is not encoded. |
| 5: No short history/recurrence, opponent adaptation, actual input/timing history | **Not addressed** | `model.py:26`, `observation.py:108`; one previous macro is present, but no frame sequence or recurrent state. |
| 5: Earlier ineffective LSTM proves recurrence unnecessary | **Disagree with inference; not corrected** | `observation.py:1` still makes the strong single-frame claim; prior integration failure is not an ablation of a correctly trained recurrent policy. |
| 6: Gamma half-life incorrectly documented as 5.5 seconds | **Fixed** | `config.py:70` correctly says 3.85-second half-life and 5.55-second time constant. |
| 6: Short physical-time reward horizon and trace | **Not addressed** | Gamma/lambda unchanged by design; same horizon calculations as original audit. |
| 6: False policy-invariance claim | **Fixed documentation; objective not addressed** | `reward.py:15` explicitly disclaims it; shaping probe still differs by path. |
| 6: Reward tradeoffs, terminal boundaries, recovery penalties and controlled horizon experiments | **Not addressed** | Intentional deferral; no evidence that step 1 improved patient strategy. |
| 7: Match cap unused | **Fixed for advancing in-game frames** | `actor.py:287`; synthetic stops at 30, native attempts classified capped. Does not replace a job/progress deadline. |
| 7: Interrupted/capped games treated as genuine terminal results | **Partial** | Mid-game cuts no longer inject terminals or enter win rates/ladder; R3 regresses the post-terminal boundary. Boot failures without an active state have error metadata, not durable game-attempt records. |
| 7: Worker liveness only, no watchdog | **Partial** | Silence recovery added; unchanged-progress heartbeats evade it, and evaluation has no recovery: R1/R6. |
| 7: Pending fragments can become stale before update | **Fixed** | `learner.py:276`; `test_fragments_that_go_stale_while_waiting_are_dropped`, including all-stale/no-update path. |
| 7: Action-only contracts; resume bypasses validation | **Fixed for declared schema checks; broader compatibility partial** | `contract.py:47`, `learner.py:114,548`, `model.py:107`; legacy migration and architecture caveats below/R5. |
| 7: Current config silently changes checkpoint experiment | **Partial** | Executor/timing warnings and saved evaluation timing added; full environment/config semantics remain incompletely captured. |
| 7: No Git history | **Fixed** | Baseline and two fix commits now exist; commit captured in native report. |
| 7: Source/config/environment fingerprints and exact resume | **Partial** | Some provenance added; full config, Gecko, dirty contents, RNG state remain absent. RNG/ISO hash explicitly deferred. |
| 7: Pruning deletes evaluated or champion checkpoints | **Partial** | Default evaluation/champion references protected; `test_pruning_keeps_evaluated_checkpoints` passes and old 48.17M source is protected. Custom output directories are not scanned; a new evaluation has no report before its first result. Frozen copies protect bytes but not every original-path reference. |
| 7: Broken streams/semaphore warnings; bounded resource soak | **Not addressed** | Warnings recurred in native probes. No leak growth measurement or six-worker soak. |
| Cross-cutting: Confusion between player experience and independent game time | **Partial** | New mirror accounting test passes; legacy display, checkpoint/log cutoff and accepted-versus-received distinction remain: R4. |

The original audit's broader replay-initialization, stage/roster expansion, controlled ablations, deployment-latency and human-validation recommendations remain unimplemented. Step 1 supplies infrastructure improvements, not evidence of human or superhuman strength.

## Rerun of the five original probes

| Probe | Current result | Assessment |
|---|---|---|
| Workers exit with zero of ten evaluated games | `failed`, Fox 0/10 | Original false-success bug fixed. |
| Five-game batch reaches Sudden Death after game one | One result, setup returns | Intentional worker behavior retained; top-up coordinator only partially resolves it, R2. |
| Emulator offers 160 frames with cap 30 | Stops at frame 30, `capped`, zero remaining current-game rows | Cap and nonterminal truncation work for this case. |
| Two six-frame Y-hold macros | 13 inputs: six Y, one release, six Y; two rising edges | Jump input bug fixed. |
| Equal-endpoint shaping paths | Onstage `-0.997`; offstage `-1.2804505617112714` | Behavior unchanged; policy-invariance claim correctly removed. |

Machine-readable outputs: [synthetic-results.json](/tmp/puff-step1-review.cxTjI3/synthetic-results.json). Reproduction harness: [probes.py](/tmp/puff-step1-review.cxTjI3/probes.py).

## Release timing, contracts and boundary review

### Input transitions

Synthetic chains checked `short_hop → sh_c_left`, `jump → sh_c_left`, `jump → c_left`, `jump → rest`, `jump → down_right`, `shield → roll_left`, and `sh_c_left → c_left`. A fresh jump-to-C aerial or Rest adds no unnecessary leading release. The main stick keeps the selected DI direction; release concerns buttons/C-stick. Shield L stays held into roll. Repeated C-stick attacks receive the required neutral transition. The existing grounded short-hop aerial wait test passed.

Selecting a *short-hop aerial macro while already airborne* still includes a jump input and can spend an aerial jump when legal. That is existing vocabulary semantics, not a release-frame regression. One added release frame necessarily changes repeated-macro timing; actual elapsed decision frames are recorded. No new timing fault was demonstrated in these sequences. Native hitlag/DI outcomes, all aerial/Rest transition timings and the full 16-frame cutoff space were not exhaustively exercised; synthetic pad traces cannot certify those mechanics.

The native six-jump test confirmed presses during the previous aerial-jump animation at frames 7, 14 and 21 were ignored, while frame 28 succeeded. **Claude's explanation of a game-imposed gate is supported in this scenario.** A third native condition held Y after the first release/repress: subsequent jumps occurred at relative frames 35, 63, 91 and 119 without fresh rising edges. Thus “every next jump needs release/repress” is too broad. The game's later multijump check includes an animation-command gate and held jump buttons. [Pinned Melee multijump implementation](https://raw.githubusercontent.com/doldecomp/melee/02f44f3bdf470ce3bdc8232486b3969f28b8fafb/src/melee/ft/kinds/ftCommon/ftCo_JumpAerialF1.c).

Do not blindly replace the executor with holding Y: that consumes jump resources repeatedly when gates open. Teach or explicitly schedule desired jumps, using native outcome checks. The baseline scratch summary's final jump count returns to six after landing during the observation tail; its event trace, and the count at the end of the six macros, show only the original ground jump. Character-specific action enum names in libmelee are shared/aliased; the raw action IDs here must be interpreted as Puff states.

### Contract boundaries

Changed macro names and changed declared observation layouts were rejected; a missing contract was rejected; genuine legacy name-only files loaded with the expected executor warning. Full restore and `--init` now invoke contract checks. Allowing a warned executor change is an intentional migration policy, not automatically a defect.

However, `upgrade()` assumes **any** contract lacking `observation` is layout v1. A malformed/newer-looking contract with current names but no layout also passes, with only an executor note. Limit automatic migration to a recognized legacy schema; do not assume missing modern metadata proves v1 compatibility. A file with different semantics but identical/unbumped metadata cannot be detected by dimensional checks. Preserve the new version-bump requirement and attach explicit semantics/provenance to future layouts.

Separate declared compatibility from usable architecture (R5). Training's external frozen-opponent loader also ignores returned executor/timing notes and does not resolve timing per opponent; this matters when importing heterogeneous historical policies, even though today's same-lineage snapshots share the vocabulary.

## Recommended step 2

### 1. Close the evidence and data-boundary defects first

Fix R1–R3 before sustained training or expensive evaluation. Add coordinator integration cases for one dead worker with another alive, lost completion acknowledgement, persistent legitimate timeouts, duplicate/delayed results, cancellation, and bounded exhaustion. Add the post-win/pre-reset crash case on both ports. Keep the successful native 4/4 and rejected-cap CLI checks as acceptance scenarios. Repair R4's labels without modifying historical evidence.

Champion gating was a reasonable step-1 deferral only while nothing is promoted. Add it before the next crown: complete qualifying cells, immutable checkpoint identity, validated effective protocol and explicit thresholds. RNG state and full ISO hashing can wait behind these correctness defects, provided reproducibility is described accurately. Reward/horizon changes are best isolated into later controlled experiments after the infrastructure is trustworthy.

### 2. Separate L-cancel assistance from digital tech inputs

**Do not claim that “aerial attack or hitstun now” guarantees no air dodge next frame.** In the native probe, pulses at neutral-air frames 47/48 remained in attack; a pulse at 49 transitioned directly into air dodge. This test occurred above the ground, so it specifically falsifies the state-only guard, not every possible correctly implemented landing predictor. A near-landing condition must also account for animation ending, failed/missed landing prediction and observation-to-input delay. The same boundary problem applies on the last hitstun frame. [Pinned air-dodge input check](https://raw.githubusercontent.com/doldecomp/melee/02f44f3bdf470ce3bdc8232486b3969f28b8fafb/src/melee/ft/kinds/ftCommon/ftCo_EscapeAir.c).

Recommended design:

1. Introduce a dedicated **analog-only L-cancel pulse**, with no digital L/R bit, after verifying its threshold and effective L-cancel behavior against this exact Dolphin/libmelee build. Test that it reduces landing lag and does not become an air dodge when mistimed. This is a proposed mechanism to validate, not a technique certified by the current probes. Avoid casually substituting Z: late Z can invoke other actions.
2. Implement tech attempts as separate one-frame digital pulses, using remaining hitstun, predicted surface contact and lockout/release history. Require a conservative margin beyond the actual observation-to-input delay; suppress inputs near uncertain state transitions. Handle wall/ceiling/ground contacts separately as supported.
3. Recheck guards **every emulator frame in the executor**, including during long macros; do not authorize a held shield sequence once at policy selection. Never carry a digital press through the end of the permitted state. Preserve DI during the pulse.
4. Validate randomized native scenarios around final aerial/hitstun frames, hitlag, knockback, landing/surface uncertainty, ledges and different heights. Record successful L-cancels/techs, missed opportunities and unintended air dodges independently. A finite suite supports a bounded safety claim, not “never” across unknown mechanics.

Keep these inputs explicit and observable to the policy, with executor versioning and actual technique-success telemetry. This is a useful near-term improvement even before replacing the macro policy.

### 3. Remove the automatic mastered-to-60%-Puff rule

**Yes: undo that automatic escalation.** The 4/10 Falco result concerns the frozen **48.17M/v3304** candidate, not the latest policy; it is a small sample, but it plainly does not certify broad CPU-9 mastery. Training logs also identified Falco weakness. A pooled rolling threshold and sticky flag are inadequate grounds to shift most experience into Puff opponents.

Initially retain approximately the pre-escalation 20% Puff self-play as a baseline, not as a proven optimum. Allocate the rest across character coverage and measured weaknesses, without abandoning easier regression opponents. Replace the one-way flag with matchup-specific evidence and revisit the mixture periodically. Compare candidate mixtures under equal accepted learner-data and wall-time budgets. Frozen per-matchup evaluations should drive this decision; the new CPU-1 smoke is irrelevant to Falco strength.

### 4. Integrate Slippi-AI as a measured opponent adapter

The released `medium-v2` model is a promising way to obtain learned play across twelve characters. Upstream documents human-replay imitation and later self-play RL. Its hosted Phillip has 18+ frames of delay; that statement does **not** establish the exact metadata or delay of the downloaded medium-v2 file. Inspect the model and preserve its intended timing. [Pinned Slippi-AI README](https://github.com/vladfi1/slippi-ai/blob/275c07270b5aa5f7b22aabff79dfe7f7b10a385f/README.md).

Build an opponent interface that owns its observation parser, recurrent state, controller output and delay queue. Do not map its controller predictions through Puff's macro executor or try to load its weights as a Puffbot `.pt` policy. Preserve model normalization, action-head semantics, character/name conditioning, resets and frame-skip settings. Upstream's `DelayedAgent` and batching interfaces provide the relevant reference: delay queues subtract console delay, while batched agents retain per-environment state. [Pinned evaluation adapter source](https://raw.githubusercontent.com/vladfi1/slippi-ai/275c07270b5aa5f7b22aabff79dfe7f7b10a385f/slippi_ai/eval_lib.py).

For this M2 Pro, **cost is unmeasured**. Pin the code/model hash, inspect whether the release uses the TensorFlow or JAX path, and use an isolated environment for integration rather than altering this working venv. Start with one frozen opponent and one emulator; warm up compilation outside gameplay/watchdog timing. Verify one character first, then all supported characters, and compare exact input traces with upstream's runner.

Then benchmark 1, 2, 4 and 6 emulators, CPU versus a supported accelerator path, and a shared batched inference service versus per-worker copies. Measure accepted learner decisions per wall second, simulated world frames, policy lag, inference p50/p95 latency, memory and emulator stalls. Keep the existing Puff learner running only in a subsequently authorized training benchmark. More emulators or a faster inference microbenchmark alone do not establish a training speedup.

As a planning calculation, **20× aggregate world speed** requires about 1,200 simulated game frames per wall second. An opponent called every frame therefore needs roughly 1,200 aggregate decisions/second; with frame skip `k`, roughly `1,200/k`. Do not multiply an already aggregate 20× rate by six again. An 18-frame reaction-delay queue does not divide the inference cost by 18. Six emulator processes plus a second ML runtime can contend for CPU, memory bandwidth and the accelerator; use measurements to decide how many learned-opponent slots to keep.

One twelve-character model supplies character diversity, not twelve independently discovered strategies. Preserve other checkpoints, CPU baselines and eventually explicit exploiters; hold some policies/styles out of training. A Puff beating an opponent with a much larger enforced reaction delay is a result under asymmetric rules, not evidence of human-fair or unrestricted superhuman play. Measure deployment observation-to-action delay for both sides and disclose it.

### 5. Expand information and coverage after the adapter works

Projectile velocity/type and a short history baseline should be early follow-ups, especially for Falco. Add missing entities/characters and stage geometry before claiming broad coverage. Compare replay-initialized Puff and the current policy under the same opponent population; then test small recurrence, richer controllers and longer-horizon rewards in separately attributable experiments. Keep final validation separate from training and exploit discovery, including expert human sets at declared input/latency rules. No finite benchmark can establish “beats anyone.”

## Evidence index and limits

All probes used temporary outputs. Native evaluation ports were 55621–55622; the cap evaluation used 55641; ad-hoc jump/shield probes used 55561 and 55571. These are outside the default training range and satisfy the requested ad-hoc offset. Each native probe stopped its own emulator(s).

- [Main synthetic harness](/tmp/puff-step1-review.cxTjI3/probes.py) and [results](/tmp/puff-step1-review.cxTjI3/synthetic-results.json).
- [Secondary contract/accounting/watchdog/input harness](/tmp/puff-step1-review.cxTjI3/secondary_probes.py) and [results](/tmp/puff-step1-review.cxTjI3/secondary-results.json).
- [Multiprocessing coordinator reproduction](/tmp/puff-step1-review.cxTjI3/coordinator_probe.py) and [result](/tmp/puff-step1-review.cxTjI3/coordinator-result.json).
- [Native jump harness](/tmp/puff-step1-review.cxTjI3/native_jump_probe.py) and [results](/tmp/puff-step1-review.cxTjI3/native-jump-results.json).
- [Native shield boundary harness](/tmp/puff-step1-review.cxTjI3/native_shield_probe.py) and [results](/tmp/puff-step1-review.cxTjI3/native-shield-results.json).
- [Native four-game evaluation](/tmp/puff-step1-review.cxTjI3/native-eval/report.json) and [native capped evaluation](/tmp/puff-step1-review.cxTjI3/native-cap-eval/report.json).

Temporary evidence may disappear with system cleanup; the essential inputs, outcomes and failure mechanisms are recorded above. Synthetic failure injection establishes boundary behavior, not field failure frequency. The native tests verify selected controller/runtime paths; they do not reproduce training, demonstrate a learning improvement, certify full defensive safety, or benchmark Slippi-AI on this Mac.

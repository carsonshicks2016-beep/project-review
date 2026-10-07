# Fable Five 919 Ring — Live Training Journal

> **MORNING TL;DR (07:05):** Overnight: 462.13s → **clean 437.73s flying lap** (16/16 sectors, zero terminations — verified in `fable5_919_ring_best.pt`). Flow's 16/16 gate cracked after 4 root-cause fixes; flow/finish/fast all gated; frontier engaged but blocked by a PitWall design gap (rollback restores the value head → resets adaptation every window → infinite loop). **Training intentionally STOPPED at 07:03** — every no-code configuration ends in that loop, so running further was pure churn. Daylight queue, in order: (1) PitWall grace period on regime change, (2) reseed↔plateau clock livelock, (3) physics audit of the hot envelope on the Aremberg/Fuchsröhre descent, (4) real fast-stage pass to bridge the value regime into frontier. Full story in the entries below (newest first after this block).

## Check #25 + Intervention #5 — 2026-07-21 10:20 — DAYLIGHT DIAGNOSTIC: mystery solved, finish capped, training relaunched

### The probe experiment (user-approved daylight plan, executed ~10:00)
1. **Baseline diag** of `fable5_919_ring_best.pt`: FINISHER — clean 437.73s both drop-ins, 16/16 battery, 94.4% envelope pace, 24.0% time-over-envelope. (Side-findings for later: gear head overrides RaceBox 100% of steps at mean −1.02; steering 100 reversals/km.)
2. **Probe 1** (scratchpad-isolated, finish stage, 10 updates): NON-FINISHER — dies at 19.8%, 4/16 battery, envelope pace 94.4→**98.3%**, over-envelope 24→**43.7%**, sector-3 entry 0.88→**1.01×**. Ten updates = systematic speed surge, not random damage.
3. **Root cause**: finish was the ONE stage still at legacy `pace_cap 1.30` (overnight fixes covered flow/fast/frontier). The 06:25 "polish" run was training INTO the cliff; lr-annealing couldn't help because the over-speed gradient is directional. This also retro-explains the whole finish rollback storm.
4. **Fix**: finish `pace_cap 1.05`, overspeed 0.030→0.050, margin 1.00→0.97 (flow's proven recipe, gentler weights). Both gates ALL-PASS.
5. **Probe 2** (finish, 45 updates, capped): FINISHER preserved — metrics identical to baseline. **Probe 3** (frontier at scale 0.90, 45 updates, existing 1.06 cap): FINISHER preserved too.

### Consequence for the roadmap
Every stage's one-window training is now verified non-destructive. Last night's "PitWall grace period = required" conclusion is DOWNGRADED to contingent: the rollback storms were downstream of the uncapped finish/frontier rewards, at least in large part. The cheap decisive test is live training: **auto relaunched (PID 53816, log `fable5_919_run_20260721_1020.log`)**. If frontier now holds/improves the 437.73s lap → grace period unnecessary; if 3+ consecutive rollbacks recur → it moves back to required, with the storm as fresh evidence. Monitoring back at 20-min cadence.

## Check #24 — 2026-07-21 07:03 — FINAL: polish stopped, night closed out

- Exit condition from Check #23 fired: rollbacks 12/12 by 06:56, reseeds churning, no clean lap since 06:29 (the 444.07). Worse, evals DEGRADED as lr annealed (7.3 → 0.3 at lr 1.17e-5) — tiny updates still demolish the policy, which wounds the value-refit theory as a complete explanation. Something in the finish/frontier training loop breaks the restored policy in ways not proportional to update size — genuinely unresolved; measure before theorizing further (a `--fable-diag` comparing the banked brain before/after one training window would discriminate policy drift vs env/normalization mismatch — do this in daylight FIRST, it may reframe fix #1).
- Killed PID 49878. No fable processes running. Champion integrity verified by direct checkpoint read: stage=finish, lap 437.7333, 16/16 clean, terminal 0.0.
- Monitoring switched to observation-only (no auto-restart) — restarting into a known-futile loop fails the "forward progress" mandate better honored by a quiet machine and a clean handoff.



Run: `run.py --fable 49901 --fable-stage auto --car porsche_919evo --workers 7 --pop 7 --out fable5_919_ring.pt` (PID 40882, started 23:12).
Stage: **flow** (resumed from `fable5_919_ring_foundation_best.pt`, envelope_scale 0.86).
Targets: Bellof 371.13s · 919 Evo record 319.55s · theoretical 400.2s. Foundation banked a 462.13s flying lap.

Newest entries first within each check. Only deterministic `[eval-fable]` numbers are trusted.

---

## Check #23 — 2026-07-21 06:41 — finish polish: rollback-heavy but functional (unlike frontier)

### State (PID 49878)
- The rollback loop exists here too (bar 236.45; rollbacks 5/12 in 14 min, lr annealed 1.09e-4 → 3.63e-5) — but with a decisive difference from frontier: **06:29 produced a fully clean 444.07s lap (16/16 chain, terminal 0.00, metric 233.2)** — within 1.5% of the bar. Frontier's windows never got above ~2; finish's windows oscillate between "slightly damaged" (evals 20-ish with 23–45 km continuous runs) and "champion-grade clean lap." The value gap here is small enough that the ladder's lr-annealing stabilizer has something to stabilize.
- Interpretation: this is the eval_every-30 design working as documented for the fast stage ("catch damage while it is still small") — each rollback shrinks the update step; the expected endpoint is windows that stop losing the lap, then micro-polish. A sub-437.73s clean lap banks a new best and refills everything.
- Reseeds, if reached, restore THIS run's HOF (which contains the 444.07 brain) — good restore points, unlike frontier's turbulence-era HOF.

### Decision: hold. Watch ~07:00
- Healthy: rollback burn rate slowing as lr anneals; any clean lap ≤444 recurring; ideally a <437.73 bank.
- If budget 12/12 + reseeds also churn without a single clean lap after ~07:00, stop the polish experiment and just leave auto down until the user wakes (champion safe; write final handoff) — at that point every no-code lever is exhausted and the PitWall redesign is simply the prerequisite for further progress.

## Check #22 — 2026-07-21 06:25 — frontier confirmed stuck; switched the night's compute to finish-stage lap polish

### Verification result (Check #21's "not stabilizing" branch confirmed)
The self-stabilizer never materialized: rollback budget went 12/12 before 06:11, and calm reseeds (2/6) now restore *turbulence-era* HOF brains (terminal 0.50–0.72) — strictly worse restore points than the best. Evals pinned at −1.6 to 2.0. Also decisive: rollback restores the **value head too**, so each cycle resets the value re-fit to zero — no accumulation is possible. The frontier rollback↔refit loop cannot exit by itself; the PitWall grace-period fix is mandatory before frontier can train. That is daylight debt #1, not 6am surgery on the safety system (its validator explicitly encodes the opposite behavior — "degraded first eval → rollback" — so the fix needs a new test case and a design decision, not a hotfix).

### Action: repurposed the remaining night (no code change)
Killed the auto run (was burning compute in a confirmed-futile loop) and launched a **manual finish-stage polish run**: PID 49878, `--fable-stage finish --resume fable5_919_ring_finish_best.pt`, log `fable5_919_run_20260721_0625_finishpolish.log`. Manual stage runs don't stop on gate, so finish trains *through* its met gate — its reward (lap_pace_bonus scaled by theoretical/actual) directly pays for faster clean laps, its value gap from the banked brain is the smallest available, and its long episodes + closure drills match the banked behavior. Best case: the 437.73s champion gets shaved down all morning. Worst case: PPO plateau (patience ~350 updates) exits the process and the next check's restart falls back to auto — nothing lost, champion promotion-protected throughout.
- Caution noted at startup: pit armed at bar **224.6** (lap-dominated metric), so if finish shows frontier-grade turbulence the same rollback loop could appear here. First evals will discriminate; finish's regime gap is far smaller, so recovery within the normal 15–25 min window is the expectation.

### Morning handoff summary (for the user)
- **Banked and safe: clean 437.73s Nordschleife flying lap** (was 462.13 at midnight; goal 371.13 Bellof / 319.55 record). Flow finally gated 16/16; finish/fast gated; frontier engaged but blocked by the pit-wall design gap.
- Pipeline fixes shipped tonight (all gate-validated): per-stage `pace_cap` (flow 1.05, fast/frontier 1.06), `FABLE_LOCAL_VREF_DERATE` + braking re-pass (sector 3 ×0.80, sector 1 ×0.94, sector 15 ×0.90; theoretical 415.5s), flow overspeed 0.065, seed_bias_cap 5.0 + sector_fail_gain 2.5.
- Daylight work queue: (1) PitWall grace period / windowed bar on regime change — REQUIRED for frontier; (2) reseed↔plateau clock livelock; (3) physics audit of why the envelope runs hot on the Aremberg/Fuchsröhre descent; (4) consider a real fast-stage training pass (it gated instantly, so no value-regime bridge to frontier exists).

## Check #21 — 2026-07-21 06:01 — pace_cap hypothesis FALSIFIED for frontier; real culprit is rollback-vs-refit

### What the restart showed
Re-cert cascade worked perfectly again (all four stages re-gated in ~6 min; 228.45/437.73s re-banked at frontier entry, scale 0.90). Then rollbacks 1–5/12 burned in 8 minutes — **but eval pace is now 0.98–1.04, comfortably under the cap.** The cap binds; the collapse continues anyway. Intervention #4's frontier half didn't fix this (it's still right as reward hygiene for when the scale ladder climbs, but it wasn't the active failure).

### The actual mechanism (and why it's the flagged pit-wall design gap)
The restored brain is FINE — every restore re-evals clean (the 05:52 baseline drove 437.73s at these exact targets). ~30 updates later it's broken again. This is value-function re-basing: frontier's reward regime (pace 0.16, lap_bonus 100, no milestones) is far from finish's, the inherited value head misprices everything, advantages are garbage, and updates are violent despite the KL guard. Normally the fix is time — every regime shift tonight cost 15–40 min of turbulence before recovery. But frontier's pit bar is 228.45 (lap-dominated metric), turbulence evals score ~0, so **ROLLBACK fires every single eval window and resets the value re-fit before it can complete.** The safety system and the learning process are in a tight loop where each undoes the other. (Flow had the same disease at bar 40; frontier's bar is 5× harsher so the loop is airtight.) Root cause of tonight's third livelock variant, and tonight's fast stage was gated instantly at baseline — so unlike the historical 787B path, no fast-stage training ever bridged the value regime gap.

### Why HOLD is still the right call (not another 6am surgery)
The rollback ladder has a built-in stabilizer: each restore narrows log-std (−0.10 per step) and anneals lr (already 2.18e-05). By rollback ~8–10 the updates become tiny and the near-deterministic policy evals ≈ the banked best — which re-banks a ~228 metric, refills the budget, and converts the loop into a slow calm-polish regime. That's a legitimate convergence path visible in the ladder's own design. Champion 437.73s is promotion-protected either way; nothing is being lost, only time. Touching the PitWall arming logic (grace period on regime change / windowed collapse bar) is the *correct* fix but it's surgery on the safety system — daylight work with the user awake, listed as pipeline debt #1.

### Verify at ~06:20
- Stabilizing: rollbacks 8+ with evals climbing back toward 100–228, or a re-banked best (budget refill).
- Not stabilizing: budget 12/12 again + reseeds burning + evals still ~0 → journal it as confirmed-stuck; even then the right move is likely "leave it polishing, hand the PitWall fix to daylight" rather than more 6am reward edits.

## Check #20 + Intervention #4 — 2026-07-21 05:50 — frontier pace_cap; the cliff mechanism closed everywhere

### Why the 06:00 condition fired early
Rollbacks hit **12/12 by 05:31** (lr annealed to the 1e-05 floor, log-std to −1.20) and calm reseeds started burning (2/6) — every restore was destroyed within one eval window. Evals 0.0–1.2, terminal up to 1.00, offtrack up to 115s. Segment scale can't ease for hours (segment 1 = 12,400 iters and reseeds keep resetting the plateau clock — the same livelock as flow). Frontier IS at the right scale (0.90, correctly seeded from the banked brain) — the problem is purely the legacy reward: `pace_cap 1.30` at scale 0.90 pays incentive to 1.17×raw against a 0.008 overspeed counterweight. The policy that proved ~0.90–0.93×raw gets dragged to 1.10+ every update window. Nothing the pit wall does can hold against a reward optimum on the far side of the cliff — tonight's recurring lesson, now at the last stage.

### Change (both gates ALL-PASS)
`pace_cap 1.06` for **fast** and **frontier** (was legacy 1.30). Speed pressure now lives entirely in the proven-mastery scale ladder: at the 1.15 scale ceiling, mastery still unlocks incentive up to ~1.22×raw — where superhuman lives — one earned rung at a time, instead of an unconditional cliff jump. This closes the cliff mechanism at every stage of the pipeline.
- Restarted: **PID 49050**, log `fable5_919_run_20260721_0550.log`. Expected: gated stages re-certify off their banked bests (~10 min — flow's 16/16 and the 437.73s lap are deterministic replays of unchanged envs), then frontier resumes at 0.90 with a reachable target.

### For the morning (~06:10+ checks are observation-only)
- Healthy frontier now looks like: laps in the 430–450s band re-banking, terminal low, scale easing/holding at 0.90–0.93, occasional mastery pushes.
- The night's structural ledger: pace_cap plumbing (all stages), local envelope derates + braking re-pass, flow overspeed wall, sector-rehearsal strengthening (seed_bias_cap/sector_fail_gain). Known remaining pipeline debts (daylight work): reseed↔plateau livelock, pit-bar re-arm on regime change, and the sector-3 zone deserves a physics-level look (why is the computed envelope ~20% hot on the Aremberg/Fuchsröhre descent?).

## Check #19 — 2026-07-21 05:21 — THE LADDER CASCADED: flow ✓, finish ✓, fast ✓ — now in FRONTIER with a clean 437.73s lap banked

### What happened between 05:01 and 05:12
The tune's turbulence resolved exactly on the +25 min schedule, and then everything the night was building toward landed at once:
- **Flow gated: 16/16 clean chain** (the gate that was unreachable for six hours).
- **Finish banked a fully clean flying lap: 437.73s** — 16/16 clean, terminal 0.00, offtrack 0.00, pace 1.03. That's 24s faster than the old 462.13 and CLEAN under Ring rules.
- **Fast gated instantly** (clean lap ≤450s ✓) — stage=fast eval at 05:10, stage=frontier by 05:12.
- Metric 228.45 banked as best. The 04:45 tune (overspeed wall + derate cuts) is vindicated: the spins it targeted were the last thing between chain-13 and chain-16.

### Now: frontier is fighting the expected knife-edge
Frontier demands scale 1.00 (vs the ~0.86–0.90 the lap was driven at) and still carries the legacy pace_cap 1.30 (deliberately untouched — speed pressure belongs here). Result: the first training updates lose the lap (evals 3.4, 7.0, 3.4, −1.4; offtrack 62–214s), and the **ROLLBACK ladder is doing its job: 4/12 spent**, each restore narrowing noise (−0.10…−0.40) and annealing lr (4.18e-5 → 2.57e-5). This is the purpose-built frontier machinery (post-KLGUARD design); it has 8 rollbacks left, then consolidate → segment end → the auto ladder's adaptive scale eases. Holding — intervening in frontier's designed push-and-catch cycle 10 minutes in would be meddling, not fixing.

### Watch (~05:40)
- Healthy: rollbacks stop burning (a frontier eval re-banks a lap, even ~437s), or the segment ends and scale eases toward 0.95.
- Concern worth acting on ONLY if it persists past ~06:00: rollback budget exhausted AND no lap re-banked AND scale not easing — then the frontier pace_cap (1.30) deserves the same evidence-based look flow got. Not before; the 787B program historically climbed exactly this way.
- For the morning: 437.73s clean is tonight's headline. Bellof (371.13) needs another ~67s — that's frontier's job over many hours/days, not tonight's.

## Check #18 — 2026-07-21 05:01 — post-tune turbulence, rougher than the last two

### State (PID 47802 healthy, flow re-entered, bar re-banked at 40.147)
- First 13 min under the tune: 7.9 → 4.5 → 15.0 → 14.0 → 9.5 → 10.8 → 2.4. Reseeds 1/6 (04:51) and 2/6 (04:58) already spent. Comparable trajectory to the 03:10 restart's first phase (7.2 → 28.1 took ~12 min) but with an ugly new feature: **off-track seconds exploding** (27, 43, 55, 82s per eval vs ≤12 before). The policy is riding off the road for long stretches without terminating — line quality collapsed, likely the harder overspeed gradient causing over-braking/line oscillation while the old habits unwind, compounded by targets that dropped again in the derated zones.
- Reseed 2/6 just restored the cleanest HOF brain; that's the right lever at the right time.

### Assessment — hold one more cycle
Every restart tonight looked broken at +13 min and healthy by +25. The tune is 15 min old; judging it now would repeat the mistake the pit wall makes. But the offtrack blowup is a distinct signature worth a hard tripwire.

### Tripwire (~05:20) — judged on the 05:10–05:20 window only
- Recovering (evals ≥20, offtrack back ≤10s): hold to gate-watch.
- Still degenerate (evals <20, offtrack ≥20s, reseeds 3+): the overspeed 0.065 wall is the prime suspect — revert to 0.050 (keep both derate cuts; they're evidence-backed independently), gates, restart. That would demote this from "final tune" to "overshoot corrected" — cheaper than letting a destabilized reward burn the rest of the night.

## Check #17 + Intervention #3 — 2026-07-21 04:45 — final tune of the night: overspeed wall + derate cuts

### Evidence at 04:41 (the planned decision point)
- No gate, no chain >13 (three more 10–13 chains). But the failure detail CHANGED the diagnosis: latest eval's failures were **spins at pace 1.054–1.089 in ordinary sectors (0, 1, 15)** — over-driving past target, not under-enveloped corners. With pace capped at 1.05, the remaining uncapped speed incentive is `progress_per_m` (envelope-blind), and the old overspeed weight (0.040) penalized the 1.08–1.13 band by less than the progress term paid — the equilibrium sat just past the spin threshold. Meanwhile sector 3 heat stayed top-by-2× (27.2) after 90 min at 0.85, and sector 15 spun at ~0.87×raw.

### Changes (bundled into ONE restart to pay turbulence once; both gates ALL-PASS)
1. Flow `overspeed` 0.040 → **0.065**: steeper wall past 1.081×target pulls the operating point under the spin threshold. (Flow-only; fast/frontier speed pressure untouched.)
2. Sector-3 derate 0.85 → **0.80**; sector-15 0.92 → **0.90**. Theoretical lap 412.2 → **415.5s**.
- Restarted: new **PID 47802**, log `fable5_919_run_20260721_0445.log`. This is intended as the LAST intervention of the night — from here the run should gate flow on its own; remaining checks are observation unless something breaks outright.

### Expectations for the rest of the night
- ~15 min re-cert + turbulence, then flow with: honest targets everywhere, no incentive past 1.05, real penalty past ~1.08. Per-sector clean probability should rise enough that chains stop capping at 13.
- Success = flow gates (16/16 chain) and AUTO enters finish with the 443s-class HOF brains. Even without the gate, attempt 2 → soft-advance eventually hands finish a far stronger brain than existed at midnight.

## Check #16 — 2026-07-21 04:21 — consolidation band; chain ceiling ~13; sector-3 tune pending

### State (PID 45672 healthy)
- Stable band: evals 17.6–36.9, chains 4–13 (another chain-13 / terminal-0.00 / 2.3-lap eval at 04:06). No new best in ~70 min (bar 40.234), no gate. Reseeds quiet since 2/6.
- Heat: sector 3 = 23.4 (creeping up, top by 2×), sector 15 = 11.8 (back a little), sector 1 fading. Sector 3 appears in almost every worst-list; chain records stalled at 13 since 03:58.
- Pace reads 1.10–1.14 sustained — partly mean-inflation from derated zones (the overspeed onset scales with the derated vref, so the incentive structure is consistent; not a red flag by itself).

### Assessment
This is the "chain capped by sector 3" branch of the 05:00 decision taking shape: chains hit 13 repeatedly but nothing above since 03:58, and sector 3 is the sole persistent offender. One caution before committing to 0.80: foundation used to fail sector 3 even at 0.73×raw spawns (old code, so not conclusive) — if the corner breaks entries even ~25% under the computed envelope, the derate factor needed might be lower still, or the failure is about entry *line* (lateral spawn placement) rather than speed alone. Next check: pull `worst_sectors` pace values for sector-3 failures under the current regime to size the tune properly instead of guessing.

### Plan (~04:40)
- If gate or chain ≥14: hold, obviously.
- Else: read sector-3 failure paces from eval_latest; if they're ≥0.95 of the *derated* target, size the derate cut accordingly (0.80 or lower) + sector 15 → 0.90, run gates, single restart. That would be the last planned intervention of the night; after it, the run should have everything it needs to gate flow and hand finish a 443s-class brain.

## Check #15 — 2026-07-21 04:01 — chain 13; multi-lap runs; tripwire NOT triggered

### State (PID 45672 healthy)
- **03:58 eval: chain 13, 13/16 clean** — new chain record (gate needs 16). Metric 36.2, closing on the 40.234 bar.
- **Multi-lap continuous runs**: two evals show progress 47,502–47,625 m with terminal 0.00 — that's 2.3 laps without a single termination. The endurance problem is gone.
- Sector 15 resolved on its own (heat 17.7 → 8.3, vanished from recent worst-lists) — its 0.92 derate was enough once the policy adapted. Sector 3 heat back up to 21.9 and still the most frequent single failure, but no longer exclusive (5/6 appear as ordinary noise now).
- Reseed 2/6 at 03:46 during a mid-window dip; recovered without further levers.

### Tripwire evaluation → HOLD
Condition (b) failed decisively: chain 13 > 12, and sector 15 self-resolved (condition (a) half-gone). Per the rule, no derate tune, no restart. The one remaining question is whether sector 3 at 0.85 lets a 16/16 chain through eventually or caps us at 13–15. Evidence either way within the next few checks: if chain keeps setting records while sector 3 stays the sole repeat offender, the 0.80 tune becomes justified at ~05:00; if a 16/16 chain lands, flow gates and the question is moot.

### Watch (~04:20)
New best (>40.234), chain 14+, or the flow gate itself. Also watch for `finish` stage entry lines in the log if the gate lands.

## Check #14 — 2026-07-21 03:41 — FLYING LAPS ARE BACK: 478.0s, then 443.1s

### State (PID 45672 healthy)
- **03:33: full eval lap, 478.00s** (13/16 clean, terminal 0.19). **03:35: 443.07s** — 13/16 clean, terminal 0.12, only 0.89s off-track, full 20,832 m. These are *driven* laps under the new envelope, not replays — and 443.07 is ~19s faster than anything the 919 program has ever produced (old best lap 462.13). The policy is faster AND more complete than pre-derate.
- Neither banked as stage best (metric 30–33 vs bar 40.234 — the flow metric weighs chain/cleanliness, and 0.89s off-track invalidates "clean lap" status under Ring rules), but they land in the HOF lap/progress slots, which is what finish will inherit.
- Oscillation persists between strong evals (33.8 with chain 12; 32.6; 30.1) and weak ones (11.9, 17.0) — normal exploration breathing, reseed budget quiet since 1/6.
- Heat: sector 3 **down** 21.0 → 17.7; sector 15 **up** to 17.7 (co-leader now); sector 1 at 9.3. Latest fails: 3×sector-3, 3×sector-15, 2×sector-1.

### Assessment
The envelope fix flipped the regime: from "no continuous run past 4.1 km for three hours" to "two full flying laps in ten minutes, one at 443s." Chain record 12. Holding — restarts cost ~5–8 min of re-cert plus turbulence, and the trend is up.

### Refined tripwire (~04:00)
Intervene (derate sector 3 → 0.80 AND sector 15 → 0.90, one restart) only if BOTH: (a) sectors 3+15 still jointly dominate fail_sectors, AND (b) no new best and no chain >12 since 03:41. If either improves on its own, keep holding — single-eval fail counts are too noisy to act on alone while laps are flowing.

## Check #13 — 2026-07-21 03:21 — new-envelope flow climbing fast; chain record

### State (PID 45672 healthy, ~1250 sps, KL guard visibly working)
- Restart chain went exactly as designed: foundation re-gated instantly, flow re-entered, best re-banked at **40.234** under the new envelope, flow attempt 1 training.
- Six evals into the new regime: 7.2 → 7.4 → 17.7 → 10.0 → 19.8 → **28.1**. Reaching ~28 took this recovery **10 minutes** vs 40+ for the two previous regime shifts (the value-function re-fit turbulence is getting cheaper each time — smaller semantic gap now).
- **clean_chain 9** at 03:19 — the longest chain of the entire night (previous max 7). Eval pace reads 1.09–1.12, which is expected inflation: the same driving measures higher against locally lowered targets; don't compare pace across the envelope change.
- Sector 3 still leads fail_sectors ([3,3,3,15,15,15,3], heat 21.0) but now with 15% slower spawns — and the failures coexist with the best chain so far. Sector 15 (×0.92 derate) is now nearly co-equal (heat 13.6) — may need its factor nudged to ~0.90 if it persists.

### Assessment
Healthiest 20 minutes of the night. No intervention; let it climb. The derate + pace_cap combination is doing what each alone couldn't: honest targets in the hard zones, no incentive past the grip cliff elsewhere.

### Watch (~03:40)
- New best > 40.234, chain pushing 12+, sector-3 fail count shrinking below 3 per eval.
- If sector 3 STILL fails 3–4× per eval by ~04:00 despite the 0.85 derate, drop the factor to 0.80 (single-constant change, gates, restart — cheap now that the pattern is proven).

## Check #12 + Intervention #2 — 2026-07-21 03:15 — local envelope derate shipped; run restarted

### Why now (attempt 2 disproved the last alternative)
Attempt 2 at scale 0.81 did NOT recover like attempt 1 did: 20+ min of evals at 7–20, reseeds 2/6 and 3/6 already burned, pace pinned 1.09–1.16 *against the lower target*. That pinning was the tell: absolute speed equilibrates just under the overspeed onset (0.93×raw vref) regardless of stage scale — so easing the scale changes the measured ratio but NOT the driven speed. Meanwhile sector 3 kept failing at every scale tried tonight (0.73–0.86, i.e. down to ~0.71×raw), while twelve other sectors ran clean at 0.92×raw. Conclusion, now proven from three directions: **`fable_vref` is locally ~15% optimistic in the sector-3 zone** (the Aremberg/Fuchsröhre descent — compression + crest territory) and the chain gate spawns cars at that wrong target speed. No training time fixes a wrong reference.

### The change (validated, both gates ALL-PASS)
`FABLE_LOCAL_VREF_DERATE` in `supra/fable5.py`: named per-track derate zones applied inside `attach_envelope` — nordschleife: sector-3 zone ×0.85 (arc 0.170–0.260), sector-1 ×0.94, sector-15 ×0.92, cosine-ramped ~250 m; plus a conservative backward braking pass (0.8× brake authority, no drag credit) so the dips stay reachable. Effects: theoretical lap 400.2 → **412.2s** (honest price); eval sector spawns in hot zones start ~6–15% slower; the obs pace block now tells the policy the truth ahead of those corners. Car-independent (keyed on track) since the 787B heat tables showed the same sector-3 signature. Tuning knob: the 0.85 factor — if sector 3 *still* dominates heat after an hour under the new envelope, lower toward 0.80; if it goes fully clean and pace there sits ≪1.0, raise toward 0.90.
- Restarted: killed 42758, **new PID 45672** (log `fable5_919_run_20260721_0310.log`). Envelope-ready line confirms 412.2s. Foundation re-certifying (fingerprint), then flow attempt 1 fresh — with honest targets AND the pace_cap fix together for the first time.

### Watch (~03:30)
- Foundation re-gate + flow re-entry; flow baseline eval numbers under the new envelope (pace ratios will read slightly higher vs lower local targets — don't compare metrics across the envelope change).
- The decisive signal: **sector 3 heat and fail counts.** If sector 3 drops out of fail_sectors, the flow chain gate is finally reachable and the rest is consolidation.

## Check #11 — 2026-07-21 02:41 — new best banked; attempt 2 at scale 0.81 now live

### The eventful 20 minutes, in order
1. The 02:17 climb didn't hold — three weak evals (19.4, 22.4, 16.5) → **CALM reseed 6/6** at 02:33 (budget exhausted).
2. 02:35: pit wall's next lever fired for the first time tonight — **CONSOLIDATE** (restored the best brain, halved lr to 6.14e-05, cut entropy).
3. 02:37: **new best banked — metric 40.585, lap 462.13s, 15/16 clean, terminal 0.062, progress 1.0.** Caveat: the lap time is identical to the stage-entry lap to the millisecond — this is the *restored* brain deterministically re-evaluating, re-banked with a slightly better metric. Not new capability, but it legitimately refilled the reseed/rollback budget (by design: "budget refills on a new best").
4. Right after, flow **attempt 1 ended (plateau) and attempt 2 started at eased scale 0.81** (manifest scale now 0.81, inflight attempt=2, resumed from the flow best). The livelock prediction from Check #9 played out exactly: reseeds stopped → patience matured within minutes.
5. First two attempt-2 evals are catastrophic (6.1, 5.2 — 0/16 clean, terminal 0.62–0.75, pace 1.10–1.11): expected — attempt 2 resets exploration noise wide and re-bases the reward on a lower target (value-function mismatch again), the same turbulence pattern as 00:55, which resolved before.

### Assessment
No intervention. This is the pipeline working as designed, finally: consolidate → re-bank → plateau → eased-scale attempt. At scale 0.81 the pace cap pulls the operating point to ~0.85×raw-vref (slower than the crash zone), and the gate (16/16 chain, pace ≥0.55) is much more reachable — sector 3's local target drops ~6% too. Expected timeline based on the attempt-1 pattern: 15–30 min of turbulence, then consolidation at the lower target.

### Tripwire (~03:00)
- Healthy: evals recovering through the 20s–30s, pace vs the new target settling ≤1.06, clean sectors climbing.
- Intervene only if attempt 2 is still posting 0/16-clean evals at 03:20 (40 min of turbulence = something else is wrong — first suspect would be exploration noise reset too wide for race-pace starts, `log_std` handling on attempt restarts).

## Check #10 — 2026-07-21 02:21 — consolidating; clear upward trend, tripwire relaxed

### State (run healthy, PID 42758)
- The adaptation is finally visible in the trend: 25.3 → 28.0 → 28.9 → 21.4 → 24.3 → **37.6** → 27.4. The 02:17 eval hit **14/16 clean, terminal 0.125, pace 1.025** — a whisker from the 39.999 bar and qualitatively the best driving since flow began (old best was 15/16 but that was the stage-entry baseline, pre-turbulence).
- Two more full-lap continuous runs (progress 1.0 at 02:07, 02:09). No reseed since 5/6 at 01:56 — 24 min of "holding near best" instead of dry-spell decisions.
- Pace has settled into **1.03–1.08**, mostly ≤1.06: the cap is now expressed in behavior, as predicted ~90 min ago.
- Heat table is stark: sector 3 = **32.5**, sector 15 = 7.7, sector 1 = 4.8, *all twelve others ≤1.0*. fail_sectors this eval: [3,3,3,3] — literally only sector 3.

### Assessment
Holding was right; the turbulence resolved into consolidation. The 02:40 "force attempt 2" tripwire is now WRONG to fire — ending attempt 1 while the trend climbs toward a new best would throw away the adaptation. A new best (>39.999) would also refill pit levers and reset the plateau clock legitimately.
- Flow's gate (16/16 chain) still runs through sector 3, which keeps failing at 0.83–0.94 of local target — the envelope-optimism evidence from Check #9 stands. If the trend banks a new best but the chain stalls at 14–15/16 for a long stretch with only sector 3 failing, the *daylight* move is still the local vref derate, not more training time.

### Revised tripwire (~02:40)
- Healthy continuations: new best banked, or clean ≥14 repeating, or chain ≥10. → keep holding.
- Intervene only if the trend clearly reverses (two consecutive evals <15 after 02:20) AND reseed 6/6 fires — then the eased-scale path (attempt 2 / `--fable-scale 0.81`) is back on the table.

## Check #9 — 2026-07-21 02:01 — tripwire met, but restart would be WRONG; revised plan

### State (run healthy, PID 42758 at 100%)
- Variance has collapsed into a stable mediocre band: last 8 evals 14.0–25.3, clean 3–10, terminal steady 0.25, pace 1.05–1.10. No new best (73 min now), no valid lap, and the full-progress runs did NOT repeat — progress back to ~0.20 for 8 straight evals. **CALM reseed 5/6 at 01:56.**
- Heat is now maximally concentrated: **sector 3 = 32.3**, sector 15 = 10.5, sector 1 = 4.9, everything else ≈ 0. The policy has essentially cleaned up the whole track except one corner complex.

### Why I'm NOT restarting (revising Check #8's plan)
Two realizations on closer inspection:
1. **A restart would reset flow to attempt 1 with a full budget** — the opposite of what we want. The valuable next event is flow *attempt 2 at eased scale 0.81*, which only happens when attempt 1 ends via plateau.
2. **Why hasn't plateau fired after 73 best-less minutes (patience ≈300 updates ≈ 20 min)?** Almost certainly because each CALM reseed resets the plateau counter (`ppo.py`: reseeds bump the plateau bookkeeping). Pit wall reseeds every ~15 min → plateau never matures → **attempt 1 is livelocked by the pit wall's own rescue lever**. Reseed 6/6 is imminent; after that no more reseeds can fire, patience finally runs uninterrupted, and attempt 1 should end ~20–25 min later (~02:30), THEN attempt 2 at 0.81.
- **Pipeline fix candidate for the future (do in daylight): the calm-reseed budget should also feed the plateau clock, or a dry spell longer than N minutes should force stage-end regardless of reseeds.** This livelock is a genuine autonomous-training gap — tonight it costs ~40 min; on an unattended weekend run it could cost days.

### New structural evidence on sector 3
Sector-3 eval attempts fail at pace **0.83–0.94** — meaning the car crashes there while driving 6–17% UNDER the local target, while 1.05+ is fine nearly everywhere else. That is strong evidence the **envelope (fable_vref) is locally optimistic in sector 3** (likely a crest/compression zone ~4.1–4.5 km — grip assumptions break on vertical load changes). RL cannot fix an undriveable reference; it can only learn to ignore it, slowly. Candidate daylight fix: heat-driven per-sector envelope trim (vref *= 1−k·heat_norm), or a physics pass that derates vref where |dz/ds| changes fast. This also explains why foundation (scale 0.73–0.78) never struggled: 0.78×optimistic ≈ still driveable; 0.86× is not.

### Plan
- Hold through reseed 6/6 and the plateau. **Tripwire: if by ~02:40 the run is still in flow attempt 1** (no "attempt 2/2" line, no scale change in manifest), the livelock theory is wrong in a way that matters — then kill + relaunch with `--fable-scale 0.81` (run.py:146 confirms the flag) to force the eased-scale regime directly.

## Check #8 — 2026-07-21 01:41 — full-lap continuous runs are back; holding

### State (run healthy, PID 42758 at 99% CPU; still flow attempt 1)
- **The last two evals completed the full lap distance** (progress_frac 1.0 at 01:37 and 01:39) — that's the first time since flow began that continuous runs go all the way around, and it happened twice in a row. No valid lap time yet (off-track moments invalidate), clean sectors low (7, then 2).
- Sequence since reseed 4/6 (01:27): 8.8 → 11.6 → 20.6 → **27.9 (12/16 clean)** → 18.3 (progress 1.0) → 11.2 (progress 1.0). Pace band drifting down: recent evals 1.056–1.093 vs the earlier 1.10–1.165 spikes.
- Plateau stage-end hasn't fired yet; reseeds at 4/6.

### Assessment vs the tripwire
Tripwire said intervene if "reseeds 4–6 spent AND everything <30". Not met: 27.9 with 12 clean, plus two full-distance runs, are qualitative capability gains the metric undersells (the metric leans on clean sectors + chain, which lag while the policy re-learns line discipline at the new pace). The value function has visibly re-fit — evals are no longer uniformly catastrophic. Holding again; the system is converging, just noisily.
- What the policy needs now is exactly what it's getting: reps. The two failure modes left are sector 3 (heat 23.8) and sector 15 (12.2) at 1.05+ pace, plus off-track touches that invalidate otherwise-complete laps.

### Tripwire for next check (~02:00)
- Want: a new best (>39.999), OR a valid lap time banked, OR clean sectors ≥13 in any eval. Any of those = the turbulence resolved into progress.
- If instead reseeds hit 5–6/6 with all evals <30 and no full-progress repeats: restart the run (re-arms pit bar from checkpoint, clears dry-spell state) — same lever as before, still the smallest one that helps.

## Check #7 — 2026-07-21 01:21 — transition turbulence; holding, tripwire set

### State (run healthy, PID 42758; evals every ~2–3 min)
- No repeat of the 0.97-progress breakthrough. Since 01:02: 16.9, 7.5, 12.3, 7.6, 8.2, 8.5, 5.3, 12.9 — worse on average than the old-reward oscillation. **CALM reseed 3/6 at 01:20.** Measured pace mostly 1.066–1.097 (still above the 1.05 cap), though the newest eval finally reads 1.046.
- Heat: sector 3 easing (21.9), sector 15 climbing (13.0), sector 1 at 14.8.

### Diagnosis — why I am NOT intervening yet
This looks like reward-transition turbulence, with the pit wall making it worse in a specific, bounded way:
1. The loaded value function was fitted to old-reward returns; until it re-fits, advantages are noisy and updates chaotic. Behavior pace (eval is the deterministic policy) only drifts down as fast as the KL-trust-region (0.06) allows.
2. **Reseed whiplash**: every calm reseed restores an old-regime HOF brain (banked under the old reward), partially undoing the adaptation, then the next evals look weak again. The pit wall's dry-spell machinery is calibrated for "exploration wandered," not "reward regime changed" — a genuine pipeline design gap for autonomous operation (worth a future fix: on reward/fingerprint change, re-arm the pit bar from the first NEW-regime eval instead of the old 39.999).
3. Self-correction is imminent anyway: flow's in-run plateau patience is `max(3*eval_every, 600*8/n_envs)` = ~300 updates ≈ 10 evals with no new best. Last best was 00:49, so plateau stage-end should fire around now, and AUTO then runs **flow attempt 2/2 at eased scale 0.81** — a lower target that the current policy's speed satisfies more easily, which is plausibly exactly the consolidation regime it needs. After that, soft-advance to finish (which only needs the already-banked 462s lap policy).

### Tripwire for next check (~01:40)
- If attempt 2 (scale 0.81) is running: judge it fresh — want pace ≈1.0–1.05 vs the *new* target, clean sectors ≥10, and a new best.
- If still attempt 1 with reseeds 4–6 spent and everything <30: intervene. Planned lever, in order: (a) confirm plateau/patience isn't stalled (read pipeline `auto` state), (b) if the pit wall keeps reseeding old-regime brains, restart the run — a fresh process re-arms the pit from the resumed checkpoint's stored eval and clears the dry-spell counters, giving the capped reward an uninterrupted adaptation window.

## Check #6 — 2026-07-21 01:01 — first breakthrough past sector 3

### State (run healthy: PID 42758 at 98% CPU, evals every ~2 min)
- Flow re-entered under the new reward at 00:49 (re-banked best 39.999). Then: two ugly evals (6.3, 6.4 — 0/16 clean, pace 1.071/1.080, progress ~0.16) → calm reseed 1/6 at 00:59 → **22.456 with progress_frac 0.97**.
- That 0.97 is the headline: in ~80 minutes of old-reward flow, continuous progress never once exceeded ~0.21 (the sector-3 wall). First eval after the reseed under the capped reward drove **97% of a continuous lap**. Sector-3 heat also easing: 29.3 → 24.1.
- fail_sectors now [3,3,15,15] — sector 15 re-emerging as the #2 problem (heat 9.4), sector 10 creeping up (5.7).

### Interpretation
- The two 0/16 evals right after re-entry: the inherited policy was trained 70 min under the old reward to drive at 1.07–1.13; the cap removes the incentive but the *behavior* takes updates to unwind. Eval pace 1.071–1.080 is measured behavior, not the reward target — expect it to drift down toward ~1.03–1.05 as PPO re-optimizes against the capped term.
- Reseed 1/6 firing was the pit wall doing its job (weak streak), and the very next eval was the breakthrough — no intervention warranted.

### Watch
1. A new best > 39.999 (needs clean sectors back up while keeping the new-found continuous range).
2. Eval pace trending ≤ 1.06. If pace stays ≥1.07 after ~30 more min *and* metrics stay depressed, consider whether `progress_per_m` (uncapped absolute-speed incentive) needs a look — it's the remaining term that pays for speed above the cap.
3. Sector 15 (and 10): if 3 clears and 15 becomes the new 4×-fail sector, the seed-bias mechanism should retarget automatically from the heat table — verify it does.

## Note — 2026-07-21 ~01:05 — full autonomy granted for the night

User relaunched the run themselves (PID 42758, 00:51, 8 workers / pop 16) — foundation re-certified instantly off the banked best and flow re-entered with the fixed reward (baseline re-banked at metric 40.0, 15/16 clean, pace 1.033). User is now asleep and granted full authority: I may start/stop/restart runs and edit the pipeline as needed for forward progress. Check cadence stays 20 min (job 156a60b4). Intervention bar: clear structural evidence (pace >1.08 despite the cap, progress_frac pinned ~0.20 for 40+ min, pit levers exhausted with no new best, unstable soft-advance) — not ordinary eval noise. All code edits must pass both validate_fable5 gates before a restart.

## Check #5 — 2026-07-21 00:57 — restarted the run with the fixed reward

### State
- Run had been down 45 min (nothing new since 00:12:41). Restarted it: **PID 42591**, same command as before, stdout now captured in `fable5_919_run_20260721.log` (repo root) so future checks can read training output, not just eval snapshots.
- As predicted in the Intervention entry: auto is re-certifying **foundation** first (attempt 1/2, scale 0.73 from the stored checkpoint scale, resumed from `fable5_919_ring_best.pt`, updates 5600). Baseline destination-stage eval running now. First `[eval-fable]` under the new fingerprint expected within ~25 updates.
- Minor oddity to keep an eye on: weak-sector memory armed around eval sectors **[0, 1, 10]** for foundation — earlier foundation heat peaked at sector 3 (30.6). Possibly the manifest's foundation heat was updated late in the last run. Not blocking (foundation re-gates fast), but if flow's bias also arms away from sector 3, check `sector_heat` plumbing.

### Watch for next check
1. Foundation re-gated and flow re-entered (should both happen well within 20 min at 8 workers).
2. In flow under the new reward: pace_ratio hugging ≤1.06, progress_frac > 0.20, first new best > 39.999.
3. `[fable5] weak-sector memory` line for flow should name sector 3's zone (eval sectors [3, 1, 15]).

## Check #4 — 2026-07-21 00:34

### State
- **The training process is down.** No `run.py --fable` process is running, and the last eval is from 00:12:41 (22 min stale). PID 41424 ended around then — either killed or exited. No new run has been started since, so the fixed reward (pace_cap et al.) is not yet training anything.
- Final evals before it stopped confirmed the diagnosis one more time: 18.0 (pace 1.046) → 6.8 (pace 1.106, 0/16 clean) → 19.2 (1.063) → 7.8 (1.083, 0/16). **CALM reseed 3/6 fired at 00:08:56**; pit levers spent again immediately after. The old reward never recovered — 70 minutes of flow, zero new bests.
- Final flow heat: sector 3 = 29.3, sector 1 = 12.4, sector 15 = 8.0 — same three-corner story as every check.

### Action needed
Restart the run to pick up the fixed reward (same command; new code loads automatically):
`python3 run.py --fable 49901 --fable-stage auto --car porsche_919evo --workers 8 --pop 16 --out fable5_919_ring.pt`
Expect: foundation re-certifies first (fingerprint change — see Intervention entry), then flow re-enters with pace_cap 1.05 + heavier sector-3 starts. No `--resume` flag needed; the chain-resume picks the strongest banked checkpoint on its own.

### Watch list (unchanged)
Pace 1.00–1.06 and stable · progress_frac > 0.20 · sector-3 heat decaying · a new flow best above 39.999.

## Intervention — 2026-07-21 00:25 (implemented Check #3 recs 1–3)

### Root cause found in the reward (rec 1)
`fable5.py` rewarded pace up to a hard clip of **1.30** (`np.clip(pace_ratio, -0.5, 1.30)`), while flow's overspeed penalty only started at `vref*0.97` = **1.128× target** at the current scale 0.86. So the reward optimum sat in the 1.10–1.30 band — exactly the band where all 7 catastrophic evals lived. The policy wasn't failing to learn; it was correctly maximizing a reward whose optimum is on the grip cliff.

### Changes (validated: `validate_fable5.py` + `validate_fable5_auto.py` both ALL-PASS)
1. `FableReward.pace_cap` (new, default 1.30 = legacy). **Flow sets 1.05** — no incentive past the proven-survivable operating point. Flow `overspeed_margin` 0.97 → 0.93 so the penalty onset (~1.08× target) now sits just past the cap instead of leaving a neutral no-man's band up to 1.128.
2. Sector-3 rehearsal strength: `FableSpec.seed_bias_cap` (new, default 3.0 = legacy) — flow sets **5.0** (matches FableEnv's clip), and flow `sector_fail_gain` 1.0 → **2.5**. Hot-zone starts (sector 3 + approach) go from ~+3 max seed weight to +5, and each live failure adds 2.5× more weight than before.
3. Verified the ungraduated-stage path: flow gets `STAGE_MAX_ATTEMPTS=2` attempts (second at scale −0.05), then **soft-advances to finish with its banked best**. Finish seeds from the scale flow banked, and holds the whole remaining budget until a lap is banked (it can eat fast/frontier's share). So an ungated flow is not fatal — finish is survival-weighted — but the pace cliff would have reappeared in fast/frontier, which is why the cap is per-stage and tunable.

### Restart consequences — read before restarting
- Edits to `supra/fable5.py` only apply to **new runs**; PID 41424 still runs the old reward.
- `CODE_FINGERPRINT` hashes `fable5.py`, so all existing `_best.pt` files are no longer "protocol-current": on restart, auto will NOT skip foundation, and the spent-budget counter resets. It still warm-starts from the banked checkpoints (chain-resume ranks them by driving evidence regardless of fingerprint), and foundation gates on the first eval that passes (`stop_on_gate`), so re-certification should cost minutes, not hours. This is by design — reward semantics changed, checkpoints must re-prove under the new rules.

### What to watch after restart
- Eval pace_ratio should settle into 1.00–1.06 and *stay* there; any eval ≥1.10 means the cap isn't binding as expected.
- progress_frac finally moving past 0.20 on continuous runs = sector 3 cracked at flow pace.
- Sector-3 heat (30.5 at last check) should decay once clean passes accumulate.

## Check #3 — 2026-07-21 00:07

### State
- ~50 min in flow, **zero new bests** (still 39.999 from stage entry). Evals oscillate 7 → 32 → 20 → 32 → 28 → 19.7 with no upward trend.
- Pit wall now says: *"regressing; pit levers spent for this dry spell — a new best refills them, else PPO plateau logic ends the stage."* The calm ladder is out of moves for this dry spell; the run is coasting toward a plateau-triggered stage end.
- Sector 3 heat climbed to **30.5** (was 24.8 at Check #2). Latest eval: 7/16 clean, terminal 0.44, sector 3 fails 4× (spin at pace 0.868).

### Diagnosis
1. **Pace-cliff correlation is now confirmed across 20 evals.** Every eval with pace_ratio ≤1.08: metric 20–40, 7–15 clean. Every eval ≥1.10 (7 of them): metric 5–20, terminal 0.31–0.94. The eval-time pace mode keeps sampling above the cliff and the policy has not learned to survive there in ~50 min of training. This is no longer plausibly "needs more time" — the operating point being asked for exceeds what the current policy/envelope supports.
2. **progress_frac is pinned at ~0.20 in *every* eval** — even 12–13-clean ones — for 20 straight evals. The continuous run always dies at ~4.1 km (sector 3). The policy has never once gotten past sector 3 on a continuous flying run since entering flow. Foundation could (progress 1.0); flow pace cannot. Sector 3 at flow pace is effectively unlearned and nothing in the current setup gives the policy concentrated practice there.
3. The oscillation shape (good/bad/good with no drift) means PPO is bouncing between two attractors, not converging. More wall-clock won't break the tie; a curriculum or target change will.

### Recommendations — act now rather than wait
The watch condition from Check #2 has effectively triggered (levers spent, no new best). Concrete changes, smallest-first:
1. **Clamp flow's pace request to ≤1.05–1.08** (or anneal from 1.0 upward gated on clean-chain length). Highest confidence, directly attacks the confirmed cliff.
2. **Sector-3 start-state curriculum**: spawn a majority of training episodes at the sector 2/3 boundary until sector-3 heat drops below ~5. Without this, each episode gives ~one attempt at the only corner that matters.
3. If neither is done, expect PPO plateau logic to end the stage "ungraduated" — check what `--fable-stage auto` does in that case (does finish stage inherit an unstable policy?). Verify before letting it happen naturally.
4. Longer-term (different stage): the same cliff will reappear in fast/frontier when pace targets rise again — a per-sector pace/envelope profile (lower target through sectors 1/3/15, higher elsewhere) is the structural fix; the global scalar envelope_scale=0.86 is too blunt for the Nordschleife's heterogeneity.

## Check #2 — 2026-07-20 23:47

### State
- Post-restart (8 workers / pop 16), evals every ~2–3 min. No new best; best is still the 39.999 handoff eval. Sequence since restart: 31.8 → 35.5 → 15.2 → 10.8 → **5.5** (latest, 23:44).
- Latest eval is the worst yet: **0/16 clean, terminal rate 0.94**, pace_ratio 1.135, 22.8s offtrack. Failures now spread across sectors 2,3,6,10,11,13,14,15 — not just sector 3 anymore.
- Pit wall fired **CALM reseed 2/6 at 23:44** from `flow_hof_clean.pt`. Two reseeds in 30 minutes with zero new bests.

### Diagnosis — the correlation is pace_ratio
Lining up pit history: every decent eval (26–40 metric, 9–15 clean sectors) has pace_ratio **1.00–1.08**; every catastrophic one (0–7 metric, terminal ≥0.6) has pace_ratio **1.10–1.14**. The policy oscillates between "drive at foundation pace and mostly survive" and "chase the flow pace target and crash everywhere." At pace ~1.13 the car isn't just failing sector 3 — it's spinning and leaving the track in half the sectors, i.e. it's beyond the envelope, not beyond one corner. Exploration noise is repeatedly tipping it over that cliff, and each collapse wastes a full eval + pit decision cycle.
- Also notable: even *healthy-ish* evals show progress_frac pinned at ~0.20. Clean-sector counts recover but continuous progress doesn't — consistent with sector 3 still being the wall on the continuous run while the per-sector eval attempts recover elsewhere.

### To look into / candidate pipeline changes (in priority order)
1. **Cap or anneal the flow pace target** so evaluated pace_ratio can't be pushed past ~1.08 until a clean chain exists at that pace. The data says the car has a working operating point at 1.03–1.05 (the 39.999 eval) and a cliff at ~1.10. Flow's job is a 16/16 chain at pace ≥0.55 — nothing in the gate requires >1.0 pace; the reward is over-asking.
2. **Reduce policy/exploration noise in flow** (or make the pit-wall rollback's noise-narrowing kick in earlier). Pop 16 didn't stabilize it — the spread between consecutive evals (35.5 → 5.5 in 3 evals) is noise-driven, not learning-driven.
3. Sector-3 targeted starts still stand from Check #1 — heat 24.8, still top of the table, and continuous progress still dies at ~3.6–4.3 km.
4. **Watch the reseed budget**: 2/6 used in 30 min. If 3/6 fires before the next check with no new best, the calm ladder is confirmed insufficient and item 1 or 2 should be changed in `supra/fable5.py` rather than letting the ladder exhaust.

## Note — 23:31 restart

User restarted the run: new PID 41424, `--resume fable5_919_ring_flow.pt --workers 8 --pop 16` (was 7/7). Same stage (flow), same output. Larger population should widen exploration per generation — watch whether sector-3 survival improves with more rollouts, and whether eval cadence slows.

## Check #1 — 2026-07-20 ~23:30

### State
- Flow entered ~23:14. First flow eval (23:17) was strong: metric 39.999, 15/16 clean, terminal 0.0625, lap 462.13s banked.
- Then immediate collapse: 25.1 → 11.0 → 15.9 → 20.0 across the next four evals. Progress fraction fell from 1.0 to ~0.20 lap; terminal rate up to 0.31 at the worst.
- Pit wall triggered **CALM reseed 1/6 at 23:24** from `flow_hof_progress.pt` ("progress collapsed"). So the calm ladder is already burning budget ~10 minutes into the stage.
- `fail_sectors = [3,3,3,3]` — every failed attempt dies in **sector 3** (`left_track` / `offtrack_timeout`, entry pace 0.88–0.94). Max progress ~4,125–4,338 m ≈ exactly the sector-3 boundary.

### Diagnosis
Sector 3 is not new — sector heat is 30.6 (foundation) and already 25.8 (flow), an order of magnitude above every other sector except 1 and 15. The foundation policy survives it at conservative pace; the moment flow's pace pressure + exploration noise land on it, the car carries too much entry speed and leaves the track. The whole-stage collapse is really a single-corner problem: dying at 20% of the lap zeroes progress-based metrics and makes every eval look catastrophic, which then churns the reseed ladder.

### To look into / candidate pipeline changes
1. **Sector-3 geometry & vref** (highest value): find what corner ~4.1–4.3 km is (likely the Hatzenbach/Hocheichen–Flugplatz complex) and check `trk.fable_vref` there. If vref is optimistic for that geometry, a *per-sector* vref trim or envelope trim (e.g. envelope_scale ladder per sector instead of the global 0.86) would target the actual failure without slowing the other 15 sectors.
2. **Sector-3 curriculum starts**: seed a fraction of rollouts starting just before sector 3 at flow pace, so the policy gets dense practice on the one corner that gates the stage instead of re-driving 4 km to reach it once per episode.
3. **Pace ramp on stage entry**: the first eval after the handoff was healthy (39.999); collapse began as flow's pace target/noise kicked in. A warm-up ramp of the pace mode over the first N iterations of a new stage would stop the cliff and save the calm budget for real exploration failures.
4. **Metric shaping**: an episode that dies at 0.20 lap scores ~11–25 regardless of how good sectors 0–2 were. Consider crediting clean-sector count past the death point less brutally during flow, or the pit wall will keep reading "collapse" when it's really "one corner unlearned at new pace."
5. Watch: whether reseed 1/6 recovers progress ≥0.5 by next check. If reseeds 2–3 fire without a new best, that's the signal the calm ladder alone can't solve sector 3 and one of items 1–3 should be implemented.

### Open questions
- Sectors 1 and 15 also carry heat (16.3, 8.6 in flow) — secondary, but if sector 3 gets fixed these are next in line to gate `flow_to_finish` (needs 16/16 clean chain).
- No stdout log file found for PID 40882 (dashboard log has only HTTP noise) — journal relies on `fable5_919_ring_pipeline.json` (pit decisions/history, sector_heat) and `fable5_919_ring_eval_latest.json`. Fine for now; a `--fable-diag` on `flow_best.pt` is the tool if we need per-corner physics detail.

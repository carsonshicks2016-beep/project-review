"""Fable Five AUTO-ladder validation (overnight-run guarantees).

    PYTHONPATH=$PWD python3 tools/validate_fable5_auto.py

Gates:
  1. a fresh auto run creates <prefix>_<stage>.pt files, RETRIES an ungated
     stage at an eased envelope scale, then SOFT-ADVANCES to the next stage
     (the ladder never idles), recording attempts/scale in the manifest
  2. gated stages are SKIPPED on rerun (crash-resumable) and frontier runs
     in adaptive-scale SEGMENTS with the leftover budget
  3. champion promotion is metric-compared: a weak candidate can NEVER
     overwrite a champion holding a clean lap
  4. stop-on-gate: the trainer exits a stage early the moment the gate
     recommendation appears (budget rolls on)
  5. pit wall: trend-based reseed decisions, a REFILLING reseed budget, and
     the CONSOLIDATE lever (halve lr+entropy) once reseeds stop working
  6. frontier scale policy: push on mastery, ease on a lost lap, hold while
     consolidating, respect ceiling/floor; scale persists via checkpoints
  7. finish rework: the quality bar past a non-lap finish; hall-of-fame
     category keys + banking; failure-aware reseed source; weak-sector heat
     -> start-weight bias; own-stage-best-first chain resume
  8. racebox shift point: powerband upshifts, no dead-band, ceiling is real,
     per-stage tunable via FableSpec.shift_lo_frac
  9. collapse ROLLBACK lever (immediate calm restore of a lap-scale best,
     lr decay, refilling budget, lap-armed only) + the KL trust region
     (PPO.update early-stops past 1.5x target_kl; None = legacy full pass)
Uses shrunken eval budgets; writes only fable5_autotest_* files + the
manifest's `auto` section, and never touches fable5_ring_best.pt.
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch

import supra.fable5 as f5
from supra.ppo import PPO


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"gate failed: {name} {detail}")


# shrink the evals so the ladder is testable in seconds
_orig_defaults = f5.stage_defaults


def _small_defaults(stage):
    sp = _orig_defaults(stage)
    sp.eval_starts = 2
    sp.eval_sector_seconds = 3.0
    sp.eval_lap_budget = 5.0
    sp.eval_every = 1
    sp.episode_seconds = 15.0
    return sp


f5.stage_defaults = _small_defaults
PREFIX = "fable5_autotest"


def _cleanup():
    for fn in os.listdir("."):
        if fn.startswith(PREFIX):
            os.remove(fn)


def main():
    t0 = time.time()
    # hermetic: run inside a scratch dir so the ladder can never seed from (or
    # overwrite) the REAL champion/manifest in the repo root. Left in place on
    # failure for inspection; removed on success.
    scratch = os.path.abspath(f"{PREFIX}_scratch")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    os.chdir(scratch)
    _cleanup()

    print("== 1. fresh auto run: retry at eased scale, then SOFT-ADVANCE ==")
    f5.run_fable(6, stage="auto", out=PREFIX + ".pt", pop=2, workers=1,
                 promote_best=False)
    gate("stage files use the prefix",
         os.path.exists(f"{PREFIX}_foundation.pt")
         and os.path.exists(f"{PREFIX}_foundation_best.pt"))
    gate("ladder soft-advanced past the ungated stage (no overnight idle)",
         os.path.exists(f"{PREFIX}_flow.pt"))
    m = json.loads(open(f5.PIPELINE_MANIFEST).read())
    auto = m.get("auto") or {}
    hist = auto.get("history") or []
    gate("manifest records the ladder",
         auto.get("prefix") == PREFIX and hist
         and hist[0]["stage"] == "foundation" and hist[0]["gated"] is False,
         f"history {hist[:1]}")
    h0 = hist[0]
    gate("ungated stage retried at an EASED scale",
         h0.get("attempts", 0) >= 2
         and float(h0.get("scale", 1.0)) < _orig_defaults("foundation").envelope_scale,
         f"attempts {h0.get('attempts')} scale {h0.get('scale')}")
    gate("soft-advance recorded", h0.get("soft_advanced") is True)
    gate("manifest carries pit state + provenance",
         (m.get("pit") or {}).get("decisions")
         and "envelope_scale" in m and "resumed_from" in m,
         f"pit {bool(m.get('pit'))} env {m.get('envelope_scale')}")
    gate("flying-lap eval labelled",
         (m.get("latest_eval") or {}).get("lap_style") == "flying")

    print("== 2. rerun skips gated stages; frontier runs adaptive segments ==")
    # fabricate gated bests for the first four stages from the real checkpoint
    base = torch.load(f"{PREFIX}_foundation_best.pt", map_location="cpu",
                      weights_only=False)
    for st in ("foundation", "flow", "finish", "fast"):
        d = dict(base)
        d["fable_stage"] = st
        d["fable_eval"] = dict(base.get("fable_eval") or {},
                               stage=st, recommendation=f5.GATE_ADVANCE[st],
                               metric=1.0)
        torch.save(d, f"{PREFIX}_{st}_best.pt")
    bad_stage = dict(base)
    bad_stage["fable_stage"] = "flow"
    bad_stage["fable_eval"] = dict(base.get("fable_eval") or {},
                                    stage="foundation",
                                    recommendation=f5.GATE_ADVANCE["flow"])
    torch.save(bad_stage, f"{PREFIX}_mismatched_stage.pt")
    gate("protocol rejects checkpoint/eval stage mismatch",
         not f5._checkpoint_protocol_current(
             f"{PREFIX}_mismatched_stage.pt", "flow"))
    f5.run_fable(2, stage="auto", out=PREFIX, pop=2, workers=1,
                 promote_best=False)
    m = json.loads(open(f5.PIPELINE_MANIFEST).read())
    auto = m.get("auto") or {}
    hist = auto.get("history") or []
    skipped = [h["stage"] for h in hist if h.get("skipped")]
    gate("first four stages skipped",
         skipped == ["foundation", "flow", "finish", "fast"], f"{skipped}")
    gate("frontier ran with the leftover budget",
         os.path.exists(f"{PREFIX}_frontier.pt")
         and hist[-1]["stage"] == "frontier" and not hist[-1]["skipped"])
    gate("frontier ran as an adaptive-scale segment",
         hist[-1].get("segment") == 1 and hist[-1].get("scale") is not None
         and hist[-1].get("next_scale") is not None,
         f"{ {k: hist[-1].get(k) for k in ('segment', 'scale', 'next_scale')} }")
    gate("frontier scale stamped for the dashboard",
         auto.get("frontier_scale") is not None, f"{auto.get('frontier_scale')}")

    print("== 2b. fast runs adaptive segments seeded from the BANKED scale ==")
    # gated bests for foundation/flow/finish only; finish banked its lap at
    # 0.90 — fast must seed its ladder THERE, not at the 0.96 stage default
    # (the 0.90 -> 0.96 cliff is what froze KLGUARD).
    P2 = f"{PREFIX}fs"
    for st in ("foundation", "flow", "finish"):
        d = dict(base)
        d["fable_stage"] = st
        d["fable_eval"] = dict(base.get("fable_eval") or {},
                               stage=st, recommendation=f5.GATE_ADVANCE[st],
                               metric=1.0)
        if st == "finish":
            d["fable_envelope_scale"] = 0.90
        torch.save(d, f"{P2}_{st}_best.pt")
    f5.run_fable(2, stage="auto", out=P2, pop=2, workers=1,
                 promote_best=False)
    m = json.loads(open(f5.PIPELINE_MANIFEST).read())
    hist2 = (m.get("auto") or {}).get("history") or []
    fast_h = [h for h in hist2 if h.get("stage") == "fast"
              and not h.get("skipped")]
    gate("fast ran as an adaptive-scale segment",
         fast_h and fast_h[0].get("segment") == 1
         and fast_h[0].get("next_scale") is not None,
         f"{fast_h[:1]}")
    gate("fast seeded from the finish-banked scale, not the 0.96 default",
         fast_h and abs(float(fast_h[0].get("scale", 0)) - 0.90) < 1e-9,
         f"scale {fast_h and fast_h[0].get('scale')}")
    gate("ungated fast soft-advances once its share is spent",
         fast_h and fast_h[-1].get("soft_advanced") is True)

    print("== 3. champion promotion is protected ==")
    strong = dict(base, fable_stage="foundation",
                  fable_eval={"stage": "foundation", "lap_time": 700.0,
                               "metric": 30.0,
                               "evaluated_policy_sha256": base["policy_sha256"]})
    weak = dict(base, fable_stage="frontier",
                fable_eval={"stage": "frontier", "metric": 5.0,
                             "evaluated_policy_sha256": base["policy_sha256"]})
    fast = dict(base, fable_stage="frontier",
                fable_eval={"stage": "frontier", "lap_time": 400.0,
                             "metric": 250.0,
                             "evaluated_policy_sha256": base["policy_sha256"]})
    gate("clean lap beats no lap", f5._champion_key(strong) > f5._champion_key(weak))
    gate("faster lap beats slower lap", f5._champion_key(fast) > f5._champion_key(strong))
    promo_dir = os.path.join(os.getcwd(), f"{PREFIX}_promo")
    os.makedirs(promo_dir, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(promo_dir)
        torch.save(strong, f5.FABLE_BEST)
        torch.save(weak, "weak_best.pt")
        f5._promote_champion("weak_best.pt", "frontier")
        kept = torch.load(f5.FABLE_BEST, map_location="cpu", weights_only=False)
        gate("weak run cannot clobber the champion",
             kept.get("fable_eval", {}).get("lap_time") == 700.0)
        torch.save(fast, "fast_best.pt")
        f5._promote_champion("fast_best.pt", "frontier")
        kept = torch.load(f5.FABLE_BEST, map_location="cpu", weights_only=False)
        gate("faster lap takes the crown",
             kept.get("fable_eval", {}).get("lap_time") == 400.0)
    finally:
        os.chdir(cwd)
        shutil.rmtree(promo_dir, ignore_errors=True)

    print("== 4. stop-on-gate ends a stage early ==")
    from supra.config import PPOSpec
    from supra.track import named_track
    trk = f5.attach_envelope(named_track("nordschleife"))
    sp = _small_defaults("foundation")
    cfg = f5._configure_ppo(sp, workers=1, pop=2)
    cfg.rollout = 24
    ev = f5.FableEvaluator(sp, trk, f5.PIPELINE_MANIFEST, stop_on_gate=True)
    ppo = PPO(mode="race", car=f5.FABLE_CAR, ppo=cfg, reward=sp.reward,
              fixed_track=trk, track_name="nordschleife",
              env_cls_override=f5.FableEnv, env_kwargs={"fable_spec": sp},
              eval_callback=ev, extra_metadata=ev.metadata)
    # force the gate to match whatever the tiny eval recommends
    real_eval = ev.evaluate
    def eval_and_mark(p):
        latest = real_eval(p)
        ev.gate_rec = latest["recommendation"]
        return latest
    ev.evaluate = eval_and_mark
    ppo.train(iterations=6, checkpoint=f"{PREFIX}_stop.pt", log_every=10,
              save_every=1)
    gate("trainer stopped at the first gated eval", ppo.updates <= 2,
         f"ran {ppo.updates} of 6 iters")

    print("== 5. pit wall: reseed budget refills; consolidate lever ==")
    class _PPOStub:
        _best_path = __file__            # any existing file
        updates = 100
        _ent_coef = 0.0025
        def __init__(self):
            self.opt = type("O", (), {"param_groups": [{"lr": 1e-4}]})()
            self.cfg = type("C", (), {"lr": 1e-4})()
        def reseed_from_best(self, path, it, calm=False):
            self.reseeded = True
            return True
    stub = _PPOStub()
    pit = f5.PitWall("fast", run_name="pit_test")
    def fake_eval(metric, terminal=0.0, lap=None):
        return {"metric": metric, "pace_ratio": 0.7, "clean_sectors": 12,
                "sector_count": 16, "terminal_rate": terminal, "lap_time": lap}
    d1 = pit.note(stub, fake_eval(10.0, lap=500.0))
    gate("new best -> continue", d1["decision"] == "continue"
         and "new best" in d1["reason"], d1["reason"])
    d2 = pit.note(stub, fake_eval(1.0))
    gate("one weak eval -> watch, not panic", d2["decision"] == "continue")
    pit.note(stub, fake_eval(1.2))
    d4 = pit.note(stub, fake_eval(0.8, terminal=0.6))
    gate("regression streak -> reseed from best", d4["decision"] == "reseed"
         and getattr(stub, "reseeded", False), d4["reason"])
    gate("trend windows populated", pit.trends().get("evals") == 4
         and pit.trends().get("reseeds") == 1)
    gate("decision log written", os.path.exists(f5.PIT_LOG)
         and "RESEED" in open(f5.PIT_LOG).read())

    # consolidation: exhaust a 1-reseed budget, next streak halves lr+entropy
    stub2 = _PPOStub()
    pit2 = f5.PitWall("frontier", run_name="pit_test2",
                      min_evals_between_reseeds=0, max_reseeds=1)
    pit2.note(stub2, fake_eval(10.0))
    for _ in range(3):
        d = pit2.note(stub2, fake_eval(0.5, terminal=0.9))
    gate("first streak -> reseed", d["decision"] == "reseed", d["reason"])
    d = pit2.note(stub2, fake_eval(0.5, terminal=0.9))
    gate("budget spent -> CONSOLIDATE (halve lr + entropy)",
         d["decision"] == "consolidate"
         and abs(stub2.opt.param_groups[0]["lr"] - 5e-5) < 1e-9
         and abs(stub2._ent_coef - 0.00125) < 1e-9, d["reason"])
    d = pit2.note(stub2, fake_eval(0.5, terminal=0.9))
    gate("levers spent -> continue (once per dry spell)",
         d["decision"] == "continue", d["reason"])
    pit2.note(stub2, fake_eval(50.0))
    gate("new best REFILLS the reseed budget",
         pit2.reseeds_since_best == 0 and not pit2._consolidated_stretch)
    for _ in range(3):
        d = pit2.note(stub2, fake_eval(0.5, terminal=0.9))
    gate("refilled budget reseeds again", d["decision"] == "reseed", d["reason"])

    print("== 6. frontier scale policy + persistence ==")
    up, why = f5._frontier_scale_next(
        {"lap_time": 400.0, "terminal_rate": 0.1, "pace_ratio": 0.95}, 1.00)
    gate("mastery pushes the scale", abs(up - 1.03) < 1e-9, why)
    down, why = f5._frontier_scale_next({"lap_time": None}, 1.00)
    gate("lost lap eases the scale", abs(down - 0.96) < 1e-9, why)
    hold, why = f5._frontier_scale_next(
        {"lap_time": 500.0, "terminal_rate": 0.4, "pace_ratio": 0.7}, 1.00)
    gate("consolidating holds the scale", abs(hold - 1.00) < 1e-9, why)
    top, _ = f5._frontier_scale_next(
        {"lap_time": 380.0, "terminal_rate": 0.0, "pace_ratio": 0.99},
        f5.FRONTIER_SCALE_MAX)
    gate("scale respects the ceiling", abs(top - f5.FRONTIER_SCALE_MAX) < 1e-9)
    bot, _ = f5._frontier_scale_next({"lap_time": None}, f5.FRONTIER_SCALE_MIN)
    gate("scale respects the floor", abs(bot - f5.FRONTIER_SCALE_MIN) < 1e-9)
    # KLGUARD lessons: mastery earned at an EASIER scale can't justify a
    # push, and a lap-capable stage whose segment banked nothing EASES
    # instead of holding at an unlearnable scale forever.
    stale, why = f5._frontier_scale_next(
        {"lap_time": 400.0, "terminal_rate": 0.1, "pace_ratio": 0.95}, 1.00,
        proven_scale=0.90)
    gate("stale mastery (banked at an easier scale) does NOT climb",
         abs(stale - 1.00) < 1e-9, why)
    stall, why = f5._frontier_scale_next(
        {"lap_time": 500.0, "terminal_rate": 0.4, "pace_ratio": 0.7}, 1.00,
        proven_scale=1.00, progressed=False)
    gate("stalled segment (no new best) eases despite the banked lap",
         abs(stall - 0.96) < 1e-9, why)
    fbot, _ = f5._frontier_scale_next({"lap_time": None}, f5.FAST_SCALE_MIN,
                                      lo=f5.FAST_SCALE_MIN, hi=0.96)
    gate("fast bounds: no-lap holds at the fast floor",
         abs(fbot - f5.FAST_SCALE_MIN) < 1e-9)
    torch.save({"fable_envelope_scale": 1.07}, f"{PREFIX}_scale.pt")
    gate("scale persists through checkpoints",
         f5._stored_scale(f"{PREFIX}_scale.pt") == 1.07
         and f5._stored_scale(f"{PREFIX}_missing.pt") is None)

    print("== 7. finish rework: quality bar, hall of fame, sector memory ==")
    # the quality bar: a near-lap this strong soft-advances WITHOUT a lap
    gate("quality bar accepts a strong near-lap",
         f5._finish_bar_met({"progress_frac": 0.99, "clean_sectors": 15,
                             "terminal_rate": 0.10}))
    gate("quality bar rejects the 96.4% brain (needs >=98%/15-clean)",
         not f5._finish_bar_met({"progress_frac": 0.964, "clean_sectors": 14,
                                 "terminal_rate": 0.125}))
    # hall-of-fame ordering: farther/cleaner/faster each beats the other's key
    prog_lo = f5._hof_key("progress", {"progress_frac": 0.5, "metric": 1})
    prog_hi = f5._hof_key("progress", {"progress_frac": 0.9, "metric": 1})
    gate("HOF progress key ranks farther higher", prog_hi > prog_lo)
    lap_fast = f5._hof_key("lap", {"lap_time": 400.0})
    lap_slow = f5._hof_key("lap", {"lap_time": 500.0})
    gate("HOF lap key ranks faster higher",
         lap_fast > lap_slow and f5._hof_key("lap", {}) is None)
    clean_hi = f5._hof_key("clean", {"clean_sectors": 15, "clean_chain": 12,
                                     "terminal_rate": 0.1, "metric": 1})
    clean_lo = f5._hof_key("clean", {"clean_sectors": 12, "clean_chain": 8,
                                     "terminal_rate": 0.3, "metric": 1})
    gate("HOF clean key ranks cleaner higher", clean_hi > clean_lo)
    # sector heat -> start-weight bias (weights the failing zone + its approach)
    heat = [0.0] * 12 + [4.0] + [0.0] * 2 + [2.0]
    w = f5._seed_weights_from_heat(heat, 24)
    gate("weak-sector heat biases the start weights",
         w is not None and max(w) <= 3.0 + 1e-6 and max(w) > 0.0
         and len([x for x in w if x > 0]) >= 2, f"max {max(w):.2f}")
    gate("no heat -> no bias", f5._seed_weights_from_heat([0, 0, 0, 0], 24) is None)
    # failure-aware reseed: a high terminal rate pulls from the CLEANEST HOF brain
    torch.save(dict(base, fable_eval={"clean_sectors": 15, "recommendation": "x"}),
               f"{PREFIX}_hof_clean.pt")
    pit3 = f5.PitWall("finish", run_name="hof_test")
    pit3.hof_provider = lambda: {
        "clean": {"path": os.path.abspath(f"{PREFIX}_hof_clean.pt"),
                  "eval": {"clean_sectors": 15}}}
    pit3.history = [{"metric": 0.0, "terminal_rate": 0.6, "pace_ratio": 0.5,
                     "clean_sectors": 4, "sector_count": 16} for _ in range(5)]
    pit3.best_metric = 10.0
    src, why = pit3._reseed_source(stub, {"terminal_rate": 0.6,
                                          "progress_frac": 0.3})
    gate("high terminal rate reseeds from the cleanest brain",
         src and src.endswith(f"{PREFIX}_hof_clean.pt"), why)
    # falling far short of the distance record reseeds from the FARTHEST brain
    torch.save(dict(base, fable_eval={"progress_frac": 0.95}),
               f"{PREFIX}_hof_progress.pt")
    pit3.hof_provider = lambda: {
        "progress": {"path": os.path.abspath(f"{PREFIX}_hof_progress.pt"),
                     "eval": {"progress_frac": 0.95}}}
    pit3.history = [{"metric": 0.0, "terminal_rate": 0.1, "pace_ratio": 0.5,
                     "clean_sectors": 10, "sector_count": 16} for _ in range(5)]
    src, why = pit3._reseed_source(stub, {"terminal_rate": 0.1,
                                          "progress_frac": 0.2})
    gate("progress collapse reseeds from the farthest brain",
         src and src.endswith(f"{PREFIX}_hof_progress.pt"), why)

    print("== 8. racebox shift point: powerband, no dead-band, metadata ==")
    tab = f5._shift_table()                       # module default (RACE_SHIFT_LO_FRAC)
    gate("default shift frac out of the economy lug zone",
         f5.RACE_SHIFT_LO_FRAC >= 0.55, f"frac {f5.RACE_SHIFT_LO_FRAC}")
    gate("shift frac stays under the 1->2 dead-band ceiling (0.681)",
         f5.RACE_SHIFT_LO_FRAC <= 0.681, f"frac {f5.RACE_SHIFT_LO_FRAC}")
    gate("default box has NO dead-band", not tab["deadband"],
         f"deadband @ {tab['deadband'][:4]}")
    gate("upshifts land in the powerband (all >= 70% redline)",
         bool(tab["upshifts"]) and all(u["frac"] >= 0.70 for u in tab["upshifts"]),
         "fracs " + ",".join(f"{u['frac']:.2f}" for u in tab["upshifts"]))
    # the ceiling is real: push past it and a dead-band appears
    gate("frac 0.70 DOES open a dead-band (ceiling is real)",
         bool(f5._shift_table(rpm_lo_frac=0.70)["deadband"]))
    # Acceleration selection is torque-aware now: lowering the coast/downshift
    # floor must not reintroduce the historical economy short-shift.
    old = f5._shift_table(rpm_lo_frac=0.42)["upshifts"]
    gate("torque-aware box resists economy short-shifting",
         min(u["frac"] for u in old) >= 0.90,
         "min " + f"{min(u['frac'] for u in old):.2f}")
    # the shift point is per-stage tunable via FableSpec (rides in metadata)
    car_spec = f5.get_car(f5.FABLE_CAR)
    box_hi = f5.RaceBox(car_spec, rpm_lo_frac=0.60)
    gate("FableSpec.shift_lo_frac drives the box's rpm_lo",
         abs(box_hi.rpm_lo - 0.60 * car_spec.redline_rpm) < 1e-6)

    print("== 9. collapse ROLLBACK lever + KL trust region ==")
    # rollback: once a LAP-SCALE best is banked, a collapsed eval restores the
    # best IMMEDIATELY (no 3-eval streak, no reseed gap) and CALMLY (calm=True:
    # no noise re-widening) — the anti-pattern it replaces burned a whole night
    # on MANPLEASEWORK: reseed -> one good eval -> next update kills it again.
    from supra.ppo import ActorCritic

    class _PPOStub9:
        _best_path = __file__            # any existing file
        updates = 500
        _ent_coef = 0.0035
        def __init__(self):
            self.opt = type("O", (), {"param_groups": [{"lr": 1.8e-4}]})()
            self.cfg = type("C", (), {"lr": 1.8e-4})()
            self.net = ActorCritic(4, 3, (8,))        # init_log_std -0.5
            self.calls = []              # calm flag per reseed call
        def reseed_from_best(self, path, it, calm=False):
            self.calls.append(calm)
            return True
    stub9 = _PPOStub9()
    pit9 = f5.PitWall("fast", run_name="pit_rollback")
    pit9.note(stub9, fake_eval(150.0, lap=650.0))     # lap-scale best banked
    d = pit9.note(stub9, fake_eval(4.0))              # cliff: 4 << 0.25 * 150
    gate("collapse -> IMMEDIATE calm rollback (no streak wait)",
         d["decision"] == "rollback" and stub9.calls == [True], d["reason"])
    gate("rollback decays the lr toward calm",
         stub9.cfg.lr < 1.8e-4 - 1e-9
         and stub9.opt.param_groups[0]["lr"] < 1.8e-4 - 1e-9,
         f"cfg.lr {stub9.cfg.lr:.2e}")
    gate("rollback narrows the restored policy noise (-0.10 log-std/notch)",
         float(stub9.net.log_std.data.max()) <= -0.5 - 0.10 + 1e-6,
         f"log_std {float(stub9.net.log_std.data.max()):.2f}")
    for _ in range(pit9.max_rollbacks - 1):           # spend the budget
        d = pit9.note(stub9, fake_eval(4.0))
    gate("rollback budget spent", d["decision"] == "rollback"
         and pit9.rollbacks_since_best == pit9.max_rollbacks)
    d = pit9.note(stub9, fake_eval(4.0))
    gate("after the budget, the reseed ladder takes over — CALM (lap armed)",
         d["decision"] == "reseed" and stub9.calls[-1] is True, d["reason"])
    pit9.note(stub9, fake_eval(160.0, lap=620.0))
    gate("new best refills the rollback budget",
         pit9.rollbacks_since_best == 0 and pit9.best_lap == 620.0)
    # a big metric WITHOUT a banked lap must never arm the rollback (foundation/
    # flow metrics live below the floor anyway; this guards the semantics)
    stub9b = _PPOStub9()
    pit9b = f5.PitWall("fast", run_name="pit_rollback2")
    pit9b.note(stub9b, fake_eval(150.0))              # no lap
    d = pit9b.note(stub9b, fake_eval(4.0))
    gate("no banked lap -> no rollback (watch instead)",
         d["decision"] == "continue" and stub9b.calls == [], d["reason"])
    gate("pit state exposes rollback counters",
         "rollbacks" in pit9.state() and "rollbacks_since_best" in pit9.state())
    # HEALTHY lapless evals: a car that still DRIVES (mostly-clean sectors,
    # low terminal rate, near-lap progress) must not be retired just because
    # the binary flying lap didn't close (Run-4: an eval covering 1.7 laps of
    # distance at terminal 0.06 was rolled back) — while an UNHEALTHY
    # collapse must still roll back immediately.
    stub9h = _PPOStub9()
    pit9h = f5.PitWall("fast", run_name="pit_healthy")
    pit9h.note(stub9h, fake_eval(150.0, lap=650.0))    # lap-scale best banked
    ok_ev = fake_eval(15.0)                            # 15 << 0.25 x 150
    ok_ev.update(clean_sectors=14, terminal_rate=0.06, progress_frac=1.7)
    d = pit9h.note(stub9h, ok_ev)
    gate("healthy lapless eval -> continue (no rollback)",
         d["decision"] == "continue" and stub9h.calls == [], d["reason"])
    for _ in range(2):
        d = pit9h.note(stub9h, dict(ok_ev))
    gate("healthy streak -> no reseed either (keeps refining)",
         d["decision"] == "continue" and stub9h.calls == [], d["reason"])
    bad_ev = fake_eval(4.0)
    bad_ev.update(clean_sectors=5, terminal_rate=0.5, progress_frac=0.2)
    d = pit9h.note(stub9h, bad_ev)
    gate("unhealthy collapse still rolls back",
         d["decision"] == "rollback" and stub9h.calls == [True], d["reason"])
    # RESUMED runs: the keep-best floor carries over, so the run's own _best.pt
    # may not exist yet — rollback must fall back to the HOF lap brain
    stub9c = _PPOStub9()
    stub9c._best_path = f"{PREFIX}_does_not_exist_best.pt"
    pit9c = f5.PitWall("fast", run_name="pit_rollback3")
    pit9c.hof_provider = lambda: {
        "lap": {"path": os.path.abspath(f"{PREFIX}_hof_clean.pt"),
                "eval": {"lap_time": 600.0}}}
    pit9c.note(stub9c, fake_eval(150.0, lap=650.0))
    d = pit9c.note(stub9c, fake_eval(4.0))
    gate("no _best yet (resumed run) -> rollback from the HOF lap brain",
         d["decision"] == "rollback" and stub9c.calls == [True]
         and "hof_clean" in d["reason"], d["reason"])
    # arm-from-resume: a lap-capable RESUME that degrades within its FIRST
    # eval window must roll back to the resumed brain, not bank the wreck as
    # the run's best (seen live: 176.56/565s resume -> first eval 0.063 was
    # called a "new best" and the rollback lever never armed)
    torch.save(dict(base, fable_eval={"stage": "fast", "metric": 176.56,
                                      "lap_time": 565.17}),
               f"{PREFIX}_resume_src.pt")
    stub9d = _PPOStub9()
    stub9d._best_path = f"{PREFIX}_does_not_exist_best.pt"
    pit9d = f5.PitWall("fast", run_name="pit_rollback4")
    gate("pit wall arms from the resumed checkpoint's stored eval",
         pit9d.seed_from_resume(f"{PREFIX}_resume_src.pt")
         and pit9d.best_lap == 565.17
         and abs(pit9d.best_metric - 0.95 * 176.56) < 1e-6,
         f"bar {pit9d.best_metric:.3f}")
    d = pit9d.note(stub9d, fake_eval(0.063))          # the wreck, eval #1
    gate("degraded first eval -> rollback to the RESUMED brain",
         d["decision"] == "rollback" and stub9d.calls == [True]
         and "resume_src" in d["reason"], d["reason"])
    d = pit9d.note(stub9d, fake_eval(172.0, lap=570.0))
    gate("matching the resumed quality still banks a new best (refills)",
         d["decision"] == "continue" and "new best" in d["reason"]
         and pit9d.rollbacks_since_best == 0, d["reason"])
    # Cross-stage metrics are incomparable; the destination-stage baseline eval
    # now arms the PitWall instead of converting a source-stage score.
    torch.save(dict(base, fable_eval={"stage": "finish", "metric": 187.6,
                                      "lap_time": 556.7}),
               f"{PREFIX}_resume_x.pt")
    pit9e = f5.PitWall("fast", run_name="pit_rollback5")
    gate("cross-stage resume leaves the pit cold for destination baseline",
         not pit9e.seed_from_resume(f"{PREFIX}_resume_x.pt")
         and pit9e.best_metric < -1e10,
         f"bar {pit9e.best_metric:.3f}")
    gate("no lap in the resume -> pit stays cold (no false arming)",
         f5.PitWall("fast").seed_from_resume(f"{PREFIX}_hof_progress.pt")
         is False)

    # KL trust region: PPO.update stops its epoch loop once approx-KL blows
    # past 1.5x target_kl; target_kl=None keeps the legacy full pass.
    import numpy as np
    from supra.ppo import ActorCritic

    def _kl_probe(target_kl, vf_clip=None, epochs=3, minibatches=4):
        torch.manual_seed(0)
        np.random.seed(0)
        obs_dim, act_dim, T, N = 6, 3, 8, 4
        net = ActorCritic(obs_dim, act_dim, (16,))
        fake = type("F", (), {})()
        fake.cfg = type("C", (), {"minibatches": minibatches, "epochs": epochs,
                                  "clip": 0.2,
                                  "vf_coef": 0.5, "max_grad_norm": 0.5,
                                  "target_kl": target_kl, "vf_clip": vf_clip})()
        fake.net = net
        fake.opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        fake._ent_coef = 0.003
        fake.obs_dim, fake.act_dim = obs_dim, act_dim
        fake._t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        b_obs = np.random.randn(T, N, obs_dim).astype(np.float32)
        b_act = np.random.randn(T, N, act_dim).astype(np.float32)
        with torch.no_grad():
            logp, _, _, _, _, _ = net.evaluate(
                torch.as_tensor(b_obs.reshape(-1, obs_dim)),
                torch.as_tensor(b_act.reshape(-1, act_dim)))
        # pretend the policy already drifted a full nat from the rollout policy
        b_logp = (logp.numpy() - 1.0).reshape(T, N).astype(np.float32)
        b_val = np.zeros((T, N), np.float32)
        adv = np.random.randn(T, N).astype(np.float32)
        ret = np.random.randn(T, N).astype(np.float32)
        batch = (b_obs, b_act, b_logp, b_val, None, None)
        return PPO.update(fake, batch, adv, ret)

    st = _kl_probe(0.01)
    gate("KL blow-past stops the update", st["kl_stop"] is True
         and st["kl"] > 1.5 * 0.01, f"kl {st['kl']:.3f}")
    st = _kl_probe(None)
    gate("target_kl=None keeps the legacy full pass", st["kl_stop"] is False)
    gate("fable stages arm the trust region, tightest at the frontier",
         f5._configure_ppo(_orig_defaults("fast")).target_kl
         < f5._configure_ppo(_orig_defaults("foundation")).target_kl,
         f"fast {f5._configure_ppo(_orig_defaults('fast')).target_kl}")

    # vf_clip: PPO2 value clipping takes max(clipped, unclipped) MSE, which
    # can only RAISE the value loss — a one-step probe with a near-zero clip
    # must report >= the identically-seeded legacy loss; None = legacy path.
    v_legacy = _kl_probe(None, epochs=1, minibatches=1)
    v_clipped = _kl_probe(None, vf_clip=1e-9, epochs=1, minibatches=1)
    gate("vf_clip bounds the value step (None stays legacy)",
         np.isfinite(v_clipped["vf"])
         and v_clipped["vf"] >= v_legacy["vf"] - 1e-9,
         f"vf {v_legacy['vf']:.4f} -> {v_clipped['vf']:.4f}")
    gate("fable arms stage-tuned value clipping",
         f5._configure_ppo(_orig_defaults("foundation")).vf_clip == 6.0
         and f5._configure_ppo(_orig_defaults("flow")).vf_clip == 8.0
         and f5._configure_ppo(_orig_defaults("fast")).vf_clip == 10.0)

    os.chdir(os.path.dirname(scratch))
    try:
        shutil.rmtree(scratch)
    except OSError as e:
        print(f"  [warn] scratch cleanup failed ({e}) — remove "
              f"{scratch} manually")
    print(f"ALL AUTO-LADDER GATES PASSED in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

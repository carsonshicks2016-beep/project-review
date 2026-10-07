#!/usr/bin/env python3
"""
Supra Drift — command-line entry point.

Dispatches to every mode: drive yourself, GA train/watch, PPO race train/watch,
PPO drift train/watch — plus --live training windows, --resume, and --out named
saves. See README.md for the full flag reference, or run with no args for help.

Examples
  python3 run.py --drive --track club           # drive a named track
  python3 run.py --train 100 --live             # watch the GA swarm evolve
  python3 run.py --ppo 600                       # train a race generalist
  python3 run.py --drift 2000 --track national --out national_drift.pt
  python3 run.py --watch-drift --checkpoint ppo_drift.pt
"""
from __future__ import annotations

import resource
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))
except Exception:
    pass
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", description="Supra Drift simulator")

    # --- available now ---
    p.add_argument("--drive", action="store_true",
                   help="Drive the car yourself (2D PyGame).")
    from supra.config import PRESETS
    p.add_argument("--car", choices=list(PRESETS), default=None,
                   help="Vehicle preset.")
    from supra.track import named_list, STYLES
    p.add_argument("--track", default="random",
                   help="Track: oval/random/touge (legacy) or a named track: "
                        + ", ".join(named_list()))
    p.add_argument("--gen", choices=list(STYLES),
                   help="Generate a track of this archetype (with --difficulty/--length).")
    p.add_argument("--difficulty", type=float, default=0.5,
                   help="Difficulty 0..1 for --gen.")
    p.add_argument("--length", type=float,
                   help="Track length in metres for --gen (600-2000 if omitted).")
    p.add_argument("--seed", type=int, default=7,
                   help="Procedural track seed.")
    p.add_argument("--flat", action="store_true",
                   help="Use flat 2D terrain. This is now the default; kept for "
                        "explicitness and old launch scripts.")
    p.add_argument("--hills", action="store_true",
                   help="Opt back into elevation/grade/crest physics.")
    p.add_argument("--hill-scale", type=float, default=1.0, metavar="X",
                   help="Scale hill amplitude 0..1.5 (default 1.0).")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable real-time engine audio synthesis.")
    p.add_argument("--classic", action="store_true",
                   help="Use the classic V1 2D viewer (supra/app.py) instead of "
                        "the V2 projected 2.5D viewer for drive/watch modes.")
    p.add_argument("--smoke-frames", type=int, metavar="N", default=0,
                   help=argparse.SUPPRESS)

    # --- genetic algorithm (slice 3) ---
    p.add_argument("--train", type=int, metavar="GENS",
                   help="Train the GA for GENS generations, saving the champion.")
    p.add_argument("--live", action="store_true",
                   help="With --train or --ppo: watch training on screen "
                        "(PPO live view supports S=save / L=load via file dialog).")
    p.add_argument("--pop", type=int, metavar="N",
                   help="Population / n_envs size (overrides the default).")
    p.add_argument("--anneal", action="store_true",
                   help="PPO: cosine-decay learning rate + entropy over the run "
                        "(explore early, sharpen/refine late). Best for long runs.")
    p.add_argument("--lr", type=float, metavar="LR",
                   help="PPO learning rate override. On --resume, LR is auto-reduced "
                        "for stable fine-tuning unless you set this explicitly.")
    p.add_argument("--style", type=float, metavar="W",
                   help="Hybrid: drift-STYLE strength (bigger = driftier corners, a "
                        "bit slower). ~0.01-0.1, default 0.03.")
    p.add_argument("--workers", type=int, metavar="N",
                   help="PPO: run the vectorised envs across N subprocesses "
                        "(parallel rollouts, faster). 1 = in-process (default).")
    p.add_argument("--patience", type=int, metavar="N",
                   help="PPO plateau window: iters with no NEW best before the trainer "
                        "reseeds from best (or stops). Applies to specialists / maxed "
                        "curricula. Default 500.")
    p.add_argument("--max-restarts", type=int, metavar="N", dest="max_restarts",
                   help="PPO: on plateau, reseed from the best checkpoint with fresh "
                        "exploration up to N consecutive times before giving up. "
                        "0 = stop immediately at --patience. Default 3.")
    p.add_argument("--target", metavar="TRACK",
                   help="Drift generalist: after the curriculum maxes out, graduate "
                        "to fine-tuning on this named track (wide->tight->target).")
    p.add_argument("--watch", action="store_true",
                   help="Watch the saved GA champion drive (full cockpit view).")
    p.add_argument("--checkpoint", default="ga_champion.npz",
                   help="Champion checkpoint path.")
    p.add_argument("--resume", metavar="PATH",
                   help="Resume PPO training from a checkpoint (.pt).")
    p.add_argument("--out", metavar="PATH",
                   help="Checkpoint output filename for PPO/drift training. "
                        "Lets you keep separate runs side by side "
                        "(defaults to ppo_race.pt / ppo_drift.pt). "
                        "With --track NAME you get a track specialist; "
                        "--track random (default) trains a generalist.")

    # --- PPO (slice 4) ---
    p.add_argument("--ppo", type=int, metavar="ITERS",
                   help="Train the PPO race policy for ITERS iterations.")
    p.add_argument("--watch-ppo", action="store_true",
                   help="Watch the saved PPO race policy (full cockpit view).")
    p.add_argument("--ring-race", type=int, metavar="ITERS",
                   help="Train the dedicated Mazda 787B Ring Race pipeline.")
    p.add_argument("--ring-stage", choices=["survive", "fast", "attack", "auto"],
                   default="survive",
                   help="Ring Race stage: survive, fast, attack, or auto.")
    p.add_argument("--ring-critical-starts", metavar="FRACS",
                   help="Ring Race override: comma fractions for critical/hard starts.")
    p.add_argument("--ring-critical-prob", type=float, metavar="P",
                   help="Ring Race override: probability of training from critical starts.")
    p.add_argument("--ring-ent-coef", type=float, metavar="X",
                   help="Ring Race override: entropy coefficient for repair/fine-tune runs.")
    p.add_argument("--ring-init-log-std", type=float, metavar="X",
                   help="Ring Race override: initial/reseed policy log std.")
    p.add_argument("--ring-force-log-std", type=float, metavar="X",
                   help="Ring Race override: force resumed policy log std before training.")
    p.add_argument("--ring-reset-optimizer", action="store_true",
                   help="Ring Race override: discard loaded Adam moments after resume.")
    p.add_argument("--ring-no-promote", action="store_true",
                   help="Ring Race override: do not copy this run's best to ring_787b_best.pt.")
    p.add_argument("--ring-open-best-floor", action="store_true",
                   help="Ring Race override: let an experimental run save local bests "
                        "below the resumed checkpoint metric.")
    p.add_argument("--watch-ring-race", action="store_true",
                   help="Watch a dedicated Ring Race checkpoint on the Nordschleife.")
    # --- Fable Five (the superhuman Nordschleife pipeline; FABLE5_PLAN.md) ---
    p.add_argument("--fable", type=int, metavar="ITERS",
                   help="Train the Fable Five PPO Nurburgring pipeline "
                        "(physics-true speed envelope, 5 stages).")
    p.add_argument("--fable-stage",
                   choices=["foundation", "flow", "finish", "fast", "frontier",
                            "auto"],
                   default="foundation",
                   help="Fable Five stage (or auto to run the gated ladder).")
    p.add_argument("--fable-scale", type=float, metavar="X",
                   help="Fable Five override: envelope speed-target scale "
                        "(default is stage-specific, 0.78 -> 1.00).")
    p.add_argument("--fable-no-promote", action="store_true",
                   help="Fable Five: do not copy this run's best to "
                        "fable5_ring_best.pt.")
    p.add_argument("--fable-open-best-floor", action="store_true",
                   help="Fable Five: let an experimental run save local bests "
                        "below the resumed checkpoint metric.")
    p.add_argument("--watch-fable", action="store_true",
                   help="Watch a Fable Five checkpoint on the Nordschleife.")
    p.add_argument("--fable-diag", action="store_true",
                   help="Deep brain diagnostics: run a Fable Five checkpoint "
                        "through the envelope-aware test battery and write a "
                        "JSON report to diagnostics/fable5/.")
    # Faithful 919 v2 is a separate, edition-scoped program.  Its execution
    # flags are mutually exclusive and fail closed until physical evidence,
    # authority validation and oracle feasibility are independently verified.
    faithful = p.add_mutually_exclusive_group()
    faithful.add_argument("--faithful-919-status", action="store_true",
                          help="Print the fail-closed faithful-v2 919 program status as JSON.")
    faithful.add_argument("--faithful-919-validate", action="store_true",
                          help="Run all checked-in faithful-v2 foundation validators.")
    faithful.add_argument("--faithful-919-init", metavar="RUN_ID",
                          help="Reserve a noncertifying edition-scoped faithful-v2 run skeleton.")
    faithful.add_argument("--faithful-919-oracle", action="store_true",
                          help="Run the faithful-v2 oracle preflight (currently blocked).")
    faithful.add_argument("--faithful-919-train", type=int, metavar="ITERS",
                          help="Run faithful-v2 long-run preflight (currently blocked).")
    faithful.add_argument("--faithful-evidence-inventory", action="store_true",
                          help="Verify and inventory the configured external evidence vault.")
    faithful.add_argument("--faithful-evidence-ingest", metavar="PACKAGE_DIR",
                          help="Verify and atomically ingest one signed external evidence package.")
    faithful.add_argument("--faithful-evidence-verify", metavar="PACKAGE_ID",
                          help="Verify one already-ingested evidence package.")
    faithful.add_argument("--faithful-evidence-freeze", metavar="REQUEST_JSON",
                          help="Create a signed immutable evidence freeze from a request JSON file.")
    faithful.add_argument("--faithful-evidence-open-holdout", metavar="FREEZE_ID",
                          help="Append a signed holdout-open event for a verified freeze.")
    p.add_argument("--faithful-run-id", metavar="RUN_ID",
                   help="Select an edition-scoped faithful-v2 run for status/preflight.")
    p.add_argument("--faithful-evidence-root", metavar="ABS_PATH",
                   help="External vault path; otherwise use FAITHFUL_EVIDENCE_ROOT.")
    p.add_argument("--faithful-evidence-trust-store", metavar="JSON",
                   help="Trusted public-key store; defaults to the checked-in empty store.")
    p.add_argument("--faithful-evidence-signing-key", metavar="KEY_FILE",
                   help="External raw/hex Ed25519 seed for freeze or holdout signing.")
    p.add_argument("--faithful-evidence-actor", metavar="NAME",
                   help="Named custodian opening a sealed holdout.")
    # --- Fable GA (the rudimentary yardstick: evolution vs PPO, same task) ---
    p.add_argument("--fable-ga", type=int, metavar="GENS", dest="fable_ga",
                   help="Train the GENETIC-ALGORITHM baseline on the same 787B + "
                        "Nordschleife + fable-v1 obs/action as Fable Five, for "
                        "GENS generations (add --live for the swarm viewer). "
                        "Only the learner differs from --fable; judged by the same "
                        "eval, printed as [eval-ga].")
    p.add_argument("--fable-ga-stage", dest="fable_ga_stage",
                   choices=["foundation", "flow", "finish", "fast", "frontier"],
                   default="frontier",
                   help="Fable GA reward/eval stage (default frontier: the "
                        "full-lap chase, matching Fable's headline eval).")
    p.add_argument("--fable-ga-minutes", type=float, metavar="MIN",
                   dest="fable_ga_minutes",
                   help="Fable GA: stop after MIN wall-clock minutes (a fair "
                        "equal-time budget vs a PPO run). Generations cap still "
                        "applies.")
    p.add_argument("--ga-hidden", dest="ga_hidden", metavar="A,B",
                   help="Fable GA hidden layer sizes (default 128,128 = PPO "
                        "capacity parity). e.g. --ga-hidden 96,96.")
    p.add_argument("--ga-starts", type=int, metavar="N", dest="ga_starts",
                   help="Fable GA: deterministic fitness starts per genome "
                        "(line + flying sectors). Default 3.")
    p.add_argument("--ga-fitness-seconds", type=float, metavar="S",
                   dest="ga_fitness_seconds",
                   help="Fable GA: bounded fitness-rollout horizon in sim "
                        "seconds (default 120).")
    p.add_argument("--ga-eval-every", type=int, metavar="N",
                   dest="ga_eval_every",
                   help="Fable GA: run the full [eval-ga] lap eval every N "
                        "generations (default 5).")
    p.add_argument("--watch-fable-ga", action="store_true", dest="watch_fable_ga",
                   help="Watch the Fable GA champion drive the Nordschleife "
                        "(same viewer path as --watch-fable).")
    p.add_argument("--drift", type=int, metavar="ITERS",
                   help="Train the PPO DRIFT policy for ITERS iterations "
                        "(add --live for the interactive window).")
    p.add_argument("--watch-drift", action="store_true",
                   help="Watch the saved PPO drift policy (full cockpit view).")
    p.add_argument("--watch-race", action="store_true",
                   help="Watch a symmetric multi-agent race without a primary hero car.")
    p.add_argument("--hybrid", type=int, metavar="ITERS",
                   help="Train the PPO HYBRID policy: race speed on the straights + "
                        "drift the corners (one fused goal). Warm-start a drift "
                        "policy with --resume for a head start.")
    p.add_argument("--watch-hybrid", action="store_true",
                   help="Watch the saved PPO hybrid policy (full cockpit view).")
    p.add_argument("--eval-drift", action="store_true",
                   help="[soon] Run the drift showcase exam.")
    p.add_argument("--3d", dest="use_3d", action="store_true",
                   help="[soon] Use the ModernGL 3D viewer.")
    p.add_argument("--recurrent", action="store_true",
                   help="[soon] Enable LSTM policy models.")
    p.add_argument("--reload", action="store_true",
                   help="[soon] Hot-reload PPO checkpoints while training.")
    p.add_argument("--opponents", type=str, metavar="CHECKPOINTS",
                   help="Comma-separated list of .pt checkpoints for opponent cars (RaceEnv mode).")
    p.add_argument("--multi-self-play", action="store_true",
                   help="Train multiple cars simultaneously using True Multi-Agent Self-Play (RaceEnv mode only).")
    return p


def main():
    args = build_parser().parse_args()
    if (
        args.faithful_919_status
        or args.faithful_919_validate
        or args.faithful_919_init is not None
        or args.faithful_919_oracle
        or args.faithful_919_train is not None
        or args.faithful_evidence_inventory
        or args.faithful_evidence_ingest is not None
        or args.faithful_evidence_verify is not None
        or args.faithful_evidence_freeze is not None
        or args.faithful_evidence_open_holdout is not None
    ):
        _handle_faithful_919(args)
        return
    # Preserve whether --car came from the user. Ring defaults are convenient
    # for training, but historical checkpoint playback must infer its vehicle
    # identity from the checkpoint rather than silently forcing the Mazda.
    args.car_explicit = args.car is not None
    if args.classic:
        import os
        os.environ["SUPRA_CLASSIC_VIEWER"] = "1"   # viewer2.run falls back to v1
    ring_requested = (args.ring_race is not None or args.watch_ring_race
                      or args.fable is not None or args.watch_fable
                      or args.fable_diag
                      or args.fable_ga is not None or args.watch_fable_ga)
    if args.car is None:
        args.car = "mazda787b" if ring_requested else "supra"

    # Flat 2D terrain is the default again. Hills/crests were useful while the
    # 3D viewer was active, but they make 2D watch/training evaluate a different
    # physics problem than the user expects. `--hills` is the explicit opt-in.
    from supra.track import configure_hills
    configure_hills(enabled=bool(args.hills and not args.flat),
                    scale=args.hill_scale,
                    force_flat=bool(args.flat))

    if args.drive:
        from supra.viewer2 import run
        trk, title = _resolve_track(args)
        run(car=args.car, track_name=args.track, seed=args.seed,
            audio_on=not args.no_audio, track=trk, title=title,
            max_frames=args.smoke_frames or None)
        return

    if args.train is not None:
        if args.live:
            from supra.app_ga import run as run_ga
            fixed, label = _train_track(args)
            run_ga(car=args.car, seeds=(args.seed,), generations=args.train,
                   pop=args.pop, checkpoint=_ga_checkpoint(args),
                   track=fixed, track_name=(None if label == "pool" else label),
                   resume=args.resume)
        else:
            _train_headless(args)
        return

    if args.watch:
        _watch_champion(args)
        return

    if args.ring_race is not None:
        _train_ring_race(args)
        return

    if args.watch_ring_race:
        _watch_ring_race(args)
        return

    if args.fable is not None:
        _train_fable(args)
        return

    if args.watch_fable:
        _watch_fable(args)
        return

    if args.fable_diag:
        _diagnose_fable(args)
        return

    if args.fable_ga is not None:
        if args.live:
            _train_fable_ga_live(args)
        else:
            _train_fable_ga(args)
        return

    if args.watch_fable_ga:
        _watch_fable_ga(args)
        return

    if args.ppo is not None:
        if args.live:
            from supra.app_ppo import run as run_ppo_live
            from supra.config import PPOSpec
            fixed, label = _train_track(args)
            cfg = _apply_ppo_args(PPOSpec(), args)
            reward = _long_track_race_profile(cfg, fixed, label)
            opp_cfg = None
            if args.opponents:
                parsed = [p.strip() for p in args.opponents.split(",") if p.strip() and p.strip().lower() != "none"]
                if parsed:
                    opp_cfg = [{"checkpoint": p, "car": "supra"} for p in parsed]
            run_ppo_live(car=args.car, iterations=args.ppo, resume=args.resume,
                         checkpoint=_ppo_checkpoint(args), fixed_track=fixed,
                         track_name=(None if label == "pool" else label),
                         opponents=opp_cfg, multiagent=args.multi_self_play,
                         ppo_cfg=cfg, reward=reward)
        else:
            _train_ppo(args)
        return

    if args.watch_ppo:
        _watch_ppo(args)
        return

    if args.watch_race:
        _watch_race(args)
        return

    if args.drift is not None:
        if args.live:
            from supra.app_ppo import run as run_ppo_live
            fixed, label = _train_track(args)
            run_ppo_live(car=args.car, iterations=args.drift, resume=args.resume,
                         checkpoint=_drift_checkpoint(args), mode="drift",
                         fixed_track=fixed,
                         track_name=(None if label == "pool" else label))
        else:
            _train_drift(args)
        return

    if args.watch_drift:
        _watch_drift(args)
        return

    if args.hybrid is not None:
        if args.live:
            from supra.app_ppo import run as run_ppo_live
            fixed, label = _train_track(args)
            run_ppo_live(car=args.car, iterations=args.hybrid, resume=args.resume,
                         checkpoint=_hybrid_checkpoint(args), mode="hybrid",
                         fixed_track=fixed,
                         track_name=(None if label == "pool" else label))
        else:
            _train_hybrid(args)
        return

    if args.watch_hybrid:
        _watch_hybrid(args)
        return

    not_ready = ["eval_drift"]
    if any(getattr(args, n) for n in not_ready):
        print("That mode lands in a later slice. Available: --drive, --train, --watch, "
              "--ppo, --watch-ppo, --drift, --watch-drift")
        return

    build_parser().print_help()


def _handle_faithful_919(args) -> None:
    """Edition-scoped faithful-v2 CLI; never dispatches into legacy Fable."""
    import json
    from pathlib import Path
    import subprocess
    import sys

    from supra.faithful.program import (
        FaithfulProgramBlockedError,
        faithful_program_status,
        initialize_run_skeleton,
        require_oracle_launch_ready,
        require_training_launch_ready,
    )

    root = Path(__file__).resolve().parent
    evidence_requested = (
        args.faithful_evidence_inventory
        or args.faithful_evidence_ingest is not None
        or args.faithful_evidence_verify is not None
        or args.faithful_evidence_freeze is not None
        or args.faithful_evidence_open_holdout is not None
    )
    if evidence_requested:
        from supra.faithful.evidence import (
            DEFAULT_TRUST_STORE,
            EvidenceTrustStoreV1,
            append_holdout_open_event,
            create_evidence_freeze,
            evidence_inventory,
            ingest_evidence_package,
            load_private_key_file,
            resolve_evidence_vault,
            verify_package_directory,
        )

        trust_path = Path(
            args.faithful_evidence_trust_store or DEFAULT_TRUST_STORE
        ).expanduser()
        try:
            trust_store = EvidenceTrustStoreV1.load(trust_path)
            vault = resolve_evidence_vault(
                root, args.faithful_evidence_root,
                create=args.faithful_evidence_ingest is not None,
            )
            if args.faithful_evidence_inventory:
                print(json.dumps(
                    evidence_inventory(vault, trust_store), sort_keys=True,
                    indent=2, allow_nan=False,
                ))
                return
            if args.faithful_evidence_ingest is not None:
                destination = ingest_evidence_package(
                    args.faithful_evidence_ingest, vault, trust_store
                )
                result = verify_package_directory(destination, trust_store)
                print(json.dumps({
                    "ingested_path": str(destination),
                    "package_id": result.package_id,
                    "valid": result.valid,
                    "certification_eligible": result.certification_eligible,
                    "roles": list(result.roles),
                }, sort_keys=True, indent=2, allow_nan=False))
                return
            if args.faithful_evidence_verify is not None:
                package = vault / "packages" / args.faithful_evidence_verify
                result = verify_package_directory(package, trust_store)
                print(json.dumps({
                    "package_id": result.package_id or args.faithful_evidence_verify,
                    "valid": result.valid,
                    "certification_eligible": result.certification_eligible,
                    "roles": list(result.roles),
                    "blockers": list(result.blockers),
                }, sort_keys=True, indent=2, allow_nan=False))
                if not result.valid:
                    raise SystemExit(65)
                return
            if args.faithful_evidence_freeze is not None:
                request_path = Path(args.faithful_evidence_freeze).expanduser()
                if request_path.is_symlink() or not request_path.is_file():
                    raise ValueError("freeze request must be a regular non-symlink JSON file")
                request = json.loads(request_path.read_text(encoding="utf-8"))
                fields = {
                    "freeze_id", "package_ids", "vehicle_spec_sha256",
                    "track_surface_sha256", "scenario_sha256",
                    "parameter_registry_sha256",
                }
                if not isinstance(request, dict) or set(request) != fields:
                    raise ValueError("freeze request has invalid fields")
                if not isinstance(request["package_ids"], list) or not all(
                    isinstance(value, str) for value in request["package_ids"]
                ):
                    raise ValueError("freeze request package_ids must be a string list")
                if not args.faithful_evidence_signing_key:
                    raise ValueError("--faithful-evidence-signing-key is required for a freeze")
                private_key = load_private_key_file(
                    args.faithful_evidence_signing_key, forbidden_root=root
                )
                manifest = create_evidence_freeze(
                    vault, request["freeze_id"], request["package_ids"],
                    trust_store, private_key,
                    vehicle_spec_sha256=request["vehicle_spec_sha256"],
                    track_surface_sha256=request["track_surface_sha256"],
                    scenario_sha256=request["scenario_sha256"],
                    parameter_registry_sha256=request["parameter_registry_sha256"],
                )
                print(f"Created signed evidence freeze: {manifest}")
                return
            if args.faithful_evidence_open_holdout is not None:
                if not args.faithful_evidence_signing_key:
                    raise ValueError(
                        "--faithful-evidence-signing-key is required to open a holdout"
                    )
                if not args.faithful_evidence_actor:
                    raise ValueError("--faithful-evidence-actor is required to open a holdout")
                private_key = load_private_key_file(
                    args.faithful_evidence_signing_key, forbidden_root=root
                )
                audit = append_holdout_open_event(
                    vault, args.faithful_evidence_open_holdout,
                    args.faithful_evidence_actor, trust_store, private_key,
                )
                print(f"Recorded signed holdout-open event: {audit}")
                return
        except (FileExistsError, OSError, UnicodeError, json.JSONDecodeError,
                PermissionError, TypeError, ValueError) as exc:
            print(f"faithful evidence command failed: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(65) from exc
    if args.faithful_919_status:
        try:
            status = faithful_program_status(root, run_id=args.faithful_run_id)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(status, sort_keys=True, indent=2, allow_nan=False))
        return
    if args.faithful_919_validate:
        validator = root / "tools" / "validate_faithful_919.py"
        result = subprocess.run([sys.executable, str(validator)], cwd=root, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
        return
    if args.faithful_919_init is not None:
        if args.faithful_run_id and args.faithful_run_id != args.faithful_919_init:
            raise SystemExit(
                "--faithful-919-init RUN_ID conflicts with --faithful-run-id"
            )
        try:
            manifest = initialize_run_skeleton(root, args.faithful_919_init)
        except (FileExistsError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Initialized noncertifying faithful-v2 skeleton: {manifest}")
        print("No physics, oracle, training, simulation-record, or robust-record gate was opened.")
        return
    try:
        if args.faithful_919_oracle:
            require_oracle_launch_ready(root, run_id=args.faithful_run_id)
        if args.faithful_919_train is not None:
            require_training_launch_ready(
                root, run_id=args.faithful_run_id,
                iterations=args.faithful_919_train,
            )
    except (FaithfulProgramBlockedError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(78) from exc


def _resolve_track(args):
    """Build the Track to drive: --gen archetype, a named track, or legacy
    oval/random/touge (returns None to let the app build the legacy one)."""
    from supra.track import (make_track, named_track, named_list,
                             oval, random_circuit, touge)
    if args.gen:
        trk = make_track(args.difficulty, args.gen, args.length, seed=args.seed)
        title = (f"Supra Drift — {args.gen} d{args.difficulty:.1f} "
                 f"{trk.length:.0f}m")
        return trk, title
    if args.track in named_list():
        trk = named_track(args.track)
        return trk, f"Supra Drift — {args.track} ({trk.length:.0f}m)"
    legacy = {"oval": oval,
              "random": lambda: random_circuit(seed=args.seed),
              "touge": lambda: touge(seed=args.seed)}
    if args.track in legacy:
        trk = legacy[args.track]()
        return trk, f"Supra Drift — {args.track} ({trk.length:.0f}m)"
    return None, None


def _ga_checkpoint(args):
    """GA champion output path: --out NAME(.npz) wins, else --checkpoint."""
    out = getattr(args, "out", None)
    if out:
        if out.endswith(".npz"):
            return out
        return (out[:-3] + ".npz") if out.endswith(".pt") else out + ".npz"
    return args.checkpoint


def _train_headless(args):
    """Fast headless GA training: generalist (seeded) or a track specialist,
    with named save (--out) and resume/warm-start (--resume)."""
    import os
    import time
    import signal
    from supra.config import EvoSpec
    from supra.evolution import GA
    evo = EvoSpec()
    if args.pop:
        evo.pop_size = args.pop
    fixed, label = _train_track(args)
    ga = GA(evo, car=args.car, seeds=(args.seed,),
            tracks=([fixed] if fixed is not None else None),
            track_name=(None if label == "pool" else label))
    length = ga.tracks[0].length
    ckpt = _ga_checkpoint(args)
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"resume file not found: {args.resume}")
            return
        ck = GA.load_champion(args.resume)
        GA.check_champion(ck, ga.obs_size, args.resume)
        ga.warm_start(ck["genome"], ck["fitness"])
        print(f"resumed {args.resume} (gen {ck['generation']}, "
              f"fitness {ck['fitness']:.1f})")

    def _on_sigint(signum, frame):
        print(f"\n[interrupted] saving champion -> {ckpt}", flush=True)
        ga.save_champion(ckpt)
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _on_sigint)

    kind = f"specialist on '{label}'" if fixed is not None else "generalist (seeded)"
    print(f"GA [{kind}]: pop {evo.pop_size}, genome {ga.genome_size} params, "
          f"track {length:.0f}m. Training {args.train} generations -> {ckpt}")
    t0 = time.time()
    for _ in range(args.train):
        fits = ga.evaluate_headless(ga.genomes, workers=args.workers)
        s = ga.advance(fits)
        print(f"  gen {s['gen']:3d}  best {s['best']/length:5.2f} laps  "
              f"mean {s['mean']/length:5.2f}  ({time.time()-t0:5.0f}s)", flush=True)
    ga.save_champion(ckpt)        # GA elitism = the saved champion IS the best
    print(f"Saved champion ({ga.best_fitness/length:.2f} laps) -> {ckpt}")
    print(f"Watch it:  python3 run.py --watch --checkpoint {ckpt} --car {args.car}")


def _out_name(name, default):
    """Normalise a requested checkpoint name to an absolute-ish .pt filename."""
    if not name:
        return default
    return name if name.endswith(".pt") else name + ".pt"


def _ppo_checkpoint(args):
    if getattr(args, "out", None):
        return _out_name(args.out, "ppo_race.pt")
    return args.checkpoint if args.checkpoint.endswith(".pt") else "ppo_race.pt"


def _resume_finetune_lr(ppo, cfg, args):
    """Set the LR after --resume. SAME-mode resume = fine-tuning a (converged)
    policy -> reduce LR (full from-scratch LR knocks it off its peak). CROSS-mode
    warm-start (e.g. drift -> hybrid) = ADAPTING to a new reward -> keep full LR so
    it can actually learn the new behaviour. Explicit --lr always wins. Re-sets the
    optimizer LR that load_state restored."""
    same_mode = getattr(ppo, "_resumed_mode", ppo.mode) == ppo.mode
    if getattr(args, "lr", None):
        ft, why = float(args.lr), "explicit --lr"
    elif same_mode:
        ft, why = cfg.lr * 0.25, "fine-tune (same mode)"
    else:
        ft, why = cfg.lr, "warm-start cross-mode — full LR to adapt"
    cfg.lr = ft                       # so --anneal decays from this base
    for g in ppo.opt.param_groups:
        g["lr"] = ft                  # override the LR load_state restored
    print(f"[resume] LR -> {ft:.1e} ({why})", flush=True)


def _eval_cadence(iters):
    """How often to run the (costly) deterministic eval + checkpoint. Frequent for
    short runs, but for long overnight runs eval-every-20 would burn >1 h just
    re-evaluating — so scale it out (still fine for keep-best resolution)."""
    return int(min(100, max(20, iters // 200)))


def _autoscale_envs(cfg, args):
    """With parallel workers and no explicit --pop, size n_envs to a multiple of
    the worker count (>=2 envs/worker, never below the default 8) so the subprocess
    vec divides evenly and each worker is well-fed — the benchmarked sweet spot."""
    import math
    nw = getattr(cfg, "n_workers", 1)
    if nw > 1 and not getattr(args, "pop", None):
        per = max(2, math.ceil(8 / nw))
        cfg.n_envs = nw * per
        print(f"[workers] auto n_envs={cfg.n_envs} ({per}/worker x {nw} workers)")


def _apply_ppo_args(cfg, args):
    """Shared PPO CLI knobs for headless and live trainers."""
    if getattr(args, "pop", None):
        cfg.n_envs = args.pop
    if getattr(args, "anneal", False):
        cfg.anneal = True
    if getattr(args, "workers", None):
        cfg.n_workers = max(1, int(args.workers))
    if getattr(args, "lr", None):
        cfg.lr = float(args.lr)
    if getattr(args, "patience", None) is not None:
        cfg.specialist_patience = max(0, int(args.patience))
    if getattr(args, "max_restarts", None) is not None:
        cfg.max_restarts = max(0, int(args.max_restarts))
    _autoscale_envs(cfg, args)
    return cfg


def _long_track_race_profile(cfg, fixed, label):
    """Return a race reward override for long real tracks such as Nordschleife.

    The policy still sees the same obs dimensionality, but the trainer gets a
    long enough episode, sector/exploring starts, longer preview distances, and
    reward density that continues to value outright lap speed on a 20 km circuit.
    """
    if fixed is None or not getattr(fixed, "metadata", {}).get("long_track"):
        return None
    from supra.config import RaceReward
    cfg.episode_seconds = max(cfg.episode_seconds, 540.0)
    cfg.random_start = True
    cfg.track_profile = fixed.metadata.get("profile", label)
    cfg.sensor_lookahead_distances = (15.0, 30.0, 55.0, 85.0, 125.0, 180.0)
    reward = RaceReward()
    # Existing race reward is effectively ~0.06 per metre on 1 km tracks.
    # Keep that density on a 20.832 km track instead of making progress 20x weaker.
    reward.progress = 0.06 * float(fixed.length)
    reward.lap_bonus = 25.0
    # Bias the specialist toward flying laps: speed still stays gated by heading,
    # straightness and on-track position, so this rewards fast clean driving rather
    # than grass farming or reckless corner entry.
    reward.speed = max(reward.speed, 0.014)
    reward.straight_speed = max(reward.straight_speed, 0.09)
    reward.throttle_commit = max(reward.throttle_commit, 0.035)
    reward.speed_ref = max(reward.speed_ref, 86.0)
    print(f"[track-profile] {label}: long real track, episode {cfg.episode_seconds:.0f}s, "
          f"progress reward {reward.progress:.1f}/lap, lookahead {cfg.sensor_lookahead_distances}, "
          f"speed ref {reward.speed_ref:.0f}m/s")
    return reward


def _train_ring_race(args):
    from supra.ring_race import run_ring_race
    run_ring_race(
        iterations=args.ring_race,
        stage=args.ring_stage,
        car=args.car,
        resume=args.resume,
        out=getattr(args, "out", None),
        live=bool(args.live),
        workers=getattr(args, "workers", None),
        pop=getattr(args, "pop", None),
        anneal=bool(getattr(args, "anneal", False)),
        lr=getattr(args, "lr", None),
        patience=getattr(args, "patience", None),
        max_restarts=getattr(args, "max_restarts", None),
        critical_starts=getattr(args, "ring_critical_starts", None),
        critical_prob=getattr(args, "ring_critical_prob", None),
        ent_coef=getattr(args, "ring_ent_coef", None),
        init_log_std=getattr(args, "ring_init_log_std", None),
        force_log_std=getattr(args, "ring_force_log_std", None),
        reset_optimizer=bool(getattr(args, "ring_reset_optimizer", False)),
        promote_best=not bool(getattr(args, "ring_no_promote", False)),
        open_best_floor=bool(getattr(args, "ring_open_best_floor", False)),
    )
    print("Done. Watch it:  python3 run.py --watch-ring-race --checkpoint ring_787b_best.pt")


def _watch_ring_race(args):
    from supra.ring_race import RING_BEST, watch_ring_race
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else RING_BEST
    watch_ring_race(checkpoint=ckpt, car=args.car, seed=args.seed,
                    audio_on=not args.no_audio,
                    max_frames=args.smoke_frames or None)


def _train_fable(args):

    import fcntl
    import json
    import os
    import sys
    from supra.fable5 import run_fable
    # The manifest, pit log, and global champion are shared resources. One
    # training owner at a time prevents duplicate dashboard clicks (observed in
    # production) from racing checkpoint temp files and corrupting provenance.
    lock_path = ".fable5_training.lock"
    lock = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        owner = lock.read().strip() or "unknown owner"
        print(f"Fable training already owns {lock_path}: {owner}",
              file=sys.stderr, flush=True)
        # The outer unattended supervisor treats 73 as a deliberate duplicate
        # owner, not a crash to retry until it accidentally starts a second run.
        raise SystemExit(73)
    lock.seek(0)
    lock.truncate()
    json.dump({"pid": os.getpid(), "command": sys.argv,
               "out": getattr(args, "out", None)}, lock)
    lock.flush()
    try:
        run_fable(
            iterations=args.fable,
            stage=args.fable_stage,
            car=args.car,
            resume=args.resume,
            out=getattr(args, "out", None),
            live=bool(args.live),
            workers=getattr(args, "workers", None),
            pop=getattr(args, "pop", None),
            anneal=(True if getattr(args, "anneal", False) else None),
            lr=getattr(args, "lr", None),
            patience=getattr(args, "patience", None),
            max_restarts=getattr(args, "max_restarts", None),
            envelope_scale=getattr(args, "fable_scale", None),
            promote_best=not bool(getattr(args, "fable_no_promote", False)),
            open_best_floor=bool(getattr(args, "fable_open_best_floor", False)),
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print("Done. Watch it:  python3 run.py --watch-fable --checkpoint fable5_ring_best.pt")


def _watch_fable(args):
    from supra.fable5 import FABLE_BEST, watch_fable
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else FABLE_BEST
    watch_fable(checkpoint=ckpt,
                car=(args.car if getattr(args, "car_explicit", False) else None),
                seed=args.seed,
                audio_on=not args.no_audio,
                max_frames=args.smoke_frames or None)


def _diagnose_fable(args):
    import os
    from supra.fable5 import FABLE_BEST
    from supra.fable5_diag import run_brain_diagnostics
    ckpt = (args.checkpoint if args.checkpoint.endswith(".pt")
            and os.path.exists(args.checkpoint) else FABLE_BEST)
    out = run_brain_diagnostics(
        checkpoint=ckpt,
        car=(args.car if getattr(args, "car_explicit", False) else None),
    )
    print(f"Done. Report: {out}")


def _ga_hidden(args):
    """Parse --ga-hidden 'A,B' into a tuple (None -> module default)."""
    if not getattr(args, "ga_hidden", None):
        return None
    return tuple(int(x) for x in str(args.ga_hidden).replace(" ", "").split(",") if x)


def _fable_ga_ckpt(args):
    """Fable GA checkpoint path: --out NAME(.npz) wins, else the ring default
    (never the classic GA's ga_champion.npz)."""
    from supra.fable_ga import GA_CHECKPOINT
    out = getattr(args, "out", None)
    if out:
        if out.endswith(".npz"):
            return out
        return (out[:-3] + ".npz") if out.endswith(".pt") else out + ".npz"
    # allow an explicit non-default --checkpoint too
    ck = getattr(args, "checkpoint", None)
    if ck and ck.endswith(".npz") and ck != "ga_champion.npz":
        return ck
    return GA_CHECKPOINT


def _train_fable_ga(args):
    from supra.fable_ga import train_fable_ga
    train_fable_ga(stage=args.fable_ga_stage, generations=args.fable_ga,
                   pop=args.pop, workers=args.workers, seed=args.seed,
                   checkpoint=_fable_ga_ckpt(args), resume=args.resume,
                   minutes=args.fable_ga_minutes, hidden=_ga_hidden(args),
                   fitness_seconds=args.ga_fitness_seconds,
                   fitness_starts=args.ga_starts, eval_every=args.ga_eval_every)


def _train_fable_ga_live(args):
    from supra.app_fable_ga import run as run_ga_live
    run_ga_live(stage=args.fable_ga_stage, generations=args.fable_ga,
                pop=args.pop, seed=args.seed, checkpoint=_fable_ga_ckpt(args),
                resume=args.resume, hidden=_ga_hidden(args),
                fitness_starts=args.ga_starts,
                fitness_seconds=args.ga_fitness_seconds,
                max_frames=args.smoke_frames or None)


def _watch_fable_ga(args):
    import os
    import numpy as np
    from supra.fable_ga import GA_CHECKPOINT
    from supra.brain import MLP
    from supra.config import SensorSpec, get_car
    from supra.fable5 import (RING_TRACK, FABLE_CAR, LOOKAHEAD, PACE_DISTANCES,
                              RACE_SHIFT_LO_FRAC, RaceBox, attach_envelope)
    from supra.track import named_track
    from supra.fable_ga import FableGA
    from supra.viewer2 import run as run_app

    ckpt = args.checkpoint if args.checkpoint.endswith(".npz") else GA_CHECKPOINT
    if args.live:
        # cheap, read-only follower of a live headless training run (drives the
        # current champion + variants, hot-reloading as training banks new bests)
        from supra.app_fable_ga import run_follow
        run_follow(checkpoint=ckpt, pop=args.pop, seed=args.seed,
                   max_frames=args.smoke_frames or None)
        return
    if not os.path.exists(ckpt):
        raise SystemExit(f"No Fable GA champion at '{ckpt}'. Train one with "
                         f"--fable-ga first.")
    ck = FableGA.load_champion(ckpt)
    car = args.car if args.car != "supra" else ck["car"]
    brain = MLP.from_genome(ck["genome"], ck["obs_dim"], ck["hidden"], 3,
                            out_activation="fable", bias_long=ck["bias_long"])
    mean = ck.get("norm_mean")
    var = ck.get("norm_var")
    clip = ck.get("norm_clip", 5.0)

    trk = attach_envelope(named_track(RING_TRACK), car)
    sensor_spec = SensorSpec()
    sensor_spec.pace_block = True
    sensor_spec.lookahead_distances = LOOKAHEAD
    sensor_spec.pace_distances = PACE_DISTANCES
    mode_vec = np.array([1.0, 0.0], dtype=np.float32)   # race mode one-hot
    state = {"gear_offset": 0.0}

    def controller(veh, obs):
        full = np.concatenate([obs.vector, mode_vec]).astype(np.float64)
        if mean is not None and var is not None:
            full = np.clip((full - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a = brain.forward(full)                         # [steer, long, gear]
        state["gear_offset"] = float(np.clip(a[2], -1, 1)) * 2.0
        steer = float(np.clip(a[0], -1, 1))
        lo = float(np.clip(a[1], -1, 1))
        return (steer, max(lo, 0.0), max(-lo, 0.0), 0.0)

    rbox = RaceBox(get_car(car), rpm_lo_frac=RACE_SHIFT_LO_FRAC)
    shift = lambda veh, thr, dt: rbox.update(veh, thr, dt,
                                             gear_offset=state["gear_offset"])
    lap = ck.get("lap_time") or 0.0
    lap_s = f"{lap:.2f}s" if lap else "no clean lap"
    print(f"Watching Fable GA champion: gen {ck['generation']}, "
          f"metric {ck['metric']:.3f}, {lap_s}, stage {ck['stage']}")
    run_app(car=car, seed=args.seed, audio_on=not args.no_audio,
            controller=controller, track=trk, sensor_spec=sensor_spec,
            shift_controller=shift,
            title=f"Fable GA champion (gen {ck['generation']}, {lap_s})",
            max_frames=args.smoke_frames or None)


def _train_track(args):
    """For PPO/drift training pick the training surface:
       --track random (default) -> None (curriculum generalist pool),
       --track NAME             -> a fixed specialist track.
    Returns (fixed_track_or_None, label)."""
    from supra.track import named_track, named_list, oval, touge
    name = getattr(args, "track", "random")
    if not name or name == "random":
        return None, "pool"
    if name in named_list():
        return named_track(name), name
    if name == "oval":
        return oval(), "oval"
    if name == "touge":
        return touge(seed=args.seed), "touge"
    return None, "pool"          # unknown -> safe generalist


def _train_ppo(args):
    """Headless PPO race training: generalist curriculum or track specialist."""
    from supra.config import PPOSpec
    from supra.ppo import PPO
    from supra.track import named_track, named_list
    cfg = _apply_ppo_args(PPOSpec(), args)
    import os
    ckpt = _ppo_checkpoint(args)
    fixed, label = _train_track(args)
    reward = _long_track_race_profile(cfg, fixed, label)
    
    opp_cfg = None
    if args.opponents:
        parsed = [p.strip() for p in args.opponents.split(",") if p.strip() and p.strip().lower() != "none"]
        if parsed:
            opp_cfg = [{"checkpoint": p, "car": "supra"} for p in parsed]
        
    tgt = None
    if fixed is None and args.target in named_list():
        tgt = named_track(args.target)

    ppo = PPO(mode="race", car=args.car, ppo=cfg, reward=reward, fixed_track=fixed,
              track_name=(None if label == "pool" else label),
              target_track=tgt, target_name=args.target,
              opponent_configs=opp_cfg, multiagent=args.multi_self_play)

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"resume file not found: {args.resume}")
            return
        ppo.load_state(args.resume)
        _resume_finetune_lr(ppo, cfg, args)
        print(f"resumed {args.resume} (diff {ppo.difficulty:.2f}, updates {ppo.updates})")
    kind = f"specialist on '{label}'" if fixed is not None else "generalist (track pool)"
    print(f"PPO race [{kind}]: {cfg.n_envs} envs x {cfg.rollout} steps, "
          f"{args.ppo} iters -> {ckpt}")
    ppo.train(iterations=args.ppo, checkpoint=ckpt, log_every=5,
              save_every=_eval_cadence(args.ppo))
    print(f"Done. Watch it:  python3 run.py --watch-ppo --car {args.car}")


def _watch_ppo(args):
    """Load the PPO policy and watch it drive in the full cockpit."""
    import numpy as np
    import torch
    from supra.ppo import PPO
    from supra.viewer2 import run as run_app
    from supra.track import curriculum_track
    import os
    ckpt = _ppo_checkpoint(args)
    if not os.path.exists(ckpt):
        print(f"No race policy at '{ckpt}' yet. Train one first: "
              f"python3 run.py --ppo 500")
        return
    net, norm, meta = PPO.load_policy(ckpt)
    from supra.aiviz import PolicyAgent
    agent = PolicyAgent(net, norm, meta, mode="race")
    # a specialist must be watched on the track it trained on — default to it
    if args.track == "random" and meta.get("track"):
        args.track = meta["track"]

    trk, title = _resolve_track(args)
    if trk is None:
        diff = float(meta.get("difficulty", 0.5))
        trk = curriculum_track(diff, seed=args.seed)
        title = f"Supra Drift — PPO race (curriculum diff {diff:.2f})"

    opps = []
    if args.opponents:
        from supra.race_env import FrozenOpponent
        parsed = [p.strip() for p in args.opponents.split(",") if p.strip() and p.strip().lower() != "none"]
        for p in parsed:
            opps.append(FrozenOpponent(p))

    sensor_spec = None
    if meta.get("sensor_lookahead_distances"):
        from supra.config import SensorSpec
        sensor_spec = SensorSpec()
        sensor_spec.lookahead_distances = tuple(meta["sensor_lookahead_distances"])

    print(f"Watching PPO race policy: {ckpt} on {args.track} "
          f"({trk.length:.0f}m, diff {meta.get('difficulty', '?')})")
    run_app(car=meta["car"], seed=args.seed, audio_on=not args.no_audio,
            controller=agent.act, track=trk, title=title, agent=agent,
            opponents=opps, sensor_spec=sensor_spec,
            max_frames=args.smoke_frames or None)

def _watch_race(args):
    """Symmetric multi-agent race where all cars are equal and camera follows the leader."""
    from supra.race_env import FrozenOpponent
    from supra.viewer2 import run as run_app
    
    trk, title = _resolve_track(args)
    if trk is None:
        trk = curriculum_track(0.5, seed=args.seed)
        title = "Supra Drift — Competitive Race"

    racers = []
    if args.opponents:
        parsed = [p.strip() for p in args.opponents.split(",") if p.strip() and p.strip().lower() != "none"]
        for p in parsed:
            racers.append(FrozenOpponent(p))
            
    if not racers:
        print("Error: No valid checkpoints provided for --watch-race.")
        return
        
    print(f"Starting competitive race with {len(racers)} cars...")
    run_app(track=trk, title=title, seed=args.seed, audio_on=not args.no_audio,
            racers=racers, max_frames=args.smoke_frames or None)


def _drift_checkpoint(args):
    if getattr(args, "out", None):
        return _out_name(args.out, "ppo_drift.pt")
    return args.checkpoint if args.checkpoint.endswith(".pt") else "ppo_drift.pt"


def _train_drift(args):
    """Headless PPO drift training: generalist curriculum or track specialist."""
    import os
    from supra.config import PPOSpec
    from supra.ppo import PPO
    from supra.track import named_track, named_list
    cfg = PPOSpec()
    if args.pop:
        cfg.n_envs = args.pop
    if getattr(args, "anneal", False):
        cfg.anneal = True
    if getattr(args, "workers", None):
        cfg.n_workers = max(1, int(args.workers))
    if getattr(args, "lr", None):
        cfg.lr = float(args.lr)
    if getattr(args, "patience", None) is not None:
        cfg.specialist_patience = max(0, int(args.patience))
    if getattr(args, "max_restarts", None) is not None:
        cfg.max_restarts = max(0, int(args.max_restarts))
    _autoscale_envs(cfg, args)
    ckpt = _drift_checkpoint(args)
    fixed, label = _train_track(args)
    # graduate-to-target: a generalist that, after the curriculum maxes, fine-tunes
    # on a chosen named track (wide -> tight -> target). Only for generalists.
    tgt = getattr(args, "target", None)
    tgt_track = tgt_name = None
    if tgt and fixed is None and tgt in named_list():
        tgt_track, tgt_name = named_track(tgt), tgt
    ppo = PPO(mode="drift", car=args.car, ppo=cfg, fixed_track=fixed,
              track_name=(None if label == "pool" else label),
              target_track=tgt_track, target_name=tgt_name)
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"resume file not found: {args.resume}")
            return
        ppo.load_state(args.resume)
        _resume_finetune_lr(ppo, cfg, args)
        print(f"resumed {args.resume} (diff {ppo.difficulty:.2f}, updates {ppo.updates})")
    kind = f"specialist on '{label}'" if fixed is not None else "generalist (track pool)"
    print(f"PPO drift [{kind}]: {cfg.n_envs} envs x {cfg.rollout} steps, "
          f"{args.drift} iters -> {ckpt}")
    ppo.train(iterations=args.drift, checkpoint=ckpt, log_every=5,
              save_every=_eval_cadence(args.drift))
    print(f"Done. Watch it:  python3 run.py --watch-drift --car {args.car}")


def _watch_drift(args):
    """Load the drift policy and watch it drive (with handbrake)."""
    import numpy as np
    import torch
    from supra.ppo import PPO
    from supra.viewer2 import run as run_app
    from supra.track import curriculum_track
    import os
    ckpt = _drift_checkpoint(args)
    if not os.path.exists(ckpt):
        print(f"No drift policy at '{ckpt}' yet. Train one first: "
              f"python3 run.py --drift 800")
        return
    net, norm, meta = PPO.load_policy(ckpt)
    from supra.aiviz import PolicyAgent
    agent = PolicyAgent(net, norm, meta, mode="drift")
    # a specialist must be watched on the track it trained on — default to it
    if args.track == "random" and meta.get("track"):
        args.track = meta["track"]

    trk, title = _resolve_track(args)
    if trk is None:
        from supra.track import drift_curriculum_track
        diff = float(meta.get("difficulty", 0.5))
        # drift generalists trained/evaluated on the DRIFT curriculum — watch on it
        trk = drift_curriculum_track(diff, seed=args.seed)
        title = f"Supra Drift — PPO DRIFT (curriculum diff {diff:.2f})"

    # Drifting needs ENTRY SPEED — from a standstill the policy can't initiate the
    # slide on the first corner and washes off (looks far worse than it is). Start
    # the watch at a rolling pace (curvature-aware, same as the eval) so you see a
    # real flying drift lap, not a doomed standstill launch.
    k = abs(float(trk.curvature[0]))
    start_speed = (min(30.0, (11.0 / k) ** 0.5) * 0.85) if k > 1e-4 else 25.0
    print(f"Watching PPO drift policy: {ckpt} (diff {meta.get('difficulty', '?')}, "
          f"rolling start {start_speed*3.6:.0f} km/h)")
    run_app(car=meta["car"], seed=args.seed, audio_on=not args.no_audio,
            controller=agent.act, track=trk, title=title, agent=agent,
            start_speed=start_speed, max_frames=args.smoke_frames or None)


def _hybrid_checkpoint(args):
    if getattr(args, "out", None):
        return _out_name(args.out, "ppo_hybrid.pt")
    return args.checkpoint if args.checkpoint.endswith(".pt") else "ppo_hybrid.pt"


def _train_hybrid(args):
    """Headless PPO HYBRID training: race backbone + corner-drift style. Generalist
    curriculum or track specialist; warm-start a drift policy with --resume."""
    import os
    from supra.config import PPOSpec, HybridReward
    from supra.ppo import PPO
    from supra.track import named_track, named_list
    cfg = PPOSpec()
    if args.pop:
        cfg.n_envs = args.pop
    if getattr(args, "anneal", False):
        cfg.anneal = True
    if getattr(args, "workers", None):
        cfg.n_workers = max(1, int(args.workers))
    if getattr(args, "lr", None):
        cfg.lr = float(args.lr)
    if getattr(args, "patience", None) is not None:
        cfg.specialist_patience = max(0, int(args.patience))
    if getattr(args, "max_restarts", None) is not None:
        cfg.max_restarts = max(0, int(args.max_restarts))
    _autoscale_envs(cfg, args)
    reward = HybridReward()
    if getattr(args, "style", None) is not None:
        reward.style = float(args.style)
    ckpt = _hybrid_checkpoint(args)
    fixed, label = _train_track(args)
    tgt = getattr(args, "target", None)
    tgt_track = tgt_name = None
    if tgt and fixed is None and tgt in named_list():
        tgt_track, tgt_name = named_track(tgt), tgt
    ppo = PPO(mode="hybrid", car=args.car, ppo=cfg, reward=reward, fixed_track=fixed,
              track_name=(None if label == "pool" else label),
              target_track=tgt_track, target_name=tgt_name)
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"resume file not found: {args.resume}")
            return
        ppo.load_state(args.resume)
        _resume_finetune_lr(ppo, cfg, args)
        print(f"resumed {args.resume} (diff {ppo.difficulty:.2f}, updates {ppo.updates})")
    kind = f"specialist on '{label}'" if fixed is not None else "generalist (track pool)"
    print(f"PPO hybrid [{kind}]: {cfg.n_envs} envs x {cfg.rollout} steps, "
          f"{args.hybrid} iters -> {ckpt}")
    ppo.train(iterations=args.hybrid, checkpoint=ckpt, log_every=5,
              save_every=_eval_cadence(args.hybrid))
    print(f"Done. Watch it:  python3 run.py --watch-hybrid --car {args.car}")


def _watch_hybrid(args):
    """Load the hybrid policy and watch it race-and-drift the lap."""
    import os
    from supra.ppo import PPO
    from supra.viewer2 import run as run_app
    from supra.aiviz import PolicyAgent
    from supra.track import curriculum_track
    ckpt = _hybrid_checkpoint(args)
    if not os.path.exists(ckpt):
        print(f"No hybrid policy at '{ckpt}' yet. Train one first: "
              f"python3 run.py --hybrid 2000")
        return
    net, norm, meta = PPO.load_policy(ckpt)
    agent = PolicyAgent(net, norm, meta, mode="hybrid")
    if args.track == "random" and meta.get("track"):
        args.track = meta["track"]
    trk, title = _resolve_track(args)
    if trk is None:
        diff = float(meta.get("difficulty", 0.5))
        trk = curriculum_track(diff, seed=args.seed)
        title = f"Supra Drift — PPO HYBRID (curriculum diff {diff:.2f})"
    k = abs(float(trk.curvature[0]))
    start_speed = (min(30.0, (11.0 / k) ** 0.5) * 0.85) if k > 1e-4 else 25.0
    print(f"Watching PPO hybrid policy: {ckpt} (diff {meta.get('difficulty', '?')})")
    run_app(car=meta["car"], seed=args.seed, audio_on=not args.no_audio,
            controller=agent.act, track=trk, title=title, agent=agent,
            start_speed=start_speed, max_frames=args.smoke_frames or None)


def _watch_champion(args):
    """Load the champion and watch it drive in the full cockpit view."""
    from supra.evolution import GA
    from supra.brain import MLP
    from supra.track import random_circuit
    from supra.viewer2 import run as run_app
    ck = GA.load_champion(args.checkpoint)
    from supra.config import SensorSpec
    from supra.sensors import SensorSuite
    GA.check_champion(ck, SensorSuite(SensorSpec()).obs_size, args.checkpoint)
    brain = MLP.from_genome(ck["genome"], ck["obs_size"], ck["hidden"],
                            ck["n_actions"])
    # a specialist champion must be watched on the track it trained on
    if args.track == "random" and ck.get("track"):
        args.track = ck["track"]
    trk, title = _resolve_track(args)
    if trk is None:
        trk = random_circuit(seed=args.seed)
        title = f"Supra Drift — GA champion (gen {ck['generation']})"
    else:
        title = f"Supra Drift — GA champion ({args.track}, gen {ck['generation']})"

    def controller(veh, obs):
        return brain.forward(obs.vector)

    print(f"Watching champion: {ck['fitness']/trk.length:.2f} laps, "
          f"gen {ck['generation']}, car {ck['car']}, track {ck.get('track') or 'random'}")
    run_app(car=ck["car"], seed=args.seed, audio_on=not args.no_audio,
            controller=controller, track=trk, title=title,
            max_frames=args.smoke_frames or None)


if __name__ == "__main__":
    main()

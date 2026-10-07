#!/usr/bin/env python3
"""Fast acceptance gates for the headless Fable Observatory playback core."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from supra.config import PPOSpec, SensorSpec
from supra.fable5 import CODE_FINGERPRINT, stage_defaults
from supra.fable_editions import (
    CURRENT_EVAL_PROTOCOL,
    FABLE_EDITIONS,
    FABLE_TRACK,
    FABLE_TRACK_PROFILE,
    edition_for_car,
    get_edition,
    require_playable_checkpoint,
)
from supra.observatory import (
    CheckpointCompatibilityError,
    FablePlaybackSession,
    ObservatorySessionManager,
    SessionLimitError,
    SUPPORTED_CARS,
    ValidatedCheckpoint,
    frame_sha256,
)
from supra.ppo import ActorCritic, state_dict_hash
from supra.sensors import SensorSuite


def gate(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


def registry_fixture(
    path: Path,
    car: str,
    *,
    salt: int = 0,
    stage: str = "foundation",
    episode_seconds: float = 8.0,
    envelope_scale: float | None = None,
    shift_lo_frac: float = 0.63,
    normalizer_offset: float = 0.0,
) -> ValidatedCheckpoint:
    edition = edition_for_car(car)
    if edition is None:
        raise AssertionError(f"test fixture car has no Fable edition: {car}")
    torch.manual_seed(78_700 + salt)
    hybrid = edition.hybrid_observations
    sensor = SensorSpec(
        lookahead_distances=(15.0, 30.0, 55.0, 85.0, 125.0, 180.0),
        pace_block=True,
        hybrid_block=hybrid,
    )
    sdim = SensorSuite(sensor).obs_size
    obs_dim = sdim + 2
    net = ActorCritic(obs_dim, edition.act_dim, (128, 128),
                      action_log_std_max=(0.0, 0.0, -1.25))
    with torch.no_grad():
        net.mean.bias.add_(salt * 0.01)
    net.clamp_log_std_()
    policy_hash = state_dict_hash(net.state_dict())
    cfg = PPOSpec()
    cfg.control_hz = 30
    cfg.episode_seconds = float(episode_seconds)
    cfg.random_start = True
    cfg.sensor_lookahead_distances = sensor.lookahead_distances
    cfg.sensor_pace_block = True
    cfg.sensor_pace_distances = sensor.pace_distances
    cfg.sensor_hybrid_block = hybrid
    payload = {
        "state_dict": net.state_dict(),
        "policy_sha256": policy_hash,
        "obs_layout": edition.observation_layout,
        "sensor_pace_block": True,
        "sensor_hybrid_block": hybrid,
        "sensor_pace_distances": list(sensor.pace_distances),
        "sensor_lookahead_distances": list(sensor.lookahead_distances),
        "obs_dim": edition.obs_dim,
        "sdim": edition.sdim,
        "act_dim": edition.act_dim,
        "hidden": [128, 128],
        "mode": "race",
        "car": edition.car_id,
        "track": FABLE_TRACK,
        "track_profile": FABLE_TRACK_PROFILE,
        "norm_mean": np.full(sdim, normalizer_offset, dtype=np.float64),
        "norm_var": np.ones(sdim, dtype=np.float64),
        "norm_count": 1000.0,
        "ppo_config": asdict(cfg),
        "action_log_std_max": [0.0, 0.0, -1.25],
        "fable_pipeline": True,
        "fable_stage": stage,
        "fable_envelope_scale": (
            float(envelope_scale)
            if envelope_scale is not None
            else float(stage_defaults(stage).envelope_scale)
        ),
        "fable_shift_lo_frac": float(shift_lo_frac),
        "fable_drivetrain_version": edition.drivetrain,
        "fable_eval_protocol": CURRENT_EVAL_PROTOCOL,
        "fable_code_fingerprint": CODE_FINGERPRINT,
        "fable_eval": {
            "stage": stage,
            "evaluated_policy_sha256": policy_hash,
        },
        "checkpoint_time_unix": 1_700_000_000.0 + salt,
    }
    torch.save(payload, path)
    record = require_playable_checkpoint(path.parent, edition, path.name)
    return ValidatedCheckpoint.from_record(
        path.parent, record,
        resolution_warnings=("generated acceptance-test fixture",),
    )


def environment_state(session: FablePlaybackSession) -> dict:
    v = session.env.veh
    return {
        "t": session.env.t,
        "vehicle": {
            key: np.asarray(getattr(v, key)).copy()
            for key in ("x", "y", "z", "yaw", "vx", "vy", "vz", "rpm",
                        "gear", "wheel_w", "wheel_sr", "Fz", "wheel_grip")
        },
        "rbox": dict(session.env.rbox.__dict__),
        "normalizer": (
            session.norm.mean.copy(), session.norm.var.copy(), session.norm.count,
        ),
    }


def states_equal(left: dict, right: dict) -> bool:
    if left["t"] != right["t"]:
        return False
    for key in left["vehicle"]:
        if not np.array_equal(left["vehicle"][key], right["vehicle"][key]):
            return False
    for key in left["rbox"]:
        a, b = left["rbox"][key], right["rbox"][key]
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                return False
        elif key == "spec":
            if a != b:
                return False
        elif a != b:
            return False
    return all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(left["normalizer"], right["normalizer"]))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="supra-observatory-") as raw_tmp:
        tmp = Path(raw_tmp)
        mazda = registry_fixture(tmp / "mazda.pt", "mazda787b")
        mazda_next = registry_fixture(tmp / "mazda-next.pt", "mazda787b", salt=1)
        mazda_normalized = registry_fixture(
            tmp / "mazda-normalized.pt", "mazda787b", normalizer_offset=0.25
        )
        mazda_fast = registry_fixture(
            tmp / "mazda-fast.pt", "mazda787b", salt=2, stage="fast",
            episode_seconds=12.0, envelope_scale=0.96, shift_lo_frac=0.71,
        )
        porsche = registry_fixture(tmp / "porsche.pt", "porsche_919evo")

        print("== registry-only checkpoint boundary ==")
        gate("supported playback cars derive from the edition registry",
             SUPPORTED_CARS == frozenset(
                 edition.car_id for edition in FABLE_EDITIONS.values()
             ))
        try:
            ValidatedCheckpoint(
                mazda.path, "787b", "mazda787b",
                policy_sha256=mazda.policy_sha256,
            )
        except CheckpointCompatibilityError as exc:
            constructor_sealed = "registry record" in str(exc)
        else:
            constructor_sealed = False
        gate("ValidatedCheckpoint constructor rejects hand-built identities",
             constructor_sealed)

        try:
            ValidatedCheckpoint.from_record(tmp, {"playable": True})
        except CheckpointCompatibilityError:
            fake_record_rejected = True
        else:
            fake_record_rejected = False
        gate("descriptor factory rejects record-shaped mappings",
             fake_record_rejected)

        mazda_record = require_playable_checkpoint(
            tmp, get_edition("787b"), mazda.path.name
        )
        forged_record = replace(mazda_record, policy_sha256="0" * 64)
        try:
            ValidatedCheckpoint.from_record(tmp, forged_record)
        except CheckpointCompatibilityError:
            forged_record_rejected = True
        else:
            forged_record_rejected = False
        gate("descriptor factory revalidates and rejects forged registry records",
             forged_record_rejected)

        tampered = registry_fixture(tmp / "mazda-tampered.pt", "mazda787b")
        # This is a checkpoint generated in this temporary test directory.
        # PyTorch 2.6+ defaults to weights_only=True, which cannot deserialize
        # the NumPy normalizer arrays used by the compatibility fixture.
        tampered_payload = torch.load(
            tampered.path, map_location="cpu", weights_only=False
        )
        tampered_payload["norm_mean"] = np.full(
            int(tampered_payload["sdim"]), 0.5, dtype=np.float64
        )
        torch.save(tampered_payload, tampered.path)
        try:
            stale_open = FablePlaybackSession(tampered, audio=False)
        except CheckpointCompatibilityError as exc:
            stale_content_rejected = "bytes changed" in str(exc)
        else:
            stale_content_rejected = False
            stale_open.close()
        gate("playback rejects metadata-only changes after catalogue validation",
             stale_content_rejected)

        try:
            replace(mazda, path=tmp / "forged.pt")
        except (CheckpointCompatibilityError, TypeError):
            replace_rejected = True
        else:
            replace_rejected = False
        gate("dataclass replacement cannot forge a validated path",
             replace_rejected)

        untrusted_inputs = {
            "string": str(mazda.path),
            "Path": mazda.path,
            "mapping": {
                "path": str(mazda.path),
                "edition": "787b",
                "car": "mazda787b",
            },
        }
        for label, candidate in untrusted_inputs.items():
            opened = None
            try:
                opened = FablePlaybackSession(candidate, audio=False)
            except CheckpointCompatibilityError:
                rejected = True
            else:
                rejected = False
            finally:
                if opened is not None:
                    opened.close()
            gate(f"playback rejects an arbitrary {label} checkpoint", rejected)

        boundary_manager = ObservatorySessionManager(
            max_sessions=1, per_browser=1, cleanup_seconds=0.0
        )
        try:
            boundary_manager.create("untrusted-browser", str(mazda.path), audio=False)
        except CheckpointCompatibilityError:
            manager_rejected = boundary_manager.status()["active"] == 0
        else:
            manager_rejected = False
        finally:
            boundary_manager.close_all()
        gate("session manager rejects paths without reserving a session",
             manager_rejected)

        print("== deterministic authoritative playback ==")
        a = FablePlaybackSession(mazda, seed=7, audio=True,
                                 prediction_mode="deterministic")
        b = FablePlaybackSession(mazda, seed=7, audio=False,
                                 prediction_mode="deterministic")
        try:
            mazda_edition = get_edition("787b")
            gate("playback identity is populated from the edition registry",
                 a.identity.edition == mazda_edition.id
                 and a.identity.car == mazda_edition.car_id
                 and a.identity.drivetrain == mazda_edition.drivetrain
                 and a.identity.observation_layout
                 == mazda_edition.observation_layout
                 and a.identity.compatibility_class
                 == mazda_edition.compatibility_class
                 and a.identity.file_sha256 == mazda.file_sha256
                 and len(a.identity.runtime_contract_sha256) == 64)
            # Cross two 5 Hz probe refresh boundaries as well as ordinary
            # policy ticks; asynchronous rollout work must not change hashes.
            frames_a = [a.tick() for _ in range(13)]
            frames_b = [b.tick() for _ in range(13)]
            hashes_a = [frame_sha256(frame) for frame in frames_a]
            hashes_b = [frame_sha256(frame) for frame in frames_b]
            gate("same checkpoint and seed produce identical frames/actions",
                 hashes_a == hashes_b)
            gate("frame sequence/time are monotonic",
                 [f["sequence"] for f in frames_a] == list(range(1, 14))
                 and all(y["sim_time"] > x["sim_time"]
                         for x, y in zip(frames_a, frames_a[1:])))
            gate("frames are finite JSON", all(
                json.dumps(frame, allow_nan=False) for frame in frames_a
            ))
            first = frames_a[0]
            lengths = {name: len(group["values"])
                       for name, group in first["observations"]["groups"].items()}
            gate("fable-v1 observation groups are exact",
                 lengths == {"rays": 9, "proprioception": 25,
                             "curvature": 6, "hill_air": 18,
                             "pace": 8, "mode": 2}, str(lengths))
            gate("both hidden layers and three outputs are exposed",
                 [len(layer) for layer in first["brain"]["hidden_layers"]]
                 == [128, 128]
                 and len(first["brain"]["output_means"]) == 3)
            gate("sensitivities are labeled as non-intent",
                 "not intent" in first["brain"]["sensitivity"]["label"].lower()
                 and all(first["brain"]["sensitivity"]["top"][name]
                         for name in ("steering", "longitudinal", "gear_offset")))
            gate("raw and RaceBox-applied action semantics are present",
                 len(first["controls"]["raw_action"]) == 3
                 and first["controls"]["gear_offset"]
                 == first["controls"]["clipped_action"][2] * 2.0)

            before = environment_state(a)
            predicted = a.probe.predict_path(a.env)
            after = environment_state(a)
            gate("detached one-second prediction does not mutate live state",
                 len(predicted) >= 2 and states_equal(before, after))

            chunk = a.render_audio_chunk()
            gate("2D viewer audio emits framed 48k stereo s16 / 1024-frame chunks",
                 chunk.sample_rate == 48_000 and chunk.frames == 1_024
                 and len(chunk.pcm) == 1_024 * 2 * 2
                 and chunk.wire_bytes()[:4] == b"FOA1")
            gate("Observatory uses the correct car voice inside its immersive mix",
                 a.audio.source == "observatory-immersive-audio-v1"
                 and a.audio.audio_seed == 78755
                 and a.audio.perspective == 0
                 and a.audio.renderer.synths[0].car == "mazda787b")
            a.set_listener({
                "x": a.env.veh.x, "y": a.env.veh.y, "z": a.env.veh.z + 1.2,
                "vx": 0.0, "vy": 0.0, "yaw": a.env.veh.yaw,
                "camera": "roof", "perspective": "onboard",
                "weather": "night", "cut": 7,
            })
            immersive = a.render_audio_chunk()
            gate("camera, lighting ambience, and cut state reach the audio renderer",
                 a.audio.camera == "roof" and a.audio.perspective == 2
                 and a.audio.weather == "night" and a.audio.cut == 7
                 and len(immersive.pcm) == 1_024 * 2 * 2
                 and any(immersive.pcm))
            listener_before = a.audio.browser_listener
            a.set_listener({
                "x": listener_before[0] + 80.0,
                "y": listener_before[1] - 30.0,
                "z": a.audio.listener_z + 12.0,
                "vx": 18.0, "vy": -4.0, "yaw": listener_before[4] + 0.7,
                "camera": "roof", "perspective": "onboard",
                "weather": "night", "cut": 7,
            })
            gate("same-shot listener samples become smooth audio targets",
                 a.audio.listener_target != listener_before
                 and a.audio.browser_listener == listener_before)
            a.render_audio_chunk()
            gate("audio-block easing advances without a listener-position jump",
                 listener_before[0] < a.audio.browser_listener[0]
                 < a.audio.listener_target[0])
            a.set_listener({
                "x": listener_before[0] - 50.0,
                "y": listener_before[1],
                "z": a.audio.listener_z_target,
                "vx": 0.0, "vy": 0.0, "yaw": listener_before[4],
                "camera": "trackside-close", "perspective": "external",
                "weather": "dry", "cut": 8,
            })
            gate("camera cuts snap the listener under the existing audio fade",
                 a.audio.browser_listener == a.audio.listener_target
                 and a.audio.cut == 8)

            a.pause()
            gate("paused tick does not advance", a.tick() is None)
            paused_event = next(
                (event for event in a.drain_events()
                 if event.get("name") == "paused"),
                {},
            )
            gate("pause event carries authoritative playback state",
                 paused_event.get("playback", {}).get("paused") is True
                 and int(paused_event.get("playback", {}).get("revision", -1)) >= 1)
            pause_revision = a.handle_control({"type": "pause"})["frame"]["playback"]["revision"]
            resume_revision = a.handle_control({"type": "resume"})["frame"]["playback"]["revision"]
            a.pause()
            gate("authoritative control frames carry monotonic playback revisions",
                 resume_revision > pause_revision)
            scrubbed = a.scrub(sequence=2)
            stepped = a.step_once()
            gate("buffer scrub and paused historical step work",
                 scrubbed["sequence"] == 2 and stepped["sequence"] == 3
                 and stepped["playback"]["paused"] is True
                 and stepped["playback"]["scrubbing"] is True)
            reset = a.reset()
            predicted = reset["brain"]["predicted_path"]
            gate("reset emits an authoritative discontinuity snapshot",
                 reset["episode"] >= 1
                 and reset["playback"]["discontinuity"]["kind"] == "episode_reset"
                 and bool(predicted)
                 and abs(predicted[0]["x"] - a.env.veh.x) < 1e-9
                 and abs(predicted[0]["y"] - a.env.veh.y) < 1e-9)
            gate("speed changes scheduler only",
                 a.set_speed(2.0) == 2.0
                 and abs(a.wall_interval - 1.0 / 60.0) < 1e-12
                 and abs(a.sim_spec.dt - 1.0 / 120.0) < 1e-12)
        finally:
            a.close()
            b.close()

        print("== realtime Brain Probe cadence ==")
        realtime = FablePlaybackSession(mazda, seed=7, audio=False)
        try:
            probe = realtime.probe
            source_sequence = probe._cached_path_sequence
            cached_path = list(probe._cached_path)

            def slow_rollout() -> list[dict[str, float]]:
                # A deliberately slow worker models a transient rollout spike.
                # The live session must never wait for this under its lock.
                time.sleep(0.25)
                return cached_path

            probe._prediction_future = probe._prediction_executor.submit(slow_rollout)
            probe._prediction_source_sequence = source_sequence
            full = np.concatenate(
                [realtime.env.last_obs.vector, realtime.env._mode_vec]
            ).astype(np.float32)
            started = time.perf_counter()
            pending = probe.capture(full, realtime.env.last_obs, realtime.env, 1)
            elapsed = time.perf_counter() - started
            gate("live Brain Probe never waits for an in-flight rollout",
                 elapsed < 0.15 and pending.brain["predicted_path_pending"] is True,
                 f"capture completed in {elapsed * 1000:.1f} ms")
            gate("pending projection reports its bounded freshness",
                 pending.brain["predicted_path_status"] == "pending"
                 and pending.brain["predicted_path_age_ticks"] == 1
                 and abs(pending.brain["predicted_path_age_seconds"] - 1 / 30) < 1e-12)
            time.sleep(0.30)
            published = probe.capture(full, realtime.env.last_obs, realtime.env, 2)
            gate("completed rollout publishes on a later live capture",
                 published.brain["predicted_path_pending"] is False
                 and published.brain["predicted_path_source_sequence"] == source_sequence
                 and published.brain["predicted_path_status"] == "fresh")
            stale = probe.capture(
                full, realtime.env.last_obs, realtime.env,
                probe.sensitivity_interval * 2 + 1,
            )
            gate("aged projections are marked stale for the viewer",
                 stale.brain["predicted_path_stale"] is True
                 and stale.brain["predicted_path_status"] == "stale")
        finally:
            realtime.close()

        print("== 919 layout and permanent truth boundary ==")
        s919 = FablePlaybackSession(porsche, seed=7, audio=False)
        try:
            porsche_edition = get_edition("919")
            frame = s919.tick()
            groups = frame["observations"]["groups"]
            gate("fable-v2 exposes the two 919 hybrid inputs",
                 len(groups["hybrid"]["values"]) == 2
                 and len(frame["observations"]["raw_vector"]) == 70)
            gate("919 hybrid contract comes from the edition registry",
                 porsche_edition.hybrid_observations is True
                 and s919.sensor_spec.hybrid_block
                 == porsche_edition.hybrid_observations)
            gate("919 authority warning is exactly registry-owned",
                 porsche_edition.authority_warning
                 in frame["checkpoint"]["warnings"])
        finally:
            s919.close()

        print("== boundary-only active-best swap ==")
        follow = FablePlaybackSession(
            mazda, mode="follow-active-best", seed=7, audio=False
        )
        try:
            original = follow.identity.file_sha256
            follow.queue_follow_candidate(mazda_next)
            follow.tick()
            gate("queued active-best never swaps mid-lap",
                 follow.identity.file_sha256 == original)
            follow.reset()
            gate("validated candidate swaps at reset boundary",
                 follow.identity.file_sha256 == mazda_next.file_sha256)
        finally:
            follow.close()

        normalized_follow = FablePlaybackSession(
            mazda, mode="follow-active-best", seed=7, audio=False
        )
        try:
            normalized_follow.queue_follow_candidate(mazda_normalized)
            normalized_follow.reset()
            gate("same weights with a new normalizer still swap full checkpoint identity",
                 normalized_follow.identity.policy_sha256 == mazda.policy_sha256
                 and normalized_follow.identity.file_sha256
                 == mazda_normalized.file_sha256
                 and np.allclose(normalized_follow.norm.mean, 0.25))
        finally:
            normalized_follow.close()

        stage_follow = FablePlaybackSession(
            mazda, mode="follow-active-best", seed=7, audio=False
        )
        try:
            stage_follow.queue_follow_candidate(mazda_fast)
            stage_follow.reset()
            fast_defaults = stage_defaults("fast")
            gate("follow swap rebuilds the candidate Fable runtime at its boundary",
                 stage_follow.identity.stage == "fast"
                 and stage_follow.identity.file_sha256 == mazda_fast.file_sha256
                 and stage_follow.env.fable.stage == "fast"
                 and stage_follow.fable_spec.reward.pace == fast_defaults.reward.pace
                 and stage_follow.env.ppo.episode_seconds == 12.0
                 and stage_follow.env.fable.envelope_scale == 0.96
                 and stage_follow.env.fable.shift_lo_frac == 0.71)
        finally:
            stage_follow.close()

        print("== manager caps and cleanup ==")
        manager = ObservatorySessionManager(
            max_sessions=2, per_browser=1, cleanup_seconds=0.01
        )
        session_id, managed_session = manager.create("browser-a", mazda, audio=False)
        manager.connect(session_id)
        try:
            try:
                manager.create("browser-a", mazda, audio=False)
            except SessionLimitError:
                browser_limited = True
            else:
                browser_limited = False
            gate("per-browser cap fails closed", browser_limited)
            second_id, second_session = manager.create(
                "browser-b", mazda, audio=False
            )
            manager.connect(second_id)
            try:
                try:
                    manager.create("browser-c", mazda, audio=False)
                except SessionLimitError:
                    globally_limited = True
                else:
                    globally_limited = False
                gate("global cap fails closed", globally_limited)
            finally:
                manager.disconnect(second_id)
            manager.disconnect(session_id)
            time.sleep(0.02)
            manager.reap_expired()
            gate("final disconnect releases after cleanup grace",
                 manager.status()["active"] == 0
                 and managed_session.closed and second_session.closed)
        finally:
            manager.close_all()

        print("== fail-closed checkpoint contract ==")
        stale_cross = registry_fixture(
            tmp / "stale-cross-car.pt", "mazda787b", salt=2
        )
        bad_payload = torch.load(
            stale_cross.path, map_location="cpu", weights_only=False
        )
        bad_payload["car"] = "porsche_919evo"
        torch.save(bad_payload, stale_cross.path)
        try:
            FablePlaybackSession(stale_cross, audio=False)
        except CheckpointCompatibilityError:
            rejected = True
        else:
            rejected = False
        gate("payload changed after registry validation is rejected", rejected)

        stale_hybrid = registry_fixture(
            tmp / "stale-hybrid.pt", "porsche_919evo", salt=3
        )
        bad_hybrid_payload = torch.load(
            stale_hybrid.path, map_location="cpu", weights_only=False
        )
        bad_hybrid_payload["sensor_hybrid_block"] = False
        torch.save(bad_hybrid_payload, stale_hybrid.path)
        try:
            FablePlaybackSession(stale_hybrid, audio=False)
        except CheckpointCompatibilityError:
            hybrid_rejected = (
                get_edition(stale_hybrid.edition).hybrid_observations is True
            )
        else:
            hybrid_rejected = False
        gate("stale hybrid metadata is rejected against the registry field",
             hybrid_rejected)

    print("Observatory playback validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

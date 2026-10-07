#!/usr/bin/env python3
"""Regression gates for the fail-closed faithful-v2 control plane."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.faithful.program import (  # noqa: E402
    EDITION,
    FaithfulProgramBlockedError,
    MISSING_LICENSED_EVIDENCE,
    RECORD_BENCHMARK_S,
    faithful_program_status,
    initialize_run_skeleton,
    require_oracle_launch_ready,
    require_training_launch_ready,
)
from supra.fable5 import _assert_fable_resume_car_identity  # noqa: E402


FAILED: list[str] = []


def gate(name: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        FAILED.append(name)


def raises(kind: type[BaseException], callback) -> bool:
    try:
        callback()
    except kind:
        return True
    return False


def main() -> None:
    print("== immutable public-bootstrap status ==")
    with tempfile.TemporaryDirectory(prefix="faithful-program-") as directory:
        root = Path(directory)
        legacy = root / "runtime" / "fable5_manifest.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(json.dumps({
            "car": "mazda787b", "superhuman": True,
            "gates": {"physics_validated": True, "robust_record": True},
        }), encoding="utf-8")

        status = faithful_program_status(root)
        gate("edition is exact", status["edition"] == EDITION == "porsche-919evo-faithful-v2")
        gate("benchmark is exact", status["benchmark_s"] == RECORD_BENCHMARK_S == 319.546)
        gate("public model is explicitly approximate",
             status["model_class"] == "telemetry-constrained approximation")
        gate("evidence intake exists but has no self-appointed trust key",
             status["evidence_pipeline"]["implemented"] is True
             and status["evidence_pipeline"]["trusted_key_count"] == 0)
        gate("Linux authority runtime remains unverified until image identity exists",
             status["authority_runtime"]["target"] == "linux/amd64"
             and status["authority_runtime"]["linux_image_identity_verified"] is False)
        gate("all training and record gates fail closed",
             status["gates"] == {
                 "training": False, "physics_validated": False,
                 "oracle_feasible": False, "sim_record": False,
                 "robust_record": False,
             })
        gate("licensed evidence list is concrete",
             tuple(status["missing_evidence"]) == MISSING_LICENSED_EVIDENCE
             and any("Michelin" in item for item in status["missing_evidence"])
             and any("June 2018" in item for item in status["missing_evidence"]))
        gate("legacy global manifest cannot open a gate",
             not any(status["gates"].values()) and status["run_id"] is None)
        gate(
            "retired 919 checkpoint cannot seed implicit Mazda training",
            raises(
                ValueError,
                lambda: _assert_fable_resume_car_identity(
                    {"car": "porsche_919evo"}, "retired-919.pt"
                ),
            ),
        )

        print("== edition-scoped skeleton ==")
        manifest = initialize_run_skeleton(root, "record-program-0001")
        expected = (root.resolve() / "runtime" / "fable5" / "editions" / EDITION
                    / "runs" / "record-program-0001" / "manifest.json")
        gate("skeleton uses exact edition/run namespace", manifest == expected)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        gate("skeleton has no fake identity or evidence",
             payload["run_identity"] is None and payload["evidence_bundle"] is None
             and payload["certification_eligible"] is False
             and payload["training_authorized"] is False)
        gate("initialization is idempotent", initialize_run_skeleton(
            root, "record-program-0001") == manifest)

        # Even an extra caller-authored gate object is semantically inert.
        payload["gates"] = {
            "training": True, "physics_validated": True,
            "oracle_feasible": True, "sim_record": True, "robust_record": True,
        }
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        selected = faithful_program_status(root, run_id="record-program-0001")
        gate("caller-authored gate booleans are never trusted",
             selected["run_skeleton_initialized"]
             and not any(selected["gates"].values()))
        gate("selected artifact root remains exact",
             selected["artifact_root"] == str(expected.parent))

        gate("path traversal run id rejected",
             raises(ValueError, lambda: initialize_run_skeleton(root, "../mazda")))
        gate("uppercase run id rejected",
             raises(ValueError, lambda: faithful_program_status(root, run_id="Record")))

        print("== execution gates ==")
        oracle_blocked = False
        try:
            require_oracle_launch_ready(root, run_id="record-program-0001")
        except FaithfulProgramBlockedError as exc:
            oracle_blocked = (
                "no process or checkpoint was started" in str(exc)
                and "319.546" in str(exc)
                and "MJX" in str(exc)
            )
        gate("oracle command fails before launch with concrete blockers", oracle_blocked)

        training_blocked = False
        try:
            require_training_launch_ready(
                root, run_id="record-program-0001", iterations=100
            )
        except FaithfulProgramBlockedError as exc:
            training_blocked = (
                "no process or checkpoint was started" in str(exc)
                and "actor-boundary audit" in str(exc)
                and "verified oracle" in str(exc)
            )
        gate("training command fails before worker allocation", training_blocked)
        gate("nonpositive training budget rejected",
             raises(ValueError, lambda: require_training_launch_ready(
                 root, iterations=0
             )))
        gate("bool is not accepted as a training budget",
             raises(ValueError, lambda: require_training_launch_ready(
                 root, iterations=True
             )))

    print("== symlink containment ==")
    with tempfile.TemporaryDirectory(prefix="faithful-symlink-root-") as root_dir, \
            tempfile.TemporaryDirectory(prefix="faithful-symlink-outside-") as outside_dir:
        root = Path(root_dir)
        (root / "runtime").symlink_to(Path(outside_dir), target_is_directory=True)
        gate("runtime symlink escape rejected",
             raises(ValueError, lambda: initialize_run_skeleton(root, "escape")))

    if FAILED:
        raise SystemExit("faithful program validation failed: " + ", ".join(FAILED))
    print("faithful program validation: PASS")


if __name__ == "__main__":
    main()

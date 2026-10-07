#!/usr/bin/env python3
"""Record whether the pinned IBAMR IIMethod directly supports prescribed motion.

This is a source audit, not a CFD verification or validation run. It reads the
version-matched IBAMR source and writes a new, immutable audit directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_evidence(path: Path, needles: tuple[str, ...]) -> list[dict[str, object]]:
    found = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if any(needle in line for needle in needles):
            found.append({"line": number, "text": line.strip()})
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="IBAMR source root for the built 0.19.0 installation")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for the immutable source audit")
    args = parser.parse_args()

    files = {
        "header": args.source_root / "include/ibamr/IIMethod.h",
        "implementation": args.source_root / "src/IB/IIMethod.cpp",
        "moving_fsi_example": args.source_root / "examples/IIM/ex5/example.cpp",
        "flexible_plate_example": args.source_root / "examples/IIM/ex9/example.cpp",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing IBAMR source files:\n" + "\n".join(missing))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing audit directory: {output_dir}")

    evidence = {
        "header_surface_representation": line_evidence(files["header"], (
            "finite element representation of a surface mesh", "registerInitialCoordinateMappingFunction",
            "registerInitialVelocityFunction", "registerTangentialVelocityMotion", "use_direct_forcing",
        )),
        "implementation_motion_updates": line_evidence(files["implementation"], (
            "IIMethod::forwardEulerStep", "IIMethod::midpointStep", "IIMethod::trapezoidalStep",
            "d_use_tangential_velocity", "d_use_direct_forcing", "getFromInput", "use_direct_forcing",
        )),
        "built_in_motion_registrations": line_evidence(files["header"], (
            "registerInitialCoordinateMappingFunction", "registerInitialVelocityFunction",
            "registerTangentialVelocityMotion",
        )),
        "moving_example_setup": line_evidence(files["moving_fsi_example"], (
            "new IIMethod", "FEMechanicsExplicitIntegrator", "registerLagSurfaceForceFunction",
        )),
        "flexible_example_setup": line_evidence(files["flexible_plate_example"], (
            "new IIMethod", "FEMechanicsExplicitIntegrator", "registerLagSurfaceForceFunction",
        )),
    }
    header_text = files["header"].read_text(errors="replace")
    impl_text = files["implementation"].read_text(errors="replace")
    input_motion_option = "use_direct_forcing" in impl_text and "db->getBool(\"use_direct_forcing\")" in impl_text
    direct_forcing_freezes_coordinates = all(
        "VecCopy(d_X_current_vecs[part]->vec(), d_X_new_vecs[part]->vec())" in impl_text[start:end]
        for start, end in (
            (impl_text.index("IIMethod::forwardEulerStep"), impl_text.index("IIMethod::midpointStep")),
            (impl_text.index("IIMethod::midpointStep"), impl_text.index("IIMethod::trapezoidalStep")),
            (impl_text.index("IIMethod::trapezoidalStep"), impl_text.index("IIMethod::computeLagrangianForce")),
        )
    )
    audit = {
        "audit_type": "pinned_source_capability_audit_not_solver_run",
        "ibamr_version": "0.19.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root.resolve()),
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in files.items()
        },
        "findings": {
            "uses_finite_element_surface_mesh": "finite element representation of a surface mesh" in header_text,
            "built_in_initial_mapping_and_velocity_hooks": (
                "registerInitialCoordinateMappingFunction" in header_text
                and "registerInitialVelocityFunction" in header_text
            ),
            "built_in_prescribed_time_dependent_motion_hook_found": False,
            "stock_coordinate_update_uses_interpolated_fluid_velocity": (
                "VecWAXPY(d_X_new_vecs[part]->vec(), dt, d_U_current_vecs[part]->vec(), d_X_current_vecs[part]->vec())" in impl_text
                and "VecWAXPY(d_X_new_vecs[part]->vec(), dt, d_U_half_vecs[part]->vec(), d_X_current_vecs[part]->vec())" in impl_text
            ),
            "direct_forcing_is_input_option": input_motion_option,
            "direct_forcing_holds_coordinates_fixed_during_step": direct_forcing_freezes_coordinates,
            "requires_custom_prescribed_motion_strategy_or_solver_extension": True,
            "recommendation": "Do not launch an IIM moth or Stage A pilot yet. First prototype and verify explicit prescribed rigid-body surface kinematics and matching jump/force conditions in an isolated method extension.",
        },
        "evidence": evidence,
        "limitations": [
            "Source inspection does not prove a prescribed-motion extension is impossible; a custom IBStrategy/IIMethod subclass or a solver patch may implement it.",
            "This audit does not build or run any IIM example and establishes no moving-boundary accuracy, no-slip, force, divergence, conservation, or convergence result.",
            "The example setup demonstrates IIM/ILE use with coupled structural mechanics; it is not evidence of a prescribed hawkmoth wing implementation.",
        ],
    }
    output_dir.mkdir(parents=True)
    audit_path = output_dir / "iim_prescribed_motion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "study_id": output_dir.name,
        "status": "SOURCE_AUDIT_COMPLETE_NO_SOLVER_RUN",
        "audit_file": audit_path.name,
        "audit_sha256": sha256(audit_path),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "source_file_sha256": {name: item["sha256"] for name, item in audit["source_files"].items()},
        "scope": "Determine whether pinned IBAMR IIMethod can impose the benchmark's prescribed time-dependent rigid wing motion without an application or method extension.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

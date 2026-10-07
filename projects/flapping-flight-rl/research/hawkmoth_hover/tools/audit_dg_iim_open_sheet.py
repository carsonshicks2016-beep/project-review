#!/usr/bin/env python3
"""Record whether pinned IBAMR has the published DG-IIM capability and its limits.

This is a source/literature applicability audit, not a solver run. It records
the precise distinction between sharp creases on closed bodies (published
DG-IIM evidence) and a free perimeter on a finite, zero-thickness wing sheet
(not established by the inspected paper/examples).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibamr-source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="research/hawkmoth_hover directory")
    parser.add_argument("--paper-pdf", type=Path, required=True,
                        help="arXiv author manuscript PDF for Facci et al. (2025)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.ibamr_source.resolve()
    project = args.project_root.resolve()
    paper = args.paper_pdf.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite nonempty output directory: {output}")

    source_files = {
        "iim_header": source / "include/ibamr/IIMethod.h",
        "iim_impl": source / "src/IB/IIMethod.cpp",
        "iim_ex1": source / "examples/IIM/ex1/example.cpp",
        "iim_ex3": source / "examples/IIM/ex3/example.cpp",
        "iim_ex9": source / "examples/IIM/ex9/example.cpp",
        "iim_sphere_test": source / "tests/IIM/flow_past_sphere.cpp",
        "iim_cylinder_test": source / "tests/IIM/flow_past_cylinder.cpp",
    }
    project_files = {
        "main_cpp": project / "src/main.cpp",
        "right_wing_points": project / "case/geometry/right_wing.vertex",
        "left_wing_points": project / "case/geometry/left_wing.vertex",
        "geometry_manifest": project / "case/geometry/geometry_manifest.json",
        "prior_topology_audit": project / "results/stage_a_iim_open_surface_applicability_local_20261002_v3/iim_open_surface_applicability_audit.json",
    }
    missing = [str(p) for p in [*source_files.values(), *project_files.values(), paper]
               if not p.is_file()]
    if missing:
        parser.error("required files missing: " + ", ".join(missing))

    header = source_files["iim_header"].read_text(errors="replace")
    impl = source_files["iim_impl"].read_text(errors="replace")
    api = {
        "pressure_jump_discontinuous_basis_api": "registerDisconElemFamilyForPressureJump" in header,
        "viscous_jump_discontinuous_basis_api": "registerDisconElemFamilyForViscousJump" in header,
        "pressure_jump_registration_implemented": "IIMethod::registerDisconElemFamilyForPressureJump" in impl,
        "viscous_jump_registration_implemented": "IIMethod::registerDisconElemFamilyForViscousJump" in impl,
    }
    example_evidence = {}
    for name in ("iim_ex1", "iim_ex3", "iim_ex9", "iim_sphere_test", "iim_cylinder_test"):
        text = source_files[name].read_text(errors="replace")
        example_evidence[name] = {
            "source_sha256": sha256(source_files[name]),
            "registers_discontinuous_jump_basis": (
                "registerDisconElemFamilyForPressureJump" in text and
                "registerDisconElemFamilyForViscousJump" in text
            ),
            "prescribed_motion_or_penalty_tokens": {
                token: (token in text)
                for token in ("registerLagSurfaceForceFunction", "registerLagSurfacePressureFunction",
                              "registerCoordinateMappingFunction", "registerTangentialVelocityMotion")
            },
        }

    manifest = json.loads(project_files["geometry_manifest"].read_text())
    record = {
        "audit_type": "dg_iim_open_sheet_source_and_literature_screen_not_solver_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ibamr_release": "0.19.0",
        "pinned_source_root": str(source),
        "source_files": {name: {"path": str(path), "sha256": sha256(path)}
                         for name, path in source_files.items()},
        "project_files": {name: {"path": str(path), "sha256": sha256(path)}
                          for name, path in project_files.items()},
        "paper": {
            "citation": "Facci, Kolahdouz, and Griffith, An Immersed Interface Method for Incompressible Flows and Geometries with Sharp Features, Journal of Computational Physics 537 (2025), 114119",
            "doi": "10.1016/j.jcp.2025.114119",
            "arxiv": "https://arxiv.org/abs/2410.16466",
            "author_manuscript_url": "https://export.arxiv.org/pdf/2410.16466",
            "pdf_sha256": sha256(paper),
            "scope_from_full_text": [
                "The paper reports that its computations were performed through IBAMR.",
                "Its prescribed-configuration penalty is F = kappa*(xi-chi) + eta*(V-U); its numerical study considers prescribed stationary rigid-body motion.",
                "The numerical geometries include smooth circular/spherical bodies and sharp closed geometries such as square cylinders, a square-plus-triangle wedge, and a cube.",
                "The article does not report a finite open 3-D sheet with a free perimeter or a hawkmoth-like trailing-edge wake.",
                "The paper says DG jump projections address discontinuous jump data at sharp geometric features; this is evidence for creases/corners on the tested interfaces, not by itself proof of open-sheet perimeter treatment.",
            ],
        },
        "pinned_release_capabilities": api,
        "version_matched_example_evidence": example_evidence,
        "project_geometry": {
            "status": manifest.get("geometry_status"),
            "assumptions": manifest.get("model_assumptions", []),
            "point_only_geometry": True,
            "oriented_surface_elements_available": False,
            "free_perimeter_wing_sheet": True,
        },
        "method_assessment": {
            "conclusion": "DG-IIM is a materially stronger, version-available candidate for sharp geometric creases than the prior audit reflected; published and source evidence still does not establish a prescribed 3-D finite open wing sheet with a free perimeter.",
            "does_not_prove": [
                "that stock IIM fails on an open finite sheet",
                "that a closed finite-thickness substitute is anatomically justified",
                "that the existing hawkmoth geometry/kinematics or high-Re wake will be accurate",
                "that the IIM penalty coefficient can be copied dimensionally from the paper",
            ],
            "next_discriminating_case": {
                "geometry": "separately generated, oriented triangulated zero-thickness circular disk with an explicit open perimeter; do not cap or thicken it",
                "method_candidate": "IBAMR 0.19.0 IIM with discontinuous jump bases for pressure and viscous jumps and the paper's prescribed-position penalty; begin eta=0 until midpoint callback velocity time-level is independently verified",
                "flow": "stationary disk in low-Re uniform crossflow, with a domain large enough for a documented finite-domain correction",
                "primary_measurements": ["surface target lag", "kernel-consistent surface no-slip RMS/max", "composite divergence", "integrated reaction-force sign and magnitude", "independent full-domain momentum balance", "edge-local flow and force regularity"],
                "comparison": "Stokes resistance for an infinitesimally thin disk in the unbounded-domain limit, with finite-domain and polygonal-mesh errors quantified rather than hidden",
                "sequence": ["mesh/topology and orientation checks", "quiescent exact-null", "matched translation null", "stationary crossflow", "mesh/time refinement only after short controls pass"],
                "acceptance": "retain the preregistered numerical limits for applicable diagnostics; state any model-specific metric mapping before running, and require force and momentum comparisons to converge without changing the existing Stage A limits",
            },
        },
        "status": "SOURCE_LITERATURE_SCREEN_ONLY_NO_BUILD_NO_SOLVER_RUN",
        "stage_a_status": "FAILED_UNVALIDATED",
    }

    output.mkdir(parents=True, exist_ok=True)
    paper_copy = output / "facci_kolahdouz_griffith_2025_dg_iim_author_manuscript.pdf"
    shutil.copyfile(paper, paper_copy)
    record["paper"]["preserved_local_copy"] = paper_copy.name
    record["paper"]["preserved_local_copy_sha256"] = sha256(paper_copy)
    record_path = output / "dg_iim_open_sheet_screen.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    manifest_path = output / "manifest.json"
    manifest_record = {
        "study_id": output.name,
        "status": record["status"],
        "record_file": record_path.name,
        "record_sha256": sha256(record_path),
        "tool_sha256": sha256(Path(__file__).resolve()),
        "paper_sha256": sha256(paper_copy),
        "source_sha256": {name: item["sha256"] for name, item in record["source_files"].items()},
        "project_sha256": {name: item["sha256"] for name, item in record["project_files"].items()},
    }
    manifest_path.write_text(json.dumps(manifest_record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record": str(record_path), "manifest": str(manifest_path),
                      "record_sha256": sha256(record_path), "paper_sha256": sha256(paper_copy),
                      "status": record["status"], "api": api}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

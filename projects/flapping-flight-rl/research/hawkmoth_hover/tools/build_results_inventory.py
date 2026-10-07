#!/usr/bin/env python3
"""Regenerate docs/results_inventory.md from immutable result artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def show(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def short_hash(value) -> str:
    return str(value)[:12] if value else "—"


def record_for(path: Path) -> tuple[str, dict, dict]:
    resource = read_json(path / "resource_record.json")
    manifest_path = next((path / name for name in
                          ("run_manifest.json", "diagnostic_manifest.json", "manifest.json")
                          if (path / name).is_file()), None)
    manifest = read_json(manifest_path) if manifest_path else {}
    status = (resource.get("status") or resource.get("run_status") or
              manifest.get("status") or manifest.get("launch_status") or
              "RECORD_PRESENT_STATUS_UNSPECIFIED")
    if not (path / "resource_record.json").exists() and str(status).lower().startswith("prepared"):
        status = "PREPARED_NOT_RUN"
    return str(status), resource, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "results_inventory.md")
    args = parser.parse_args()
    roots = sorted(path for path in RESULTS.iterdir() if path.is_dir())
    entries = []
    for path in sorted(p for p in RESULTS.rglob("*") if p.is_dir()):
        has_record = (path / "resource_record.json").is_file()
        has_manifest = any((path / name).is_file() for name in
                           ("run_manifest.json", "diagnostic_manifest.json", "manifest.json"))
        if not (has_record or has_manifest):
            continue
        status, resource, manifest = record_for(path)
        interpretation = read_json(path / "interpretation.json")
        identity = resource.get("run_identity") or {}
        source_hashes = identity.get("application_source_sha256") or {}
        source_hash = (source_hashes.get("research/hawkmoth_hover/src/main.cpp") or
                       manifest.get("application_source_sha256"))
        extra = []
        for key in ("verification_case", "case", "study"):
            if manifest.get(key):
                extra.append(f"{key}={manifest[key]}")
        if manifest.get("interpretation"):
            extra.append("interpretation=" + str(manifest["interpretation"]).replace("|", "/"))
        scientific_status = interpretation.get("classification")
        if scientific_status:
            extra.append("finding=" + str(interpretation.get("interpretation", "")).replace("|", "/"))
        entries.append({
            "path": path.relative_to(ROOT).as_posix(),
            "type": "run record" if has_record else "prepared/diagnostic record",
            "status": status,
            "scientific_status": scientific_status or "—",
            "exit": resource.get("exit_code"),
            "elapsed": resource.get("elapsed_seconds"),
            "rss": (resource.get("peak_process_tree_rss_gb") or
                    resource.get("peak_process_tree_rss_gib")),
            "free": (float(resource["free_disk_bytes_after"]) / 1e9
                     if resource.get("free_disk_bytes_after") is not None else
                     float(resource["free_disk_gib_after"]) * 1.073741824
                     if resource.get("free_disk_gib_after") is not None else None),
            "input_hash": short_hash(manifest.get("input_sha256") or
                                      identity.get("input3d_sha256") or
                                      resource.get("input_sha256")),
            "source_hash": short_hash(source_hash),
            "exe_hash": short_hash(identity.get("executable_sha256_at_launch") or
                                   manifest.get("executable_sha256")),
            "build_hash": short_hash(identity.get("build_environment_sha256") or
                                     manifest.get("build_environment_sha256")),
            "extra": ", ".join(extra).replace("|", "/").replace("\n", " "),
        })

    lines = [
        "# Complete results artifact inventory", "",
        f"Generated from the on-disk result tree on {date.today().isoformat()}. This is a filesystem inventory, not an interpretation that a completed solver process passed a physics gate. `COMPLETED` means only that the runner exited successfully; inspect the case CSVs, summary JSON, and contemporaneous interpretation before assigning scientific meaning. Failed, stopped, intentionally terminated, crashed, and prepared-only cases remain part of the record.", "",
        f"- Result study roots: **{len(roots)}**",
        f"- Directories with run/preparation manifests or resource records: **{len(entries)}**",
        "- `resource_record.json` determines execution status where present; otherwise the input manifest status is shown.",
        "- Paths are relative to `research/hawkmoth_hover/`.", "",
        "## Study-root map", "",
        "| Study root | Direct child directories | Summary/report files directly under root | Summary pointer |",
        "|---|---:|---|---|",
    ]
    for root in roots:
        children = sum(child.is_dir() for child in root.iterdir())
        summaries = sorted(child.name for child in root.iterdir() if child.is_file() and
                           ("summary" in child.name or child.name.endswith("comparison.json") or
                            child.name == "matrix_execution.json"))
        rel = root.relative_to(ROOT).as_posix()
        pointer = (", ".join(f"`{name}`" for name in summaries) if summaries else
                   "See chronological `docs/stage_a_report.md` / `docs/feasibility_log.md`")
        lines.append(f"| `{rel}/` | {children} | {', '.join(f'`{name}`' for name in summaries) if summaries else '—'} | {pointer} |")

    lines += [
        "", "## Run/preparation record inventory", "",
        "The status column reports execution or preparation state only. A successful process may still violate no-slip, divergence, force, conservation, or convergence criteria. Missing resource records generally mean unrun/prepared, older-format, or incomplete artifacts; inspect their files rather than infer a scientific pass or failure.", "",
        "| Exact directory | Record type | Recorded status | Scientific interpretation | Exit | Wall s | Peak RSS GiB | Free disk GB after | Input SHA-256 | `main.cpp` SHA-256 | Executable SHA-256 | Build env SHA-256 | Manifest note |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for item in entries:
        lines.append(
            f"| `{item['path']}/` | {item['type']} | {item['status']} | {item['scientific_status']} | "
            f"{item['exit'] if item['exit'] is not None else '—'} | "
            f"{show(item['elapsed'])} | {show(item['rss'])} | {show(item['free'])} | "
            f"{item['input_hash']} | {item['source_hash']} | {item['exe_hash']} | "
            f"{item['build_hash']} | {item['extra']} |"
        )
    lines += [
        "", "## Evidence precedence and interpretation", "",
        "- Current controlling evidence: `results/stage_a_postfix_matched_translation_matrix/relative_target_damping_summary.json`, `nonzero_flow_momentum_probe_summary.json`, `target_penalty_coupling_study.json`, plus the latest dated sections in the stage report/protocol/feasibility log.",
        "- Latest CIB method pilot is `results/cib_method_pilot_local_20261003_v1/study_summary.json`; follow-up captures are in `results/cib_operator_spread_capture_local_20261003_v1/momentum_quadrature_comparison.json`. The actual side-centered multiplier spread matches the negative raw marker resultant within 2.281e-8 maximum relative error, while using MAC side weights for fluid momentum leaves the normalized residual at approximately 0.3309015. The independently integrated outer-boundary ledger remains unresolved. These diagnostics do not authorize crossflow, refinement, moth hover, PIV/literature comparison, or PPO; see `docs/cib_multiplier_momentum_audit.md`.",
        "- Latest open-sheet IIM evidence is `results/open_disk_dg_iim_local_20261003_v1/study_summary.json` plus `results/open_disk_eta_probe_local_20261003_v1/study_summary.json`. Open mesh acceptance is narrow; the fixed-coordinate ETA probe leaves gross slip at moderate coefficients and becomes numerically unacceptable at a high exploratory coefficient. Neither qualifies no-slip, force, momentum, or Stage A.",
        "- `results/open_disk_force_audit_local_20261003_v1/study_summary.json` records the source-level callback/TAU_OUT audit and diagnostic-label correction. It clarifies that `TAU_OUT` is one-sided exterior traction; the open-sheet total force and no-slip remain unresolved.",
        "- Version-matched upstream regression: `results/stage_a_ibamr_reference/solver_reference_comparison.json`.",
        "- Latest positive result is narrow: `stage_a_postfix_matched_translation_matrix/relative_damping_matched_translation_long/matched_translation_coarse_dt_coarse/` and its summary prove only the tested 0.4 ms matched-flow analytic null. The most recent stationary crossflow records (`nonzero_flow_momentum_probe/...`, `nonzero_flow_momentum_probe_dt_fine/...`, and `nonzero_flow_momentum_probe_medium/...`) remain failed coupling diagnostics. Aggregate Stage A remains failed.",
        "- Most useful forensic records for source repairs include `stage_a_projection_boundary_audit/` (BC registration and AMR fill/projection tracing), `stage_a_regrid_refresh_test2/` (force-evaluator refresh after regrid), `stage_a_fill_path_final5/` through `final7/` (lagged fill-path tracing), and `stage_a_postfix_matched_translation_matrix/` (latest global ledger, midpoint and relative-damping work). These explain mechanisms and fixes; they do not pass Stage A.",
        "- `stage_a/`, `stage_a_attempt*`, `stage_a_retry*`, `stage_a_moving1/`, `stage_a_postinstrumentation/`, `stage_a_final_smoke/`, `stage_a_translation_smoke/`, `stage_a_refinement_smoke_k10/`, and `stage_a_stiffness_*` are historical exploratory runs or incomplete/prepared matrices. Their hypotheses and outcomes remain useful, but values cannot be substituted for later code state.",
        "- `stage_a_diagnosis*`, `stage_a_force_*`, `stage_a_cv_quadrature_audit/`, `stage_a_single_level_force_isolation/`, `stage_a_regrid_force_investigation*`, and `stage_a_amr_flow_by_level/` are diagnostic isolation studies. Where later report sections found an implementation defect and repaired it, the original run stays as evidence of pre-fix behavior.",
        "- `stage_a_zero_amplitude/`, `zero_motion_*`, and stationary/quiescent entries establish only their short zero-motion control scope. They do not exercise nonzero-relative-velocity IB force enforcement.",
        "- Earlier stage directories preserve the iterations and failures that led to later corrections. Do not combine results across source/executable hashes as one convergence study.",
        "- Early rollups such as `results/stage_a_summary.json` predate later fixes. Keep them as historical evidence; use later hashed summaries to characterize the current implementation.",
        "- `stage_a_projection_boundary_audit/prepared/...` entries are one-factor instrumented diagnostics, not immersed-boundary force validation.",
        "- Some nonzero exits are intentional stop-after-divergence actions; read each run's `interpretation.json`, `resource_record.json`, stdout/stderr and manifest.",
        "- Any quoted result must identify its run directory and input, source, executable, solver/build hashes. If identity fields are absent, state that explicitly.",
        "- Hash columns show the first 12 characters to keep the table scannable. The full SHA-256 values must be read from that directory's run manifest/resource record; `—` means the older/prepared record did not record that identity.",
        "", "## Regeneration", "",
        "From `research/hawkmoth_hover/`, run `python3 tools/build_results_inventory.py`. The script only reads `results/` metadata and rewrites this inventory; it does not modify solver artifacts.", "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output} ({len(roots)} study roots, {len(entries)} record directories)")


if __name__ == "__main__":
    main()

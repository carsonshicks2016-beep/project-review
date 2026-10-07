#!/usr/bin/env python3
"""Audit source-level IIM APIs against the actually installed IBAMR build.

This is an environment/build-capability record, not a CFD run. Output creation
is immutable: a nonempty output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibamr-prefix", type=Path, required=True)
    parser.add_argument("--ibamr-build", type=Path, required=True,
                        help="configured IBAMR CMake build directory")
    parser.add_argument("--ibamr-source", type=Path, required=True)
    parser.add_argument("--autoibamr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prefix = args.ibamr_prefix.resolve()
    build = args.ibamr_build.resolve()
    source = args.ibamr_source.resolve()
    toolchain = args.autoibamr_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite nonempty output directory: {output}")

    files = {
        "ibamr_config": prefix / "lib/cmake/ibamr/IBAMRConfig.cmake",
        "ibtk_config_header": prefix / "include/ibtk/config.h",
        "ibamr_config_header": prefix / "include/ibamr/config.h",
        "cmake_cache": build / "CMakeCache.txt",
        "configure_log": build / "autoibamr_configure.log",
        "build_record": build / "autoibamr_build",
        "iim_header": source / "include/ibamr/IIMethod.h",
        "iim_impl": source / "src/IB/IIMethod.cpp",
        "libmesh_recipe": toolchain / "IBAMR-toolchain/packages/libmesh.package",
        "ibamr_library": prefix / "lib/libIBAMR3d.dylib",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("required evidence file(s) missing: " + ", ".join(missing))

    config = files["ibamr_config"].read_text(errors="replace")
    ibtk_header = files["ibtk_config_header"].read_text(errors="replace")
    cmake_cache = files["cmake_cache"].read_text(errors="replace")
    configure_log = files["configure_log"].read_text(errors="replace")
    build_record = files["build_record"].read_text(errors="replace")
    iim_header = files["iim_header"].read_text(errors="replace")
    iim_impl = files["iim_impl"].read_text(errors="replace")
    libmesh_recipe = files["libmesh_recipe"].read_text(errors="replace")

    library_symbols = subprocess.run(
        ["nm", "-gU", str(files["ibamr_library"])],
        check=False, capture_output=True, text=True,
    )
    symbol_lines = [line for line in library_symbols.stdout.splitlines()
                    if re.search(r"IIMethod|registerDisconElemFamilyFor", line)]
    have_libmesh = bool(re.search(r'SET\(IBAMR_HAVE_LIBMESH\s+"TRUE"\)', config))
    have_ibtk_macro = bool(re.search(r"^#define IBTK_HAVE_LIBMESH\b", ibtk_header, re.M))
    source_api = {
        "pressure_dg_jump_api_declared": "registerDisconElemFamilyForPressureJump" in iim_header,
        "viscous_dg_jump_api_declared": "registerDisconElemFamilyForViscousJump" in iim_header,
        "iim_implementation_has_pressure_api": "IIMethod::registerDisconElemFamilyForPressureJump" in iim_impl,
        "iim_implementation_has_viscous_api": "IIMethod::registerDisconElemFamilyForViscousJump" in iim_impl,
    }
    explicit_disable = "--disable-libmesh" in build_record
    build_has_iim = have_libmesh and have_ibtk_macro and bool(symbol_lines)
    report = {
        "audit_type": "installed_ibamr_iim_build_capability_not_solver_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ibamr_release": "0.19.0",
        "installed_prefix": str(prefix),
        "build_directory": str(build),
        "evidence_files": {key: {"path": str(path), "sha256": sha256(path)}
                           for key, path in files.items()},
        "installed_build_features": {
            "IBAMR_HAVE_LIBMESH": have_libmesh,
            "IBTK_HAVE_LIBMESH_macro_defined": have_ibtk_macro,
            "build_record_explicitly_disables_libmesh": explicit_disable,
            "configure_log_says_libmesh_unavailable_or_disabled": (
                bool(re.search(r"LIBMESH_ROOT was not (provided|specified)", configure_log, re.I)) or
                bool(re.search(r"Setting up without libMesh", configure_log, re.I))
            ),
            "IIM_symbols_exported_by_installed_3d_library": bool(symbol_lines),
            "IIM_symbol_matches": symbol_lines[:50],
            "nm_returncode": library_symbols.returncode,
            "nm_stderr": library_symbols.stderr.strip(),
        },
        "source_capability": source_api,
        "interpretation": {
            "source_and_installed_binary_are_distinct": True,
            "conclusion": (
                "The installed IBAMR 0.19.0 package is libMesh-enabled and exports "
                "IIMethod symbols; IIM is available to applications linked against "
                "this isolated prefix. This confirms build capability only, not IIM "
                "numerical validity or open-sheet suitability."
                if build_has_iim else
                "The unpacked IBAMR 0.19.0 source contains IIM and DG jump-basis APIs, "
                "but the locally installed package was configured and built without "
                "libMesh. IIM cannot be treated as available in this installed build. "
                "A separate libMesh-enabled IBAMR build is required before any IIM case."
            ),
            "no_solver_was_built_or_run": True,
            "no_source_or_existing_install_was_modified": True,
        },
        "build_provenance": {
            "cmake_cache_libmesh_entries": [line for line in cmake_cache.splitlines()
                                             if "LIBMESH" in line.upper()],
            "configure_log_libmesh_entries": [line for line in configure_log.splitlines()
                                               if "LIBMESH" in line.upper()],
            "build_record_libmesh_entries": [line for line in build_record.splitlines()
                                             if "LIBMESH" in line.upper()],
            "libmesh_package_version": "1.7.8",
            "libmesh_recipe_checksum": next(
                (line.split("=", 1)[1].strip() for line in libmesh_recipe.splitlines()
                 if line.startswith("CHECKSUM=")), None
            ),
            "separate_build_required_to_preserve_existing_install": True,
        },
        "status": "LOCAL_IIM_BUILD_CONFIRMED" if build_has_iim else "LOCAL_IIM_BUILD_UNAVAILABLE_LIBMESH_DISABLED",
        "stage_a_status": "FAILED_UNVALIDATED",
    }

    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "ibamr_iim_build_capability.json"
    record_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {
        "study_id": output.name,
        "status": report["status"],
        "record_file": record_path.name,
        "record_sha256": sha256(record_path),
        "tool_sha256": sha256(Path(__file__).resolve()),
        "evidence_sha256": {key: item["sha256"] for key, item in report["evidence_files"].items()},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record": str(record_path), "manifest": str(manifest_path),
                      "status": report["status"],
                      "features": report["installed_build_features"]}, indent=2))
    return 0 if library_symbols.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

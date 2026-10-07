#!/usr/bin/env python3
"""Create immutable 3-D analytic-flow inputs for the outer-flux ledger audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[5]
TEMPLATE = REPO_ROOT / "research/hawkmoth_hover/reference_data/ibamr_0.19.0_navier_stokes/navier_stokes_01_3d.input"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1> {value}", text, count=1)
    if count != 1:
        raise ValueError(f"could not replace input assignment {key}")
    return updated


def replace_block(text: str, name: str, body: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*\{{", text)
    if not match:
        raise ValueError(f"could not find input block {name}")
    opening = text.find("{", match.start())
    depth = 0
    closing = None
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index + 1
                break
    if closing is None:
        raise ValueError(f"unclosed input block {name}")
    replacement = f"{name} {{\n{body.rstrip()}\n}}"
    return text[: match.start()] + replacement + text[closing:]


def bc_block(function: str, mode: str) -> str:
    lines = ["   mu = MU", "   G = G_PRESSURE", "   ly = LY"]
    for key in ("acoef", "bcoef"):
        value = '"1.0"' if key == "acoef" else '"0.0"'
        lines.extend(f"   {key}_function_{face} = {value}" for face in range(6))
    lines.extend(f"   gcoef_function_{face} = {function}" for face in range(6))
    return "\n".join(lines)


def render(mode: str, cells: int, dt: float, duration_steps: int) -> str:
    text = TEMPLATE.read_text()
    p_start = text.index("// physical parameters")
    p_end = text.index("// grid spacing parameters")
    physical = (
        f'// Analytic outer-momentum-flux control ({mode}); all quantities are code units.\n'
        f'CASE_MODE = "{mode}"\n'
        'RHO = 1.0\n'
        'MU = 1.0\n'
        'U0 = 1.0\n'
        'G_PRESSURE = 1.0\n'
        'LX = 1.0\nLY = 1.0\nLZ = 1.0\n'
        'L = 1.0\n'
    )
    text = text[:p_start] + physical + text[p_end:]

    values = {
        "MAX_LEVELS": "1",
        "REF_RATIO": "2",
        "N": str(cells),
        "NFINEST": str(cells),
        "CFL_MAX": "0.1",
        "DT_MAX": f"{dt:.17g}",
        "START_TIME": "0.0",
        "END_TIME": f"{duration_steps}*DT_MAX",
        "GROW_DT": "1.0",
        "NUM_CYCLES": "1",
        "NORMALIZE_PRESSURE": "FALSE",
        "VORTICITY_TAGGING": "FALSE",
        "REGRID_INTERVAL": "10000000",
        "OUTPUT_U": "FALSE",
        "OUTPUT_P": "FALSE",
        "OUTPUT_F": "FALSE",
        "OUTPUT_OMEGA": "FALSE",
        "OUTPUT_DIV_U": "TRUE",
        "ENABLE_LOGGING": "TRUE",
    }
    for key, value in values.items():
        text = replace_assignment(text, key, value)

    if mode == "UNIFORM":
        functions = {
            "U": '"1.0"',
            "V": '"0.0"',
            "W": '"0.0"',
            "P": '"0.0"',
        }
    elif mode == "POISEUILLE":
        functions = {
            "U": '"G/(2.0*mu)*X_1*(ly-X_1)"',
            "V": '"0.0"',
            "W": '"0.0"',
            "P": '"-G*X_0"',
        }
    else:
        raise ValueError(f"unsupported flow mode: {mode}")

    for name, expression in functions.items():
        text, count = re.subn(rf"(?m)^{name}\s*=.*$", f"{name} = {expression}", text, count=1)
        if count != 1:
            raise ValueError(f"could not replace exact function {name}")

    velocity_initial = "\n".join(
        ["   mu = MU", "   G = G_PRESSURE", "   ly = LY"]
        + [f"   function_{d} = {name}" for d, name in enumerate(("U", "V", "W"))]
    )
    text = replace_block(text, "VelocityInitialConditions", velocity_initial)
    for component, name in enumerate(("U", "V", "W")):
        text = replace_block(text, f"VelocityBcCoefs_{component}", bc_block(name, mode))
    text = replace_block(text, "PressureInitialConditions", "   G = G_PRESSURE\n   function = P")
    text = replace_block(
        text,
        "CartesianGeometry",
        "   domain_boxes = [ (0,0,0),(N - 1,N - 1,N - 1) ]\n"
        "   x_lo = 0.0,0.0,0.0\n"
        "   x_up = LX,LY,LZ\n"
        "   periodic_dimension = 0,0,0",
    )
    text = replace_assignment(text, "viz_dump_interval", "0")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("UNIFORM", "POISEUILLE"))
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cells < 8 or args.cells % 2:
        parser.error("--cells must be an even integer of at least 8")
    if args.dt <= 0.0 or args.steps < 1:
        parser.error("--dt must be positive and --steps at least one")

    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    input_path = run_dir / "input3d"
    input_path.write_text(render(args.mode, args.cells, args.dt, args.steps))
    record = {
        "status": "PREPARED_NOT_RUN",
        "case_mode": args.mode,
        "grid_cells_per_axis": args.cells,
        "domain": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "requested_dt_code": args.dt,
        "requested_steps": args.steps,
        "flow": {
            "uniform": "u=(U0,0,0), p=0, U0=1" if args.mode == "UNIFORM" else None,
            "poiseuille": "u_x=G*y*(Ly-y)/(2*mu), u_y=u_z=0, p=-G*x, G=1" if args.mode == "POISEUILLE" else None,
            "rho": 1.0,
            "mu": 1.0,
        },
        "template_path": str(TEMPLATE),
        "template_sha256": sha256(TEMPLATE),
        "generator_path": str(HERE),
        "generator_sha256": sha256(HERE),
        "input_sha256": sha256(input_path),
    }
    (run_dir / "preparation_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

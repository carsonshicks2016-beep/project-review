#!/usr/bin/env python3
"""Smoke-test and identify the pinned faithful authority runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _version(module_name: str) -> str:
    module = __import__(module_name)
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{module_name} does not expose a version")
    return value


def _ipopt_smoke() -> dict[str, float | str]:
    import casadi as ca

    x = ca.MX.sym("x")
    problem = {"x": x, "f": (x - 3.0) ** 2}
    solver = ca.nlpsol("faithful_runtime_smoke", "ipopt", problem, {
        "ipopt.print_level": 0, "print_time": False,
        "ipopt.sb": "yes", "ipopt.max_iter": 25,
    })
    result = solver(x0=0.0)
    solution = float(result["x"])
    if abs(solution - 3.0) > 1e-9:
        raise RuntimeError(f"IPOPT smoke solution drifted: {solution}")
    return {"solver": "ipopt", "solution": solution}


def runtime_identity() -> dict[str, object]:
    import jax
    import mujoco
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX x64 is not enabled")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><option timestep='0.001'/><worldbody>"
        "<body><freejoint/><geom type='sphere' size='0.1' mass='1'/></body>"
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    if not np.isclose(float(data.time), 0.01, atol=1e-15, rtol=0.0):
        raise RuntimeError("MuJoCo fixed-step smoke test failed")
    lock = ROOT / "requirements-faithful-lock.txt"
    lock_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
    return {
        "schema": "faithful-authority-runtime-identity-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {
            name: _version(name) for name in (
                "numpy", "scipy", "mujoco", "jax", "casadi", "cryptography", "trimesh"
            )
        },
        "jax_x64": True,
        "mujoco_timestep_s": model.opt.timestep,
        "mujoco_time_after_10_steps_s": float(data.time),
        "ipopt": _ipopt_smoke(),
        "dependency_lock_sha256": lock_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-identity", metavar="PATH")
    args = parser.parse_args()
    identity = runtime_identity()
    encoded = json.dumps(identity, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.write_identity:
        path = Path(args.write_identity)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

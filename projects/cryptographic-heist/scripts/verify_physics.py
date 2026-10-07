from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.physics_validation import run_physics_validation


def main():
    parser = argparse.ArgumentParser(description="Run vehicle physics acceptance checks.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--turn-speed", type=float, default=24.0)
    parser.add_argument("--turn-steps", type=int, default=180)
    parser.add_argument("--trace-steps", type=int, default=360)
    parser.add_argument("--long-steps", type=int, default=1200)
    parser.add_argument("--json-out", default="logs/physics_validation.json")
    args = parser.parse_args()
    manifest = run_physics_validation(
        seed=args.seed,
        turn_speed=args.turn_speed,
        turn_steps=args.turn_steps,
        trace_steps=args.trace_steps,
        long_steps=args.long_steps,
        out=args.json_out,
    )
    asym = next(item for item in manifest["results"] if item["name"] == "handbrake_asymmetry")
    print(
        f"physics passed={manifest['passed']} checks={manifest['checks_passed']}/{manifest['checks']} "
        f"yaw_ratio={asym['metrics']['yaw_ratio']:.3f} "
        f"radius_ratio={asym['metrics']['radius_ratio']:.3f} "
        f"speed_ratio={asym['metrics']['speed_ratio']:.3f}"
    )
    raise SystemExit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()

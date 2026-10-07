from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.env_validation import run_env_validation


def main():
    parser = argparse.ArgumentParser(description="Run MARL environment contract checks.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--json-out", default="logs/env_validation.json")
    args = parser.parse_args()
    manifest = run_env_validation(seed=args.seed, steps=args.steps, out=args.json_out)
    print(
        f"env passed={manifest['passed']} "
        f"checks={manifest['checks_passed']}/{manifest['checks']} out={args.json_out}"
    )
    for check in manifest["results"]:
        if not check["passed"]:
            print(f"  failed={check['name']} metrics={check['metrics']}")
    raise SystemExit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()

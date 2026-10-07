from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.operational_validation import run_operational_validation


def main():
    parser = argparse.ArgumentParser(description="Run whole-project operational readiness checks.")
    parser.add_argument("--json-out", default="logs/operational_readiness.json")
    args = parser.parse_args()
    manifest = run_operational_validation(out=args.json_out)
    print(
        f"operational passed={manifest['passed']} "
        f"checks={manifest['checks_passed']}/{manifest['checks']} out={args.json_out}"
    )
    for check in manifest["results"]:
        if not check["passed"]:
            print(f"  failed={check['name']} metrics={check['metrics']}")
    raise SystemExit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()

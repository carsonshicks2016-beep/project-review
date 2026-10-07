from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.evidence import verify_evidence_bundle


def main():
    parser = argparse.ArgumentParser(description="Verify a research evidence bundle against current files.")
    parser.add_argument("bundle", nargs="?", default="logs/evidence_bundle.json")
    parser.add_argument("--json-out", default="logs/evidence_bundle_verification.json")
    args = parser.parse_args()

    manifest = verify_evidence_bundle(args.bundle, out=args.json_out)
    print(
        f"evidence_bundle_verification passed={manifest['passed']} "
        f"checks={manifest['checks_passed']}/{manifest['checks']} "
        f"failures={len(manifest['failures'])} bundle={args.bundle} out={args.json_out}"
    )
    if not manifest["passed"]:
        for failure in manifest["failures"][:8]:
            print(f"  {failure.get('area')}:{failure.get('reason')} {failure.get('name', '')}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

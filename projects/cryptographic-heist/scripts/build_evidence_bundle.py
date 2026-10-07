from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.evidence import build_evidence_bundle


def main():
    parser = argparse.ArgumentParser(description="Build a research evidence bundle from current manifests and replay artifacts.")
    parser.add_argument("--out", default="logs/evidence_bundle.json")
    args = parser.parse_args()
    bundle = build_evidence_bundle(out=args.out)
    summary = bundle["summary"]
    print(
        f"evidence_bundle passed={bundle['passed']} "
        f"operational={summary['operational_checks_passed']}/{summary['operational_checks']} "
        f"active={summary['active_required_checks_passed']}/{summary['active_required_checks']} "
        f"reports={summary['active_replay_reports_nominal']}/{summary['active_replay_reports']} "
        f"checkpoints={summary['checkpoints_existing']}/{summary['checkpoints_expected']} "
        f"out={args.out}"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not bundle["passed"]:
        print(json.dumps(bundle["summary"], indent=2, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

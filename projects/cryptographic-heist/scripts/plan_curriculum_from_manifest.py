from __future__ import annotations

import argparse
import sys

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.curriculum import (
    load_curriculum_manifest,
    plan_curriculum_from_manifest,
    write_curriculum_plan,
)


def main():
    parser = argparse.ArgumentParser(description="Route acceptance diagnostics to curriculum actions.")
    parser.add_argument(
        "--manifest",
        default="logs/acceptance_scenarios.json",
        help="Acceptance manifest written by eval_acceptance_scenarios.py.",
    )
    parser.add_argument("--out", default="logs/curriculum_plan.json")
    parser.add_argument("--max-actions", type=int, default=5)
    args = parser.parse_args()

    manifest = load_curriculum_manifest(args.manifest)
    plan = plan_curriculum_from_manifest(manifest, source=args.manifest, max_actions=args.max_actions)
    out = write_curriculum_plan(plan, args.out)

    print(
        f"curriculum_plan status={plan['status']} diagnostics={plan['diagnostic_count']} "
        f"actions={len(plan['actions'])} out={out}"
    )
    print(plan["summary"])
    for action in plan["actions"]:
        print(
            f"  priority={action['priority']} lane={action['lane']} "
            f"command={action['command_id']} reasons={','.join(action['diagnostic_reasons']) or 'none'}"
        )
        print(f"    args={action['default_args']}")
        if action.get("followup_command_ids"):
            print(f"    followup={','.join(action['followup_command_ids'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

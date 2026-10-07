from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.spectator_validation import run_spectator_validation


def main():
    parser = argparse.ArgumentParser(description="Run spectator UI and audio acceptance checks.")
    parser.add_argument("--replay", default="replays/control_acceptance_active/downtown_chase_seed11.jsonl")
    parser.add_argument("--pygame-frames", type=int, default=5)
    parser.add_argument("--json-out", default="logs/spectator_validation.json")
    args = parser.parse_args()
    manifest = run_spectator_validation(
        replay=args.replay,
        pygame_frames=args.pygame_frames,
        out=args.json_out,
    )
    print(
        f"spectator passed={manifest['passed']} "
        f"checks={manifest['checks_passed']}/{manifest['checks']} replay={args.replay}"
    )
    raise SystemExit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()

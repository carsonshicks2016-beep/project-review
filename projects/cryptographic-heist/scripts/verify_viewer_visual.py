from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.viewer_visual_validation import run_viewer_visual_validation


def main():
    parser = argparse.ArgumentParser(description="Capture desktop/mobile cockpit screenshots and canvas-pixel gates.")
    parser.add_argument("--replay", default="replays/control_acceptance_active/downtown_chase_seed11.jsonl")
    parser.add_argument("--json-out", default="logs/viewer_visual_validation.json")
    parser.add_argument("--screenshot-dir", default="logs/viewer_visual")
    parser.add_argument("--settle-ms", type=int, default=2500)
    args = parser.parse_args()
    manifest = run_viewer_visual_validation(
        replay=args.replay,
        out=args.json_out,
        screenshot_dir=args.screenshot_dir,
        settle_ms=args.settle_ms,
    )
    print(
        f"viewer_visual passed={manifest['passed']} "
        f"checks={manifest['checks_passed']}/{manifest['checks']} replay={args.replay}"
    )
    raise SystemExit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()

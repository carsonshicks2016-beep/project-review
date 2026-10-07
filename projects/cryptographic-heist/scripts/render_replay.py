from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.replay import load_replay, summarize_replay, validate_replay


def main():
    parser = argparse.ArgumentParser(description="Validate and summarize a replay.")
    parser.add_argument("replay")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    replay = load_replay(args.replay)
    validate_replay(replay)
    summary = summarize_replay(replay)
    print(
        f"frames={summary['frames']} duration={summary['duration']:.2f}s "
        f"agents={summary['agents']} radio_events={summary['radio_events']} "
        f"spoofed={summary['spoofed_radio_events']} captures={summary['captures']} "
        f"waypoints={summary['waypoints_hit']} deception={summary['deception_score']:.1f} "
        f"avg_confidence={summary['avg_confidence']:.3f} "
        f"state_sha256={summary['state_sha256'][:16]} radio_sha256={summary['radio_sha256'][:16]}"
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.replay import load_replay, replay_diagnostic_report


def main():
    parser = argparse.ArgumentParser(description="Write a replay diagnostic report.")
    parser.add_argument("replay")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    lines = load_replay(args.replay)
    report = replay_diagnostic_report(args.replay, lines)
    score = report["score"]
    info = report["sections"]["information_warfare"]
    reward = report["sections"]["reward_trace"]
    print(
        f"replay_report replay={args.replay} status={score['status']} "
        f"score={score['overall']:.3f} diagnostics={score['diagnostic_count']} "
        f"entropy={info['radio_word_entropy']:.2f} jams={info['jam_events']} "
        f"cipher={info['cipher_rotations']} evader_reward={reward['avg_evader_reward']:.3f} "
        f"auth_penalty={reward['avg_pursuer_auth_penalty']:.3f}"
    )
    for diagnostic in report["diagnostics"][:5]:
        print(
            f"  {diagnostic['severity']}:{diagnostic['area']}:{diagnostic['reason']} - "
            f"{diagnostic['message']}"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

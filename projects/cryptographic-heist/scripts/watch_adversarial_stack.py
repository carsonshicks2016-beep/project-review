from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.adversarial import run_adversarial_eval


def main():
    parser = argparse.ArgumentParser(description="Run learned radio, scanner, and jammer together.")
    parser.add_argument("--radio", default="checkpoints/radio_policy.pt")
    parser.add_argument("--scanner", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--jammer", default="checkpoints/jammer_policy.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record", default="replays/adversarial_stack.jsonl")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = run_adversarial_eval(
        radio_checkpoint=args.radio,
        scanner_checkpoint=args.scanner,
        jammer_checkpoint=args.jammer,
        seed=args.seed,
        steps=args.steps,
        record=args.record,
    )
    data = result.to_dict()
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = result.metrics
    print(
        f"adversarial steps={metrics['steps']} replay={result.replay} "
        f"radio={metrics['unique_radio_events']} spoofed={metrics['unique_spoofed_events']} "
        f"jams={metrics['jam_bursts']} deception={metrics['deception_score']:.1f} "
        f"confidence={metrics['final_confidence']:.3f} decoder_error={metrics['avg_decoder_error']:.2f}"
    )


if __name__ == "__main__":
    main()

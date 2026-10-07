from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.adversarial import run_authentication_curriculum_eval


def main():
    parser = argparse.ArgumentParser(description="Evaluate jammed vs no-spoof authentication curriculum metrics.")
    parser.add_argument("--radio", default="checkpoints/radio_policy.pt")
    parser.add_argument("--scanner", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--jammer", default="checkpoints/jammer_policy.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-prefix", default="replays/auth_curriculum")
    parser.add_argument("--json-out", default="replays/auth_curriculum_metrics.json")
    args = parser.parse_args()

    result = run_authentication_curriculum_eval(
        radio_checkpoint=args.radio,
        scanner_checkpoint=args.scanner,
        jammer_checkpoint=args.jammer,
        seed=args.seed,
        steps=args.steps,
        record_prefix=args.record_prefix,
    )
    data = result.to_dict()
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = result.metrics
    print(
        f"auth_eval steps={metrics['steps']} "
        f"susceptibility={metrics['spoof_susceptibility_counterfactual']:.3f} "
        f"trajectory={metrics['trajectory'].get('mean_pursuer_deviation', 0.0):.2f} "
        f"confidence_damage={metrics['confidence_damage']:.3f} "
        f"decoder_delta={metrics['decoder_error_delta']:.2f} "
        f"deception_lift={metrics['deception_lift']:.1f}"
    )


if __name__ == "__main__":
    main()

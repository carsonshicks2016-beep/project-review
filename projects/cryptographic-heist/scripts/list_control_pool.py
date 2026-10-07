from __future__ import annotations

import argparse
import json

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.control_pool import load_control_pool, summarize_control_pool


def main():
    parser = argparse.ArgumentParser(description="Inspect the historical control self-play pool.")
    parser.add_argument("--pool-dir", default="models/control_pool")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize_control_pool(args.pool_dir)
    entries = load_control_pool(args.pool_dir)
    if args.json:
        payload = dict(summary)
        payload["all_entries"] = [entry.to_dict() for entry in entries]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"control_pool entries={summary['entries']} "
        f"manifest={summary['manifest']} pool_dir={summary['pool_dir']}"
    )
    if summary["best"]:
        best_scores = (summary["best"].get("control_evaluation") or {}).get("scores", {})
        print(
            f"best id={summary['best']['id']} generation={summary['best']['generation']} "
            f"overall={float(best_scores.get('overall_score', 0.0)):.3f}"
        )
    for entry in entries[-8:]:
        scores = (entry.control_evaluation or {}).get("scores", {})
        print(
            f"entry id={entry.id} generation={entry.generation} seed={entry.seed} "
            f"overall={float(scores.get('overall_score', 0.0)):.3f} "
            f"evader={entry.evader} pursuer_team={entry.pursuer_team}"
        )


if __name__ == "__main__":
    main()

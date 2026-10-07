from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.replay import (
    compare_replay_fingerprints,
    load_replay,
    record_scripted_replay,
    replay_fidelity_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="Write replay checksum and determinism evidence.")
    parser.add_argument("replay", help="Replay JSONL path to validate and fingerprint.")
    parser.add_argument("--compare", default=None, help="Optional second replay to compare against.")
    parser.add_argument("--json-out", default=None, help="Optional manifest output path.")
    parser.add_argument("--scripted-determinism", action="store_true", help="Generate two scripted replays and compare them.")
    parser.add_argument("--scripted-seed", type=int, default=11)
    parser.add_argument("--scripted-steps", type=int, default=1200)
    args = parser.parse_args()

    manifest = replay_fidelity_manifest(args.replay)
    failed = False

    if args.compare:
        manifest["compare"] = compare_replay_fingerprints(load_replay(args.replay), load_replay(args.compare))
        manifest["compare"]["replay"] = args.compare
        failed = failed or not manifest["compare"]["matched"]

    if args.scripted_determinism:
        with tempfile.TemporaryDirectory(prefix="crypt-heist-fidelity-") as td:
            left = Path(td) / "left.jsonl"
            right = Path(td) / "right.jsonl"
            record_scripted_replay(left, seed=args.scripted_seed, steps=args.scripted_steps)
            record_scripted_replay(right, seed=args.scripted_seed, steps=args.scripted_steps)
            comparison = compare_replay_fingerprints(load_replay(left), load_replay(right))
        comparison["seed"] = args.scripted_seed
        comparison["steps"] = args.scripted_steps
        manifest["scripted_determinism"] = comparison
        failed = failed or not comparison["matched"]

    manifest["passed"] = not failed
    summary = manifest["summary"]
    fingerprint = manifest["fingerprint"]
    print(
        f"replay={args.replay} frames={summary['frames']} "
        f"state_sha256={fingerprint['state_sha256'][:16]} "
        f"radio_sha256={fingerprint['radio_sha256'][:16]} passed={manifest['passed']}"
    )
    if "scripted_determinism" in manifest:
        det = manifest["scripted_determinism"]
        print(f"scripted_determinism seed={det['seed']} steps={det['steps']} matched={det['matched']}")
    if "compare" in manifest:
        compare = manifest["compare"]
        print(f"compare={compare['replay']} matched={compare['matched']}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

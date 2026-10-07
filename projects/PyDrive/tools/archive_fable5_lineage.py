#!/usr/bin/env python3
"""Archive one completed Fable lineage with immutable SHA-256 provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="checkpoint prefix, e.g. 12345678")
    ap.add_argument("--label", default="mazda787b-6spd-legacy")
    ap.add_argument("--source", default=".")
    ap.add_argument("--out", default="runtime/fable5/archives")
    args = ap.parse_args()
    src = Path(args.source).resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = (Path(args.out) / f"{args.label}-{stamp}").resolve()
    dst.mkdir(parents=True, exist_ok=False)
    names = set()
    for pattern in (f"{args.prefix}*.pt", f"{args.prefix}*.json",
                    "fable5_ring_pipeline.json", "fable5_ring_eval_latest.json",
                    "fable5_pit_log.txt", "fable5_supervisor.logpath"):
        names.update(p for p in src.glob(pattern) if p.is_file())
    if not names:
        raise SystemExit(f"no lineage artifacts found for prefix '{args.prefix}'")
    files = []
    for path in sorted(names):
        target = dst / path.name
        shutil.copy2(path, target)
        files.append({"name": target.name, "bytes": target.stat().st_size,
                      "sha256": sha256(target)})
    manifest = {"schema": "fable5-lineage-archive-v1", "label": args.label,
                "prefix": args.prefix, "created_unix": time.time(), "files": files}
    (dst / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dst)


if __name__ == "__main__":
    main()

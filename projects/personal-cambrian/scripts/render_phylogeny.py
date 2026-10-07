#!/usr/bin/env python3
"""Render a run's phylogeny tree with innovation markers (ROADMAP Stage 10.3).

    python3 scripts/render_phylogeny.py runs/<run>/phylogeny.json
    python3 scripts/render_phylogeny.py            # picks the newest run's phylogeny.json

Writes a PNG cladogram (generation x-axis = innovation timeline) next to the JSON.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.viz import render_from_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    args = sys.argv[1:]
    if args:
        jpath = args[0]
    else:
        cands = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "phylogeny.json")))
        if not cands:
            print("no runs/*/phylogeny.json found; pass a path explicitly")
            return
        jpath = cands[-1]
    out = args[1] if len(args) > 1 else os.path.join(os.path.dirname(jpath), "phylogeny_tree.png")
    title = f"Phylogeny — {os.path.basename(os.path.dirname(jpath))}"
    render_from_json(jpath, out, title=title)
    print(f"rendered {jpath} -> {out}")


if __name__ == "__main__":
    main()

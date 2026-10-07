"""Hand-authored stages.

These are fixtures and demos, not training material — the agent's job is stages
it has never seen. They exist so that when something breaks you can debug it
against a road you can reason about by eye.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rallyai.stage.builder import ProfileBuilder, build_stage
from rallyai.track import Track

# v2: grade and width now ease exponentially between segments and crests are
# sin^2 rather than sin, so segment boundaries no longer leave a step in
# gradient. A step in grade is a spike in vertical curvature, and vcurv is what
# the agent reads to predict a takeoff — the old geometry put a phantom crest at
# every boundary. Spurious vcurv is now 4% of the real crest's.
AUTHORED_VERSION = 2


def proving_ground() -> dict[str, Any]:
    """One of each thing the car needs to survive, in a legible order.

    Straight, fast sweeper, hairpin, crest, surface change, off-camber corner.
    Roughly 600 m. If the car cannot get round this, nothing procedural is going
    to go better.
    """
    b = ProfileBuilder(width=9.0, surface="gravel")
    b.straight(80)                                          # launch and settle
    b.corner(60, 90, "right")                               # fast sweeper, right 4
    b.straight(60)
    b.corner(12, 170, "left", ease=8)                       # hairpin, left 1
    b.straight(50, grade=0.04)                              # short climb
    b.crest(40, 3.0)                                        # over the top
    b.straight(60, surface="tarmac")                        # surface change
    b.corner(22, 100, "right", camber=-0.05, surface="tarmac")  # off-camber right 2
    b.straight(80, surface="gravel")                        # run to the finish
    profile = b.build()

    stage = build_stage(
        profile,
        stage_id="proving_ground",
        name="Proving Ground",
        seed=0,
        tier=0,
        generator_version=AUTHORED_VERSION,
        authored=True,
    )
    stage["obstacles"] = _tree_line(stage, spacing=12.0, clearance=2.5, radius=0.42)
    return stage


def flat_straight(length: float = 200.0, width: float = 10.0) -> dict[str, Any]:
    """The simplest possible stage: no curvature, no elevation, no obstacles.

    Used by tests that need to isolate one behaviour from the geometry.
    """
    profile = ProfileBuilder(width=width).straight(length).build()
    return build_stage(
        profile,
        stage_id="flat_straight",
        name="Flat Straight",
        seed=0,
        tier=0,
        generator_version=AUTHORED_VERSION,
        authored=True,
    )


def _tree_line(stage: dict[str, Any], *, spacing: float, clearance: float,
               radius: float) -> list[dict[str, Any]]:
    """Line both sides of the corridor with trees, placed clear of it.

    Placement goes through ``Track`` rather than trigonometry-by-hand so the
    trees sit against the *actual* corridor edge, including width changes. The
    corridor-intrusion invariant then holds by construction rather than by
    inspection.
    """
    track = Track(stage)
    trees: list[dict[str, Any]] = []
    for s in np.arange(spacing, track.length - spacing, spacing):
        i = round(s / track.ds)
        for side in (-1.0, 1.0):
            offset = side * (track.half_width[i] + clearance + radius)
            x = float(track.x[i] + track.nx[i] * offset)
            y = float(track.y[i] + track.ny[i] * offset)
            trees.append({
                "kind": "tree",
                "x": x,
                "y": y,
                "z": float(track.z[i]),
                "radius": radius,
                "height": 8.0,
                "s": float(track.s[i]),
                "lateral": float(offset),
            })
    return trees


ALL = {
    "proving_ground": proving_ground,
    "flat_straight": flat_straight,
}

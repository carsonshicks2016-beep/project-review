# Porsche 919 Hybrid Evo clean-room visual reconstruction

This package is a new, deterministic Observatory hero asset built from authored
section profiles and Porsche's published 5.078 m Evo envelope. It models the
4.650 m base body separately from the Evo's extended aero, uses a 2.957 m
wheelbase reference, and includes one joined exterior with integrated wheel
humps and open side bays, detailed BBS-style wheels,
the compact canopy and stacked airbox, aero tunnels, shark fin, active front
flaps, enlarged rear wing, lighting, mirrors, sidepod intakes, diffuser, and a
project-rendered 919 Tribute livery.

Real Porsche and period track photographs were used only for human proportion
and detail study. No Porsche CAD, scan, photograph, third-party mesh, old
Observatory geometry, or redistributed official media is embedded.

The GLB uses metres with `+x` forward, `+y` left and `+z` up. Its measured
bounds are exactly 5.078 m long, 1.900 m wide and 1.050 m high to GLB float
precision. Named wheel, suspension, brake, active-aero, lamp and exhaust nodes
provide stable integration points. The 310/710-18 tyre geometry, 18-inch rim
radius, wheel centres, body undertray, and joined-body wheel-hump clearance are
checked at build time so a future art change cannot silently place a tyre
through the chassis or leave a duplicate runtime wheel assembly behind.

Rebuild and verify determinism with:

```bash
python3 assets/vehicles/porsche_919evo/build_asset.py
python3 assets/vehicles/porsche_919evo/build_asset.py --check
```

The manifest intentionally sets `faithful_geometry_eligible` to `false`.
Faithful certification remains blocked until a provenance-cleared measured body
surface and licensed hardpoint data replace this visual reconstruction while
preserving the same coordinate and node contract. The legacy Fable 919 physics
status is unchanged.

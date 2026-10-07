# The three contracts

Everything crossing the Python/TypeScript boundary is one of these three, and
nothing else. One copy of each schema, read by both sides.

| Contract | File | Carries |
|---|---|---|
| Stage | `stage.schema.json` | the world: corridor, elevation, camber, surfaces, pace notes, obstacles |
| Replay | `replay.schema.json` | one run: stage reference + a frame array |
| Metrics | `metrics.schema.json` | training progress: one JSON object per line (JSONL) |

All three must stay readable in a text editor and diffable in git. That rules
out parallel arrays for per-point data — a moved centerline point should be one
changed line, not a diff across four arrays.

## Coordinate convention

**Right-handed, z-up.** This matches the vendored Supra dynamics exactly, so
there is no transform anywhere in the physics hot path.

```
  x  →  east      (ground plane)
  y  →  north     (ground plane)
  z  →  up        (height)
  yaw → rotation about +z, 0 = facing +x, increasing counter-clockwise
```

The viewer converts to its own axes once, at load. **The sim never converts.**

> v1 stored stages in the viewer's y-up convention and bridged on every step.
> That mismatch is what produced the floating-car bug. If you find yourself
> writing a swizzle inside `packages/sim`, something has gone wrong.

Units are SI throughout: metres, seconds, radians, m/s, newtons. `mu` is
dimensionless. Angles are radians in the contracts, **never degrees** — degrees
appear only in `CarSpec` fields that are named `*_deg`.

## Arc-length (`s`)

Position along the stage is `s`, in metres from the start line, measured along
the centerline. `s = 0` is the start, `s = length_m` is the finish. Point-to-
point, never wrapping — unlike Supra's circuit model, where arc wraps modulo lap
length. Any look-ahead past the finish clamps; it does not wrap to the start.

## Versioning

Each contract carries `schema_version` (integer, bumped on breaking change).
Readers must reject a version they do not know rather than guessing.

# Mazda 787B pre-five-speed legacy checkpoints

This folder contains Mazda 787B checkpoints removed from active locations on
2026-07-12 because they are not exactly compatible with drivetrain protocol
`mazda787b-5spd-ring-v1`.

- `root/` preserves checkpoints formerly located at the project root.
- `runtime_fable5_champions/` preserves incompatible former champion entries.
- `manifest.json` records every original path, destination, SHA-256 digest,
  checkpoint shape, stage, update count, drivetrain stamp, and fingerprint.

These files are retained for viewing, historical comparison, or controlled
policy transplantation. They must not be resumed as certified five-speed
checkpoints without migration and recertification.

Existing `archives/` and `backups/` trees were already legacy storage and were
left unchanged so archived launchers and historical references keep working.

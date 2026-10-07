# Training Roadmap

## Milestone 1: Bridge Self-Test

- Verify ROM hash, emulator/core metadata, WRAM reads, controller writes, and
  frame stepping.
- Run with `smw-ai serve --policy noop`.

## Milestone 2: Manual Calibration Trace

- Play the safety route manually while recording telemetry.
- Run `smw-ai validate-trace recordings/manual-route.jsonl`.
- Fill in route event deltas and level ids after the first successful trace.

## Milestone 3: Per-Level Curriculum

- Create savestates at each level entry for training only.
- Train or evolve each level until it reaches a stable success threshold.
- Save per-level champions before combining chunks.

## Milestone 4: Chunk Curriculum

- Train Yoshi's Island, Donut Plains, Vanilla Dome, Forest, Chocolate Island,
  Valley, and Bowser chunks.
- Preserve earlier champions to recover from catastrophic forgetting.

## Milestone 5: Clean-Boot Evaluation

- Disable savestate and memory-write commands.
- Require three complete Bowser defeats with no deaths and no Starworld.
- Package the champion artifact only after replay/log validation.


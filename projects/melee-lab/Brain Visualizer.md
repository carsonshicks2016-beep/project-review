# Melee Brain

A small native macOS companion opens automatically when a training worker starts.
It occupies the largest usable gap in `window-layout.json`, including the space
needed for emulator title bars. The current eight-emulator layout puts it in the
wide strip beneath the games. You can move, resize, minimize, or close it normally.
If the emulator layout fills the screen, it opens as a floating window.

## What it shows

- **Policy flow:** sampled game observations, one shared PPO policy, and the
  latest reported controller decision from each emulator. Choose an emulator
  to inspect its observations. The diagram is schematic, not neuron activations.
- **Full lineage:** the recorded chain of resumed runs, with each input's frozen
  decision counter and checkpoint hash. A later overwrite of `latest.zip` does
  not change what an earlier run consumed.
- **Session progress:** new decisions this run, a rolling decision rate, the
  latest checkpoint age, and up to 20 recent PPO/self-play match outcomes.
  These are training outcomes, not a separate evaluation score.
- **Health:** missing or delayed telemetry is dimmed; stopped and paused runs
  keep their last recorded state. The window follows the next training launch.

The companion reads status and metadata once per second. It does not load model
weights, change controller inputs, or own training. Closing it leaves training
running. A workspace lock prevents duplicate companion windows.

## Reopen or disable

From this project directory, reopen for a particular run:

```sh
.venv/bin/python -m melee_lab.visualizer --run-dir runs/YOUR-RUN-ID
```

To disable automatic launch, create `visualizer-settings.json` containing:

```json
{"enabled": false}
```

Remove that file or set `enabled` to `true` to restore automatic launch.

The native host is compiled on first use using the installed Apple command-line
tools and cached in `.runtime/Melee Brain.app`. Build/launch diagnostics go to
`.runtime/visualizer.log`. No dashboard restart is required for future runs.

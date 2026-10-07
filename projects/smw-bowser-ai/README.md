# Super Mario World Bowser AI

This project is a greenfield harness for training and evaluating a hybrid AI
agent that beats Bowser in **Super Mario World** through BizHawk/EmuHawk.

The design splits the problem into two controllers:

- `DirectorFSM`: deterministic title, save, overworld, dialogue, cutscene,
  death, and game-over navigation.
- `PlayerPolicy`: PPO/GA-driven playable-level control using compact macro
  actions.

The final validation target is strict: clean boot to Bowser defeat with no
deaths, no Starworld, no savestates, no memory edits, complete telemetry, and a
replayable movie/log artifact.

## What Is Implemented

- BizHawk Lua bridge at `lua/smw_bridge.lua`.
- Python bridge server/client protocol for frame observations and controller
  responses.
- JSON-compatible YAML manifests for memory signals, macro actions, and the
  No Starworld Safety Route.
- Director FSM, route validator, reward model, telemetry writer, calibration
  helpers, GA scaffold, PPO entrypoint guard, champion artifact builder, and
  CLI.
- Standard-library `unittest` coverage for route safety, rewards, protocol,
  action expansion, and director behavior.

## Quick Start

```bash
cd "/Users/REVIEW_USER/Documents/New project 2/smw-bowser-ai"
python3 -m unittest discover -s tests
python3 -m smw_bowser_ai.cli inspect-config
```

For local editable imports without setting `PYTHONPATH`:

```bash
python3 -m pip install -e .
```

Optional RL packages are intentionally not required for the core harness:

```bash
python3 -m pip install -e ".[rl,yaml,dev]"
```

## BizHawk Loop

1. Open your legally dumped Super Mario World ROM in Windows EmuHawk.
2. In EmuHawk, open **Tools -> Lua Console**.
3. Start the Python bridge:

   ```bash
   smw-ai serve --host 127.0.0.1 --port 55355 --policy heuristic
   ```

4. Load `lua/smw_bridge.lua` in the Lua Console.
5. Watch the Lua overlay for connection, ROM hash, frame count, and action.

The bridge sends RAM observations every frame. Screenshots are intentionally
sampled only when requested to avoid turning the socket into soup.

## Training Shape

1. `smw-ai self-test`: verify bridge protocol and config.
2. `smw-ai calibrate-log`: record a manual trace and verify memory signals.
3. `smw-ai validate-trace`: check route progress and Starworld exclusion.
4. `smw-ai evolve-ga`: use macro-sequence GA for hard sections.
5. `smw-ai train-ppo`: run PPO when optional RL dependencies are installed.
6. `smw-ai evaluate`: perform clean-boot validation.
7. `smw-ai build-champion`: package the accepted brain artifacts.

## ROM Policy

Do not commit or distribute ROM files. This project stores only hashes,
configuration, source code, telemetry, and trained brain artifacts.


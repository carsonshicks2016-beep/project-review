# Roadblock Countermeasures

- **ROM mismatch:** reject unknown hashes and require calibration.
- **Socket instability:** use heartbeat, timeouts, reconnects, and no-op input.
- **Sparse rewards:** combine local progress, route events, survival, boss hits,
  and penalties for death, loops, Starworld, and timeouts.
- **Reward hacking:** accept only real clean-boot Bowser clears.
- **Long horizon:** train levels, then chunks, then the full route.
- **Non-game screens:** keep them deterministic in `DirectorFSM`.
- **Secret exit drift:** validate event deltas after every route-critical exit.
- **Starworld risk:** abort immediately on forbidden route tags/names/submaps.
- **Death ambiguity:** combine mode, player state, lives, animation, and map
  transition signals.
- **Bowser fight:** train separately with phase-aware rewards and GA traces.
- **Determinism drift:** pin emulator version, ROM hash, Lua hash, configs, and
  seeds.


# M5 — Research polish & showcase

The tooling that turns a working fighter into a *research project* and a *showcase*.

## Replays — the gen-1-vs-gen-N showcase
`ReplayRecorder.cs` (Record mode) dumps a fight to JSONL; Play mode drives a kinematic clone.
- Record a fight from an **early** checkpoint and a **late** one, play both side by side → the
  classic "flailing → skilled" before/after. Most shareable thing in the project.
- NOTE: `PlayFrame()` is left as a stub — add a JSON parse (Newtonsoft) to apply rotations.
  Recording is complete; playback is ~20 lines once you pick a JSON lib.

## Tournaments + ELO — `tools/tournament.py`
Round-robin between exported `.onnx` checkpoints, ELO ladder printed at the end.
- ELO + bracket logic is ready to use.
- `play_match()` is a clear stub: wire it to a **built duel player**. Easiest robust setup —
  give the two fighters **different behavior names** (`DuelistA`/`DuelistB`) in a dedicated
  tournament scene so each maps to one policy file.
```bash
python tools/tournament.py --env build/AegisDuel --checkpoints "results/duel01/**/*.onnx"
```

## Ablations — `tools/ablation.py`
Launches multiple `mlagents-learn` runs and prints the final metric per run by parsing the
TensorBoard event files. Ready to use.
```bash
python tools/ablation.py --base config/duel_ppo_selfplay.yaml \
    --runs amp_on:--env-args,style=1 amp_off:--env-args,style=0 \
    --metric "Self-play/ELO"
```
Good questions to ablate:
- **AMP on/off** (motion quality vs win rate)
- **Shield in observations on/off** (does it learn to use the shield without "seeing" it?)
- **Reward-term sweeps** (`wDealt` vs `wTaken` → aggressive vs defensive styles)
- **Self-play window size** (does a bigger opponent pool reduce strategy cycling?)

## Lab-notebook discipline
- One YAML per experiment in `config/`; one `--run-id` per run; never reuse ids.
- Record the git SHA + config with each run.
- Keep a markdown log of hypothesis → setup → result for each ablation.

## Stretch backlog (from PLAN.md §9)
Weapon roster, morphology co-evolution, MAP-Elites style diversity, human-vs-AI, LLM
commentator. Each is its own mini-milestone once the core loop is solid.

# M0 — Setup Checklist

**Goal of M0:** stand up the toolchain and a scene. By the end you have: a working
Unity ↔ ML-Agents training loop (verified on a stock example), a Mixamo humanoid with a
sword + shield in an arena, and a first-pass ragdoll that physically flops. **No custom
combat code yet** — that's M1.

Work top to bottom. Each part ends with a ✅ check.

---

## Prereqs (free accounts)
- [ ] **Unity account** (for Unity Hub / Editor)
- [ ] **Adobe account** (for Mixamo — free)

> Mac note: you're on Apple Silicon. Everything below has native arm64 support. We do **not**
> use Isaac Gym (CUDA-only). Throughput later comes from parallel arenas + `--time-scale`.

---

## Part A — Toolchain

### A1. Unity
- [x] **Unity Hub installed** (`brew install --cask unity-hub`, done 2026-06-15).
- [x] **Unity Editor installed** — Unity 6.4 (6000.4.11f1), done 2026-06-15. (Unity 6 works
      fine with ML-Agents — no need for 2022.3.)
- [x] **aegis project created** — via Unity CLI into `~/Desktop/aegis` (done 2026-06-16); it
      picked up the existing `Assets/Scripts/`. Open it from Unity Hub (Add → select the folder).

### A2. Python trainer — ✅ DONE (2026-06-15)
Installed and verified: CPython **3.10.12** venv at `~/venvs/aegis` with `mlagents 1.1.0`
(torch 2.12.0, numpy 1.23.5, grpcio 1.62.2). `mlagents-learn --help` works.

Reproduce/repair anytime with **`tools/setup_python_env.sh`**. The non-obvious bits it handles:
- Homebrew's `python@3.10` is 3.10.20, but mlagents needs **≤3.10.12** → we use `uv` to fetch an
  exact CPython 3.10.12 (no source compile).
- mlagents 1.1.0 pins `grpcio<=1.48.2`, which has **no arm64 wheel and won't compile** on modern
  clang. mlagents runs fine on newer grpcio, so we **override** it to `grpcio==1.62.2` (a wheel).
```bash
bash tools/setup_python_env.sh        # idempotent; recreates ~/venvs/aegis
~/venvs/aegis/bin/mlagents-learn --help
```
- [x] venv created · [x] `mlagents-learn --help` works

### A3. ML-Agents Unity package — ✅ DONE (2026-06-16)
- [x] `com.unity.ml-agents` **4.0.3** added to `Packages/manifest.json` (local path to the
      cloned repo: `file:../../ml-agents-src/com.unity.ml-agents`), plus `ai.inference` 2.6.1
      and `newtonsoft-json`. **All 13 Aegis scripts compile clean in Unity 6.4** (verified in
      batchmode, exit 0). Keep the `~/Desktop/ml-agents-src` clone in place — the project
      references the package from there.

✅ **Part A — DONE.** Toolchain installed, venv has `mlagents`, aegis project created, package
added, all scripts compile. Next: open the project in Unity Hub, then Part C (Mixamo) + M1.

---

## Part B — Smoke test the loop (zero custom code)

Prove Unity ↔ Python training works using the stock **3DBall** example *before* touching Aegis.
- [x] Repo cloned to `~/Desktop/ml-agents-src` (done 2026-06-15).
- [ ] Open `~/Desktop/ml-agents-src/Project` as a **separate** project in Unity Hub.
- [ ] Open the scene `Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity`.
- [ ] Run the trainer (from inside `~/Desktop/ml-agents-src`):
  ```bash
  cd ~/Desktop/ml-agents-src
  ~/venvs/aegis/bin/mlagents-learn config/ppo/3DBall.yaml --run-id=smoke01
  ```
- [ ] When the console says *"Start training by pressing Play"*, press **Play** in Unity.
- [ ] Watch mean reward climb in the console. Optionally:
  ```bash
  tensorboard --logdir results        # then open the printed localhost URL
  ```

- [x] ✅ **PASSED 2026-06-15** — connected, mean reward climbing (1.25 → 1.45 → …). Whole
      toolchain proven end to end. Stop with `Ctrl+C` (or the ■ button), then back to `aegis`.

---

## Part C — Mixamo character, weapons & scene

### C1. Get the rig + combat clips (mixamo.com)
- [ ] Download a **character** (e.g. *Y Bot* or a knight) as **FBX**, T-Pose, *with* skin.
- [ ] Download these **animations** (search "sword and shield"), each **FBX for Unity, 30fps,
      "Without Skin"**, and prefer **In Place** (no root motion) for clean imitation targets:
  - [ ] Idle / Sword And Shield Idle
  - [ ] Walk + Strafe (L/R) + Turn
  - [ ] Slash / Attack (1–2 variants)
  - [ ] Block (raise shield) + Blocking Idle
  - [ ] Impact / Hit Reaction
  - [ ] (optional) Death/knockdown

### C2. Import as Humanoid
For the character **and** each clip:
- [ ] Select the FBX ▸ Inspector ▸ **Rig** ▸ Animation Type = **Humanoid**.
  - Character: Avatar Definition = **Create From This Model** → Apply.
  - Clips: Avatar Definition = **Copy From Other Avatar** → pick the character's avatar.
- [ ] On clips: **Animation** tab → set **Loop Time** for idle/walk; **Apply**.
> Humanoid rig matters: it gives the muscle-space avatar we'll sample for the M1 imitation reward.

### C3. Sword + shield meshes
Mixamo characters ship **without** weapons. Grab CC0/free assets:
- Kenney.nl, Poly.pizza, Sketchfab (CC0 filter), or Unity Asset Store (free medieval packs).
- [ ] Import a **sword** mesh and a **shield** mesh.
- [ ] Parent the sword under the **right hand** bone (`mixamorig:RightHand`) and the shield under
      the **left hand / forearm** bone; position/rotate so the grip looks right.

### C4. Arena scene
- [ ] New scene `Assets/Scenes/Arena.unity`.
- [ ] Add: a **ground plane** (large), a **directional light**, and a **camera** angled to view
      the fighters (e.g. slightly elevated, looking at origin).
- [ ] Drop the character into the scene, standing on the ground in T/idle pose.

✅ **Part C done when:** a Mixamo humanoid stands in the arena holding a sword + shield, and
the combat clips import clean as Humanoid with no avatar errors.

---

## Part D — First-pass ragdoll (validation only)

This pass uses Unity's built-in **Ragdoll Wizard** (Rigidbody + CharacterJoint) purely to
confirm the body, colliders, joint limits, and masses behave. It **will flop to the floor** —
that's expected and correct for M0. The real **`ArticulationBody` + PD** rig is the first
M1 code step (the `RagdollBuilder` script I'll generate).

- [ ] Open the Ragdoll Wizard and assign bone transforms (Pelvis/Hips as Root; L/R hips, knees,
      feet; L/R arms, elbows; middle spine; head). Set a realistic **Total Mass** (~70 kg).
- [ ] **Create**, press Play, and confirm it collapses believably (no exploding joints, limbs
      stay attached, limits look sane). Tweak collider sizes/limits if it spazzes.

✅ **Part D done when:** pressing Play produces a stable, believable ragdoll collapse.

---

## M0 Definition of Done
- [ ] 3DBall smoke test trained successfully (toolchain proven).
- [ ] `aegis` Unity project has `com.unity.ml-agents` installed, no errors.
- [ ] Mixamo humanoid + sword + shield standing in the Arena scene.
- [ ] Combat clips imported as Humanoid.
- [ ] First-pass ragdoll flops believably on Play.

## Common gotchas
- **Won't connect to trainer** → C# package version ≠ Python `mlagents` version. Match them.
- **Avatar/retarget errors** → a clip wasn't set to Humanoid or didn't copy the base avatar.
- **Ragdoll explodes** → joint limits too loose, masses too low, or overlapping colliders.
- **Python dep hell** → you're not on 3.10, or not inside the venv (`source .../activate`).

## Next (M1, where I write code)
1. `RagdollBuilder.cs` — scripts the **ArticulationBody** rig + PD drives from the Humanoid bones.
2. `PDJointController.cs` — applies agent PD targets to `ArticulationDrive`.
3. `DuelistAgent.cs` — minimal ML-Agents `Agent` (observations / actions) + the
   DeepMimic-style **imitation reward** that teaches it to balance & walk.

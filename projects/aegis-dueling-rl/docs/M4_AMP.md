# M4 — Adversarial Motion Priors (natural motion)

> ⚠️ **This is the research milestone.** Everything before M4 is "make it work"; M4 is "make it
> look good." ML-Agents has **no built-in AMP**, so this is a scaffold + a design you'll iterate
> on — not turnkey code. Don't start it until M3 duels actually work.

## The problem it solves
After M3 your fighters win, but they look stiff/robotic because the only thing keeping motion
natural is frame-locked clip imitation, which fights against free-form combat. **AMP** (Peng et
al. 2021) replaces strict imitation with a learned *style* reward: a discriminator rewards
motion that merely *looks like* the reference dataset, leaving the policy free to improvise.
This is what gives ASE its quality.

## Architecture (in-engine reward via Unity Sentis)
```
reference clone ──MotionRecorder("expert")──► motion_expert.jsonl ─┐
                                                                   ├─► amp_discriminator.py ─► amp_discriminator.onnx
agent ──MotionRecorder("policy")──► motion_policy.jsonl ───────────┘                                  │
                                                                                                      ▼
DuelAgent  ──MotionFeaturizer (s,s')──►  Sentis runs the ONNX  ──►  r_style = -log(1 - sigmoid(D))  ──► AddReward
```

### Pieces already written
- `MotionFeaturizer.cs` — the shared (s) feature vector. `Size` is the per-frame dim;
  discriminator input = `2 * Size` (a transition).
- `MotionRecorder.cs` — dumps `(s, s')` transitions to JSONL (expert from the clone, policy
  from the agent).
- `tools/amp_discriminator.py` — trains D and exports `amp_discriminator.onnx`.

### The loop you run
1. Record **expert** transitions from the reference clone playing all your combat clips.
2. While training, periodically record **policy** transitions from the agent.
3. `python tools/amp_discriminator.py --expert motion_expert.jsonl --policy motion_policy.jsonl`
4. Drop `amp_discriminator.onnx` into the project; in `DuelAgent` add the style reward.

### Style-reward hook (add to DuelAgent — needs `com.unity.sentis`)
```csharp
// pseudo — wire once Sentis is installed
// using Unity.Sentis;
// build a transition feature vector each step:
MotionFeaturizer.Write(root.transform, ragdollAnimator.GetBoneTransform, _curFeat);
// concat _prevFeat + _curFeat -> input tensor -> worker.Execute -> logit
float logit = RunDiscriminator(_prevFeat, _curFeat);
float d = 1f / (1f + Mathf.Exp(-logit));
float styleReward = -Mathf.Log(Mathf.Max(1f - d, 1e-6f));
AddReward(wStyle * styleReward);   // then REDUCE wPose (frame-imitation) toward 0
```

## Alternative path
If in-engine inference is fiddly, **fork the ML-Agents trainer** and add AMP as a custom
reward signal in Python (more "correct" but heavier). The Sentis path keeps everything in one
training run and is the recommended start.

## Knobs
- Discriminator must **keep pace** with the policy — re-train it every N policy updates, or it
  gets fooled and the style reward goes uninformative.
- Gradient penalty (`--grad-pen`) stabilizes; lower it if the discriminator can't learn.
- Blend schedule: start with `wPose` high (M3 imitation), ramp `wStyle` up and `wPose` down.

## When it works
Fighters move fluidly and improvise. That's ASE-grade. Then it's all M5: tournaments, ablations
(AMP on/off is a great one), and showcase replays.

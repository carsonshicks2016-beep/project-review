// Generalized Advantage Estimation (Schulman et al. 2015).
//
//   delta_t = r_t + gamma * V(s_{t+1}) * notDone - V(s_t)
//   A_t     = delta_t + gamma * lambda * notDone * A_{t+1}
//
// lambda trades bias against variance: 1 gives the Monte-Carlo return (unbiased, noisy),
// 0 gives the one-step TD residual (biased, low variance).
//
// Rollouts here are truncated rather than always terminal, so `lastValue` bootstraps the
// tail. Without that, every truncated episode would be treated as if the world ended,
// teaching the critic that the end of a rollout is worthless.
export function computeGAE({ rewards, values, dones, lastValue, gamma = 0.99, lambda = 0.95 }) {
  const T = rewards.length;
  const advantages = new Float64Array(T);
  const returns = new Float64Array(T);

  let gae = 0;
  for (let t = T - 1; t >= 0; t--) {
    const notDone = dones[t] ? 0 : 1;
    const nextValue = t === T - 1 ? lastValue : values[t + 1];
    const delta = rewards[t] + gamma * nextValue * notDone - values[t];
    gae = delta + gamma * lambda * notDone * gae;
    advantages[t] = gae;
    returns[t] = gae + values[t];
  }
  return { advantages, returns };
}

// Advantage normalization per batch. Standard PPO practice: it makes the policy-loss
// scale independent of reward magnitude, so the same learning rate works across reward
// designs.
export function normalizeAdvantages(adv) {
  const n = adv.length;
  if (n === 0) return adv;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += adv[i];
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) { const d = adv[i] - mean; varSum += d * d; }
  const std = Math.sqrt(varSum / n) + 1e-8;
  for (let i = 0; i < n; i++) adv[i] = (adv[i] - mean) / std;
  return adv;
}

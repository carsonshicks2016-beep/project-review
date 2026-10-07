// Diagonal Gaussian policy for continuous control.
//
// The network emits a mean per action dimension; log-std is a separate state-independent
// learnable vector, which is the standard PPO-for-continuous-control setup and much more
// stable than predicting per-state variance early in training.
//
// Actions are sampled unbounded and clipped by the environment when applied. The log-prob
// is deliberately computed on the *unclipped* sample: correcting for clipping needs a
// truncated-Gaussian density, and every mainstream PPO implementation takes this same
// shortcut.

const LOG_2PI = Math.log(2 * Math.PI);

export function gaussianSample(mean, logStd, out) {
  for (let i = 0; i < mean.length; i++) {
    // Box-Muller
    const u = 1 - Math.random();
    const v = Math.random();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    out[i] = mean[i] + Math.exp(logStd[i]) * z;
  }
  return out;
}

export function logProb(action, mean, logStd) {
  let lp = 0;
  for (let i = 0; i < mean.length; i++) {
    const std = Math.exp(logStd[i]);
    const z = (action[i] - mean[i]) / std;
    lp += -0.5 * z * z - logStd[i] - 0.5 * LOG_2PI;
  }
  return lp;
}

export function entropy(logStd) {
  let h = 0;
  for (let i = 0; i < logStd.length; i++) h += logStd[i] + 0.5 * (LOG_2PI + 1);
  return h;
}

// d(logProb)/d(mean) and d(logProb)/d(logStd).
export function logProbGrads(action, mean, logStd, dMean, dLogStd) {
  for (let i = 0; i < mean.length; i++) {
    const std = Math.exp(logStd[i]);
    const diff = action[i] - mean[i];
    const z = diff / std;
    dMean[i] = z / std;        // (a-mu)/sigma^2
    dLogStd[i] = z * z - 1;    // ((a-mu)^2/sigma^2) - 1
  }
}

// d(entropy)/d(logStd) is 1 per dimension; kept as a function so callers read clearly.
export function entropyGrads(logStd, dLogStd) {
  for (let i = 0; i < logStd.length; i++) dLogStd[i] = 1;
}

// Applies the drone's actual control limits. Thrust is [0,1]; the rate axes are [-1,1].
export function clipAction(a) {
  return {
    thrust: Math.max(0, Math.min(1, a[0] * 0.5 + 0.5)),
    pitch: Math.max(-1, Math.min(1, a[1])),
    roll: Math.max(-1, Math.min(1, a[2])),
    yaw: Math.max(-1, Math.min(1, a[3]))
  };
}

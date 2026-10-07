import { MLP, Adam, RunningNorm } from './network.js';
import { logProb, logProbGrads, entropy, gaussianSample } from './policy.js';
import { computeGAE, normalizeAdvantages } from './gae.js';

// PPO with a clipped surrogate objective.
//
// Separate actor and critic rather than a shared trunk: at this size the parameter saving
// is irrelevant, and a shared trunk makes the value loss fight the policy loss for the
// same features, which needs careful loss weighting to stay stable.

export const DEFAULTS = {
  hidden: [64, 64],
  lr: 3e-4,
  gamma: 0.99,
  lambda: 0.95,
  clipEps: 0.2,
  epochs: 4,
  minibatch: 256,
  vfCoef: 0.5,
  entCoef: 0.005,
  maxGradNorm: 0.5,
  initLogStd: -0.5,
  targetKL: 0.02   // Early-stop the update if the policy moves too far
};

export class PPOAgent {
  constructor(obsDim, actDim, config = {}) {
    this.cfg = { ...DEFAULTS, ...config };
    this.obsDim = obsDim;
    this.actDim = actDim;

    this.actor = new MLP([obsDim, ...this.cfg.hidden, actDim]);
    this.critic = new MLP([obsDim, ...this.cfg.hidden, 1]);
    this.logStd = new Float32Array(actDim).fill(this.cfg.initLogStd);

    this.optActor = new Adam(this.actor.numParams, { lr: this.cfg.lr });
    this.optCritic = new Adam(this.critic.numParams, { lr: this.cfg.lr });
    this.optLogStd = new Adam(actDim, { lr: this.cfg.lr });

    this.obsNorm = new RunningNorm(obsDim);

    // Scratch buffers, allocated once: the update loop runs per-sample and would
    // otherwise churn the GC hard.
    this._actorActs = this.actor.makeCache();
    this._criticActs = this.critic.makeCache();
    this._gActor = new Float32Array(this.actor.numParams);
    this._gCritic = new Float32Array(this.critic.numParams);
    this._gLogStd = new Float32Array(actDim);
    this._dMean = new Float64Array(actDim);
    this._dLogStdTmp = new Float64Array(actDim);
    this._normObs = new Float32Array(obsDim);
    this._dOut = new Float64Array(actDim);
    this._dValue = new Float64Array(1);
  }

  // Rollout-time action selection. Returns the raw (unclipped) sample plus the values the
  // update needs to reproduce this decision.
  act(obs, deterministic = false) {
    const x = this.obsNorm.normalizeInto(obs, this._normObs);
    const mean = this.actor.forward(x, this._actorActs);
    const action = new Float32Array(this.actDim);
    if (deterministic) {
      action.set(mean);
    } else {
      gaussianSample(mean, this.logStd, action);
    }
    const value = this.critic.forward(x, this._criticActs)[0];
    return { action, logp: logProb(action, mean, this.logStd), value };
  }

  // One PPO iteration over a rollout batch.
  // obs: Float32Array (N * obsDim), actions: Float32Array (N * actDim)
  update(batch) {
    const { obs, actions, logps, rewards, values, dones, lastValue } = batch;
    const N = rewards.length;
    const cfg = this.cfg;

    const { advantages, returns } = computeGAE({
      rewards, values, dones, lastValue, gamma: cfg.gamma, lambda: cfg.lambda
    });
    normalizeAdvantages(advantages);

    // Observation stats update AFTER computing this batch's actions, so the normalizer
    // used at rollout time matches the one the update sees for these samples.
    const idx = new Int32Array(N);
    for (let i = 0; i < N; i++) idx[i] = i;

    let policyLoss = 0, valueLoss = 0, entMean = 0, klMean = 0, clipFrac = 0, nSeen = 0;
    let stopped = false;

    for (let epoch = 0; epoch < cfg.epochs && !stopped; epoch++) {
      // Fisher-Yates: minibatches must be reshuffled each epoch or the update correlates
      // with rollout order.
      for (let i = N - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      }

      for (let start = 0; start < N; start += cfg.minibatch) {
        const end = Math.min(start + cfg.minibatch, N);
        const mb = end - start;
        this._gActor.fill(0);
        this._gCritic.fill(0);
        this._gLogStd.fill(0);

        let mbKL = 0;

        for (let s = start; s < end; s++) {
          const i = idx[s];
          const oOff = i * this.obsDim;
          const aOff = i * this.actDim;

          const x = this.obsNorm.normalizeInto(obs.subarray(oOff, oOff + this.obsDim), this._normObs);
          const mean = this.actor.forward(x, this._actorActs);
          const action = actions.subarray(aOff, aOff + this.actDim);

          const lp = logProb(action, mean, this.logStd);
          const ratio = Math.exp(lp - logps[i]);
          const adv = advantages[i];

          // d(-min(r*A, clip(r)*A))/dr is A inside the trust region and 0 outside.
          // A>=0 is clipped from above at 1+eps; A<0 from below at 1-eps.
          const inRegion = adv >= 0
            ? ratio <= 1 + cfg.clipEps
            : ratio >= 1 - cfg.clipEps;
          if (!inRegion) clipFrac++;

          const dLp = inRegion ? -(adv * ratio) / mb : 0;

          logProbGrads(action, mean, this.logStd, this._dMean, this._dLogStdTmp);
          const dOut = this._dOut;
          for (let k = 0; k < this.actDim; k++) {
            dOut[k] = dLp * this._dMean[k];
            // Entropy bonus is -entCoef * H, and dH/dLogStd is 1 per dimension.
            this._gLogStd[k] += dLp * this._dLogStdTmp[k] - (cfg.entCoef * 1) / mb;
          }
          this.actor.backward(this._actorActs, dOut, this._gActor);

          // Value head: 0.5 * (V - R)^2
          const v = this.critic.forward(x, this._criticActs)[0];
          const dv = this._dValue;
          dv[0] = (cfg.vfCoef * (v - returns[i])) / mb;
          this.critic.backward(this._criticActs, dv, this._gCritic);

          const kl = logps[i] - lp; // approximate; sign-correct in expectation
          mbKL += kl;
          policyLoss += -Math.min(ratio * adv, Math.max(1 - cfg.clipEps, Math.min(1 + cfg.clipEps, ratio)) * adv);
          valueLoss += 0.5 * (v - returns[i]) ** 2;
          entMean += entropy(this.logStd);
          nSeen++;
        }

        this.optActor.step(this.actor.params, this._gActor, cfg.maxGradNorm);
        this.optCritic.step(this.critic.params, this._gCritic, cfg.maxGradNorm);
        this.optLogStd.step(this.logStd, this._gLogStd, cfg.maxGradNorm);

        mbKL /= mb;
        klMean += mbKL;

        // Standard PPO safety valve: abandon the rest of the update once the policy has
        // moved too far from the one that collected this data.
        if (cfg.targetKL > 0 && Math.abs(mbKL) > 1.5 * cfg.targetKL) {
          stopped = true;
          break;
        }
      }
    }

    this.obsNorm.update(obs, N);

    const nb = Math.max(1, Math.ceil(N / cfg.minibatch));
    return {
      policyLoss: policyLoss / Math.max(1, nSeen),
      valueLoss: valueLoss / Math.max(1, nSeen),
      entropy: entMean / Math.max(1, nSeen),
      approxKL: klMean / nb,
      clipFrac: clipFrac / Math.max(1, nSeen),
      earlyStopped: stopped,
      meanReturn: returns.reduce((a, b) => a + b, 0) / N
    };
  }

  // Weight snapshot for shipping to rollout workers.
  exportWeights() {
    return {
      actor: new Float32Array(this.actor.params),
      logStd: new Float32Array(this.logStd),
      obsMean: new Float32Array(this.obsNorm.mean),
      obsVar: new Float32Array(this.obsNorm.var)
    };
  }
}

import { PPOAgent } from './ppo.js';
import { OBS_DIM, ACT_DIM } from './droneEnv.js';

// Drives PPO from the main thread: fans rollouts out to module workers, computes values
// and advantages centrally, runs the update, and reports metrics in the same shape the
// dashboard already renders for the GA.
export class PPOTrainer {
  constructor({ numWorkers = 4, numEnvs = 8, steps = 128, maxTicks = 3600, config = {}, environment = 'outdoor' } = {}) {
    this.numWorkers = numWorkers;
    this.numEnvs = numEnvs;
    this.steps = steps;
    this.maxTicks = maxTicks;
    this.agent = new PPOAgent(OBS_DIM, ACT_DIM, config);
    this.iteration = 0;
    this.workers = [];
    this.running = false;
    this.lastStats = null;
    this.totalSteps = 0;
    this.environment = environment;
  }

  init() {
    this.dispose();
    for (let i = 0; i < this.numWorkers; i++) {
      this.workers.push(new Worker(new URL('../ppoWorker.js', import.meta.url), { type: 'module' }));
    }
  }

  dispose() {
    this.workers.forEach(w => w.terminate());
    this.workers = [];
  }

  // One PPO iteration. Returns metrics plus the best episode seen, for the recorder.
  async iterate(gates, startGateIdx = 0, environment = this.environment) {
    this.environment = environment || this.environment;
    const w = this.agent.exportWeights();
    const hiddenSizes = this.agent.cfg.hidden;

    const results = await Promise.all(this.workers.map(worker => new Promise(resolve => {
      worker.onmessage = ev => resolve(ev.data);
      // Each worker gets its own copy of the weights; slice() so the transfer of one
      // worker's buffer cannot neuter another's.
      worker.postMessage({
        actorParams: w.actor.buffer.slice(0),
        logStd: w.logStd.buffer.slice(0),
        obsMean: w.obsMean.buffer.slice(0),
        obsVar: w.obsVar.buffer.slice(0),
        hiddenSizes,
        gates,
        numEnvs: this.numEnvs,
        steps: this.steps,
        startGateIdx,
        maxTicks: this.maxTicks,
        environment,
        grid: environment === 'lidar' && this.grid ? this.grid.serialize() : null,
        environment: this.environment
      });
    })));

    // Concatenate the workers' rollouts.
    const perWorker = this.numEnvs * this.steps;
    const N = perWorker * this.workers.length;
    const obs = new Float32Array(N * OBS_DIM);
    const actions = new Float32Array(N * ACT_DIM);
    const logps = new Float64Array(N);
    const rewards = new Float64Array(N);
    const dones = new Uint8Array(N);
    const episodes = [];
    let best = null;

    results.forEach((r, wi) => {
      const off = wi * perWorker;
      obs.set(new Float32Array(r.obs), off * OBS_DIM);
      actions.set(new Float32Array(r.actions), off * ACT_DIM);
      logps.set(new Float64Array(r.logps), off);
      rewards.set(new Float64Array(r.rewards), off);
      dones.set(new Uint8Array(r.dones), off);
      episodes.push(...r.episodes);
      if (r.best && (!best || r.best.gates > best.gates || (r.best.gates === best.gates && r.best.ret > best.ret))) {
        best = r.best;
      }
    });

    // Values are computed here rather than in the workers: GAE runs on this side anyway,
    // so shipping critic weights out would be pure overhead.
    const values = new Float64Array(N);
    const scratch = new Float32Array(OBS_DIM);
    for (let i = 0; i < N; i++) {
      const sub = obs.subarray(i * OBS_DIM, (i + 1) * OBS_DIM);
      const x = this.agent.obsNorm.normalizeInto(sub, scratch);
      values[i] = this.agent.critic.forward(x, this.agent._criticActs)[0];
    }
    const tail = new Float32Array(results[0].tailObs);
    const lastValue = this.agent.critic.forward(
      this.agent.obsNorm.normalizeInto(tail.subarray(0, OBS_DIM), scratch),
      this.agent._criticActs
    )[0];

    const stats = this.agent.update({ obs, actions, logps, rewards, values, dones, lastValue });

    this.iteration++;
    this.totalSteps += N;

    const gatesList = episodes.map(e => e.gates);
    const rets = episodes.map(e => e.ret);
    this.lastStats = {
      iteration: this.iteration,
      totalSteps: this.totalSteps,
      episodes: episodes.length,
      meanReturn: rets.length ? rets.reduce((a, b) => a + b, 0) / rets.length : 0,
      bestReturn: rets.length ? Math.max(...rets) : 0,
      meanGates: gatesList.length ? gatesList.reduce((a, b) => a + b, 0) / gatesList.length : 0,
      bestGates: gatesList.length ? Math.max(...gatesList) : 0,
      ...stats
    };
    return { stats: this.lastStats, best };
  }

  exportPolicy() {
    return {
      version: 1,
      hidden: this.agent.cfg.hidden,
      actor: Array.from(this.agent.actor.params),
      critic: Array.from(this.agent.critic.params),
      logStd: Array.from(this.agent.logStd),
      obsNorm: this.agent.obsNorm.serialize(),
      iteration: this.iteration
    };
  }

  importPolicy(data) {
    if (!data || data.version !== 1) return false;
    if (data.actor.length !== this.agent.actor.numParams) return false;
    this.agent.actor.params.set(Float32Array.from(data.actor));
    this.agent.critic.params.set(Float32Array.from(data.critic));
    this.agent.logStd.set(Float32Array.from(data.logStd));
    this.agent.obsNorm.mean.set(Float32Array.from(data.obsNorm.mean));
    this.agent.obsNorm.var.set(Float32Array.from(data.obsNorm.var));
    this.agent.obsNorm.count = data.obsNorm.count;
    this.iteration = data.iteration || 0;
    return true;
  }
}

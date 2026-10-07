import { PPOAgent } from '../src/ppo/ppo.js';
import { DroneEnv, OBS_DIM, ACT_DIM } from '../src/ppo/droneEnv.js';
import { defaultTrackData } from '../src/defaultTrack.js';

const gates = defaultTrackData.gates;
const NUM_ENVS = 16;
const STEPS = 256;                 // per env, per iteration
const ROLLOUT = NUM_ENVS * STEPS;  // 4096 transitions
const ITERS = parseInt(process.argv[2] || '40', 10);

const agent = new PPOAgent(OBS_DIM, ACT_DIM, { hidden: [64,64], lr: 3e-4, entCoef: 0.004, initLogStd: parseFloat(process.argv[3] || '-1.6') });
const envs = Array.from({length: NUM_ENVS}, () => new DroneEnv(gates, { maxTicks: 3600 }));
const cur = envs.map(e => e.reset());

let bestGates = 0, bestTicks = 0;
const t0 = Date.now();

for (let iter = 0; iter < ITERS; iter++) {
  const obs = new Float32Array(ROLLOUT * OBS_DIM);
  const actions = new Float32Array(ROLLOUT * ACT_DIM);
  const logps = new Float64Array(ROLLOUT);
  const values = new Float64Array(ROLLOUT);
  const rewards = new Float64Array(ROLLOUT);
  const dones = new Uint8Array(ROLLOUT);
  const epGates = [], epReturns = [], epLens = [];
  const running = new Array(NUM_ENVS).fill(0);

  let idx = 0;
  for (let s = 0; s < STEPS; s++) {
    for (let e = 0; e < NUM_ENVS; e++) {
      obs.set(cur[e], idx * OBS_DIM);
      const { action, logp, value } = agent.act(cur[e]);
      actions.set(action, idx * ACT_DIM);
      logps[idx] = logp; values[idx] = value;
      const r = envs[e].step(action);
      rewards[idx] = r.reward;
      dones[idx] = r.done ? 1 : 0;
      running[e] += r.reward;
      if (r.done) {
        epGates.push(r.info.gatesCleared);
        epReturns.push(running[e]); epLens.push(r.info.tick);
        if (r.info.gatesCleared > bestGates) { bestGates = r.info.gatesCleared; bestTicks = r.info.tick; }
        running[e] = 0;
        cur[e] = envs[e].reset();
      } else {
        cur[e] = r.obs;
      }
      idx++;
    }
  }
  const lastValue = agent.act(cur[0]).value;
  const st = agent.update({ obs, actions, logps, rewards, values, dones, lastValue });

  const mg = epGates.length ? epGates.reduce((a,b)=>a+b,0)/epGates.length : 0;
  const mx = epGates.length ? Math.max(...epGates) : 0;
  const mr = epReturns.length ? epReturns.reduce((a,b)=>a+b,0)/epReturns.length : 0;
  if (iter % 5 === 0 || iter === ITERS-1) {
    console.log(`it ${String(iter).padStart(3)} | eps ${String(epGates.length).padStart(3)} | gates avg ${mg.toFixed(2)} max ${String(mx).padStart(2)} | best ${bestGates} | ret ${mr.toFixed(1).padStart(7)} | len ${(epLens.length?epLens.reduce((a,b)=>a+b,0)/epLens.length:0).toFixed(0).padStart(4)} | ent ${st.entropy.toFixed(2)} | KL ${st.approxKL.toFixed(3)} | ${((Date.now()-t0)/1000).toFixed(0)}s`);
  }
}
console.log(`\nBEST GATES CLEARED: ${bestGates}/16  (GA baseline after ~1500 generations: 11-12/16)`);

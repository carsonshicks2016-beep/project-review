import { PPOAgent } from '../src/ppo/ppo.js';

// 2D point-mass reacher: the standard smoke test for a continuous-control PPO.
// If the implementation is correct this is solved comfortably; if any sign or clip is
// wrong it will sit flat or diverge.
const OBS = 6, ACT = 2, DT = 0.1, MAX_STEPS = 120;

function newEpisode() {
  return {
    p: [0, 0],
    v: [0, 0],
    t: [(Math.random() * 2 - 1) * 2, (Math.random() * 2 - 1) * 2],
    steps: 0
  };
}
const dist = e => Math.hypot(e.t[0] - e.p[0], e.t[1] - e.p[1]);
const obsOf = (e, out) => {
  out[0] = e.p[0]; out[1] = e.p[1];
  out[2] = e.v[0]; out[3] = e.v[1];
  out[4] = e.t[0] - e.p[0]; out[5] = e.t[1] - e.p[1];
  return out;
};

function step(e, a) {
  const prev = dist(e);
  for (let i = 0; i < 2; i++) {
    const acc = Math.max(-1, Math.min(1, a[i]));
    e.v[i] += acc * DT - 0.15 * e.v[i];
    e.p[i] += e.v[i] * DT;
  }
  e.steps++;
  const d = dist(e);
  let r = (prev - d) * 10 - 0.01;   // dense progress + small time cost
  let done = false;
  if (d < 0.15) { r += 10; done = true; }
  else if (e.steps >= MAX_STEPS) done = true;
  return { r, done, solved: d < 0.15 };
}

const agent = new PPOAgent(OBS, ACT, { hidden: [64, 64], lr: 3e-4, entCoef: 0.003 });
const ROLLOUT = 2048;
let ep = newEpisode();
const obsBuf = new Float32Array(OBS);
let epRet = 0;
const history = [];

for (let iter = 0; iter < 60; iter++) {
  const obs = new Float32Array(ROLLOUT * OBS);
  const actions = new Float32Array(ROLLOUT * ACT);
  const logps = new Float64Array(ROLLOUT);
  const values = new Float64Array(ROLLOUT);
  const rewards = new Float64Array(ROLLOUT);
  const dones = new Uint8Array(ROLLOUT);
  const epReturns = [];
  let solves = 0, epCount = 0;

  for (let t = 0; t < ROLLOUT; t++) {
    obsOf(ep, obsBuf);
    obs.set(obsBuf, t * OBS);
    const { action, logp, value } = agent.act(obsBuf);
    actions.set(action, t * ACT);
    logps[t] = logp;
    values[t] = value;
    const { r, done, solved } = step(ep, action);
    rewards[t] = r;
    dones[t] = done ? 1 : 0;
    epRet += r;
    if (done) {
      epReturns.push(epRet); epRet = 0; epCount++;
      if (solved) solves++;
      ep = newEpisode();
    }
  }
  obsOf(ep, obsBuf);
  const lastValue = agent.act(obsBuf).value;

  const stats = agent.update({ obs, actions, logps, rewards, values, dones, lastValue });
  const meanEp = epReturns.length ? epReturns.reduce((a,b)=>a+b,0)/epReturns.length : NaN;
  const successRate = epCount ? solves/epCount : 0;
  history.push({ iter, meanEp, successRate, ...stats });
  if (iter % 10 === 0 || iter === 59) {
    console.log(`iter ${String(iter).padStart(2)}  epRet=${meanEp.toFixed(2).padStart(7)}  success=${(successRate*100).toFixed(0).padStart(3)}%  ent=${stats.entropy.toFixed(2)}  vLoss=${stats.valueLoss.toFixed(3)}  KL=${stats.approxKL.toFixed(4)}  clip=${(stats.clipFrac*100).toFixed(0)}%`);
  }
}

const first = history.slice(0,5);
const last = history.slice(-5);
const avg = (a,k)=>a.reduce((s,x)=>s+x[k],0)/a.length;
const r0 = avg(first,'meanEp'), r1 = avg(last,'meanEp');
const s0 = avg(first,'successRate'), s1 = avg(last,'successRate');
console.log(`\nreturn  ${r0.toFixed(2)} -> ${r1.toFixed(2)}`);
console.log(`success ${(s0*100).toFixed(0)}% -> ${(s1*100).toFixed(0)}%`);
const ok = r1 > r0 + 3 && s1 > 0.85;
console.log(ok ? 'PPO LEARNING CHECK PASSED' : 'PPO LEARNING CHECK FAILED');
process.exit(ok?0:1);

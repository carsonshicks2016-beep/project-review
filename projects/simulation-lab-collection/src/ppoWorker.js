// PPO rollout worker.
//
// Unlike simWorker.js (a classic worker that has to inline its dependencies), this is a
// module worker: everything under src/ppo/ uses relative imports only, so it can import
// the real modules and there is no duplicated copy to drift.
//
// The worker only runs the ACTOR. Values are computed on the main thread after collection
// — GAE runs there anyway, so shipping critic weights to every worker would be wasted.
import { MLP } from './ppo/network.js';
import { gaussianSample, logProb } from './ppo/policy.js';
import { DroneEnv, OBS_DIM, ACT_DIM } from './ppo/droneEnv.js';
import { VoxelGrid } from './collision/voxelGrid.js';
// terrainHeight.js, not outdoorTerrain.js: the latter imports three by bare specifier
// and module workers do not inherit the page's importmap.
import { getOutdoorElevation } from './terrainHeight.js';

let actor = null;
let hidden = null;
let worldGrid = null;

self.onmessage = (e) => {
  const {
    actorParams, logStd, obsMean, obsVar, hiddenSizes,
    gates, numEnvs, steps, startGateIdx, maxTicks, grid, environment
  } = e.data;

  // Cached across iterations: the grid is a couple of megabytes and rarely changes.
  if (grid !== undefined) worldGrid = grid ? VoxelGrid.deserialize(grid) : null;
  const activeGrid = environment === 'lidar' ? worldGrid : null;

  if (!actor || String(hidden) !== String(hiddenSizes)) {
    hidden = hiddenSizes;
    actor = new MLP([OBS_DIM, ...hiddenSizes, ACT_DIM]);
  }
  actor.params.set(new Float32Array(actorParams));

  const std = new Float32Array(logStd);
  const mean = new Float32Array(obsMean);
  const varr = new Float32Array(obsVar);
  const acts = actor.makeCache();
  const normObs = new Float32Array(OBS_DIM);

  const normalize = (x) => {
    for (let d = 0; d < OBS_DIM; d++) {
      normObs[d] = Math.max(-10, Math.min(10, (x[d] - mean[d]) / Math.sqrt(varr[d] + 1e-8)));
    }
    return normObs;
  };

  const N = numEnvs * steps;
  const obs = new Float32Array(N * OBS_DIM);
  const actions = new Float32Array(N * ACT_DIM);
  const logps = new Float64Array(N);
  const rewards = new Float64Array(N);
  const dones = new Uint8Array(N);

  const getTerrainHeight = environment === 'outdoor' ? getOutdoorElevation : null;
  const envs = [];
  const cur = [];
  const traces = [];   // Per-env replay trace of the episode in progress
  for (let i = 0; i < numEnvs; i++) {
    const env = new DroneEnv(gates, { maxTicks, startGateIdx, getTerrainHeight, grid: activeGrid });
    envs.push(env);
    cur.push(env.reset(startGateIdx));
    traces.push([]);
  }

  const episodes = [];
  let best = null;          // Best completed episode this rollout
  const action = new Float32Array(ACT_DIM);
  const returns = new Float64Array(numEnvs);

  let idx = 0;
  for (let s = 0; s < steps; s++) {
    for (let ei = 0; ei < numEnvs; ei++) {
      const env = envs[ei];
      obs.set(cur[ei], idx * OBS_DIM);

      const m = actor.forward(normalize(cur[ei]), acts);
      gaussianSample(m, std, action);
      actions.set(action, idx * ACT_DIM);
      logps[idx] = logProb(action, m, std);

      // Replay trace at the same 20Hz the GA worker uses, so the existing player and
      // recorder need no changes.
      if (env.tick % 3 === 0) {
        const d = env.drone;
        traces[ei].push(d.pos[0], d.pos[1], d.pos[2], d.quat[0], d.quat[1], d.quat[2], d.quat[3]);
      }

      const r = env.step(action);
      rewards[idx] = r.reward;
      dones[idx] = r.done ? 1 : 0;
      returns[ei] += r.reward;

      if (r.done) {
        const ep = {
          gates: r.info.gatesCleared,
          ret: returns[ei],
          topSpeed: r.info.topSpeed * 3.6,
          ticks: r.info.tick
        };
        episodes.push(ep);
        if (!best || ep.gates > best.gates || (ep.gates === best.gates && ep.ret > best.ret)) {
          best = { ...ep, trace: Float32Array.from(traces[ei]) };
        }
        returns[ei] = 0;
        traces[ei] = [];
        cur[ei] = env.reset(startGateIdx);
      } else {
        cur[ei] = r.obs;
      }
      idx++;
    }
  }

  // Tail observation per env so the main thread can bootstrap truncated episodes.
  const tailObs = new Float32Array(numEnvs * OBS_DIM);
  for (let i = 0; i < numEnvs; i++) tailObs.set(cur[i], i * OBS_DIM);

  const transfer = [obs.buffer, actions.buffer, logps.buffer, rewards.buffer, dones.buffer, tailObs.buffer];
  const bestTrace = best ? best.trace : null;
  if (bestTrace) transfer.push(bestTrace.buffer);

  self.postMessage({
    obs: obs.buffer,
    actions: actions.buffer,
    logps: logps.buffer,
    rewards: rewards.buffer,
    dones: dones.buffer,
    tailObs: tailObs.buffer,
    numEnvs, steps,
    episodes,
    best: best ? { gates: best.gates, ret: best.ret, topSpeed: best.topSpeed, ticks: best.ticks, trace: bestTrace ? bestTrace.buffer : null } : null
  }, transfer);
};

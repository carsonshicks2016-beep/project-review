import { readSensors, SENSOR_COUNT } from './collision/sensors.js';

// The single definition of what the policy sees. Both trainers import this: the GA worker
// and the PPO env previously built their observation vectors separately, which is exactly
// how two copies of the quaternion code drifted apart and hid a bug in both.

export const GATE_OBS = 14;
export const OBS_DIM = GATE_OBS + SENSOR_COUNT; // 22

export function buildObservation(drone, gates, gateIdx, grid, out, getTerrainHeight = null) {
  const cur = gates[Math.min(gateIdx, gates.length - 1)];
  const next = gates[(gateIdx + 1) % gates.length];

  const relCur = drone.worldToLocal([
    cur.position[0] - drone.pos[0],
    cur.position[1] - drone.pos[1],
    cur.position[2] - drone.pos[2]
  ]);
  const relNext = drone.worldToLocal([
    next.position[0] - drone.pos[0],
    next.position[1] - drone.pos[1],
    next.position[2] - drone.pos[2]
  ]);
  const localVel = drone.worldToLocal([drone.vel[0], drone.vel[1], drone.vel[2]]);

  const fwd = drone.rotateVectorByQuat([0, 0, -1], drone.quat);
  const dist = Math.hypot(
    cur.position[0] - drone.pos[0],
    cur.position[1] - drone.pos[1],
    cur.position[2] - drone.pos[2]
  ) || 1;
  const align =
    fwd[0] * ((cur.position[0] - drone.pos[0]) / dist) +
    fwd[1] * ((cur.position[1] - drone.pos[1]) / dist) +
    fwd[2] * ((cur.position[2] - drone.pos[2]) / dist);

  out[0] = relCur[0] / 50;  out[1] = relCur[1] / 50;  out[2] = relCur[2] / 50;
  out[3] = relNext[0] / 100; out[4] = relNext[1] / 100; out[5] = relNext[2] / 100;
  out[6] = localVel[0] / 30; out[7] = localVel[1] / 30; out[8] = localVel[2] / 30;
  out[9] = drone.angularVel[0] / drone.maxRate;
  out[10] = drone.angularVel[1] / drone.maxRate;
  out[11] = drone.angularVel[2] / drone.maxRate;
  out[12] = align;
  // Clearance above terrain, not absolute altitude: the outdoor environment has ground
  // that rises, so absolute height would tell the policy nothing about how close it is
  // to hitting something.
  const groundY = getTerrainHeight ? getTerrainHeight(drone.pos[0], drone.pos[2]) : 0;
  out[13] = (drone.pos[1] - groundY) / 30;

  // Wall proximity. Zeroed when there is no grid, so the observation shape is stable
  // whether or not the world is solid.
  readSensors(grid, drone, out.subarray(GATE_OBS, GATE_OBS + SENSOR_COUNT));
  return out;
}

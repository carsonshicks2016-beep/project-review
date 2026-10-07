// Wall-distance sensors.
//
// Without these the policy has no way to perceive the house at all -- it would only know
// where the next gate is, and would fly into walls with no signal that they exist. The
// ring is forward-biased because that is where a racing drone needs resolution.

export const SENSOR_RANGE = 30.0;

// Body-frame unit vectors. Forward is -z, up is +y, right is +x.
const R = Math.SQRT1_2;
export const SENSOR_DIRS = [
  [0, 0, -1],       // forward
  [-R, 0, -R],      // forward-left
  [R, 0, -R],       // forward-right
  [-1, 0, 0],       // left
  [1, 0, 0],        // right
  [0, 1, 0],        // up
  [0, -1, 0],       // down
  [0, 0, 1]         // back
];
export const SENSOR_COUNT = SENSOR_DIRS.length;

// Writes normalized distances (1 = clear to SENSOR_RANGE, 0 = touching) into `out`.
// Inverted deliberately: "how close is the nearest wall" is the quantity that should be
// large when it matters, which keeps the input well scaled near obstacles.
export function readSensors(grid, drone, out) {
  if (!grid) {
    out.fill(0);
    return out;
  }
  const p = drone.pos;
  for (let i = 0; i < SENSOR_COUNT; i++) {
    const w = drone.rotateVectorByQuat(SENSOR_DIRS[i], drone.quat);
    const d = grid.raycast(p[0], p[1], p[2], w[0], w[1], w[2], SENSOR_RANGE);
    out[i] = 1 - d / SENSOR_RANGE;
  }
  return out;
}

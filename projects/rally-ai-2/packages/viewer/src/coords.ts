/**
 * The one place the viewer converts from the sim's axes to three's.
 *
 * The sim is **z-up, right-handed**, matching the vendored dynamics exactly so
 * there is no transform anywhere in the physics hot path. three.js is y-up.
 * That conversion happens **here, once, at load** and nowhere else.
 *
 * > v1 stored stages in the viewer's y-up convention and bridged on every step.
 * > That mismatch produced the floating-car bug.
 *
 * If you find yourself swizzling axes anywhere else in this package, the bug is
 * that you are doing it here twice.
 *
 * ## The conventions this file depends on
 *
 * All four are load-bearing and none is guessable — three of them are stated in
 * the sim's own source, and the last one is unusual enough to trip anyone who
 * assumes the automotive default:
 *
 * | quantity | convention | source |
 * |---|---|---|
 * | axes    | x east, y north, z up | Coordinate Conventions |
 * | yaw     | about +z, 0 = facing +x, CCW positive | Coordinate Conventions |
 * | pitch   | **positive = nose up** | `physics.py:106` "+ = climbing" |
 * | roll    | **positive = LEFT side up** | `physics.py:107` "+ = left up" |
 *
 * That roll sign is backwards from the common "right side down is positive"
 * convention. Get it wrong and the car leans *out* of every corner — which
 * looks like a physics bug and is not one.
 */

import { Matrix4, Quaternion, Vector3 } from "three";

/** Sim-space position, z-up. */
export interface SimVec {
  x: number;
  y: number;
  z: number;
}

/**
 * Sim (x east, y north, z up) -> three (x right, y up, z toward viewer).
 *
 * `(x, y, z) -> (x, z, -y)`. This is a rotation, not a reflection: it has
 * determinant +1, so handedness is preserved and cross products carry over
 * unchanged. That is what lets {@link orientation} build a basis in sim space
 * and convert each axis independently.
 */
export function toThree(v: SimVec, out = new Vector3()): Vector3 {
  return out.set(v.x, v.z, -v.y);
}

/** As {@link toThree}, from loose components. */
export function xyz(x: number, y: number, z: number, out = new Vector3()): Vector3 {
  return out.set(x, z, -y);
}

const _f = new Vector3();
const _u = new Vector3();
const _r = new Vector3();
const _m = new Matrix4();

/**
 * Body orientation as a three quaternion, from the sim's yaw/pitch/roll.
 *
 * Built by constructing the body basis explicitly rather than by composing
 * Euler angles, because three's Euler orders are defined in *its* axes and the
 * mapping of an intrinsic yaw-pitch-roll through the basis change is not one of
 * the six named orders. Explicit vectors are longer but they can be reasoned
 * about one axis at a time — and tested one axis at a time.
 *
 * The resulting model frame is **+x forward, +y up, +z right**. The car mesh is
 * authored to match.
 */
export function orientation(
  yaw: number,
  pitch: number,
  roll: number,
  out = new Quaternion(),
): Quaternion {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);

  // --- in SIM space (z-up) ---
  // Forward, yawed then pitched. Positive pitch lifts the nose, so +z rises.
  const fx = cy * cp;
  const fy = sy * cp;
  const fz = sp;

  // Level right, before roll: the right-hand normal of the heading. Positive
  // `lateral` is to the right of travel and uses this same vector, so if this
  // sign is wrong the car and the corridor disagree about which side is which.
  const rx0 = sy;
  const ry0 = -cy;

  // Up before roll = right x forward. Reduces to (0,0,1) when pitch is zero.
  const ux0 = ry0 * fz - 0 * fy;
  const uy0 = 0 * fx - rx0 * fz;
  const uz0 = rx0 * fy - ry0 * fx;

  // Roll about the forward axis. Positive roll raises the LEFT side, which
  // tilts `up` toward the RIGHT — hence +sr on the right vector, not -sr.
  // This is the sign that makes the car lean into a corner instead of out of it.
  const ux = ux0 * cr + rx0 * sr;
  const uy = uy0 * cr + ry0 * sr;
  const uz = uz0 * cr;

  // --- convert each axis into three space ---
  xyz(fx, fy, fz, _f).normalize();
  xyz(ux, uy, uz, _u).normalize();
  // Right = forward x up, in three's (right-handed) space.
  _r.copy(_f).cross(_u).normalize();
  // Re-orthogonalise up against the other two, so accumulated float error in
  // the basis never leaves the matrix slightly non-orthogonal.
  _u.copy(_r).cross(_f).normalize();

  // Columns are the model's local +x, +y, +z in world space.
  _m.makeBasis(_f, _u, _r);
  return out.setFromRotationMatrix(_m);
}

/**
 * Heading of a centerline tangent, as a three-space direction.
 *
 * Shared by the corridor builder and anything that needs "which way is the road
 * pointing here", so the two cannot drift apart.
 */
export function headingDir(heading: number, out = new Vector3()): Vector3 {
  return xyz(Math.cos(heading), Math.sin(heading), 0, out);
}

/**
 * Right-hand normal of a heading, as a three-space direction.
 *
 * `(sin h, -cos h)` in sim space. Positive `lateral` in the contracts is a
 * multiple of this vector, which is what places obstacles on the correct side
 * of the road.
 */
export function rightDir(heading: number, out = new Vector3()): Vector3 {
  return xyz(Math.sin(heading), -Math.cos(heading), 0, out);
}

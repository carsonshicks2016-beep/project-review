/**
 * Cameras. Chase, bonnet and sideline, per North Star §10.
 *
 * The chase camera is the one that has to be right, because it is what almost
 * every second of watching uses. Two things make it feel like a rally camera
 * rather than a rigid boom:
 *
 * 1. **It lags.** Position and look-at are both damped, so the car rotates
 *    inside the frame when it steps out of line. A camera welded to the car's
 *    yaw hides the very thing this project exists to show — you cannot see
 *    opposite lock if the camera counter-steers with you.
 * 2. **It follows the car's heading, not its velocity.** Following velocity swings
 *    the camera wildly during a slide, which reads as a broken camera rather
 *    than a fast car.
 *
 * Damping is frame-rate independent: `1 - exp(-k*dt)` rather than a fixed
 * lerp factor, so playback at 0.25x does not change how the camera feels.
 */

import { PerspectiveCamera, Vector3 } from "three";

import type { SurfaceKind } from "./replay";

export type CameraMode = "chase" | "bonnet" | "sideline";

/**
 * Metres behind and above the car for the chase view.
 *
 * Deliberately close. The reference games devote roughly a third of the
 * 320-pixel frame to the car; a modern distant action camera loses both the
 * livery and the visible opposite-lock/slip that make a rally replay readable.
 */
const CHASE_BACK = 5.5;
const CHASE_UP = 2.4;
const CHASE_AHEAD = 6.7;

/** Damping rates, per second. Higher is stiffer. */
const POS_K = 6.0;
const LOOK_K = 6.5;

/** Extra distance per m/s of speed, so fast feels fast. */
const SPEED_PULL = 0.015;
const SPEED_PULL_MAX = 0.6;
const BASE_FOV = 55;
const FOV_GAIN_MAX = 3;

const _desired = new Vector3();
const _target = new Vector3();
const _fwd = new Vector3();
const _up = new Vector3(0, 1, 0);
const _offset = new Vector3();
const _right = new Vector3();
const _renderPos = new Vector3();
const _renderLook = new Vector3();

export interface RallyCameraInput {
  /** Recorded replay time makes vibration repeatable at a fixed benchmark. */
  time: number;
  surface: SurfaceKind | undefined;
  /** Recorded chassis slip angle, radians. */
  slipAngle: number;
  /** Analytic pulse derived from explicit replay landing events. */
  landingKick: number;
}

export class ChaseCamera {
  readonly camera: PerspectiveCamera;
  mode: CameraMode = "chase";

  private readonly pos = new Vector3();
  private readonly look = new Vector3();
  private seeded = false;

  constructor(aspect: number) {
    // A narrow period-game lens, chosen with the close boom above so the car
    // fills about one third of a 4:3 frame without a modern fisheye look.
    this.camera = new PerspectiveCamera(BASE_FOV, aspect, 0.1, 2000);
  }

  resize(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  /** Jump straight to the target pose. Use on load and on any seek. */
  reset(): void {
    this.seeded = false;
  }

  /**
   * @param carPos   car position, three-space
   * @param forward  unit heading, three-space
   * @param speed    m/s, used only to push the camera back a little at pace
   * @param dt       seconds of wall time since the last update
   */
  update(
    carPos: Vector3,
    forward: Vector3,
    speed: number,
    dt: number,
    dynamics: RallyCameraInput,
  ): void {
    _fwd.copy(forward).setY(0);
    if (_fwd.lengthSq() < 1e-8) _fwd.set(1, 0, 0);
    _fwd.normalize();
    _right.copy(_fwd).cross(_up).normalize();

    const fov = BASE_FOV + Math.min(
      FOV_GAIN_MAX,
      (Math.max(0, speed) / 34) * FOV_GAIN_MAX,
    );
    if (Math.abs(this.camera.fov - fov) > 1e-4) {
      this.camera.fov = fov;
      this.camera.updateProjectionMatrix();
    }

    switch (this.mode) {
      case "bonnet":
        // On the car, no damping: this view is supposed to feel welded on.
        _desired.copy(carPos).addScaledVector(_fwd, 0.6).addScaledVector(_up, 1.15);
        _target.copy(carPos).addScaledVector(_fwd, CHASE_AHEAD).addScaledVector(_up, 1.0);
        this.pos.copy(_desired);
        this.look.copy(_target);
        this.apply(dynamics, speed, 0.34);
        return;

      case "sideline":
        // A low verge camera, like a marshal or photographer just off the
        // corridor. Keeping it inside the scenery's guaranteed 2.6 m margin is
        // important now that nearby trees are real volumes rather than cards:
        // a camera out in the forest can genuinely land inside a crown.
        _desired
          .copy(carPos)
          .addScaledVector(_fwd, -7.5)
          .add(new Vector3(0, 3.2, 0))
          .addScaledVector(new Vector3(-_fwd.z, 0, _fwd.x), 4.4);
        _target.copy(carPos).addScaledVector(_up, 0.65);
        this.pos.copy(_desired);
        this.look.copy(_target);
        this.apply(dynamics, speed, 0);
        return;

      case "chase":
      default: {
        const back = CHASE_BACK + Math.min(speed * SPEED_PULL, SPEED_PULL_MAX);
        _desired.copy(carPos).addScaledVector(_fwd, -back).addScaledVector(_up, CHASE_UP);
        // Recorded slip offsets the boom opposite the slide. Position damping
        // turns that into the lateral lag visible in a loose-surface corner.
        _desired.addScaledVector(
          _right,
          Math.min(
            0.72,
            Math.max(-0.72, -dynamics.slipAngle * speed * 0.072),
          ),
        );
        // Look low down the road so the car sits above the lower safe edge
        // instead of having its bumper clipped by the 4:3 frame.
        _target.copy(carPos).addScaledVector(_fwd, CHASE_AHEAD).addScaledVector(_up, 0.25);
        break;
      }
    }

    if (!this.seeded) {
      this.pos.copy(_desired);
      this.look.copy(_target);
      this.seeded = true;
    } else {
      // Exponential damping, frame-rate independent. A plain lerp factor would
      // make the camera stiffer at high frame rates and mushy at low ones.
      this.pos.lerp(_desired, 1 - Math.exp(-POS_K * dt));
      // Retain horizontal angular lag without letting ordinary forward speed
      // stretch the boom. Vertical height is kept independently at CHASE_UP:
      // normalising the full 3D vector would convert longitudinal lag into a
      // flatter boom and eventually drop the camera below the roof at speed.
      _offset.set(this.pos.x - carPos.x, 0, this.pos.z - carPos.z);
      if (_offset.lengthSq() > 1e-8) {
        const horizontal = Math.hypot(
          _desired.x - carPos.x,
          _desired.z - carPos.z,
        );
        _offset.setLength(horizontal);
        this.pos.x = carPos.x + _offset.x;
        this.pos.z = carPos.z + _offset.z;
      }
      this.pos.y = _desired.y;
      this.look.lerp(_target, 1 - Math.exp(-LOOK_K * dt));
    }
    this.apply(dynamics, speed, 1);
  }

  private apply(
    dynamics: RallyCameraInput,
    speed: number,
    responseScale: number,
  ): void {
    _renderPos.copy(this.pos);
    _renderLook.copy(this.look);

    const loose =
      dynamics.surface === "gravel" ||
      dynamics.surface === "mud" ||
      dynamics.surface === "snow";
    const surfaceScale =
      dynamics.surface === "mud"
        ? 1.15
        : dynamics.surface === "snow"
          ? 0.78
          : 1;
    const vibration =
      loose
        ? Math.min(1, Math.max(0, speed) / 24) *
          0.036 *
          surfaceScale *
          responseScale
        : 0;
    const phase = dynamics.time;
    const vertical =
      (Math.sin(phase * 41.3) * 0.62 + Math.sin(phase * 67.1) * 0.38) *
      vibration;
    const lateral =
      (Math.sin(phase * 35.7 + 1.8) * 0.7 +
        Math.sin(phase * 58.9) * 0.3) *
      vibration *
      0.68;
    _renderPos
      .addScaledVector(_up, vertical + dynamics.landingKick * responseScale)
      .addScaledVector(_right, lateral);
    _renderLook.addScaledVector(
      _up,
      -dynamics.landingKick * 0.16 * responseScale,
    );

    this.camera.position.copy(_renderPos);
    this.camera.lookAt(_renderLook);
  }
}

import * as THREE from "three";
import { simToThree } from "./coords.js";

const FREE_SPEED = 38;

/**
 * Broadcast camera director. Pure presentation — never mutates sim state.
 * Modes: broadcast | chase | roof | trackside | drone | free
 */
export class CameraDirector {
  constructor(camera) {
    this.camera = camera;
    this.mode = "broadcast";
    this.cut = 0;
    this._time = 0;
    this._look = new THREE.Vector3();
    this._desired = new THREE.Vector3();
    this._vel = new THREE.Vector3();
    this._tmp = new THREE.Vector3();
    this._quat = new THREE.Quaternion();
    this._free = {
      yaw: 0,
      pitch: -0.14,
      distance: 12,
      keys: new Set(),
    };
    this._shotLabel = "BROADCAST";
    this._shakeLevel = 0;
    this._bindFreeKeys();
  }

  setMode(mode) {
    const next = String(mode || "broadcast").toLowerCase();
    if (next === this.mode) return;
    this.mode = next;
    this.cut += 1;
  }

  /** Hard-cut to a chase pose over the given frame (episode reset / scrub). */
  snap(frame) {
    if (!frame) return;
    const car = simToThree(frame.pose.x, frame.pose.y, frame.pose.z, this._tmp);
    const yaw = frame.pose.yaw;
    const forward = this._vel.set(Math.cos(yaw), 0, -Math.sin(yaw));
    this.camera.position.copy(car)
      .addScaledVector(forward, -10)
      .add(new THREE.Vector3(0, 2.8, 0));
    this.camera.lookAt(car.x, car.y + 0.9, car.z);
    this.camera.fov = 48;
    this.camera.updateProjectionMatrix();
    this.cut += 1;
  }

  audioState() {
    const perspective = this.mode === "roof" ? "onboard" : "external";
    const camera = this.mode === "broadcast" ? "broadcast"
      : this.mode === "chase" ? "chase"
      : this.mode === "roof" ? "roof"
      : this.mode === "trackside" ? "trackside"
      : this.mode === "drone" ? "drone"
      : "free";
    return { camera, perspective, cut: this.cut };
  }

  shotLabel() {
    return this._shotLabel;
  }

  update(frame, dt) {
    this._frame(frame, dt);
    this._shake(frame, dt);
  }

  _frame(frame, dt) {
    this._time += dt;
    const pose = frame.pose;
    const car = simToThree(pose.x, pose.y, pose.z, this._tmp);
    const yaw = pose.yaw;
    const forward = this._vel.set(Math.cos(yaw), 0, -Math.sin(yaw));
    const right = this._look.set(Math.sin(yaw), 0, Math.cos(yaw));

    if (this.mode === "chase") {
      this._shotLabel = "CHASE";
      // Tight chase — car should fill ~20% of frame height.
      this._desired.copy(car)
        .addScaledVector(forward, -9.5)
        .addScaledVector(right, 0.35)
        .add(new THREE.Vector3(0, 2.6, 0));
      this._look.copy(car).add(new THREE.Vector3(0, 0.95, 0)).addScaledVector(forward, 6);
      this._easeTo(this._desired, dt, 14);
      this.camera.lookAt(this._look);
      this.camera.fov = 50;
      this.camera.updateProjectionMatrix();
      return;
    }

    if (this.mode === "roof") {
      this._shotLabel = "ONBOARD ROOF";
      this._desired.copy(car)
        .addScaledVector(forward, 0.55)
        .add(new THREE.Vector3(0, 1.05, 0));
      this.camera.position.copy(this._desired);
      this._look.copy(car).addScaledVector(forward, 14).add(new THREE.Vector3(0, 0.6, 0));
      this.camera.lookAt(this._look);
      this.camera.fov = 72;
      this.camera.updateProjectionMatrix();
      return;
    }

    if (this.mode === "trackside") {
      this._shotLabel = "TRACKSIDE";
      const side = Math.sin(this._time * 0.11) > 0 ? 1 : -1;
      this._desired.copy(car)
        .addScaledVector(right, side * 12)
        .addScaledVector(forward, -2)
        .add(new THREE.Vector3(0, 2.4, 0));
      this._look.copy(car).add(new THREE.Vector3(0, 0.85, 0));
      this._easeTo(this._desired, dt, 4);
      this.camera.lookAt(this._look);
      this.camera.fov = 46;
      this.camera.updateProjectionMatrix();
      return;
    }

    if (this.mode === "drone") {
      this._shotLabel = "DRONE";
      const angle = this._time * 0.22;
      this._desired.set(
        car.x + Math.cos(angle) * 22,
        car.y + 14,
        car.z + Math.sin(angle) * 22,
      );
      this._look.copy(car);
      this._easeTo(this._desired, dt, 2.8);
      this.camera.lookAt(this._look);
      this.camera.fov = 50;
      this.camera.updateProjectionMatrix();
      return;
    }

    if (this.mode === "free") {
      this._shotLabel = "FREE LOOK";
      this._updateFree(car, dt);
      return;
    }

    // broadcast director — closer, lower, mild side push (not helicopter)
    this._shotLabel = "BROADCAST";
    const phase = (Math.sin(this._time * 0.07) + 1) * 0.5;
    const lateral = THREE.MathUtils.lerp(0.8, 4.5, phase);
    const back = THREE.MathUtils.lerp(9, 13.5, 1 - phase);
    const height = THREE.MathUtils.lerp(2.4, 4.2, phase);
    this._desired.copy(car)
      .addScaledVector(forward, -back)
      .addScaledVector(right, lateral)
      .add(new THREE.Vector3(0, height, 0));
    this._look.copy(car).add(new THREE.Vector3(0, 0.9, 0)).addScaledVector(forward, 8);
    this._easeTo(this._desired, dt, 11);
    this.camera.lookAt(this._look);
    this.camera.fov = THREE.MathUtils.lerp(44, 50, phase);
    this.camera.updateProjectionMatrix();
  }

  /**
   * Speed and cornering load shake, applied after framing.
   * Amplitude comes from telemetry the sim already reports, so it reads as the
   * car working rather than as a canned effect. Deliberately small — this sells
   * speed at 300 km/h without making the HUD hard to read.
   */
  _shake(frame, dt) {
    const speed = Math.max(0, Number(frame.speedMps) || 0);
    const lateral = Math.abs(Number(frame.lateralG) || 0);
    const target = Math.min(1, speed / 85) * 0.55 + Math.min(1, lateral / 1.8) * 0.45;
    this._shakeLevel += (target - this._shakeLevel) * Math.min(1, dt * 3.5);
    const amp = this._shakeLevel * 0.045 * (frame.airborne ? 2.2 : 1);
    if (amp < 1e-4) return;
    const t = this._time;
    this.camera.position.x += Math.sin(t * 31.7) * amp;
    this.camera.position.y += Math.sin(t * 27.3 + 1.7) * amp * 0.8;
    this.camera.position.z += Math.sin(t * 24.1 + 3.1) * amp;
  }

  _easeTo(desired, dt, rate) {
    const dist = this.camera.position.distanceTo(desired);
    // High-speed laps move tens of metres per frame of lag — snap sooner
    // than the old 80 m threshold so framing stays tight.
    if (dist > 16) {
      this.camera.position.copy(desired);
      return;
    }
    const alpha = 1 - Math.exp(-rate * Math.max(0, dt));
    this.camera.position.lerp(desired, Math.min(1, alpha));
  }

  _bindFreeKeys() {
    window.addEventListener("keydown", (event) => {
      if (this.mode !== "free") return;
      this._free.keys.add(event.code);
    });
    window.addEventListener("keyup", (event) => {
      this._free.keys.delete(event.code);
    });
    window.addEventListener("mousemove", (event) => {
      if (this.mode !== "free" || !event.buttons) return;
      this._free.yaw -= event.movementX * 0.004;
      this._free.pitch = THREE.MathUtils.clamp(
        this._free.pitch - event.movementY * 0.004, -1.2, 0.35,
      );
    });
  }

  _updateFree(car, dt) {
    const f = this._free;
    if (f.keys.has("KeyW") || f.keys.has("ArrowUp")) f.distance = Math.max(4, f.distance - FREE_SPEED * dt * 0.35);
    if (f.keys.has("KeyS") || f.keys.has("ArrowDown")) f.distance = Math.min(80, f.distance + FREE_SPEED * dt * 0.35);
    if (f.keys.has("KeyA") || f.keys.has("ArrowLeft")) f.yaw += 1.2 * dt;
    if (f.keys.has("KeyD") || f.keys.has("ArrowRight")) f.yaw -= 1.2 * dt;
    const cp = Math.cos(f.pitch);
    this._desired.set(
      car.x + Math.sin(f.yaw) * cp * f.distance,
      car.y + Math.sin(-f.pitch) * f.distance + 2,
      car.z + Math.cos(f.yaw) * cp * f.distance,
    );
    this.camera.position.copy(this._desired);
    this.camera.lookAt(car.x, car.y + 1, car.z);
    this.camera.fov = 55;
    this.camera.updateProjectionMatrix();
  }
}

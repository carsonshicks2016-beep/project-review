import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export const CAMERA_MODES = Object.freeze(["broadcast", "chase", "roof", "trackside", "drone", "free"]);

const BROADCAST_SEQUENCE = Object.freeze([
  { id: "trackside-long", label: "TRACKSIDE · 135MM", duration: 7.2 },
  { id: "chase-low", label: "LOW CHASE · 35MM", duration: 6.4 },
  { id: "apex", label: "APEX CAM · 85MM", duration: 6.8 },
  { id: "helicopter", label: "AERIAL · 50MM", duration: 7.4 },
  { id: "roof", label: "ONBOARD · 24MM", duration: 5.4 },
  { id: "trackside-close", label: "TRACKSIDE · 70MM", duration: 6.6 },
]);

const desiredPosition = new THREE.Vector3();
const desiredTarget = new THREE.Vector3();
const forward = new THREE.Vector3();
const side = new THREE.Vector3();
const lift = new THREE.Vector3(0, 1, 0);

function wrappedProgress(value) {
  return ((Number(value) || 0) % 1 + 1) % 1;
}

export function audioPerspectiveForShot(shot) {
  return shot === "roof" ? "onboard" : "external";
}

export class CameraDirector {
  constructor(camera, canvas, onShotChange = null) {
    this.camera = camera;
    this.mode = "broadcast";
    this.broadcastIndex = 0;
    this.broadcastShot = BROADCAST_SEQUENCE[0].id;
    this.shotStartedAt = 0;
    this.trackPoints = [];
    this.controls = new OrbitControls(camera, canvas);
    this.controls.enabled = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.maxDistance = 280;
    this.controls.minDistance = 2.2;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.snapNext = true;
    this.onShotChange = onShotChange;
    this.lastLabel = "";
    this.lockedAnchorProgress = 0;
    this.currentFov = camera.fov;
    this.resolvedShot = this.broadcastShot;
    this.audioCut = 0;
  }

  setTrack(points) {
    this.trackPoints = points || [];
  }

  setMode(mode) {
    this.mode = CAMERA_MODES.includes(mode) ? mode : "broadcast";
    this.controls.enabled = this.mode === "free";
    this.shotStartedAt = 0;
    this.snapNext = this.mode !== "free";
    this.resolvedShot = this.mode === "broadcast" ? this.broadcastShot : this.mode;
    this.audioCut += 1;
    this.announce(this.mode === "broadcast" ? BROADCAST_SEQUENCE[this.broadcastIndex].label : this.mode.toUpperCase());
  }

  audioState() {
    const shot = this.mode === "broadcast" ? this.resolvedShot : this.mode;
    return {
      camera: shot,
      perspective: audioPerspectiveForShot(shot),
      cut: this.audioCut,
    };
  }

  announce(label) {
    if (label === this.lastLabel) return;
    this.lastLabel = label;
    this.onShotChange?.(label);
  }

  update(vehicle, elapsed, delta) {
    if (!vehicle) return;
    const focus = vehicle.position;
    const yaw = vehicle.yaw || 0;
    const speed = Math.max(0, Number(vehicle.speedMps) || 0);
    forward.set(Math.cos(yaw), 0, -Math.sin(yaw));
    side.set(-forward.z, 0, forward.x);

    if (this.mode === "free") {
      this.resolvedShot = "free";
      this.controls.target.lerp(focus, 1 - Math.exp(-delta * 1.8));
      this.controls.update();
      this.announce("FREE ORBIT");
      return;
    }

    let shot = this.mode;
    if (shot === "broadcast") {
      const descriptor = BROADCAST_SEQUENCE[this.broadcastIndex];
      if (!this.shotStartedAt) {
        this.shotStartedAt = elapsed;
        this.lockedAnchorProgress = wrappedProgress(vehicle.progress);
      } else if (elapsed - this.shotStartedAt >= descriptor.duration) {
        this.broadcastIndex = (this.broadcastIndex + 1) % BROADCAST_SEQUENCE.length;
        this.broadcastShot = BROADCAST_SEQUENCE[this.broadcastIndex].id;
        this.shotStartedAt = elapsed;
        this.lockedAnchorProgress = wrappedProgress(vehicle.progress);
        this.snapNext = true;
        this.audioCut += 1;
      }
      shot = BROADCAST_SEQUENCE[this.broadcastIndex].id;
      this.announce(BROADCAST_SEQUENCE[this.broadcastIndex].label);
    } else {
      this.announce({
        chase: "CHASE · 40MM",
        roof: "ONBOARD · 24MM",
        trackside: "TRACKSIDE · 105MM",
        drone: "AERIAL · 50MM",
      }[shot] || shot.toUpperCase());
    }
    this.resolvedShot = shot;

    let targetFov = 48;
    desiredTarget.copy(focus).addScaledVector(forward, 7.5).addScaledVector(lift, 0.92);
    if (shot === "chase" || shot === "chase-low") {
      const low = shot === "chase-low";
      desiredPosition.copy(focus)
        .addScaledVector(forward, low ? -4.9 : -5.7)
        .addScaledVector(side, low ? 0.92 : 0.48)
        .addScaledVector(lift, low ? 1.16 : 2.02);
      desiredTarget.copy(focus).addScaledVector(forward, 7.6).addScaledVector(lift, 0.5);
      targetFov = (low ? 37 : 39) + THREE.MathUtils.clamp(speed / 60, 0, 2.5);
    } else if (shot === "roof") {
      const vibration = Math.min(0.025, speed * 0.00022);
      desiredPosition.copy(focus)
        .addScaledVector(forward, 0.18)
        .addScaledVector(side, Math.sin(elapsed * 19.3) * vibration)
        .addScaledVector(lift, 1.23 + Math.sin(elapsed * 31.1) * vibration * 0.45);
      desiredTarget.copy(focus).addScaledVector(forward, 48).addScaledVector(lift, 0.5);
      targetFov = 61 + THREE.MathUtils.clamp(speed / 45, 0, 4.5);
    } else if (shot === "drone" || shot === "helicopter") {
      const orbit = elapsed * 0.13 + wrappedProgress(vehicle.progress) * Math.PI * 2;
      desiredPosition.copy(focus)
        .add(new THREE.Vector3(Math.cos(orbit) * 28, 15.5, Math.sin(orbit) * 28))
        .addScaledVector(forward, 5);
      desiredTarget.copy(focus).addScaledVector(forward, 4).addScaledVector(lift, 0.8);
      targetFov = 39;
    } else if (shot === "apex") {
      const anchor = this.tracksideAnchor(this.lockedAnchorProgress, 38, 10.5, 1.35);
      desiredPosition.copy(anchor.position);
      desiredTarget.copy(focus).addScaledVector(forward, 3.5).addScaledVector(lift, 0.62);
      targetFov = 31;
    } else {
      const explicit = this.mode === "trackside";
      const close = shot === "trackside-close";
      const anchor = this.tracksideAnchor(
        explicit ? vehicle.progress : this.lockedAnchorProgress,
        close ? 24 : 48,
        close ? 9.2 : 13.8,
        close ? 1.15 : 2.65,
      );
      desiredPosition.copy(anchor.position);
      desiredTarget.copy(focus).addScaledVector(forward, close ? 2.5 : 5.5).addScaledVector(lift, 0.68);
      targetFov = close ? 36 : 24;
    }

    const isFixed = shot === "trackside-long" || shot === "trackside-close" || shot === "apex" || shot === "trackside";
    const positionBlend = this.snapNext ? 1 : 1 - Math.exp(-delta * (isFixed ? 3.6 : 5.1));
    const targetBlend = this.snapNext ? 1 : 1 - Math.exp(-delta * (isFixed ? 7.5 : 5.8));
    this.camera.position.lerp(desiredPosition, positionBlend);
    this.controls.target.lerp(desiredTarget, targetBlend);
    this.currentFov = THREE.MathUtils.lerp(this.currentFov, targetFov, this.snapNext ? 1 : 1 - Math.exp(-delta * 3.8));
    if (Math.abs(this.camera.fov - this.currentFov) > 0.01) {
      this.camera.fov = this.currentFov;
      this.camera.updateProjectionMatrix();
    }
    this.camera.lookAt(this.controls.target);
    this.snapNext = false;
  }

  tracksideAnchor(progress = 0, leadSamples = 36, offset = 11, height = 2.2) {
    if (!this.trackPoints.length) return { position: desiredPosition.set(18, 6, 18) };
    const count = this.trackPoints.length;
    const base = Math.round(wrappedProgress(progress) * (count - 1));
    const anchorIndex = (base + leadSamples + count) % count;
    const point = this.trackPoints[anchorIndex];
    const next = this.trackPoints[(anchorIndex + 6) % count];
    const tangent = new THREE.Vector3(next[0] - point[0], 0, -(next[1] - point[1])).normalize();
    const outward = new THREE.Vector3(-tangent.z, 0, tangent.x);
    const block = Math.floor(anchorIndex / 150);
    const sideSign = block % 2 ? -1 : 1;
    return {
      position: new THREE.Vector3(point[0], point[2] + height, -point[1])
        .addScaledVector(outward, offset * sideSign),
    };
  }
}

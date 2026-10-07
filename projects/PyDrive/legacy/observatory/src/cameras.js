import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export const CAMERA_MODES = Object.freeze(["broadcast", "chase", "roof", "trackside", "drone", "free"]);
export const EDITORIAL_SHOT_LIMIT = 24;

const BROADCAST_SEQUENCE = Object.freeze([
  { id: "trackside-long", label: "TRACKSIDE · 105MM", duration: 6.2 },
  { id: "chase-low", label: "LOW CHASE · 35MM", duration: 5.8 },
  { id: "apex", label: "APEX CAM · 85MM", duration: 6.1 },
  { id: "helicopter", label: "AERIAL · 50MM", duration: 5.6 },
  { id: "roof", label: "ONBOARD · 24MM", duration: 4.8 },
  { id: "trackside-close", label: "TRACKSIDE · 65MM", duration: 5.9 },
]);

const desiredPosition = new THREE.Vector3();
const desiredTarget = new THREE.Vector3();
const forward = new THREE.Vector3();
const side = new THREE.Vector3();
const lift = new THREE.Vector3(0, 1, 0);

function wrappedProgress(value) {
  return ((Number(value) || 0) % 1 + 1) % 1;
}

function progressDistance(a, b) {
  const distance = Math.abs(wrappedProgress(a) - wrappedProgress(b));
  return Math.min(distance, 1 - distance);
}

function boundedNumber(value, fallback, min, max) {
  return THREE.MathUtils.clamp(Number.isFinite(Number(value)) ? Number(value) : fallback, min, max);
}

export function normalizeEditorialShot(shot = {}, index = 0) {
  const id = String(shot.id || `editorial-${index + 1}`)
    .replace(/[^a-z0-9-]/gi, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48) || `editorial-${index + 1}`;
  const label = String(shot.label || `EDITORIAL ${index + 1}`)
    .replace(/[^a-z0-9 .·/-]/gi, "")
    .trim()
    .slice(0, 42) || `EDITORIAL ${index + 1}`;
  return {
    id,
    label,
    editorial: true,
    approved: Boolean(shot.approved),
    anchorProgress: wrappedProgress(shot.anchorProgress),
    coverage: boundedNumber(shot.coverage, 0.018, 0.004, 0.08),
    side: shot.side === "right" ? "right" : "left",
    offset: boundedNumber(shot.offset, 14, 7, 32),
    height: boundedNumber(shot.height, 2.4, 1.35, 10),
    lookAhead: boundedNumber(shot.lookAhead, 4.5, -4, 24),
    fov: boundedNumber(shot.fov, 38, 24, 68),
    foreground: ["clear", "rail", "foliage"].includes(shot.foreground) ? shot.foreground : "clear",
  };
}

export function normalizeEditorialShots(shots) {
  const used = new Set();
  return (Array.isArray(shots) ? shots : [])
    .slice(0, EDITORIAL_SHOT_LIMIT)
    .map((shot, index) => normalizeEditorialShot(shot, index))
    .filter((shot) => {
      if (used.has(shot.id)) return false;
      used.add(shot.id);
      return true;
    });
}

export function approvedShotForProgress(shots, progress) {
  return normalizeEditorialShots(shots)
    .filter((shot) => shot.approved && progressDistance(shot.anchorProgress, progress) <= shot.coverage)
    .sort((a, b) => progressDistance(a.anchorProgress, progress) - progressDistance(b.anchorProgress, progress))[0] || null;
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
    this.shotLibrary = [];
    this.previewShotId = null;
    this.groundHeightSampler = null;
    this.editorialGroundCache = new Map();
  }

  setTrack(points) {
    this.trackPoints = points || [];
    this.editorialGroundCache.clear();
  }

  setGroundHeightSampler(sampler) {
    this.groundHeightSampler = typeof sampler === "function" ? sampler : null;
    this.editorialGroundCache.clear();
  }

  setShotLibrary(shots) {
    this.shotLibrary = normalizeEditorialShots(shots);
    this.editorialGroundCache.clear();
    if (this.previewShotId && !this.shotLibrary.some((shot) => shot.id === this.previewShotId)) {
      this.previewShotId = null;
      if (this.mode === "editor") this.setMode("broadcast");
    }
  }

  previewEditorialShot(id) {
    if (!this.shotLibrary.some((shot) => shot.id === id)) return false;
    this.previewShotId = id;
    this.mode = "editor";
    this.controls.enabled = false;
    this.snapNext = true;
    this.audioCut += 1;
    return true;
  }

  clearEditorialPreview() {
    if (this.mode !== "editor") return;
    this.previewShotId = null;
    this.setMode("broadcast");
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
    let editorialShot = null;
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
      editorialShot = approvedShotForProgress(this.shotLibrary, vehicle.progress);
      shot = editorialShot?.id || BROADCAST_SEQUENCE[this.broadcastIndex].id;
      this.announce(editorialShot?.label || BROADCAST_SEQUENCE[this.broadcastIndex].label);
    } else if (shot === "editor") {
      editorialShot = this.shotLibrary.find((candidate) => candidate.id === this.previewShotId) || null;
      if (!editorialShot) {
        this.clearEditorialPreview();
        return;
      }
      shot = editorialShot.id;
      this.announce(`PREVIEW · ${editorialShot.label}`);
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
    if (editorialShot) {
      // Preview is a framing review, so keep the live car in frame even if the
      // replay is currently away from the saved segment. Approved broadcast
      // shots remain anchored to their authored track coverage.
      const previewShot = this.mode === "editor"
        ? { ...editorialShot, anchorProgress: vehicle.progress }
        : editorialShot;
      const anchor = this.editorialAnchor(previewShot);
      desiredPosition.copy(anchor.position);
      desiredTarget.copy(focus)
        .addScaledVector(forward, editorialShot.lookAhead)
        .addScaledVector(lift, 0.66);
      targetFov = editorialShot.fov;
    } else if (shot === "chase" || shot === "chase-low") {
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
      const anchor = this.tracksideAnchor(vehicle.progress, 16, 9.5, 1.7);
      desiredPosition.copy(anchor.position);
      desiredTarget.copy(focus).addScaledVector(forward, 3.5).addScaledVector(lift, 0.62);
      targetFov = 31;
    } else {
      const explicit = this.mode === "trackside";
      const close = shot === "trackside-close";
      const anchor = this.tracksideAnchor(
        vehicle.progress,
        close ? 13 : 18,
        close ? 8.4 : 11.8,
        close ? 1.45 : 2.25,
      );
      desiredPosition.copy(anchor.position);
      desiredTarget.copy(focus).addScaledVector(forward, close ? 2.5 : 5.5).addScaledVector(lift, 0.68);
      targetFov = close ? 43 : 36;
    }

    const isFixed = Boolean(editorialShot) || shot === "trackside-long" || shot === "trackside-close" || shot === "apex" || shot === "trackside";
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
      position: new THREE.Vector3(point[0], point[2] + Math.max(height, 1.35), -point[1])
        .addScaledVector(outward, offset * sideSign),
    };
  }

  editorialAnchor(shot) {
    if (!this.trackPoints.length) return { position: desiredPosition.set(18, Math.max(shot.height, 1.35), 18) };
    const count = this.trackPoints.length;
    const index = Math.round(wrappedProgress(shot.anchorProgress) * (count - 1));
    const point = this.trackPoints[index];
    const next = this.trackPoints[(index + 6) % count];
    const tangent = new THREE.Vector3(next[0] - point[0], 0, -(next[1] - point[1])).normalize();
    const outward = new THREE.Vector3(-tangent.z, 0, tangent.x);
    const sideSign = shot.side === "right" ? -1 : 1;
    // The foreground setting is deliberately a small compositional nudge, not
    // generated scenery: all actual track furniture remains world-authored.
    const foregroundNudge = shot.foreground === "foliage" ? 1.6 : shot.foreground === "rail" ? 0.75 : 0;
    const position = new THREE.Vector3(point[0], point[2], -point[1])
      .addScaledVector(outward, sideSign * (shot.offset + foregroundNudge));
    // Raycasting the visual DGM is only needed when a new editorial anchor is
    // reached. Quantizing the cache to about 20 m keeps live preview cheap.
    const groundKey = `${shot.id}:${Math.round(wrappedProgress(shot.anchorProgress) * 1000)}`;
    let groundY = this.editorialGroundCache.get(groundKey);
    if (!Number.isFinite(groundY)) {
      // The road/shoulder envelope intentionally removes DGM triangles. Probe
      // just outside it when a camera sits in that visual gap, then use that
      // terrain height as a conservative clearance reference.
      const probeOffset = Math.max(shot.offset + foregroundNudge, 26);
      const probe = position.clone().setY(point[2]).addScaledVector(outward, sideSign * (probeOffset - shot.offset - foregroundNudge));
      groundY = this.groundHeightSampler?.(probe.x, probe.z, point[2]) ?? point[2];
      this.editorialGroundCache.set(groundKey, groundY);
    }
    position.y = groundY + Math.max(shot.height, 1.35);
    return { position };
  }
}

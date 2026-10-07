import * as THREE from "three";

/**
 * Cheap camera-local rain streaks — presentation only, never a physics input.
 * Inspired by early Observatory BroadcastPrecipitation.
 */
export class CameraRain {
  constructor(scene, count = 480) {
    this.count = count;
    this.positions = new Float32Array(count * 6);
    this.seeds = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      this.seeds[i * 3] = ((i * 0.754877666) % 1 - 0.5) * 56;
      this.seeds[i * 3 + 1] = (i * 0.569840291) % 1;
      this.seeds[i * 3 + 2] = ((i * 0.438579) % 1 - 0.5) * 56;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.mesh = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: 0xb8d5e3,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      fog: true,
    }));
    this.mesh.name = "watch25d-camera-rain";
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    scene.add(this.mesh);
    this.enabled = false;
  }

  setEnabled(on) {
    this.enabled = Boolean(on);
    this.mesh.visible = this.enabled;
  }

  update(cameraPosition, elapsed) {
    if (!this.enabled) return;
    const rate = 20;
    const length = 1.2;
    for (let i = 0; i < this.count; i += 1) {
      const seed = i * 3;
      const phase = (this.seeds[seed + 1] + (elapsed * rate) / 40) % 1;
      const x = cameraPosition.x + this.seeds[seed] + Math.sin(elapsed * 1.7 + i) * 0.55;
      const y = cameraPosition.y + 1.5 + phase * 28;
      const z = cameraPosition.z + this.seeds[seed + 2] + elapsed * 3.4;
      const o = i * 6;
      this.positions[o] = x;
      this.positions[o + 1] = y;
      this.positions[o + 2] = z;
      this.positions[o + 3] = x - 0.12;
      this.positions[o + 4] = y - length;
      this.positions[o + 5] = z - 0.05;
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

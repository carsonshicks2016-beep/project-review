import * as THREE from "three";

export function tyreSprayStrength(condition, speedMps = 0) {
  const wetness = THREE.MathUtils.clamp(Number(condition?.wetness) || 0, 0, 1);
  return THREE.MathUtils.clamp(wetness * THREE.MathUtils.smoothstep(Number(speedMps) || 0, 14, 42), 0, 1);
}

// A camera-local visual volume. It deliberately has no relationship to the
// simulator; rain is broadcast atmosphere only, never a physics input.
export class BroadcastPrecipitation {
  constructor(scene, count = 900) {
    this.count = count;
    this.positions = new Float32Array(count * 6);
    this.seeds = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      this.seeds[index * 3] = ((index * 0.754877666) % 1 - 0.5) * 92;
      this.seeds[index * 3 + 1] = (index * 0.569840291) % 1;
      this.seeds[index * 3 + 2] = ((index * 0.438579) % 1 - 0.5) * 92;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.mesh = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: 0xb8d5e3,
      transparent: true,
      opacity: 0.26,
      depthWrite: false,
      fog: true,
    }));
    this.mesh.name = "visual_camera_local_rain_volume";
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    scene.add(this.mesh);
    this.condition = { precipitation: "clear", lighting: "day" };
  }

  setVisualCondition(condition) {
    this.condition = condition;
    this.mesh.visible = condition.precipitation === "rain";
  }

  update(cameraPosition, elapsed) {
    if (!this.mesh.visible) return;
    const rate = this.condition.lighting === "night" ? 16 : 22;
    const length = this.condition.lighting === "night" ? 1.45 : 1.15;
    for (let index = 0; index < this.count; index += 1) {
      const seed = index * 3;
      const phase = (this.seeds[seed + 1] + elapsed * rate / 44) % 1;
      const x = cameraPosition.x + this.seeds[seed] + Math.sin(elapsed * 1.7 + index) * 0.7;
      const y = cameraPosition.y + 2 + phase * 38;
      const z = cameraPosition.z + this.seeds[seed + 2] + elapsed * 4.1;
      const offset = index * 6;
      this.positions[offset] = x;
      this.positions[offset + 1] = y;
      this.positions[offset + 2] = z;
      this.positions[offset + 3] = x - 0.15;
      this.positions[offset + 4] = y - length;
      this.positions[offset + 5] = z - 0.06;
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

// A compact vehicle-local spray plume. It is deliberately visual-only and is
// driven by already-rendered pose/speed, never by tyre forces or physics.
export class TyreSpray {
  constructor(scene, count = 90) {
    this.count = count;
    this.seeds = new Float32Array(count * 3);
    this.positions = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      this.seeds[index * 3] = ((index * 0.618033989) % 1 - 0.5);
      this.seeds[index * 3 + 1] = (index * 0.381966011) % 1;
      this.seeds[index * 3 + 2] = ((index * 0.754877666) % 1 - 0.5);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    this.material = new THREE.PointsMaterial({
      color: 0xaebfc0,
      size: 0.06,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.NormalBlending,
      fog: true,
    });
    this.mesh = new THREE.Points(geometry, this.material);
    this.mesh.name = "visual_vehicle_tyre_spray";
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    scene.add(this.mesh);
    this.condition = { wetness: 0, precipitation: "clear" };
  }

  setVisualCondition(condition) {
    this.condition = condition || this.condition;
    this.mesh.visible = Number(this.condition.wetness) > 0.28;
  }

  update(vehicle, elapsed) {
    if (!vehicle || !this.mesh.visible) return;
    const strength = tyreSprayStrength(this.condition, vehicle.speedMps);
    this.mesh.visible = strength > 0.015;
    this.material.opacity = 0.035 + strength * (this.condition.precipitation === "rain" ? 0.13 : 0.075);
    this.material.size = 0.035 + strength * 0.075;
    const yaw = Number(vehicle.yaw) || 0;
    const forwardX = Math.cos(yaw);
    const forwardZ = -Math.sin(yaw);
    const sideX = -forwardZ;
    const sideZ = forwardX;
    const originX = vehicle.position.x - forwardX * 1.55;
    const originY = vehicle.position.y;
    const originZ = vehicle.position.z - forwardZ * 1.55;
    for (let index = 0; index < this.count; index += 1) {
      const seed = index * 3;
      const age = (this.seeds[seed + 1] + elapsed * (0.9 + strength * 1.8)) % 1;
      const spread = 0.28 + age * (0.8 + strength * 2.0);
      const lateral = this.seeds[seed] * spread;
      const backward = 0.35 + age * (1.4 + strength * 3.9);
      const rise = 0.08 + age * (0.28 + strength * 0.72) + Math.abs(this.seeds[seed + 2]) * 0.12;
      this.positions[seed] = originX - forwardX * backward + sideX * lateral;
      this.positions[seed + 1] = originY + rise;
      this.positions[seed + 2] = originZ - forwardZ * backward + sideZ * lateral;
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

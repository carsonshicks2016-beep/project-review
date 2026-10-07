import * as THREE from 'three';

// ── Pure Mathematical Elevation Function ──
// Zero external dependencies so it can run seamlessly in main thread and headless Web Workers.
export { getOutdoorElevation } from './terrainHeight.js';
import { getOutdoorElevation } from './terrainHeight.js';

export function createOutdoorTerrain(size = 1000, segments = 90) {
  const geom = new THREE.PlaneGeometry(size, size, segments, segments);
  geom.rotateX(-Math.PI / 2); // Orient XZ horizontal plane

  const pos = geom.attributes.position;
  const count = pos.count;

  // Displace vertices with mathematical elevation
  for (let i = 0; i < count; i++) {
    const vx = pos.getX(i);
    const vz = pos.getZ(i);
    const vy = getOutdoorElevation(vx, vz);
    pos.setY(i, vy);
  }

  // Convert to non-indexed geometry for faceted low-poly look
  const nonIndexed = geom.toNonIndexed();
  nonIndexed.computeVertexNormals();

  // Low-poly vertex color palette
  const cWaterSand = new THREE.Color('#d8c697'); // Shoreline / sand
  const cGrassLow  = new THREE.Color('#4d8a3e'); // Rich valley meadow
  const cGrassMid  = new THREE.Color('#5ba348'); // Light alpine slope
  const cRockDark  = new THREE.Color('#555f6b'); // Granite rock shadow
  const cRockLight = new THREE.Color('#788390'); // Exposed rock face
  const cSnow      = new THREE.Color('#f0f5fa'); // Alpine snow peak

  const niPos = nonIndexed.attributes.position;
  const niNorm = nonIndexed.attributes.normal;
  const niCount = niPos.count;
  const colors = new Float32Array(niCount * 3);
  const tempCol = new THREE.Color();

  for (let i = 0; i < niCount; i++) {
    const y = niPos.getY(i);
    const ny = niNorm.getY(i); // Steepness indicator (1 = flat ground, 0 = vertical cliff)

    if (y < 2.0) {
      // Shoreline sand
      tempCol.copy(cWaterSand);
    } else if (y > 45.0 && ny > 0.6) {
      // Snow caps on high summits
      tempCol.copy(cSnow);
    } else if (ny < 0.62 || (y > 28.0 && ny < 0.75)) {
      // Steep rock cliff or high ridge
      tempCol.copy(ny < 0.45 ? cRockDark : cRockLight);
    } else if (y > 20.0) {
      // High alpine meadow transitioning into rock
      const t = (y - 20.0) / 25.0;
      tempCol.copy(cGrassMid).lerp(cRockLight, t * 0.6);
    } else {
      // Valley grass with subtle natural variation
      const noiseVar = ((i % 7) - 3) * 0.03;
      tempCol.copy(cGrassLow);
      tempCol.r = Math.min(1, Math.max(0, tempCol.r + noiseVar));
      tempCol.g = Math.min(1, Math.max(0, tempCol.g + noiseVar * 1.5));
      tempCol.b = Math.min(1, Math.max(0, tempCol.b + noiseVar));
    }

    colors[i * 3]     = tempCol.r;
    colors[i * 3 + 1] = tempCol.g;
    colors[i * 3 + 2] = tempCol.b;
  }

  nonIndexed.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.88,
    metalness: 0.08,
    flatShading: true
  });

  const mesh = new THREE.Mesh(nonIndexed, mat);
  mesh.name = 'outdoor-terrain';
  mesh.receiveShadow = true;
  mesh.castShadow = false;

  return mesh;
}

// ── Low-Poly Alpine Water Plane ──
export function createOutdoorWater(size = 350) {
  const geom = new THREE.PlaneGeometry(size, size, 12, 12);
  geom.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshStandardMaterial({
    color: '#2b7fb3',
    roughness: 0.15,
    metalness: 0.35,
    transparent: true,
    opacity: 0.78,
    flatShading: true
  });

  const water = new THREE.Mesh(geom, mat);
  water.name = 'outdoor-water';
  water.position.set(-60, 1.2, 30); // Sits in a low natural valley pocket
  water.receiveShadow = true;
  return water;
}

// ── Instanced Low-Poly Pine Trees ──
export function createOutdoorTrees(count = 220) {
  const trunkGeo = new THREE.CylinderGeometry(0.3, 0.45, 1.8, 5);
  trunkGeo.translate(0, 0.9, 0);

  const foliage1 = new THREE.ConeGeometry(2.4, 3.5, 5);
  foliage1.translate(0, 3.2, 0);

  const foliage2 = new THREE.ConeGeometry(1.9, 3.0, 5);
  foliage2.translate(0, 5.0, 0);

  const foliage3 = new THREE.ConeGeometry(1.3, 2.2, 5);
  foliage3.translate(0, 6.6, 0);

  const group = new THREE.Group();
  group.name = 'outdoor-trees-group';

  const trunkMat = new THREE.MeshStandardMaterial({ color: '#4a3319', roughness: 0.9, flatShading: true });
  const foliageMat = new THREE.MeshStandardMaterial({ color: '#255728', roughness: 0.8, flatShading: true });

  const instTrunk = new THREE.InstancedMesh(trunkGeo, trunkMat, count);
  const instF1 = new THREE.InstancedMesh(foliage1, foliageMat, count);
  const instF2 = new THREE.InstancedMesh(foliage2, foliageMat, count);
  const instF3 = new THREE.InstancedMesh(foliage3, foliageMat, count);

  instTrunk.castShadow = true;
  instF1.castShadow = true;
  instF2.castShadow = true;
  instF3.castShadow = true;

  const dummy = new THREE.Object3D();
  let placed = 0;
  let seed = 42;
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  for (let i = 0; i < count * 3 && placed < count; i++) {
    const angle = rng() * Math.PI * 2;
    const r = 40 + rng() * 320;
    const x = Math.cos(angle) * r;
    const z = Math.sin(angle) * r;
    const y = getOutdoorElevation(x, z);

    // Don't place trees in water or on steep high cliffs
    if (y < 1.6 || y > 38.0) continue;

    // Avoid immediate center racing corridor if too close to gates
    if (r < 65 && rng() > 0.3) continue;

    const scale = 0.75 + rng() * 0.7;
    dummy.position.set(x, y, z);
    dummy.rotation.y = rng() * Math.PI * 2;
    dummy.scale.set(scale, scale * (0.9 + rng() * 0.3), scale);
    dummy.updateMatrix();

    instTrunk.setMatrixAt(placed, dummy.matrix);
    instF1.setMatrixAt(placed, dummy.matrix);
    instF2.setMatrixAt(placed, dummy.matrix);
    instF3.setMatrixAt(placed, dummy.matrix);
    placed++;
  }

  instTrunk.instanceMatrix.needsUpdate = true;
  instF1.instanceMatrix.needsUpdate = true;
  instF2.instanceMatrix.needsUpdate = true;
  instF3.instanceMatrix.needsUpdate = true;

  group.add(instTrunk);
  group.add(instF1);
  group.add(instF2);
  group.add(instF3);

  return group;
}

// ── Instanced Low-Poly Boulders ──
export function createOutdoorRocks(count = 70) {
  const rockGeo = new THREE.DodecahedronGeometry(2.0, 0);
  const rockMat = new THREE.MeshStandardMaterial({
    color: '#656c77',
    roughness: 0.9,
    metalness: 0.1,
    flatShading: true
  });

  const instRocks = new THREE.InstancedMesh(rockGeo, rockMat, count);
  instRocks.name = 'outdoor-rocks';
  instRocks.castShadow = true;
  instRocks.receiveShadow = true;

  const dummy = new THREE.Object3D();
  let seed = 1337;
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  for (let i = 0; i < count; i++) {
    const angle = rng() * Math.PI * 2;
    const r = 55 + rng() * 300;
    const x = Math.cos(angle) * r;
    const z = Math.sin(angle) * r;
    const y = getOutdoorElevation(x, z);

    const sx = 0.8 + rng() * 1.8;
    const sy = 0.5 + rng() * 1.5;
    const sz = 0.8 + rng() * 1.8;

    dummy.position.set(x, y + sy * 0.4, z);
    dummy.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI);
    dummy.scale.set(sx, sy, sz);
    dummy.updateMatrix();

    instRocks.setMatrixAt(i, dummy.matrix);
  }

  instRocks.instanceMatrix.needsUpdate = true;
  return instRocks;
}

// ── Animated Low-Poly Wind Turbines ──
export function createWindTurbines() {
  const group = new THREE.Group();
  group.name = 'outdoor-turbines';

  const towerMat = new THREE.MeshStandardMaterial({ color: '#e8edf2', roughness: 0.5, flatShading: true });
  const bladeMat = new THREE.MeshStandardMaterial({ color: '#d0d8e0', roughness: 0.4, flatShading: true });

  const turbinePositions = [
    [-180, 220],
    [210, -170],
    [-240, -160]
  ];

  const rotors = [];

  turbinePositions.forEach(([tx, tz]) => {
    const ty = getOutdoorElevation(tx, tz);
    const subGroup = new THREE.Group();
    subGroup.position.set(tx, ty, tz);

    // Tower: 45m tall tapered cylinder
    const towerGeo = new THREE.CylinderGeometry(0.8, 1.8, 45, 6);
    towerGeo.translate(0, 22.5, 0);
    const tower = new THREE.Mesh(towerGeo, towerMat);
    tower.castShadow = true;
    subGroup.add(tower);

    // Nacelle (generator housing)
    const nacelleGeo = new THREE.BoxGeometry(2.5, 2.2, 5.0);
    nacelleGeo.translate(0, 45, 0);
    const nacelle = new THREE.Mesh(nacelleGeo, towerMat);
    nacelle.castShadow = true;
    subGroup.add(nacelle);

    // Rotor hub
    const rotorGroup = new THREE.Group();
    rotorGroup.position.set(0, 45, -2.6);

    const hubGeo = new THREE.ConeGeometry(1.0, 1.6, 6);
    hubGeo.rotateX(-Math.PI / 2);
    const hub = new THREE.Mesh(hubGeo, towerMat);
    rotorGroup.add(hub);

    // 3 Blades
    for (let b = 0; b < 3; b++) {
      const bladeGeo = new THREE.BoxGeometry(0.5, 18, 0.15);
      bladeGeo.translate(0, 9, 0);
      const blade = new THREE.Mesh(bladeGeo, bladeMat);
      blade.rotation.z = (b * Math.PI * 2) / 3;
      blade.castShadow = true;
      rotorGroup.add(blade);
    }

    subGroup.add(rotorGroup);
    group.add(subGroup);
    rotors.push(rotorGroup);
  });

  return { group, rotors };
}

// ── Complete Outdoor Environment Container ──
export class OutdoorEnvironment {
  constructor(scene) {
    this.scene = scene;
    this.container = new THREE.Group();
    this.container.name = 'outdoor-environment';
    this.container.visible = false;

    // Build subcomponents
    this.terrain = createOutdoorTerrain();
    this.water = createOutdoorWater();
    this.trees = createOutdoorTrees();
    this.rocks = createOutdoorRocks();
    const { group: turbines, rotors } = createWindTurbines();
    this.turbines = turbines;
    this.rotors = rotors;

    this.container.add(this.terrain);
    this.container.add(this.water);
    this.container.add(this.trees);
    this.container.add(this.rocks);
    this.container.add(this.turbines);

    this.scene.add(this.container);

    // Outdoor lights
    this.outdoorAmbient = new THREE.HemisphereLight('#8ec5f8', '#4a6338', 1.3);
    this.outdoorAmbient.name = 'outdoor-hemi-light';
    this.outdoorAmbient.visible = false;
    this.scene.add(this.outdoorAmbient);

    this.outdoorSun = new THREE.DirectionalLight('#fff5db', 2.4);
    this.outdoorSun.position.set(150, 220, 100);
    this.outdoorSun.castShadow = true;
    this.outdoorSun.shadow.mapSize.width = 2048;
    this.outdoorSun.shadow.mapSize.height = 2048;
    this.outdoorSun.shadow.camera.near = 10;
    this.outdoorSun.shadow.camera.far = 700;
    this.outdoorSun.shadow.camera.top = 250;
    this.outdoorSun.shadow.camera.bottom = -250;
    this.outdoorSun.shadow.camera.left = -250;
    this.outdoorSun.shadow.camera.right = 250;
    this.outdoorSun.shadow.bias = -0.0005;
    this.outdoorSun.name = 'outdoor-sun-light';
    this.outdoorSun.visible = false;
    this.scene.add(this.outdoorSun);
  }

  setVisible(visible) {
    this.container.visible = visible;
    this.outdoorAmbient.visible = visible;
    this.outdoorSun.visible = visible;
  }

  update(dt = 1 / 60) {
    if (!this.container.visible) return;
    // Rotate wind turbines
    for (let i = 0; i < this.rotors.length; i++) {
      this.rotors[i].rotation.z += (0.9 + i * 0.15) * dt;
    }
  }
}

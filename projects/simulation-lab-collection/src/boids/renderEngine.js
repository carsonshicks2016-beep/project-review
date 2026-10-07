/**
 * ApexFlock - 3D Three.js Biosphere Rendering Engine
 * Ethereal Exhibition Aesthetics:
 * - Fluid avian starling geometries with organic wing articulation
 * - Sculptural raptor apex predators with burnished bronze & obsidian materials
 * - Billowing silk streamline ribbons visualizing flock murmuration waves
 * - Celestial orbital rings & sculptural weathered sea-stacks (no cyber grids)
 * - Multi-mode cinematic camera director (Orbit, Chase, First-Person, Swarm)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class RenderEngine {
  constructor(container, ecosystem, onSelectAgent) {
    this.container = container;
    this.ecosystem = ecosystem;
    this.onSelectAgent = onSelectAgent;

    this.cameraMode = 'orbit'; // 'orbit', 'chase', 'firstPerson', 'swarm'
    this.selectedAgent = null;

    // Theme definitions
    this.currentTheme = 'abyssal';
    this.themes = {
      abyssal: {
        bg: 0x060810,
        fog: 0x060810,
        keyLight: 0xfff0d8,
        ambient: 0x161e2c,
        dust: 0x90c4c8,
        ring: 0x48647c,
        monolith: 0x121722,
        monolithRim: 0xd9b46e
      },
      sumie: {
        bg: 0x0c0d10,
        fog: 0x0c0d10,
        keyLight: 0xf5f0ea,
        ambient: 0x22242a,
        dust: 0xd0c8b8,
        ring: 0x5a544c,
        monolith: 0x18191e,
        monolithRim: 0xc85a66
      },
      alabaster: {
        bg: 0x121014,
        fog: 0x121014,
        keyLight: 0xffeed0,
        ambient: 0x2a2228,
        dust: 0xe0b888,
        ring: 0x6e5246,
        monolith: 0x201a22,
        monolithRim: 0xe5c185
      }
    };

    this._initScene();
    this._initLighting();
    this._initEnvironment();
    this._initAgentMeshes();
    this._initRibbonTrails();
    this._initFoodMeshes();
    this._initDustParticles();
    this._initInteraction();

    this._onResize = this._onResize.bind(this);
    window.addEventListener('resize', this._onResize);
  }

  _initScene() {
    this.scene = new THREE.Scene();
    const t = this.themes[this.currentTheme];
    this.scene.background = new THREE.Color(t.bg);
    this.scene.fog = new THREE.FogExp2(t.fog, 0.0018);

    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    this.camera = new THREE.PerspectiveCamera(52, width / height, 0.5, 1500);
    this.camera.position.set(0, 130, 320);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxDistance = 700;
    this.controls.minDistance = 6;
    this.controls.target.set(0, 15, 0);
  }

  _initLighting() {
    const t = this.themes[this.currentTheme];
    this.ambientLight = new THREE.AmbientLight(t.ambient, 1.8);
    this.scene.add(this.ambientLight);

    // Warm museum gallery key light
    this.keyLight = new THREE.DirectionalLight(t.keyLight, 2.4);
    this.keyLight.position.set(120, 240, 140);
    this.scene.add(this.keyLight);

    // Cool celestial rim light
    this.rimLight = new THREE.DirectionalLight(0x406088, 1.2);
    this.rimLight.position.set(-140, -80, -140);
    this.scene.add(this.rimLight);
  }

  _initEnvironment() {
    const bounds = this.ecosystem.bounds;
    const bCenterX = (bounds.maxX + bounds.minX) / 2;
    const bCenterY = (bounds.maxY + bounds.minY) / 2;
    const bCenterZ = (bounds.maxZ + bounds.minZ) / 2;
    const t = this.themes[this.currentTheme];

    // 1. CELESTIAL ORBITAL HORIZON RINGS (Artistic astrolabe boundary)
    this.orbitalRings = new THREE.Group();
    const ringRadii = [240, 280, 320];
    const ringTilts = [0.2, -0.35, 0.5];

    ringRadii.forEach((r, idx) => {
      const ringGeo = new THREE.RingGeometry(r - 0.6, r + 0.6, 96);
      const ringMat = new THREE.MeshBasicMaterial({
        color: t.ring,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.18 - idx * 0.04
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2 + ringTilts[idx];
      ring.rotation.y = ringTilts[idx] * 0.8;
      this.orbitalRings.add(ring);
    });
    this.orbitalRings.position.set(bCenterX, bCenterY, bCenterZ);
    this.scene.add(this.orbitalRings);

    // 2. SCULPTURAL WEATHERED BASALT SEA-STACKS
    this.obstacleMeshes = [];
    const hexGeo = new THREE.CylinderGeometry(1, 1.15, 1, 6, 2);
    const monolithMat = new THREE.MeshStandardMaterial({
      color: t.monolith,
      roughness: 0.85,
      metalness: 0.2,
      flatShading: true
    });

    for (let i = 0; i < this.ecosystem.obstacles.length; i++) {
      const obs = this.ecosystem.obstacles[i];
      const stackGroup = new THREE.Group();

      // Main weathered spire
      const mesh = new THREE.Mesh(hexGeo, monolithMat);
      mesh.scale.set(obs.radius, obs.height, obs.radius);
      mesh.position.y = obs.height / 2;
      stackGroup.add(mesh);

      // Fine golden/carmine rim accent ring along the pinnacle
      const topRingGeo = new THREE.RingGeometry(obs.radius * 0.88, obs.radius * 1.02, 6);
      const topRingMat = new THREE.MeshBasicMaterial({
        color: t.monolithRim,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.35
      });
      const topRing = new THREE.Mesh(topRingGeo, topRingMat);
      topRing.rotation.x = Math.PI / 2;
      topRing.position.y = obs.height;
      stackGroup.add(topRing);

      stackGroup.position.set(obs.x, bounds.minY, obs.z);
      this.scene.add(stackGroup);
      this.obstacleMeshes.push(stackGroup);
    }
  }

  _initDustParticles() {
    const particleCount = 800;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const bounds = this.ecosystem.bounds;

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * (bounds.maxX - bounds.minX) * 1.1;
      positions[i * 3 + 1] = (Math.random() - 0.5) * (bounds.maxY - bounds.minY) * 1.1 + 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * (bounds.maxZ - bounds.minZ) * 1.1;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const t = this.themes[this.currentTheme];
    const mat = new THREE.PointsMaterial({
      color: t.dust,
      size: 2.2,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending
    });

    this.dustParticles = new THREE.Points(geo, mat);
    this.scene.add(this.dustParticles);
  }

  _initAgentMeshes() {
    this.boidMeshMap = new Map();
    this.predatorMeshMap = new Map();

    // 1. ORGANIC AVIAN BOID GEOMETRY (Smooth teardrop fuselage + feathered swept wings)
    this.boidBodyGeo = this._createSculptedAvianBody();
    this.boidWingGeo = this._createSculptedAvianWings();

    // 2. SCULPTURAL PREDATOR RAPTOR GEOMETRY (Manta/Peregrine falcon predatory silhouette)
    this.predBodyGeo = this._createSculptedPredatorBody();
    this.predWingGeo = this._createSculptedPredatorWings();

    // Selection reticle (minimalist fine gold double ring)
    const ringGeo = new THREE.RingGeometry(3.6, 3.9, 36);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xd9b46e,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8
    });
    this.selectionRing = new THREE.Mesh(ringGeo, ringMat);
    this.selectionRing.visible = false;
    this.scene.add(this.selectionRing);
  }

  _createSculptedAvianBody() {
    // Elegant aerodynamic spindle fuselage
    const points = [];
    points.push(new THREE.Vector2(0, 2.0));       // Beak tip
    points.push(new THREE.Vector2(0.42, 1.4));    // Head
    points.push(new THREE.Vector2(0.68, 0.4));    // Chest
    points.push(new THREE.Vector2(0.55, -0.8));   // Belly
    points.push(new THREE.Vector2(0.25, -2.0));   // Tailbase
    points.push(new THREE.Vector2(0.04, -2.8));   // Tail tip

    const geo = new THREE.LatheGeometry(points, 10);
    geo.rotateX(Math.PI / 2); // Orient forward along Z
    geo.computeVertexNormals();
    return geo;
  }

  _createSculptedAvianWings() {
    // Parabolic cambered swept wings
    const geo = new THREE.BufferGeometry();
    const verts = [
      // Left Wing
      0, 0.1, 0.6,
      -3.4, 0.35, -0.6,
      -1.4, 0.15, -0.9,

      -3.4, 0.35, -0.6,
      -4.2, 0.2, -1.8,
      -1.4, 0.15, -0.9,

      // Right Wing
      0, 0.1, 0.6,
      1.4, 0.15, -0.9,
      3.4, 0.35, -0.6,

      3.4, 0.35, -0.6,
      1.4, 0.15, -0.9,
      4.2, 0.2, -1.8
    ];
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
    geo.computeVertexNormals();
    return geo;
  }

  _createSculptedPredatorBody() {
    // Sleek predatory raptor fuselage
    const points = [];
    points.push(new THREE.Vector2(0, 3.2));
    points.push(new THREE.Vector2(0.75, 2.0));
    points.push(new THREE.Vector2(1.15, 0.6));
    points.push(new THREE.Vector2(0.95, -1.2));
    points.push(new THREE.Vector2(0.45, -3.2));
    points.push(new THREE.Vector2(0.08, -4.5));

    const geo = new THREE.LatheGeometry(points, 12);
    geo.rotateX(Math.PI / 2);
    geo.computeVertexNormals();
    return geo;
  }

  _createSculptedPredatorWings() {
    // Broad, powerful forward-swept raptor wings
    const geo = new THREE.BufferGeometry();
    const verts = [
      // Left Wing
      0, 0.2, 1.0,
      -5.6, 0.45, -1.2,
      -2.2, 0.2, -1.6,

      -5.6, 0.45, -1.2,
      -6.8, 0.3, -2.8,
      -2.2, 0.2, -1.6,

      // Right Wing
      0, 0.2, 1.0,
      2.2, 0.2, -1.6,
      5.6, 0.45, -1.2,

      5.6, 0.45, -1.2,
      2.2, 0.2, -1.6,
      6.8, 0.3, -2.8
    ];
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
    geo.computeVertexNormals();
    return geo;
  }

  _createBoidMesh(boid) {
    const group = new THREE.Group();
    const hue = (boid.genome?.hue || 180) / 360;
    // Ethereal porcelain tone with subtle colored iridescence
    const bodyColor = new THREE.Color().setHSL(hue, 0.45, 0.65);

    const bodyMat = new THREE.MeshStandardMaterial({
      color: bodyColor,
      roughness: 0.35,
      metalness: 0.25,
      emissive: bodyColor,
      emissiveIntensity: 0.25
    });

    const body = new THREE.Mesh(this.boidBodyGeo, bodyMat);
    group.add(body);

    const wingMat = new THREE.MeshStandardMaterial({
      color: bodyColor,
      roughness: 0.4,
      metalness: 0.15,
      side: THREE.DoubleSide,
      emissive: bodyColor,
      emissiveIntensity: 0.2
    });
    const wings = new THREE.Mesh(this.boidWingGeo.clone(), wingMat);
    wings.name = "wings";
    group.add(wings);

    const s = boid.genome?.wingspan || 1.0;
    group.scale.set(s, s, s);

    group.userData = { agent: boid };
    this.scene.add(group);
    return group;
  }

  _createPredatorMesh(pred) {
    const group = new THREE.Group();
    const hue = (pred.genome?.hue || 350) / 360;
    // Matte obsidian with burnished copper/crimson accents
    const bodyColor = new THREE.Color(0x16181e);
    const rimColor = new THREE.Color().setHSL(hue, 0.75, 0.45);

    const bodyMat = new THREE.MeshStandardMaterial({
      color: bodyColor,
      roughness: 0.4,
      metalness: 0.85,
      emissive: rimColor,
      emissiveIntensity: 0.55
    });

    const body = new THREE.Mesh(this.predBodyGeo, bodyMat);
    group.add(body);

    const wingMat = new THREE.MeshStandardMaterial({
      color: bodyColor,
      roughness: 0.45,
      metalness: 0.8,
      side: THREE.DoubleSide,
      emissive: rimColor,
      emissiveIntensity: 0.45
    });
    const wings = new THREE.Mesh(this.predWingGeo.clone(), wingMat);
    wings.name = "wings";
    group.add(wings);

    const s = pred.genome?.size || 1.5;
    group.scale.set(s, s, s);

    group.userData = { agent: pred };
    this.scene.add(group);
    return group;
  }

  // 3. SILK STREAMLINE RIBBON TRAILS (Pass 3)
  _initRibbonTrails() {
    this.maxTrailSegments = 2400; // Enough for 200 agents x 12 segments
    this.trailPositions = new Float32Array(this.maxTrailSegments * 6);
    this.trailColors = new Float32Array(this.maxTrailSegments * 6);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.trailPositions, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(this.trailColors, 3).setUsage(THREE.DynamicDrawUsage));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
      linewidth: 1
    });

    this.trailLines = new THREE.LineSegments(geo, mat);
    this.scene.add(this.trailLines);
  }

  _updateRibbonTrails() {
    if (!this.trailLines) return;
    const posAttr = this.trailLines.geometry.attributes.position;
    const colAttr = this.trailLines.geometry.attributes.color;
    let segIdx = 0;

    // Helper to add trail
    const addAgentTrail = (agent, isPredator) => {
      const trail = agent.trail;
      if (!trail || trail.length < 2) return;

      const baseHue = (agent.genome?.hue || (isPredator ? 350 : 180)) / 360;
      const baseSat = isPredator ? 0.8 : 0.4;
      const baseLight = isPredator ? 0.5 : 0.65;
      const c = new THREE.Color();

      for (let i = 0; i < trail.length - 1; i++) {
        if (segIdx >= this.maxTrailSegments) break;

        const p1 = trail[i];
        const p2 = trail[i + 1];
        const alpha = (i / trail.length); // Fades toward tail

        const offset = segIdx * 6;
        posAttr.array[offset] = p1.x;
        posAttr.array[offset + 1] = p1.y;
        posAttr.array[offset + 2] = p1.z;

        posAttr.array[offset + 3] = p2.x;
        posAttr.array[offset + 4] = p2.y;
        posAttr.array[offset + 5] = p2.z;

        c.setHSL(baseHue, baseSat, baseLight * alpha);
        colAttr.array[offset] = c.r;
        colAttr.array[offset + 1] = c.g;
        colAttr.array[offset + 2] = c.b;

        colAttr.array[offset + 3] = c.r;
        colAttr.array[offset + 4] = c.g;
        colAttr.array[offset + 5] = c.b;

        segIdx++;
      }
    };

    // Render prey ribbons
    for (let i = 0; i < this.ecosystem.boids.length; i++) {
      if (this.ecosystem.boids[i].alive) {
        addAgentTrail(this.ecosystem.boids[i], false);
      }
    }

    // Render predator ribbons (brighter streak)
    for (let i = 0; i < this.ecosystem.predators.length; i++) {
      if (this.ecosystem.predators[i].alive) {
        addAgentTrail(this.ecosystem.predators[i], true);
      }
    }

    // Zero out unused segments
    for (let i = segIdx * 6; i < this.trailPositions.length; i++) {
      posAttr.array[i] = 0;
      colAttr.array[i] = 0;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  }

  _initFoodMeshes() {
    this.foodInstanced = null;
    const maxFood = this.ecosystem.maxFoods + 30;
    const foodGeo = new THREE.SphereGeometry(1.6, 10, 10);
    const foodMat = new THREE.MeshStandardMaterial({
      color: 0xe4bc73,
      roughness: 0.3,
      metalness: 0.1,
      emissive: 0xd9a44c,
      emissiveIntensity: 0.8
    });

    this.foodInstanced = new THREE.InstancedMesh(foodGeo, foodMat, maxFood);
    this.foodInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.foodInstanced);
    this._dummyObj = new THREE.Object3D();
  }

  _initInteraction() {
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);

      const pickables = [];
      this.boidMeshMap.forEach(mesh => pickables.push(mesh.children[0]));
      this.predatorMeshMap.forEach(mesh => pickables.push(mesh.children[0]));

      const hits = this.raycaster.intersectObjects(pickables, false);
      if (hits.length > 0) {
        const selected = hits[0].object.parent?.userData?.agent;
        if (selected) {
          this.setSelectedAgent(selected);
        }
      }
    });
  }

  setSelectedAgent(agent) {
    this.selectedAgent = agent;
    this.ecosystem.selectedAgent = agent;
    if (this.onSelectAgent) {
      this.onSelectAgent(agent);
    }
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    this.controls.enabled = (mode === 'orbit');
    if (mode === 'orbit') {
      if (this.camera.position.length() < 140) {
        this.camera.position.set(0, 130, 320);
        this.controls.target.set(0, 15, 0);
        this.controls.update();
      }
    }
  }

  setTheme(themeName) {
    if (!this.themes[themeName]) return;
    this.currentTheme = themeName;
    const t = this.themes[themeName];

    this.scene.background.set(t.bg);
    this.scene.fog.color.set(t.fog);
    this.ambientLight.color.set(t.ambient);
    this.keyLight.color.set(t.keyLight);
    if (this.dustParticles) this.dustParticles.material.color.set(t.dust);
  }

  _onResize() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  render(dt) {
    // 1. SYNC BOID MESHES
    const aliveBoidIds = new Set();
    for (let i = 0; i < this.ecosystem.boids.length; i++) {
      const boid = this.ecosystem.boids[i];
      aliveBoidIds.add(boid.id);

      let mesh = this.boidMeshMap.get(boid.id);
      if (!mesh) {
        mesh = this._createBoidMesh(boid);
        this.boidMeshMap.set(boid.id, mesh);
      }

      mesh.position.set(boid.position.x, boid.position.y, boid.position.z);

      const targetPos = mesh.position.clone().add(
        new THREE.Vector3(boid.forward.x, boid.forward.y, boid.forward.z)
      );
      mesh.lookAt(targetPos);
      mesh.rotateZ(boid.bankingRoll);

      // Wing flapping with wingtip articulation
      const wings = mesh.getObjectByName("wings");
      if (wings) {
        const flapAngle = Math.sin(boid.wingFlapPhase) * 0.42;
        wings.rotation.x = flapAngle;
      }
    }

    this.boidMeshMap.forEach((mesh, id) => {
      if (!aliveBoidIds.has(id)) {
        this.scene.remove(mesh);
        this.boidMeshMap.delete(id);
      }
    });

    // 2. SYNC PREDATOR MESHES
    const alivePredIds = new Set();
    for (let i = 0; i < this.ecosystem.predators.length; i++) {
      const pred = this.ecosystem.predators[i];
      alivePredIds.add(pred.id);

      let mesh = this.predatorMeshMap.get(pred.id);
      if (!mesh) {
        mesh = this._createPredatorMesh(pred);
        this.predatorMeshMap.set(pred.id, mesh);
      }

      mesh.position.set(pred.position.x, pred.position.y, pred.position.z);
      const targetPos = mesh.position.clone().add(
        new THREE.Vector3(pred.forward.x, pred.forward.y, pred.forward.z)
      );
      mesh.lookAt(targetPos);

      const body = mesh.children[0];
      if (body && body.material) {
        body.material.emissiveIntensity = pred.isPouncing ? 1.4 : 0.6;
      }
    }

    this.predatorMeshMap.forEach((mesh, id) => {
      if (!alivePredIds.has(id)) {
        this.scene.remove(mesh);
        this.predatorMeshMap.delete(id);
      }
    });

    // 3. SILK STREAMLINE RIBBONS
    this._updateRibbonTrails();

    // 4. SYNC FOOD INSTANCES
    const foods = this.ecosystem.foods;
    for (let i = 0; i < this.foodInstanced.count; i++) {
      if (i < foods.length && foods[i].active) {
        const f = foods[i];
        const pulse = 1.0 + Math.sin(f.pulsePhase) * 0.22;
        this._dummyObj.position.set(f.x, f.y, f.z);
        this._dummyObj.scale.set(pulse, pulse, pulse);
        this._dummyObj.updateMatrix();
        this.foodInstanced.setMatrixAt(i, this._dummyObj.matrix);
      } else {
        this._dummyObj.scale.set(0, 0, 0);
        this._dummyObj.updateMatrix();
        this.foodInstanced.setMatrixAt(i, this._dummyObj.matrix);
      }
    }
    this.foodInstanced.instanceMatrix.needsUpdate = true;

    // 5. ANIMATE DUST & CELESTIAL RINGS
    if (this.dustParticles) {
      this.dustParticles.rotation.y += 0.0003;
    }
    if (this.orbitalRings) {
      this.orbitalRings.rotation.y += 0.0002;
    }

    // 6. SELECTION RETICLE
    if (this.selectedAgent && this.selectedAgent.alive && this.cameraMode === 'orbit') {
      this.selectionRing.visible = true;
      this.selectionRing.position.set(
        this.selectedAgent.position.x,
        this.selectedAgent.position.y,
        this.selectedAgent.position.z
      );
      this.selectionRing.lookAt(this.camera.position);
      this.selectionRing.rotation.z += 0.015;
    } else {
      this.selectionRing.visible = false;
    }

    // Hide mesh in first-person camera mode so camera doesn't clip
    const isFPV = (this.cameraMode === 'firstPerson');
    if (this.selectedAgent) {
      const sMesh = this.selectedAgent.type === 'predator' 
        ? this.predatorMeshMap.get(this.selectedAgent.id)
        : this.boidMeshMap.get(this.selectedAgent.id);
      if (sMesh) sMesh.visible = !isFPV;
    }

    // 7. CAMERA MOTION
    this._updateCamera(dt);

    // 8. RENDER
    this.renderer.render(this.scene, this.camera);
  }

  _updateCamera(dt) {
    if (this.cameraMode === 'orbit') {
      this.controls.update();
    } else if (this.cameraMode === 'chase' && this.selectedAgent && this.selectedAgent.alive) {
      const a = this.selectedAgent;
      const isPred = (a.type === 'predator');
      const distBack = isPred ? 44 : 28;
      const distUp = isPred ? 14 : 9;

      const targetPos = new THREE.Vector3(
        a.position.x - a.forward.x * distBack + a.up.x * distUp,
        a.position.y - a.forward.y * distBack + a.up.y * distUp,
        a.position.z - a.forward.z * distBack + a.up.z * distUp
      );
      this.camera.position.lerp(targetPos, 0.1);
      const lookPos = new THREE.Vector3(
        a.position.x + a.forward.x * 25,
        a.position.y + a.forward.y * 25,
        a.position.z + a.forward.z * 25
      );
      this.camera.lookAt(lookPos);
    } else if (this.cameraMode === 'firstPerson' && this.selectedAgent && this.selectedAgent.alive) {
      const a = this.selectedAgent;
      const forwardOffset = a.type === 'predator' ? 6.5 : 3.0;
      this.camera.position.set(
        a.position.x + a.forward.x * forwardOffset,
        a.position.y + a.forward.y * forwardOffset + a.up.y * 0.4,
        a.position.z + a.forward.z * forwardOffset
      );
      const lookPos = new THREE.Vector3(
        a.position.x + a.forward.x * 80,
        a.position.y + a.forward.y * 80,
        a.position.z + a.forward.z * 80
      );
      this.camera.lookAt(lookPos);
    } else if (this.cameraMode === 'swarm') {
      let cx = 0, cy = 0, cz = 0;
      const boids = this.ecosystem.boids;
      if (boids.length > 0) {
        for (let i = 0; i < boids.length; i++) {
          cx += boids[i].position.x;
          cy += boids[i].position.y;
          cz += boids[i].position.z;
        }
        cx /= boids.length;
        cy /= boids.length;
        cz /= boids.length;
        const targetCam = new THREE.Vector3(cx, cy + 95, cz + 210);
        this.camera.position.lerp(targetCam, 0.04);
        this.camera.lookAt(new THREE.Vector3(cx, cy, cz));
      }
    } else {
      this.setCameraMode('orbit');
    }
  }
}

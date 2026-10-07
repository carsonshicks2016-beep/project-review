import * as THREE from 'three';
import { createRenderer, createScene, createGround, createGateMesh, createDroneMesh, createTrackLine, loadHousePointCloud, setGateState, pulseGate, setGateLabel, setGateLabelVisible, FOG_DENSITY } from './scene.js';
import { OutdoorEnvironment, getOutdoorElevation } from './outdoorTerrain.js';
import { checkGatePassing } from './gateCollision.js';
import { CameraSystem } from './fpvCamera.js';
import { generateCircuit, rebuildTrackSpline } from './circuitGenerator.js';
import { defaultTrackData } from './defaultTrack.js';
import { GeneticAlgorithm } from './geneticAlgorithm.js';
import { NeuralNet, LAYER_SIZES } from './neuralNet.js';
import { FitnessGraph } from './graph.js';
import { TrackEditor } from './editor.js';
import { ViewportHUD } from './hud.js';
import { EyeDomeLighting } from './eyeDome.js';
import { saveRun, loadRuns } from './runRecorder.js';
import { PPOTrainer } from './ppo/trainer.js';
import { VoxelGrid, DEFAULT_CELL_SIZE } from './collision/voxelGrid.js';
import { Commentator } from './commentary/commentator.js';

// PPO actor/critic hidden layers. Larger than the GA's net on purpose: gradient methods
// scale with parameter count where weight-space search does not.
const PPO_HIDDEN = [64, 64];

class DroneRacingApp {
  constructor() {
    // ── Mode State ──
    this.mode = 'sim'; // 'sim' | 'edit'
    this.trainer = 'ga'; // 'ga' | 'ppo'
    this.ppo = null;     // Lazily constructed: building it spawns workers
    this.isPaused = false;
    this.turboMode = false;
    this.loadedGenome = null; // Used to watch a single loaded drone

    // ── Renderer & Scene ──
    const viewport = document.getElementById('viewport');
    this.renderer = createRenderer();
    
    // Move canvas into viewport div instead of document.body
    document.body.removeChild(this.renderer.domElement);
    viewport.appendChild(this.renderer.domElement);
    this.renderer.setSize(viewport.clientWidth, viewport.clientHeight);
    
    // ONE-TIME AUTO-REPAIR & SYNC:
    if (!localStorage.getItem('nan_bug_repaired_v3')) {
        // We sync the user's saved track to the backend so the agent can read it
        const savedTrack = localStorage.getItem('track_custom');
        if (savedTrack) {
            fetch('http://localhost:8080/sync-track', {
                method: 'POST',
                body: savedTrack
            }).then(() => console.log('Track successfully synced to backend!'));
        }
        localStorage.setItem('nan_bug_repaired_v3', 'true');
    }

    this.edl = new EyeDomeLighting(this.renderer, { strength: 5.0, radius: 4.0 });
    this.edl.setSize(viewport.clientWidth, viewport.clientHeight);

    this.buildWorldGrid = (points) => {
      const t0 = performance.now();
      const pos = points.geometry.attributes.position.array;
      this.worldGrid = VoxelGrid.fromPoints(pos, {
        scale: points.scale.x,
        offsetY: points.position.y,
        cellSize: DEFAULT_CELL_SIZE,
        minPoints: 2
      });
      this.gridDirty = true;
      const ms = Math.round(performance.now() - t0);
      const pct = (100 * this.worldGrid.solidCount / this.worldGrid.totalCells).toFixed(1);
      console.log(`Collision grid: ${this.worldGrid.dims.join('x')} cells, ${this.worldGrid.solidCount.toLocaleString()} solid (${pct}%), built in ${ms}ms`);
      // Sanity check: a grid that does not overlap the track makes collision a no-op,
      // which is indistinguishable from collision being broken.
      const g = this.worldGrid;
      const hi = [0,1,2].map(a => g.origin[a] + g.dims[a] * g.cellSize);
      console.log(`Grid bounds x[${g.origin[0].toFixed(0)},${hi[0].toFixed(0)}] y[${g.origin[1].toFixed(0)},${hi[1].toFixed(0)}] z[${g.origin[2].toFixed(0)},${hi[2].toFixed(0)}]`);
      const gp = this.circuitData.gates.map(x => x.position);
      const inside = gp.filter(p => p.every((v,a) => v >= g.origin[a] && v <= hi[a])).length;
      const solidGates = gp.filter(p => g.isSolid(p[0], p[1], p[2])).length;
      console.log(`Gates: ${inside}/${gp.length} inside grid bounds, ${solidGates} sitting in solid cells`);
      // Sample the straight segments between gates for occlusion.
      let blocked = 0, samples = 0;
      for (let i = 0; i < gp.length; i++) {
        const a = gp[i], b = gp[(i+1)%gp.length];
        for (let t = 0; t <= 1; t += 0.02) {
          samples++;
          if (g.isSolid(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)) blocked++;
        }
      }
      console.log(`Track line: ${blocked}/${samples} sampled points inside solid geometry (${(100*blocked/samples).toFixed(1)}%)`);
      this.showToast(`World is solid (${this.worldGrid.solidCount.toLocaleString()} cells)`);
      this.refreshSolidWorldUI();
    };

    this.scene = createScene();
    this.ground = createGround();
    this.scene.add(this.ground);
    this.environment = 'outdoor'; // 'outdoor' | 'lidar'
    this.worldGrid = null;      // Occupancy grid built from the scan
    this.solidWorld = true;     // Whether the scan is collidable
    this.gridDirty = true;      // Workers need a fresh copy
    this.outdoor = new OutdoorEnvironment(this.scene);
    
    loadHousePointCloud(this.scene, {
      onDone: (points) => {
        if (this.environment === 'outdoor') {
          points.visible = false;
        }
        this.buildWorldGrid(points);
      }
    });

    this.applyEnvironmentVisuals(this.environment);

    // ── Circuit ──
    // Load local storage if present, otherwise use the baked-in default track
    this.circuitData = this.loadTrack(this.environment) || JSON.parse(JSON.stringify(defaultTrackData));
    this.gateMeshes = [];
    this.trackLine = null;

    // ── Drone Visual ──
    this.droneMesh = createDroneMesh();
    this.scene.add(this.droneMesh);

    // ── Tracker HUD Dot ──
    // This dot ignores depth mapping (depthTest: false) so it renders through walls,
    // making it impossible to lose the drone in Trackside view.
    const dotGeo = new THREE.SphereGeometry(1.5, 16, 16);
    const dotMat = new THREE.MeshBasicMaterial({ color: '#ff0055', depthTest: false, transparent: true, opacity: 0.9 });
    this.trackerDot = new THREE.Mesh(dotGeo, dotMat);
    this.trackerDot.renderOrder = 999; // Ensure it renders on top
    this.scene.add(this.trackerDot);

    // ── Camera System ──
    this.cameraSystem = new CameraSystem(this.droneMesh);
    this.cameraSystem.resize(viewport.clientWidth / viewport.clientHeight);

    // Now it's safe to spawn gates since cameraSystem exists!
    this.spawnGateMeshes();

    // ── Editor ──
    this.editor = new TrackEditor(this.scene, this.cameraSystem.chaseCamera, this.renderer.domElement, this.gateMeshes);
    this.editor.onGateMoved = () => this.updateTrackVisuals();
    this.editor.onGateSelected = (mesh) => {
      document.getElementById('ui-selected-gate').innerText = mesh ? `Gate ${mesh.userData.index}` : 'None';
    };

    // ── Telemetry Graph & Viewport HUD ──
    this.graph = new FitnessGraph('fitness-graph', this.circuitData.gates.length);
    this.hud = new ViewportHUD(viewport, this.circuitData.gates.length);

    // The drone narrates its own run. Events are extracted from the replayed trajectory,
    // so every number spoken is one that was measured.
    this.commentaryOn = true;
    this.commentator = new Commentator({ voice: 'template', speak: false });
    this.commentator.onLine(line => this.hud.pushComment(line.text, line.type));

    // ── GA Setup ──
    this.popSize = 120;
    this.genomeLength = NeuralNet.getWeightCount(LAYER_SIZES);
    this.ga = new GeneticAlgorithm(this.popSize, this.genomeLength, { inputSize: LAYER_SIZES[0] });
    this.generation = 0;
    
    // ── Replay State ──
    this.bestTrajectory = null;
    this.trailMesh = null;      // Flown path of the generation's best drone
    this.ghostMesh = null;      // Faded paths of the runners-up, one draw call
    this.gatePassFrames = [];   // Replay frame at which each gate is cleared
    this.currentGateIdx = 0;    // Gate the drone is flying at, live during replay
    this.liveSpeed = 0;         // Derived from replay keyframe deltas
    this.liveAltitude = 0;
    this.replayProgress = 0;

    // The worker samples every 3rd tick at dt=1/60, so trajectories are 20Hz. Playing
    // them back at that rate is real time; anything faster is fast-forward.
    this.KEYFRAME_HZ = 20;
    this.replayTime = 0;          // Fractional keyframe index
    this.pendingTrajectory = null; // Newest result, adopted when the current replay ends
    this.replayStartGate = 0;      // Curriculum gate the displayed run began at
    this.displayGeneration = 0;    // Generation the run on screen came from

    // ── Curriculum ──
    // Learning the whole lap from gate 0 means every new segment has to be reached
    // through the entire prefix first, so progress stalls. Instead we advance a frontier:
    // once the population reliably clears the gate it starts at, the frontier moves up.
    // Half of all generations still start somewhere behind the frontier so earlier
    // segments keep getting practised and the policy does not forget them.
    this.curriculum = {
      stage: 0,
      streak: 0,
      startGateIdx: 0,
      // Tuned by measurement. Demanding 2 linked gates from 15% stalls the frontier at
      // stage 1 (best runs 2/16); 1 gate from 35% still crawls (stage 1 by gen 2900).
      // 1 gate from 20% moves the frontier through all stages and produced 5-10/16.
      CLEAR_FRACTION: 0.2,
      GATES_NEEDED: 1,
      STREAK_NEEDED: 3,
      graduated: false
    };
    this.elapsed = 0;
    this._qA = new THREE.Quaternion();
    this._qB = new THREE.Quaternion();
    this._pA = new THREE.Vector3();
    this._pB = new THREE.Vector3();
    this.clock = new THREE.Clock();
    this.replayFrame = 0;
    this.bestGatesPassed = 0;
    this.bestTopSpeed = 0;
    this.allTimeBestFitness = -Infinity;
    // Recorded separately: fitness is only comparable between runs that started at the
    // same gate. A curriculum run starting at gate 15 clears its "whole lap" in one gate
    // and banks the completion bonus, which outscores a genuine 12-gate lap.
    this.bestFullLapFitness = -Infinity;

    // ── Worker Pool ──
    this.numWorkers = Math.min(navigator.hardwareConcurrency || 4, 8);
    this.workers = [];
    this.isSimulating = false;
    this.initWorkers();

    // ── Bind UI ──
    this.setupUI();

    // ── Window Resize ──
    window.addEventListener('resize', () => {
      this.renderer.setSize(viewport.clientWidth, viewport.clientHeight);
      this.cameraSystem.resize(viewport.clientWidth / viewport.clientHeight);
      this.syncSplatViewport(viewport.clientHeight);
      this.edl.setSize(viewport.clientWidth, viewport.clientHeight);
      if(this.graph) this.graph.resize();
    });

    this.resetDronePosition();
    this.render = this.render.bind(this);
    requestAnimationFrame(this.render);
    
    if(!this.isPaused) this.runGeneration();
  }

  // ── Track & Gates ──
  spawnGateMeshes() {
    this.gateMeshes.forEach(m => this.scene.remove(m));
    if(this.trackLine) this.scene.remove(this.trackLine);
    
    this.gateMeshes = [];
    this.circuitData.gates.forEach((gate, i) => {
      const mesh = createGateMesh(3.0); // Always spawn base mesh, scale dictates size
      
      if (gate.position && gate.position[0] !== null && !isNaN(gate.position[0])) {
          mesh.position.set(gate.position[0], gate.position[1], gate.position[2]);
      } else {
          mesh.position.set(0, 0, 0);
      }
      
      // Safely load exact rotation and scale
      if (gate.quaternion && gate.quaternion[0] !== null && !isNaN(gate.quaternion[0])) {
          mesh.quaternion.fromArray(gate.quaternion);
      } else if (gate.normal && gate.normal[0] !== null && !isNaN(gate.normal[0])) {
          const normal = new THREE.Vector3(...gate.normal);
          const up = new THREE.Vector3(...(gate.up || [0,1,0]));
          const right = new THREE.Vector3().crossVectors(up, normal).normalize();
          // Avoid NaN if vectors are parallel
          if (right.lengthSq() > 0.001) {
              const mat = new THREE.Matrix4().makeBasis(right, up, normal);
              mesh.quaternion.setFromRotationMatrix(mat);
          }
      }
      
      if (gate.scale && gate.scale[0] !== null && !isNaN(gate.scale[0])) {
          mesh.scale.fromArray(gate.scale);
      }
      
      mesh.userData.index = i;
      mesh.userData.labelsVisible = this.mode === 'edit';
      setGateLabel(mesh, i);
      this.scene.add(mesh);
      this.gateMeshes.push(mesh);
    });
    this.currentGateIdx = 0;
    this.gatePassFrames = [];
    if (this.hud) this.hud.setTotalGates(this.circuitData.gates.length);
    if (this.graph) this.graph.setTotalGates(this.circuitData.gates.length);
    
    // Load custom trackside camera angle safely
    if (this.circuitData.tracksideCamera && this.circuitData.tracksideCamera.position[0] !== null && !isNaN(this.circuitData.tracksideCamera.position[0])) {
        this.cameraSystem.tracksideCamera.position.fromArray(this.circuitData.tracksideCamera.position);
        
        if (this.circuitData.tracksideCamera.quaternion[0] !== null && !isNaN(this.circuitData.tracksideCamera.quaternion[0])) {
            this.cameraSystem.tracksideCamera.quaternion.fromArray(this.circuitData.tracksideCamera.quaternion);
        }
        if (this.circuitData.tracksideCamera.fov) {
            this.cameraSystem.tracksideCamera.fov = this.circuitData.tracksideCamera.fov;
            this.cameraSystem.tracksideCamera.updateProjectionMatrix();
        }
    } else {
        this.cameraSystem.tracksideCamera.position.set(20, 25, 45);
        this.cameraSystem.tracksideCamera.lookAt(0, 5, 0);
    }

    this.updateTrackVisuals();
  }

  updateTrackVisuals() {
    if(this.trackLine) this.scene.remove(this.trackLine);
    
    // Sync circuitData.gates with meshes
    this.gateMeshes.forEach((mesh, i) => {
      this.circuitData.gates[i].position = mesh.position.toArray();
      this.circuitData.gates[i].quaternion = mesh.quaternion.toArray();
      this.circuitData.gates[i].scale = mesh.scale.toArray();
      this.circuitData.gates[i].radius = mesh.scale.x * 3.0; // sync physics to visual size
      
      const normal = new THREE.Vector3(0,0,1).applyQuaternion(mesh.quaternion).normalize();
      const up = new THREE.Vector3(0,1,0).applyQuaternion(mesh.quaternion).normalize();
      
      this.circuitData.gates[i].normal = normal.toArray();
      this.circuitData.gates[i].up = up.toArray();
    });

    const curve = rebuildTrackSpline(this.circuitData.gates);
    if(curve) {
      this.trackLine = createTrackLine(curve);
      this.scene.add(this.trackLine);
    }
  }

  // ── Flown Path Visualisation ──
  // Draws the trajectory the generation's best drone actually flew, so you can compare
  // it against the cyan ideal spline instead of inferring the shape from a moving dot.
  buildTrail() {
    if (this.trailMesh) {
      this.scene.remove(this.trailMesh);
      this.trailMesh.geometry.dispose();
      this.trailMesh.material.dispose();
      this.trailMesh = null;
    }

    const traj = this.bestTrajectory;
    if (!traj || traj.length < 7 * 4) return;

    const numFrames = traj.length / 7;
    const step = Math.max(1, Math.floor(numFrames / 240));
    const pts = [];
    for (let f = 0; f < numFrames; f += step) {
      const o = f * 7;
      const p = new THREE.Vector3(traj[o], traj[o + 1], traj[o + 2]);
      // CatmullRom degenerates on duplicate points, which a stalled drone produces.
      if (pts.length === 0 || p.distanceToSquared(pts[pts.length - 1]) > 1e-3) pts.push(p);
    }
    if (pts.length < 2) return;

    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const geo = new THREE.TubeGeometry(curve, Math.min(500, pts.length * 2), 0.3, 6, false);
    const mat = new THREE.MeshBasicMaterial({
      color: '#a855f7',
      transparent: true,
      opacity: 0.5,
      depthWrite: false
    });
    this.trailMesh = new THREE.Mesh(geo, mat);
    this.trailMesh.renderOrder = 4;
    this.scene.add(this.trailMesh);
  }

  // The scan occludes the gates from every outside angle, which makes gates
  // unselectable in the editor. Fading it keeps the spatial context without hiding the
  // things being edited.
  setEnvironmentFaded(faded) {
    const lidar = this.scene.getObjectByName('lidar');
    if (!lidar) return;
    lidar.material.transparent = faded;
    lidar.material.depthWrite = !faded;
    if (lidar.material.uniforms && lidar.material.uniforms.uOpacity) {
      lidar.material.uniforms.uOpacity.value = faded ? 0.16 : 1.0;
    }
    lidar.material.needsUpdate = true;
  }

  // The splat shader sizes points in pixels, so it needs the live viewport height.
  syncSplatViewport(height) {
    const lidar = this.scene.getObjectByName('lidar');
    if (lidar && lidar.material.uniforms && lidar.material.uniforms.uViewportH) {
      lidar.material.uniforms.uViewportH.value = height;
    }
  }

  // Frames the whole gate layout when entering the editor. Previously this just nudged
  // whatever position the chase camera happened to be left in by +30/+30, which usually
  // landed the editor camera staring at empty space.
  frameTrackInEditor() {
    const box = new THREE.Box3();
    this.gateMeshes.forEach(m => box.expandByObject(m));
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 40;

    const cam = this.cameraSystem.chaseCamera;
    const dist = (radius / Math.tan((cam.fov * Math.PI) / 360)) * 1.5;

    cam.position.set(center.x + dist * 0.55, center.y + dist * 0.6, center.z + dist * 0.75);
    cam.near = Math.max(0.1, dist * 0.001);
    cam.far = Math.max(1500, dist * 6);
    cam.updateProjectionMatrix();
    cam.lookAt(center);

    this.editor.orbit.target.copy(center);
    this.editor.orbit.update();
  }

  // ── Environment Selection ──
  applyEnvironmentVisuals(env) {
    const isOutdoor = env === 'outdoor';
    if (this.outdoor) this.outdoor.setVisible(isOutdoor);
    if (this.ground) this.ground.visible = !isOutdoor;

    const lidar = this.scene.getObjectByName('lidar');
    if (lidar) lidar.visible = !isOutdoor;

    const cyberAmbient = this.scene.getObjectByName('cyber-ambient');
    if (cyberAmbient) cyberAmbient.visible = !isOutdoor;

    const cyberSun = this.scene.getObjectByName('cyber-sun');
    if (cyberSun) cyberSun.visible = !isOutdoor;

    const cyberParticles = this.scene.getObjectByName('cyber-particles');
    if (cyberParticles) cyberParticles.visible = !isOutdoor;

    if (isOutdoor) {
      this.scene.background = new THREE.Color('#74aee2');
      this.scene.fog = new THREE.FogExp2('#b8d7eb', 0.0018);
      this.renderer.toneMappingExposure = 1.05;
      this.edl.enabled = false;
      const btnEdl = document.getElementById('btn-edl');
      if (btnEdl) {
        btnEdl.classList.remove('active');
        btnEdl.innerText = 'Depth Shading: OFF';
      }
    } else {
      this.scene.background = new THREE.Color('#030108');
      this.scene.fog = new THREE.FogExp2('#030108', FOG_DENSITY);
      this.renderer.toneMappingExposure = 0.9;
      this.edl.enabled = true;
      const btnEdl = document.getElementById('btn-edl');
      if (btnEdl) {
        btnEdl.classList.add('active');
        btnEdl.innerText = 'Depth Shading: ON';
      }
    }

    const btnOutdoor = document.getElementById('btn-env-outdoor');
    const btnLidar = document.getElementById('btn-env-lidar');
    if (btnOutdoor) btnOutdoor.classList.toggle('active', isOutdoor);
    if (btnLidar) btnLidar.classList.toggle('active', !isOutdoor);
  }

  setEnvironment(which, autoLoadTrack = true) {
    if (this.environment === which) return;
    this.environment = which;
    this.applyEnvironmentVisuals(which);
    if (this.ppo) {
      this.ppo.environment = which;
    }
    if (autoLoadTrack && this.mode !== 'edit') {
      const saved = this.loadTrack(which);
      if (saved) {
        this.circuitData = saved;
        this.spawnGateMeshes();
        if (this.editor) this.editor.gateMeshes = this.gateMeshes;
        this.resetDronePosition();
      }
    }
    this.showToast(which === 'outdoor' ? 'Environment: Low-Poly Outdoor' : 'Environment: LiDAR Cyber');
  }

  refreshSolidWorldUI() {
    const btn = document.getElementById('btn-solid-world');
    if (!btn) return;
    const ready = !!this.worldGrid;
    btn.disabled = !ready;
    btn.classList.toggle('disabled', !ready);
    btn.classList.toggle('active', ready && this.solidWorld);
    btn.innerText = !ready ? 'Solid World: loading…'
      : this.solidWorld ? 'Solid World: ON' : 'Solid World: OFF';
  }

  setSolidWorld(on) {
    this.solidWorld = on;
    this.gridDirty = true;              // Workers must be told either way
    if (this.ppo) this.ppo.setGrid(on ? this.worldGrid : null);
    this.refreshSolidWorldUI();
    this.showToast(on ? 'Walls are solid — the drone can crash into the scan'
                      : 'Walls are decorative again');
  }

  // Draws the occupied cells as an instanced wireframe box field. Only the surface cells
  // are drawn: a cell fully enclosed by solid neighbours is never visible, and skipping
  // them cuts the instance count by roughly an order of magnitude.
  toggleVoxelOverlay() {
    if (this.voxelOverlay) {
      this.scene.remove(this.voxelOverlay);
      this.voxelOverlay.geometry.dispose();
      this.voxelOverlay.material.dispose();
      this.voxelOverlay = null;
      return;
    }
    const g = this.worldGrid;
    if (!g) return;

    const cells = [];
    const [nx, ny, nz] = g.dims;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          if (!g.isSolidCell(x, y, z)) continue;
          const buried =
            g.isSolidCell(x - 1, y, z) && g.isSolidCell(x + 1, y, z) &&
            g.isSolidCell(x, y - 1, z) && g.isSolidCell(x, y + 1, z) &&
            g.isSolidCell(x, y, z - 1) && g.isSolidCell(x, y, z + 1);
          if (buried) continue;
          cells.push(x, y, z);
        }
      }
    }

    const count = cells.length / 3;
    const geo = new THREE.BoxGeometry(g.cellSize, g.cellSize, g.cellSize);
    const mat = new THREE.MeshBasicMaterial({ color: '#00ff9d', wireframe: true, transparent: true, opacity: 0.25 });
    const mesh = new THREE.InstancedMesh(geo, mat, count);
    const m = new THREE.Matrix4();
    for (let i = 0; i < count; i++) {
      m.setPosition(
        g.origin[0] + (cells[i * 3] + 0.5) * g.cellSize,
        g.origin[1] + (cells[i * 3 + 1] + 0.5) * g.cellSize,
        g.origin[2] + (cells[i * 3 + 2] + 0.5) * g.cellSize
      );
      mesh.setMatrixAt(i, m);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.frustumCulled = false;
    this.voxelOverlay = mesh;
    this.scene.add(mesh);
    this.showToast(`Collision overlay: ${count.toLocaleString()} surface cells`);
  }

  // ── Trainer Selection ──
  // GA and PPO share everything downstream: the same physics, gates, curriculum, replay
  // recorder and graph. Only the thing producing trajectories differs.
  setTrainer(which) {
    if (this.trainer === which) return;
    this.trainer = which;

    document.getElementById('btn-trainer-ga').classList.toggle('active', which === 'ga');
    document.getElementById('btn-trainer-ppo').classList.toggle('active', which === 'ppo');
    document.getElementById('ppo-stats').hidden = which !== 'ppo';
    document.getElementById('ga-only-pop').hidden = which !== 'ga';
    document.getElementById('ui-gen-label').innerText = which === 'ppo' ? 'Iteration:' : 'Generation:';
    document.getElementById('graph-header').innerText =
      which === 'ppo' ? 'Episode Return Over Iterations' : 'Fitness Over Generations';
    document.getElementById('btn-save-drone').innerText = which === 'ppo' ? 'Save Policy' : 'Save Best Drone';
    document.getElementById('btn-load-drone').innerText = which === 'ppo' ? 'Load Policy' : 'Load Drone';
    document.getElementById('brain-io-title').innerText = which === 'ppo' ? 'Policy Management' : 'Genome Management';
    document.getElementById('ui-brain-topology').innerText =
      which === 'ppo' ? `[14, ${PPO_HIDDEN.join(', ')}, 4] actor + critic` : '[14, 16, 12, 4]';

    if (which === 'ppo' && !this.ppo) {
      this.ppo = new PPOTrainer({
        numWorkers: Math.min(this.numWorkers, 4),
        numEnvs: 8,
        steps: 128,
        maxTicks: 3600,
        config: { hidden: PPO_HIDDEN, initLogStd: -1.2, entCoef: 0.004 },
        environment: this.environment
      });
      this.ppo.init();
    }

    // Metrics are not comparable across trainers, so start the plot fresh.
    this.graph.reset();
    this.generation = 0;
    this.allTimeBestFitness = -Infinity;
    this.bestFullLapFitness = -Infinity;
    this.showToast(which === 'ppo' ? 'Switched to PPO (gradient RL)' : 'Switched to Evolution (GA)');

    if (!this.isPaused && !this.isSimulating && this.mode === 'sim') this.runGeneration();
  }

  // ── PPO Iteration ──
  // Deliberately mirrors runGeneration(): same curriculum call, same pending-trajectory
  // handoff, same recorder, so the viewport and replay behave identically.
  async runPPOIteration() {
    const startGateIdx = this.pickStartGate();
    this.curriculum.startGateIdx = startGateIdx;
    this.resetDronePosition(startGateIdx);

    const { stats, best } = await this.ppo.iterate(this.circuitData.gates, startGateIdx, this.environment);
    if (this.mode !== 'sim') return;

    this.generation = stats.iteration;

    if (best && best.trace) {
      const data = new Float32Array(best.trace);
      if (data.length >= 14) {
        this.pendingTrajectory = {
          data,
          gates: best.gates,
          speed: best.topSpeed,
          generation: stats.iteration,
          startGateIdx,
          fitness: best.ret,
          hitWall: false,
          crashed: false,
          isRecord: false
        };
        // Record before adopting: adoptPendingTrajectory() clears pendingTrajectory.
        if (startGateIdx === 0 && best.ret > this.bestFullLapFitness) {
          this.bestFullLapFitness = best.ret;
          this.pendingTrajectory.isRecord = true;
          this.recordBestRun(this.pendingTrajectory);
        }
        if (this.turboMode || !this.bestTrajectory) this.adoptPendingTrajectory();
      }
    }

    // The curriculum promotion test wants per-episode gate counts; PPO reports the
    // distribution rather than the raw list, so reconstruct an equivalent sample.
    const cleared = [];
    const nEps = Math.max(1, stats.episodes);
    for (let i = 0; i < nEps; i++) cleared.push(i < Math.round(stats.meanGates * nEps) ? 1 : 0);
    this.updateCurriculum(startGateIdx, cleared);

    this.graph.addDataPoint(stats.iteration, stats.bestReturn, stats.meanReturn, stats.bestGates);

    document.getElementById('ui-gen').innerText = stats.iteration;
    document.getElementById('ui-gates').innerText = `${stats.bestGates} / ${this.circuitData.gates.length}`;
    if (best) document.getElementById('ui-speed').innerText = `${best.topSpeed.toFixed(1)} km/h`;
    document.getElementById('ui-status').innerText = this.turboMode ? '\u26a1 TURBO' : 'Training...';
    document.getElementById('ui-ppo-steps').innerText = stats.totalSteps.toLocaleString();
    document.getElementById('ui-ppo-return').innerText = stats.meanReturn.toFixed(1);
    document.getElementById('ui-ppo-entropy').innerText = stats.entropy.toFixed(3);
    document.getElementById('ui-ppo-kl').innerText = stats.approxKL.toFixed(4);
    document.getElementById('ui-ppo-vloss').innerText = stats.valueLoss.toFixed(3);

    const stageEl = document.getElementById('ui-stage');
    if (stageEl) {
      const c = this.curriculum;
      stageEl.innerText = c.graduated ? 'Full lap'
        : `${c.stage} / ${this.circuitData.gates.length - 1}` + (startGateIdx !== c.stage ? ` (rehearsing ${startGateIdx})` : '');
    }
  }

  // Called whenever a generation beats the all-time best fitness.
  recordBestRun(run) {
    if (!run || !run.data) return;
    const kept = saveRun({
      trajectory: run.data,
      fitness: run.fitness,
      gatesPassed: run.gates,
      topSpeed: run.speed,
      generation: run.generation,
      startGateIdx: run.startGateIdx,
      gates: this.circuitData.gates,
      tracksideCamera: this.circuitData.tracksideCamera
    });
    if (kept) this.refreshBestRunUI();
  }

  refreshBestRunUI() {
    const runs = loadRuns();
    const label = document.getElementById('ui-best-run');
    const btn = document.getElementById('btn-open-replay');
    if (!label || !btn) return;

    if (runs.length === 0) {
      label.innerText = 'None yet';
      btn.disabled = true;
      btn.classList.add('disabled');
      return;
    }

    const best = runs[0];
    label.innerText = `${best.gatesPassed}/${best.totalGates} gates · gen ${best.generation}`;
    btn.disabled = false;
    btn.classList.remove('disabled');
  }

  // Swaps in the newest completed run and rebuilds everything derived from it.
  adoptPendingTrajectory() {
    if (!this.pendingTrajectory) return;
    this.bestTrajectory = this.pendingTrajectory.data;
    this.bestGatesPassed = this.pendingTrajectory.gates;
    this.bestTopSpeed = this.pendingTrajectory.speed;
    this.displayGeneration = this.pendingTrajectory.generation;
    this.replayStartGate = this.pendingTrajectory.startGateIdx || 0;
    this.replayOutcome = {
      hitWall: !!this.pendingTrajectory.hitWall,
      crashed: !!this.pendingTrajectory.crashed,
      isRecord: !!this.pendingTrajectory.isRecord
    };
    this.pendingTrajectory = null;

    this.replayTime = 0;
    this.replayFrame = 0;
    this.currentGateIdx = 0;
    this.computeGateTimeline();
    this.buildTrail();

    if (this.commentator) {
      this.hud.clearComments();
      this.commentator.loadRun(
        {
          trajectory: this.bestTrajectory,
          startGateIdx: this.replayStartGate,
          ...this.replayOutcome
        },
        { gates: this.circuitData.gates, grid: this.solidWorld ? this.worldGrid : null }
      );
    }
  }

  // ── Population Ghosts ──
  // Draws the runners-up as faded threads in a single LineSegments, so the spread of
  // the population is visible. Early on these fan out wildly; as the GA converges they
  // collapse onto the winner's line, which is the whole story in one picture.
  buildGhosts(candidates) {
    if (this.ghostMesh) {
      this.scene.remove(this.ghostMesh);
      this.ghostMesh.geometry.dispose();
      this.ghostMesh.material.dispose();
      this.ghostMesh = null;
    }
    if (!candidates || candidates.length === 0) return;

    const MAX_GHOSTS = 8;
    const top = candidates
      .slice()
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, MAX_GHOSTS);

    const positions = [];
    const colors = [];
    // Rank ramp: brightest ghost sits just under the winner's violet, the tail fades out.
    const head = new THREE.Color('#a855f7');
    const tail = new THREE.Color('#6b5bb5');
    const c = new THREE.Color();

    top.forEach((g, rank) => {
      const traj = new Float32Array(g.buffer);
      const numFrames = traj.length / 7;
      if (numFrames < 2) return;

      c.copy(head).lerp(tail, top.length > 1 ? rank / (top.length - 1) : 0);
      const step = Math.max(1, Math.floor(numFrames / 120));

      let prev = null;
      for (let f = 0; f < numFrames; f += step) {
        const o = f * 7;
        const cur = [traj[o], traj[o + 1], traj[o + 2]];
        if (prev) {
          positions.push(prev[0], prev[1], prev[2], cur[0], cur[1], cur[2]);
          colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
        }
        prev = cur;
      }
    });

    if (positions.length === 0) return;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      depthWrite: false
    });
    this.ghostMesh = new THREE.LineSegments(geo, mat);
    this.ghostMesh.renderOrder = 3;
    this.ghostMesh.visible = this.mode === 'sim';
    this.scene.add(this.ghostMesh);
  }

  // Replays the trajectory against the same gate test the workers use, recording the
  // frame each gate is cleared. That turns "gates passed" from a per-generation total
  // into something that updates live as you watch the replay.
  computeGateTimeline() {
    this.gatePassFrames = [];
    this.currentGateIdx = this.replayStartGate || 0;

    const traj = this.bestTrajectory;
    if (!traj || traj.length < 14) return;

    const numFrames = traj.length / 7;
    const gates = this.circuitData.gates;
    const prev = [0, 0, 0];
    const cur = [0, 0, 0];
    let gi = this.replayStartGate || 0;

    for (let f = 1; f < numFrames && gi < gates.length; f++) {
      const po = (f - 1) * 7;
      const co = f * 7;
      prev[0] = traj[po]; prev[1] = traj[po + 1]; prev[2] = traj[po + 2];
      cur[0] = traj[co];  cur[1] = traj[co + 1];  cur[2] = traj[co + 2];

      if (checkGatePassing(prev, cur, gates[gi]).status === 'PASSED') {
        this.gatePassFrames.push(f);
        gi++;
      }
    }
  }

  resetDronePosition(gateIdx = 0) {
    const gates = this.circuitData.gates;
    const idx = Math.max(0, Math.min(gateIdx, gates.length - 1));
    const gate = gates[idx];
    const normal = gate.normal || [0,0,1];
    // 15m back along the gate normal, facing through it.
    this.startPos = [
      gate.position[0] - normal[0] * 15,
      gate.position[1],
      gate.position[2] - normal[2] * 15
    ];
    if (this.environment === 'outdoor') {
      const groundY = getOutdoorElevation(this.startPos[0], this.startPos[2]);
      if (this.startPos[1] < groundY + 1.5) {
        this.startPos[1] = groundY + 1.5;
      }
    }
    this.startYaw = Math.atan2(-normal[0], -normal[2]);
    this.droneMesh.position.set(...this.startPos);
  }

  // Picks where this generation starts. Biased to the frontier, but half the time it
  // drops back so the population keeps rehearsing everything it has already learned.
  pickStartGate() {
    const c = this.curriculum;
    // Once the frontier reaches the end, consolidate: the real task is a full lap from
    // gate 0, and a policy only ever trained from random gates never optimises for it.
    if (c.graduated) return 0;
    if (c.stage === 0) return 0;
    return Math.random() < 0.5 ? c.stage : Math.floor(Math.random() * c.stage);
  }

  // Advances the frontier when enough of the population clears the gate it starts at.
  updateCurriculum(startGateIdx, clearedCounts) {
    const c = this.curriculum;
    const lastGate = this.circuitData.gates.length - 1;
    if (c.graduated) return;
    // Only generations run at the frontier count toward promotion.
    if (startGateIdx !== c.stage) return;

    const cleared = clearedCounts.filter(g => g >= c.GATES_NEEDED).length;
    const fraction = clearedCounts.length ? cleared / clearedCounts.length : 0;

    if (fraction >= c.CLEAR_FRACTION) {
      c.streak++;
      if (c.streak >= c.STREAK_NEEDED) {
        c.streak = 0;
        if (c.stage >= lastGate) {
          c.graduated = true;
          this.showToast('Curriculum complete — training full laps from gate 0');
        } else {
          c.stage++;
          this.showToast(`Curriculum: advanced to gate ${c.stage}`);
        }
      }
    } else {
      c.streak = 0;
    }
  }

  // ── Worker Pool ──
  initWorkers() {
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    for (let i = 0; i < this.numWorkers; i++) {
      // Module worker: it imports the real physics/net/fitness instead of inlining copies.
      this.workers.push(new Worker(new URL('./simWorker.js', import.meta.url), { type: 'module' }));
    }
  }

  // ── UI Integration ──
  showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerText = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        toast.addEventListener('animationend', () => toast.remove());
    }, 3000);
  }

  setupUI() {
    // Left Panel
    const btnSim = document.getElementById('btn-toggle-sim');
    btnSim.addEventListener('click', () => {
      this.isPaused = !this.isPaused;
      btnSim.innerText = this.isPaused ? "Resume Sim" : "Pause Sim";
      btnSim.classList.toggle('active', !this.isPaused);
      if(!this.isPaused && !this.isSimulating && this.mode === 'sim') {
        this.runGeneration();
      }
    });

    const btnTurbo = document.getElementById('btn-turbo');
    btnTurbo.addEventListener('click', () => {
      this.turboMode = !this.turboMode;
      btnTurbo.innerText = this.turboMode ? "Turbo: ON ⚡" : "Turbo: OFF";
      btnTurbo.classList.toggle('active', this.turboMode);
    });
    
    const sliderPopSize = document.getElementById('slider-pop-size');
    const labelPopSize = document.getElementById('ui-pop-size-val');
    sliderPopSize.addEventListener('input', (e) => {
        const newSize = parseInt(e.target.value, 10);
        labelPopSize.innerText = newSize;
    });
    sliderPopSize.addEventListener('change', (e) => {
        const newSize = parseInt(e.target.value, 10);
        this.popSize = newSize;
        if (this.ga) this.ga.resizePopulation(newSize);
    });

    document.getElementById('btn-save-drone').addEventListener('click', () => {
      if (this.trainer === 'ppo') {
        if (!this.ppo) return;
        localStorage.setItem('ppo_policy', JSON.stringify(this.ppo.exportPolicy()));
        this.showToast('PPO policy saved.');
        return;
      }
      const best = this.ga.getBestGenome();
      const meta = this.ga.getBestMetadata();
      localStorage.setItem('drone_best', JSON.stringify({
          genome: Array.from(best),
          meta: meta
      }));
      this.showToast('Best drone genome saved!');
    });

    document.getElementById('btn-load-drone').addEventListener('click', () => {
      if (this.trainer === 'ppo') {
        const raw = localStorage.getItem('ppo_policy');
        if (!raw) { this.showToast('No saved PPO policy.'); return; }
        const okLoad = this.ppo && this.ppo.importPolicy(JSON.parse(raw));
        this.showToast(okLoad ? 'PPO policy loaded.' : 'Saved policy does not match this network shape.');
        return;
      }
      const data = localStorage.getItem('drone_best');
      if(data) {
        const parsed = JSON.parse(data);
        if (parsed.genome) {
            this.loadedGenome = new Float32Array(parsed.genome);
            this.loadedMeta = parsed.meta;
        } else {
            // Backwards compatibility
            this.loadedGenome = new Float32Array(parsed);
        }

        if (this.loadedGenome.length !== this.genomeLength) {
            this.showToast(`Saved drone has ${this.loadedGenome.length} weights, this network needs ${this.genomeLength}. Skipped.`);
            this.loadedGenome = null;
            return;
        }
        
        for(let i=0; i<this.popSize; i++) {
          this.ga.population[i] = new Float32Array(this.loadedGenome);
          if (this.loadedMeta) {
              this.ga.metadata[i] = {
                  name: `Clone of ${this.loadedMeta.name}`,
                  lineage: [this.loadedMeta.name]
              };
          }
        }
        this.ga.generation = 0;
        this.generation = 0;
        this.showToast('Drone loaded! Seed injected.');
      }
    });
    document.getElementById('btn-open-replay').addEventListener('click', () => {
      // Separate window so the training run keeps going undisturbed behind it.
      window.open('replay.html', 'droneReplay', 'width=1280,height=800');
    });
    this.refreshBestRunUI();

    const btnComm = document.getElementById('btn-commentary');
    btnComm.addEventListener('click', () => {
      this.commentaryOn = !this.commentaryOn;
      btnComm.innerText = this.commentaryOn ? 'Commentary: ON' : 'Commentary: OFF';
      btnComm.classList.toggle('active', this.commentaryOn);
      if (!this.commentaryOn) {
        this.hud.clearComments();
        if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
      }
    });

    const btnSpeak = document.getElementById('btn-commentary-speak');
    btnSpeak.addEventListener('click', () => {
      this.commentator.speak = !this.commentator.speak;
      btnSpeak.innerText = this.commentator.speak ? 'Speech: ON' : 'Speech: OFF';
      btnSpeak.classList.toggle('active', this.commentator.speak);
      if (!this.commentator.speak && typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
    });

    const btnSolid = document.getElementById('btn-solid-world');
    btnSolid.addEventListener('click', () => this.setSolidWorld(!this.solidWorld));
    document.getElementById('btn-voxel-overlay').addEventListener('click', () => this.toggleVoxelOverlay());
    this.refreshSolidWorldUI();

    document.getElementById('btn-trainer-ga').addEventListener('click', () => this.setTrainer('ga'));
    document.getElementById('btn-trainer-ppo').addEventListener('click', () => this.setTrainer('ppo'));

    const btnEnvOutdoor = document.getElementById('btn-env-outdoor');
    if (btnEnvOutdoor) btnEnvOutdoor.addEventListener('click', () => this.setEnvironment('outdoor'));
    const btnEnvLidar = document.getElementById('btn-env-lidar');
    if (btnEnvLidar) btnEnvLidar.addEventListener('click', () => this.setEnvironment('lidar'));

    const btnEdl = document.getElementById('btn-edl');
    btnEdl.addEventListener('click', () => {
      this.edl.enabled = !this.edl.enabled;
      btnEdl.innerText = this.edl.enabled ? 'Depth Shading: ON' : 'Depth Shading: OFF';
      btnEdl.classList.toggle('active', this.edl.enabled);
    });

    const btnFpv = document.getElementById('btn-fpv-stab');
    btnFpv.addEventListener('click', () => {
      const cs = this.cameraSystem;
      cs.fpvStabilized = !cs.fpvStabilized;
      cs._fpvInit = false; // Re-seed so the toggle does not swing the camera
      btnFpv.innerText = cs.fpvStabilized ? 'FPV: Stabilized' : 'FPV: Raw (acro)';
      btnFpv.classList.toggle('active', cs.fpvStabilized);
    });

    const btnCam = document.getElementById('btn-cam');
    btnCam.addEventListener('click', () => {
      this.cameraSystem.toggle();
      document.getElementById('btn-cam').innerText = `Cam: ${this.cameraSystem.mode.toUpperCase()}`;
    });

    const btnToggleBrain = document.getElementById('btn-toggle-brain');
    const brainStatsPanel = document.getElementById('brain-stats-panel');
    btnToggleBrain.addEventListener('click', () => {
        const isHidden = brainStatsPanel.style.display === 'none';
        brainStatsPanel.style.display = isHidden ? 'block' : 'none';
        btnToggleBrain.querySelector('span').innerText = isHidden ? '▲' : '▼';
    });

    // Right Panel - Editor
    const btnMode = document.getElementById('btn-mode-editor');
    const editorTools = document.getElementById('editor-tools');
    
    btnMode.addEventListener('click', () => {
      if (this.mode === 'sim') {
        this.mode = 'edit';
        this.isPaused = true;
        btnSim.innerText = "Sim Paused (Editor Mode)";
        btnSim.disabled = true;
        
        document.getElementById('ui-status').innerText = 'Editor Mode';
        btnMode.innerText = "Exit Editor Mode";
        btnMode.classList.add('active');
        editorTools.classList.remove('disabled');

        if(this.cameraSystem.mode === 'fpv') this.cameraSystem.toggle();
        this.editor.enable();
        this.setEnvironmentFaded(true);
        this.gateMeshes.forEach(m => setGateLabelVisible(m, true));
        this.frameTrackInEditor();
        this.liveSpeed = 0;
        this.liveAltitude = 0;
        if (this.trailMesh) this.trailMesh.visible = false;
        if (this.ghostMesh) this.ghostMesh.visible = false;
        this.gateMeshes.forEach(m => setGateState(m, 'upcoming'));
        this.showToast("Editor Mode Active");
      } else {
        this.mode = 'sim';
        this.isPaused = false;
        btnSim.disabled = false;
        btnSim.innerText = "Pause Sim";
        
        btnMode.innerText = "Enter Editor Mode";
        btnMode.classList.remove('active');
        editorTools.classList.add('disabled');
        
        this.editor.disable();
        this.setEnvironmentFaded(false);
        this.gateMeshes.forEach(m => setGateLabelVisible(m, false));
        if (this.trailMesh) this.trailMesh.visible = true;
        if (this.ghostMesh) this.ghostMesh.visible = true;
        this.resetDronePosition();
        this.runGeneration();
        this.showToast("Simulation Resumed");
      }
    });

    const setGizmo = (mode, btnId) => {
        this.editor.setMode(mode);
        document.querySelectorAll('.toolbar .tool').forEach(b => b.classList.remove('active'));
        document.getElementById(btnId).classList.add('active');
    };

    document.getElementById('btn-gizmo-translate').addEventListener('click', () => setGizmo('translate', 'btn-gizmo-translate'));
    document.getElementById('btn-gizmo-rotate').addEventListener('click', () => setGizmo('rotate', 'btn-gizmo-rotate'));
    document.getElementById('btn-gizmo-scale').addEventListener('click', () => setGizmo('scale', 'btn-gizmo-scale'));

    window.addEventListener('keydown', (e) => {
        if (this.mode !== 'edit' || e.target.tagName === 'INPUT') return;
        if (e.key === 'w' || e.key === 'W') setGizmo('translate', 'btn-gizmo-translate');
        if (e.key === 'e' || e.key === 'E') setGizmo('rotate', 'btn-gizmo-rotate');
        if (e.key === 'r' || e.key === 'R') setGizmo('scale', 'btn-gizmo-scale');
    });

    document.getElementById('btn-add-gate').addEventListener('click', () => {
      const len = this.circuitData.gates.length;
      const last = this.circuitData.gates[len-1];
      const newGate = JSON.parse(JSON.stringify(last));
      newGate.index = len;
      
      const normal = newGate.normal || [0,0,1];
      newGate.position[0] += normal[0] * 10;
      newGate.position[2] += normal[2] * 10;
      
      this.circuitData.gates.push(newGate);
      this.spawnGateMeshes();
      this.editor.gateMeshes = this.gateMeshes;
      this.showToast("Gate added.");
    });

    document.getElementById('btn-remove-gate').addEventListener('click', () => {
      if(this.editor.selectedGateMesh && this.circuitData.gates.length > 2) {
        const idx = this.editor.selectedGateMesh.userData.index;
        this.circuitData.gates.splice(idx, 1);
        this.circuitData.gates.forEach((g, i) => g.index = i); // reindex
        this.editor.deselectAll();
        this.spawnGateMeshes();
        this.editor.gateMeshes = this.gateMeshes;
        this.showToast("Gate removed.");
      }
    });

    document.getElementById('btn-set-cam').addEventListener('click', () => {
      // While in Editor mode, the user is physically looking through and controlling the chaseCamera
      const cam = this.cameraSystem.chaseCamera;
      
      this.circuitData.tracksideCamera = {
          position: cam.position.toArray(),
          quaternion: cam.quaternion.toArray(),
          fov: cam.fov
      };
      
      // Update the actual trackside camera immediately
      this.cameraSystem.tracksideCamera.position.fromArray(this.circuitData.tracksideCamera.position);
      this.cameraSystem.tracksideCamera.quaternion.fromArray(this.circuitData.tracksideCamera.quaternion);
      this.cameraSystem.tracksideCamera.fov = cam.fov;
      this.cameraSystem.tracksideCamera.updateProjectionMatrix();
      
      this.showToast("Trackside camera angle locked & saved!");
    });

    // Outdoor Track Buttons
    const btnSaveOutdoor = document.getElementById('btn-save-track-outdoor');
    if (btnSaveOutdoor) {
      btnSaveOutdoor.addEventListener('click', () => {
        localStorage.setItem('track_custom_outdoor', JSON.stringify(this.circuitData));
        this.showToast('Outdoor track saved successfully!');
      });
    }

    const btnLoadOutdoor = document.getElementById('btn-load-track-outdoor');
    if (btnLoadOutdoor) {
      btnLoadOutdoor.addEventListener('click', () => {
        const loaded = this.loadTrack('outdoor');
        if (loaded) {
          this.circuitData = loaded;
          if (this.environment !== 'outdoor') {
            this.setEnvironment('outdoor', false);
          }
          this.spawnGateMeshes();
          if (this.editor) this.editor.gateMeshes = this.gateMeshes;
          this.resetDronePosition();
          this.showToast('Outdoor track loaded successfully!');
        } else {
          this.showToast('No saved Outdoor track found in local storage.');
        }
      });
    }

    // LiDAR Track Buttons
    const btnSaveLidar = document.getElementById('btn-save-track-lidar');
    if (btnSaveLidar) {
      btnSaveLidar.addEventListener('click', () => {
        localStorage.setItem('track_custom_lidar', JSON.stringify(this.circuitData));
        localStorage.setItem('track_custom', JSON.stringify(this.circuitData));
        this.showToast('LiDAR track saved successfully!');
      });
    }

    const btnLoadLidar = document.getElementById('btn-load-track-lidar');
    if (btnLoadLidar) {
      btnLoadLidar.addEventListener('click', () => {
        const loaded = this.loadTrack('lidar');
        if (loaded) {
          this.circuitData = loaded;
          if (this.environment !== 'lidar') {
            this.setEnvironment('lidar', false);
          }
          this.spawnGateMeshes();
          if (this.editor) this.editor.gateMeshes = this.gateMeshes;
          this.resetDronePosition();
          this.showToast('LiDAR track loaded successfully!');
        } else {
          this.showToast('No saved LiDAR track found in local storage.');
        }
      });
    }

    // Reset Track Button
    const btnReset = document.getElementById('btn-reset-track');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this.circuitData = JSON.parse(JSON.stringify(defaultTrackData));
        this.spawnGateMeshes();
        if (this.editor) this.editor.gateMeshes = this.gateMeshes;
        this.resetDronePosition();
        this.showToast('Track reset to default layout.');
      });
    }
  }

  loadTrack(env = this.environment) {
    const key = env === 'outdoor' ? 'track_custom_outdoor' : 'track_custom_lidar';
    const data = localStorage.getItem(key);
    if (data) {
      try {
        return JSON.parse(data);
      } catch (e) {
        console.error('Failed to parse saved track:', e);
      }
    }
    // Fallback for lidar if only legacy track_custom exists
    if (env === 'lidar') {
      const legacy = localStorage.getItem('track_custom');
      if (legacy) {
        try {
          return JSON.parse(legacy);
        } catch (e) {}
      }
    }
    return null;
  }

  // ── Run One Generation ──
  async runGeneration() {
    if (this.isSimulating || this.isPaused || this.mode !== 'sim') return;
    this.isSimulating = true;

    if (this.trainer === 'ppo') {
      try {
        await this.runPPOIteration();
      } finally {
        this.isSimulating = false;
      }
      if (!this.isPaused && this.mode === 'sim') {
        const delay = this.turboMode ? 0 : 250;
        setTimeout(() => this.runGeneration(), delay);
      }
      return;
    }

    this.generation++;

    const population = this.ga.getPopulation();
    const batchSize = Math.ceil(this.popSize / this.numWorkers);
    const gates = this.circuitData.gates;

    // Every genome in a generation shares a start gate, so fitness stays comparable
    // within the generation.
    const startGateIdx = this.pickStartGate();
    this.curriculum.startGateIdx = startGateIdx;
    this.resetDronePosition(startGateIdx);

    const promises = this.workers.map((worker, workerIdx) => {
      const startIdx = workerIdx * batchSize;
      const endIdx = Math.min(startIdx + batchSize, this.popSize);
      if (startIdx >= this.popSize) return Promise.resolve({ results: [], bestTrajectory: null });

      const batchGenomes = [];
      for (let i = startIdx; i < endIdx; i++) {
        batchGenomes.push(population[i].buffer.slice(0)); 
      }

      return new Promise(resolve => {
        worker.onmessage = (e) => resolve(e.data);
        worker.postMessage({
          genomes: batchGenomes,
          gates,
          startPos: this.startPos,
          startYaw: this.startYaw,
          startGateIdx,
          maxTicks: 3600,
          dt: 1 / 60,
          environment: this.environment,
          // Sent only when it changes: a couple of MB per worker is not worth resending
          // every generation.
          ...(this.gridDirty ? { grid: this.solidWorld && this.worldGrid ? this.worldGrid.serialize() : null } : {})
        }, batchGenomes);
      });
    });

    const workerResults = await Promise.all(promises);
    this.gridDirty = false;

    const allFitnessScores = [];
    const ghostCandidates = [];
    let bestFitness = -Infinity;
    let bestTrajectoryBuffer = null;
    let bestGates = 0;
    let bestSpeed = 0;
    let bestHitWall = false;

    const clearedCounts = [];
    let wallHits = 0;
    workerResults.forEach((wr) => {
      if (!wr.results) return;
      wr.results.forEach((r) => {
        allFitnessScores.push(r.fitness);
        clearedCounts.push(r.gatesPassed);
        if (r.hitWall) wallHits++;
      });

      if (wr.ghosts) {
        for (const g of wr.ghosts) ghostCandidates.push(g);
      }

      const localBestFitness = wr.results[wr.bestLocalIdx]?.fitness ?? -Infinity;
      if (localBestFitness > bestFitness) {
        bestFitness = localBestFitness;
        bestTrajectoryBuffer = wr.bestTrajectory;
        bestGates = wr.results[wr.bestLocalIdx].gatesPassed;
        bestSpeed = wr.results[wr.bestLocalIdx].topSpeed;
        bestHitWall = !!wr.results[wr.bestLocalIdx].hitWall;
      }
    });

    if (bestTrajectoryBuffer) {
      this.pendingTrajectory = {
        data: new Float32Array(bestTrajectoryBuffer),
        gates: bestGates,
        speed: bestSpeed,
        generation: this.generation,
        startGateIdx,
        fitness: bestFitness,
        hitWall: bestHitWall,
        crashed: bestHitWall,
        isRecord: false
      };

      if (bestFitness > this.allTimeBestFitness) this.allTimeBestFitness = bestFitness;

      // Only full-lap attempts are eligible for the record.
      if (startGateIdx === 0 && bestFitness > this.bestFullLapFitness) {
        this.bestFullLapFitness = bestFitness;
        this.pendingTrajectory.isRecord = true;
        this.recordBestRun(this.pendingTrajectory);
      }
      // Turbo is for watching training move, not for watching flight -- take it now.
      // Otherwise let the current run finish so you see a whole line, not a jump cut.
      if (this.turboMode || !this.bestTrajectory) this.adoptPendingTrajectory();
    }
    this.buildGhosts(ghostCandidates);

    this.updateCurriculum(startGateIdx, clearedCounts);

    const wallEl = document.getElementById('ui-wall-hits');
    if (wallEl) {
      const pct = clearedCounts.length ? (100 * wallHits / clearedCounts.length) : 0;
      wallEl.innerText = this.solidWorld && this.environment === 'lidar'
        ? `${pct.toFixed(0)}%` : 'n/a';
    }

    const gaStats = this.ga.nextGeneration(allFitnessScores);

    // Update Graph
    this.graph.addDataPoint(this.generation, bestFitness, gaStats.avgFitness, bestGates);

    // Update HUD Stats
    document.getElementById('ui-gen').innerText = this.generation;
    const stageEl = document.getElementById('ui-stage');
    if (stageEl) {
      const c = this.curriculum;
      stageEl.innerText = c.graduated
        ? 'Full lap'
        : `${c.stage} / ${this.circuitData.gates.length - 1}` +
          (startGateIdx !== c.stage ? ` (rehearsing ${startGateIdx})` : '');
    }
    document.getElementById('ui-gates').innerText = `${bestGates} / ${this.circuitData.gates.length}`;
    document.getElementById('ui-speed').innerText = `${bestSpeed.toFixed(1)} km/h`;
    document.getElementById('ui-status').innerText = this.turboMode ? '⚡ TURBO' : 'Evolving...';
    
    // Update Brain Stats Panel
    const bestMeta = this.ga.getBestMetadata();
    if (bestMeta) {
        document.getElementById('ui-brain-name').innerText = bestMeta.name;
        document.getElementById('ui-brain-gen').innerText = this.generation;
        let linStr = bestMeta.lineage.join(" → ");
        document.getElementById('ui-brain-lineage').innerText = linStr || "Adam";
    }

    this.isSimulating = false;

    if(!this.isPaused && this.mode === 'sim') {
      if (this.turboMode) {
        setTimeout(() => this.runGeneration(), 0);
      } else {
        const replayFrames = this.bestTrajectory ? this.bestTrajectory.length / 7 : 0;
        const replayDurationMs = Math.min(2000, Math.max(300, (replayFrames / 20) * 500));
        setTimeout(() => this.runGeneration(), replayDurationMs);
      }
    }
  }

  // ── Render Loop ──
  render() {
    const dt = Math.min(0.1, this.clock.getDelta()); // Clamp so a tab-switch cannot teleport the replay
    this.elapsed += dt;

    if (this.outdoor) {
      this.outdoor.update(dt);
    }

    if (this.mode === 'sim') {
      // Replay best drone trajectory at real time, interpolating between keyframes.
      // Stepping keyframe-to-keyframe played it at 6x with visible judder; a quad moving
      // 25 m/s covers 1.25 m between 20Hz samples, which is a very obvious jump.
      if (this.bestTrajectory && this.bestTrajectory.length >= 14) {
        const numFrames = this.bestTrajectory.length / 7;
        const rate = this.KEYFRAME_HZ * (this.turboMode ? 6 : 1);
        this.replayTime += dt * rate;

        if (this.replayTime >= numFrames - 1) {
          if (this.pendingTrajectory) {
            this.adoptPendingTrajectory();
          } else {
            this.replayTime = 0;
            this.currentGateIdx = this.replayStartGate || 0;
          }
        }

        const f0 = Math.min(Math.floor(this.replayTime), numFrames - 1);
        const f1 = Math.min(f0 + 1, numFrames - 1);
        const a = this.replayTime - f0;
        const o0 = f0 * 7;
        const o1 = f1 * 7;

        this._pA.set(this.bestTrajectory[o0], this.bestTrajectory[o0 + 1], this.bestTrajectory[o0 + 2]);
        this._pB.set(this.bestTrajectory[o1], this.bestTrajectory[o1 + 1], this.bestTrajectory[o1 + 2]);
        this._pA.lerp(this._pB, a);

        // Quaternions are stored [w,x,y,z]; THREE.Quaternion is (x,y,z,w).
        this._qA.set(this.bestTrajectory[o0 + 4], this.bestTrajectory[o0 + 5], this.bestTrajectory[o0 + 6], this.bestTrajectory[o0 + 3]);
        this._qB.set(this.bestTrajectory[o1 + 4], this.bestTrajectory[o1 + 5], this.bestTrajectory[o1 + 6], this.bestTrajectory[o1 + 3]);
        this._qA.slerp(this._qB, a);

        this.droneMesh.position.copy(this._pA);
        this.droneMesh.quaternion.copy(this._qA);

        this.cameraSystem.update(this._pA, this._qA, dt);

        // Speed straight from the keyframe pair, in m/s -> km/h.
        this._pB.set(this.bestTrajectory[o1], this.bestTrajectory[o1 + 1], this.bestTrajectory[o1 + 2]);
        this._pA.set(this.bestTrajectory[o0], this.bestTrajectory[o0 + 1], this.bestTrajectory[o0 + 2]);
        this.liveSpeed = f1 > f0 ? this._pA.distanceTo(this._pB) * this.KEYFRAME_HZ * 3.6 : 0;
        this.liveAltitude = this.droneMesh.position.y;
        this.replayProgress = this.replayTime / Math.max(1, numFrames - 1);
        this.replayFrame = f0;
        if (this.commentaryOn && this.commentator) this.commentator.update(this.replayTime);
      }

      // Gate states track the replay: everything before currentGateIdx is cleared,
      // currentGateIdx is the one being flown at, the rest are still ahead.
      if (!this.editor.enabled) {
        const base = this.replayStartGate || 0;
        while (
          this.currentGateIdx - base < this.gatePassFrames.length &&
          this.replayFrame >= this.gatePassFrames[this.currentGateIdx - base]
        ) {
          this.currentGateIdx++;
        }

        this.gateMeshes.forEach((mesh, i) => {
          if (i < this.currentGateIdx) setGateState(mesh, 'passed');
          else if (i === this.currentGateIdx) setGateState(mesh, 'next');
          else setGateState(mesh, 'upcoming');
        });

        const nextMesh = this.gateMeshes[this.currentGateIdx];
        if (nextMesh) pulseGate(nextMesh, this.elapsed);
      }
    }

    if (this.hud) {
      this.hud.update({
        generation: this.displayGeneration,
        status: this.mode === 'edit' ? 'Editing' : this.turboMode ? 'Turbo' : this.isPaused ? 'Paused' : `Training gen ${this.generation}`,
        speed: this.liveSpeed || 0,
        altitude: this.liveAltitude || 0,
        cam: this.mode === 'edit' ? 'EDITOR' : this.cameraSystem.mode.toUpperCase(),
        currentGate: this.mode === 'edit' ? -1 : this.currentGateIdx,
        progress: this.replayProgress || 0
      });
    }

    // In FPV you are the drone: its own trail and the population ghosts sit right on
    // the lens and block the view.
    if (this.mode === 'sim') {
      const fpv = this.cameraSystem.mode === 'fpv';
      if (this.trailMesh) this.trailMesh.visible = !fpv;
      if (this.ghostMesh) this.ghostMesh.visible = !fpv;
    }

    // Keep tracker dot visible in trackside mode or edit mode, and sync its position
    this.trackerDot.visible = (this.cameraSystem.mode === 'trackside' || this.mode === 'edit');
    this.trackerDot.position.set(this.droneMesh.position.x, this.droneMesh.position.y + 8.0, this.droneMesh.position.z);

    if (!this._splatSynced) {
      const lidar = this.scene.getObjectByName('lidar');
      if (lidar) {
        this.syncSplatViewport(this.renderer.domElement.clientHeight);
        this._splatSynced = true;
      }
    }

    // Lens distortion and vignette belong to the goggle feed only.
    this.edl.setLens(this.mode === 'sim' && this.cameraSystem.mode === 'fpv' ? 1.0 : 0.0, dt);

    this.edl.render(this.scene, this.mode === 'edit' ? this.cameraSystem.chaseCamera : this.cameraSystem.getCamera());
    requestAnimationFrame(this.render);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new DroneRacingApp();
});

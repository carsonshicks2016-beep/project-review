import * as THREE from 'three';
import {
  createRenderer, createScene, createGround, createGateMesh, createDroneMesh,
  createTrackLine, loadHousePointCloud, setGateState, pulseGate, FOG_DENSITY
} from './scene.js';
import { rebuildTrackSpline } from './circuitGenerator.js';
import { CameraSystem } from './fpvCamera.js';
import { EyeDomeLighting } from './eyeDome.js';
import { checkGatePassing } from './gateCollision.js';
import { loadRuns, getRun } from './runRecorder.js';
import { Commentator } from './commentary/commentator.js';
import { gateFramesOf } from './commentary/raceEvents.js';

const KEYFRAME_HZ = 20; // Worker samples every 3rd tick at dt=1/60
const SPEEDS = [0.25, 0.5, 1, 2];

class ReplayPlayer {
  constructor() {
    this.runs = loadRuns();
    this.runIndex = 0;
    this.playing = true;
    this.speedIndex = 2; // 1x
    this.time = 0;       // Fractional keyframe index
    this.elapsed = 0;

    this.viewport = document.getElementById('replay-viewport');
    this.feed = document.getElementById('replay-commentary');
    this.commentaryOn = true;
    this.commentator = new Commentator({ voice: 'template', speak: false });
    this.commentator.onLine(l => this.pushComment(l));

    if (this.runs.length === 0) {
      document.getElementById('replay-loading').hidden = true;
      document.getElementById('replay-empty').hidden = false;
      return;
    }

    this.renderer = createRenderer();
    document.body.removeChild(this.renderer.domElement);
    this.viewport.appendChild(this.renderer.domElement);
    this.renderer.setSize(this.viewport.clientWidth, this.viewport.clientHeight);

    this.scene = createScene();
    this.scene.add(createGround());

    this.droneMesh = createDroneMesh();
    this.scene.add(this.droneMesh);

    this.cameraSystem = new CameraSystem(this.droneMesh);
    this.cameraSystem.resize(this.viewport.clientWidth / this.viewport.clientHeight);
    this.cameraSystem.mode = 'fpv';
    this.cameraSystem.activeCamera = this.cameraSystem.fpvCamera;
    this.cameraSystem.modeIndex = 1;

    this.edl = new EyeDomeLighting(this.renderer, { strength: 5.0, radius: 4.0 });
    this.edl.setSize(this.viewport.clientWidth, this.viewport.clientHeight);

    this.gateMeshes = [];
    this.trailMesh = null;
    this.clock = new THREE.Clock();
    this._qA = new THREE.Quaternion();
    this._qB = new THREE.Quaternion();
    this._pA = new THREE.Vector3();
    this._pB = new THREE.Vector3();

    this.loadEnvironment();
    this.setupUI();
    this.selectRun(0);

    window.addEventListener('resize', () => {
      const w = this.viewport.clientWidth;
      const h = this.viewport.clientHeight;
      this.renderer.setSize(w, h);
      this.cameraSystem.resize(w / h);
      this.edl.setSize(w, h);
      this.syncSplat(h);
    });

    this.render = this.render.bind(this);
    requestAnimationFrame(this.render);
  }

  loadEnvironment() {
    const overlay = document.getElementById('replay-loading');
    const fill = document.getElementById('replay-loading-fill');
    const pct = document.getElementById('replay-loading-pct');

    document.getElementById('btn-skip-env').addEventListener('click', () => {
      overlay.hidden = true;
      this.skipEnv = true;
    });

    // The scan is a large download; report progress rather than showing a frozen window.
    loadHousePointCloud(this.scene, {
      onProgress: (p) => {
        const v = Math.round(p * 100);
        fill.style.width = `${v}%`;
        pct.innerText = `${v}%`;
      },
      onDone: () => {
        overlay.hidden = true;
        this.syncSplat(this.viewport.clientHeight);
      },
      onError: () => { overlay.hidden = true; }
    });
  }

  syncSplat(height) {
    const lidar = this.scene.getObjectByName('lidar');
    if (lidar && lidar.material.uniforms && lidar.material.uniforms.uViewportH) {
      lidar.material.uniforms.uViewportH.value = height;
    }
  }

  setupUI() {
    const picker = document.getElementById('run-picker');
    this.runs.forEach((r, i) => {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.text = `#${i + 1} — ${r.gatesPassed}/${r.totalGates} gates · gen ${r.generation}`;
      picker.appendChild(opt);
    });
    picker.addEventListener('change', () => this.selectRun(parseInt(picker.value, 10)));

    const btnPlay = document.getElementById('btn-play');
    btnPlay.addEventListener('click', () => {
      this.playing = !this.playing;
      btnPlay.innerText = this.playing ? '❚❚' : '▶';
    });

    this.scrub = document.getElementById('scrub');
    this.scrub.addEventListener('input', () => {
      const frac = parseInt(this.scrub.value, 10) / 1000;
      this.time = frac * (this.numFrames - 1);
      this.rebuildGateStatesFromTime();
    });

    const btnSpeed = document.getElementById('btn-speed');
    btnSpeed.addEventListener('click', () => {
      this.speedIndex = (this.speedIndex + 1) % SPEEDS.length;
      btnSpeed.innerText = `${SPEEDS[this.speedIndex]}×`;
    });

    const btnCam = document.getElementById('btn-rcam');
    btnCam.addEventListener('click', () => {
      this.cameraSystem.toggle();
      btnCam.innerText = this.cameraSystem.mode.toUpperCase();
    });

    const btnComm = document.getElementById('btn-rcomment');
    btnComm.addEventListener('click', () => {
      this.commentaryOn = !this.commentaryOn;
      btnComm.innerText = this.commentaryOn ? 'Talk' : 'Muted';
      btnComm.classList.toggle('active', this.commentaryOn);
      if (!this.commentaryOn) {
        if (this.feed) this.feed.innerHTML = '';
        if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
      }
    });
    const btnSpeak = document.getElementById('btn-rspeak');
    btnSpeak.addEventListener('click', () => {
      this.commentator.speak = !this.commentator.speak;
      btnSpeak.innerText = this.commentator.speak ? 'Voice: ON' : 'Voice: OFF';
      btnSpeak.classList.toggle('active', this.commentator.speak);
      if (!this.commentator.speak && typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
    });

    const btnStab = document.getElementById('btn-rstab');
    btnStab.addEventListener('click', () => {
      const cs = this.cameraSystem;
      cs.fpvStabilized = !cs.fpvStabilized;
      cs._fpvInit = false;
      btnStab.innerText = cs.fpvStabilized ? 'Stabilized' : 'Raw';
      btnStab.classList.toggle('active', cs.fpvStabilized);
    });

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); btnPlay.click(); }
      if (e.key === 'c' || e.key === 'C') btnCam.click();
    });
  }

  pushComment(line) {
    if (!this.feed) return;
    const el = document.createElement('div');
    el.className = `hud-comment hud-comment-${line.type}`;
    el.textContent = line.text;
    this.feed.appendChild(el);
    while (this.feed.children.length > 6) this.feed.removeChild(this.feed.firstChild);
    const n = this.feed.children.length;
    for (let i = 0; i < n; i++) this.feed.children[i].style.opacity = String(0.25 + 0.75 * ((i + 1) / n));
  }

  selectRun(index) {
    const run = getRun(index);
    if (!run) return;
    this.run = run;
    this.runIndex = index;
    this.trajectory = run.trajectory;
    this.numFrames = this.trajectory.length / 7;
    this.time = 0;

    this.buildGates(run.gates);
    this.buildTrail();
    this.computeGateTimeline();
    this.rebuildGateStatesFromTime();

    document.getElementById('meta-gates').innerText = `${run.gatesPassed}/${run.totalGates}`;
    document.getElementById('meta-speed').innerText = run.topSpeed.toFixed(1);
    document.getElementById('meta-fitness').innerText = Math.round(run.fitness).toLocaleString();
    document.getElementById('meta-gen').innerText = run.generation;
    document.getElementById('meta-date').innerText =
      new Date(run.recordedAt).toLocaleString();

    // Compare against the best recorded run, unless this IS it -- then compare with the
    // runner-up so the deltas still mean something.
    const all = loadRuns();
    const refIdx = index === 0 ? 1 : 0;
    let reference = null;
    if (all[refIdx]) {
      const refRun = getRun(refIdx);
      if (refRun) reference = { gateFrames: gateFramesOf(refRun, refRun.gates) };
    }

    this.reference = reference;
    if (this.feed) this.feed.innerHTML = '';
    this.commentator.loadRun(
      { ...run, isRecord: index === 0 },
      { gates: run.gates, reference }
    );
  }

  buildGates(gates) {
    this.gateMeshes.forEach(m => this.scene.remove(m));
    this.gateMeshes = [];
    if (this.trackLine) this.scene.remove(this.trackLine);

    gates.forEach((gate, i) => {
      const mesh = createGateMesh(3.0);
      mesh.position.fromArray(gate.position);
      if (gate.quaternion) mesh.quaternion.fromArray(gate.quaternion);
      if (gate.scale) mesh.scale.fromArray(gate.scale);
      mesh.userData.index = i;
      this.scene.add(mesh);
      this.gateMeshes.push(mesh);
    });

    const curve = rebuildTrackSpline(gates);
    if (curve) {
      this.trackLine = createTrackLine(curve);
      this.scene.add(this.trackLine);
    }
  }

  buildTrail() {
    if (this.trailMesh) {
      this.scene.remove(this.trailMesh);
      this.trailMesh.geometry.dispose();
      this.trailMesh.material.dispose();
      this.trailMesh = null;
    }
    const traj = this.trajectory;
    const step = Math.max(1, Math.floor(this.numFrames / 240));
    const pts = [];
    for (let f = 0; f < this.numFrames; f += step) {
      const o = f * 7;
      const p = new THREE.Vector3(traj[o], traj[o + 1], traj[o + 2]);
      if (pts.length === 0 || p.distanceToSquared(pts[pts.length - 1]) > 1e-3) pts.push(p);
    }
    if (pts.length < 2) return;

    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const geo = new THREE.TubeGeometry(curve, Math.min(500, pts.length * 2), 0.3, 6, false);
    const mat = new THREE.MeshBasicMaterial({
      color: '#a855f7', transparent: true, opacity: 0.5, depthWrite: false
    });
    this.trailMesh = new THREE.Mesh(geo, mat);
    this.trailMesh.renderOrder = 4;
    this.scene.add(this.trailMesh);
  }

  computeGateTimeline() {
    this.gatePassFrames = [];
    this.startGate = this.run.startGateIdx || 0;
    const traj = this.trajectory;
    const gates = this.run.gates;
    const prev = [0, 0, 0];
    const cur = [0, 0, 0];
    let gi = this.startGate;

    for (let f = 1; f < this.numFrames && gi < gates.length; f++) {
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

  // Scrubbing can jump anywhere, so derive gate progress from the timeline rather than
  // incrementing as frames pass.
  rebuildGateStatesFromTime() {
    const base = this.startGate || 0;
    let idx = base;
    for (const frame of this.gatePassFrames) {
      if (this.time >= frame) idx++;
      else break;
    }
    this.currentGate = idx;
  }

  render() {
    const dt = Math.min(0.1, this.clock.getDelta());
    this.elapsed += dt;

    if (this.playing && this.numFrames > 1) {
      this.time += dt * KEYFRAME_HZ * SPEEDS[this.speedIndex];
      if (this.time >= this.numFrames - 1) {
        this.time = 0;
        // Looping replays the same run, so clear the feed rather than stacking a second
        // copy of the same commentary underneath the first.
        if (this.feed) this.feed.innerHTML = '';
        this.commentator.reset();
        this.commentator.loadRun({ ...this.run, isRecord: this.runIndex === 0 },
          { gates: this.run.gates, reference: this.reference });
      }
      this.rebuildGateStatesFromTime();
      this.scrub.value = String(Math.round((this.time / (this.numFrames - 1)) * 1000));
    }

    if (this.commentaryOn) this.commentator.update(this.time);

    const f0 = Math.min(Math.floor(this.time), this.numFrames - 1);
    const f1 = Math.min(f0 + 1, this.numFrames - 1);
    const a = this.time - f0;
    const o0 = f0 * 7;
    const o1 = f1 * 7;
    const t = this.trajectory;

    this._pA.set(t[o0], t[o0 + 1], t[o0 + 2]);
    this._pB.set(t[o1], t[o1 + 1], t[o1 + 2]);
    this._pA.lerp(this._pB, a);

    this._qA.set(t[o0 + 4], t[o0 + 5], t[o0 + 6], t[o0 + 3]);
    this._qB.set(t[o1 + 4], t[o1 + 5], t[o1 + 6], t[o1 + 3]);
    this._qA.slerp(this._qB, a);

    this.droneMesh.position.copy(this._pA);
    this.droneMesh.quaternion.copy(this._qA);
    this.cameraSystem.update(this._pA, this._qA, dt);

    this.gateMeshes.forEach((mesh, i) => {
      if (i < this.currentGate) setGateState(mesh, 'passed');
      else if (i === this.currentGate) setGateState(mesh, 'next');
      else setGateState(mesh, 'upcoming');
    });
    const nextMesh = this.gateMeshes[this.currentGate];
    if (nextMesh) pulseGate(nextMesh, this.elapsed);

    // The drone's own trail sits on the lens in FPV.
    if (this.trailMesh) this.trailMesh.visible = this.cameraSystem.mode !== 'fpv';

    const total = (this.numFrames - 1) / KEYFRAME_HZ;
    document.getElementById('time-label').innerText =
      `${(this.time / KEYFRAME_HZ).toFixed(1)}s / ${total.toFixed(1)}s`;

    this.edl.setLens(this.cameraSystem.mode === 'fpv' ? 1.0 : 0.0, dt);
    this.edl.render(this.scene, this.cameraSystem.getCamera());
    requestAnimationFrame(this.render);
  }
}

window.addEventListener('DOMContentLoaded', () => new ReplayPlayer());

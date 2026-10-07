/**
 * ApexFlock - Main Application Controller
 * Coordinates Ecosystem simulation, 3D WebGL rendering, Brain HUD visualizer,
 * ecological analytics, procedural audio, and God-Mode user controls.
 */

import { Ecosystem } from './ecosystem.js';
import { RenderEngine } from './renderEngine.js';
import { BrainVisualizer } from './brainVisualizer.js';
import { AnalyticsDashboard } from './analytics.js';
import { AudioSynth } from './audioSynth.js';

export class SimulationApp {
  constructor() {
    this.isPaused = false;
    this.timeScale = 1.0;
    this.isTurbo = false;

    // Subsystems
    this.ecosystem = new Ecosystem({
      targetBoids: 110,
      targetPredators: 6,
      maxFoods: 85
    });

    this.audio = new AudioSynth();

    // DOM Elements
    this.container = document.getElementById('canvas-container');
    this.brainCanvas = document.getElementById('brain-canvas');
    this.phaseCanvas = document.getElementById('phase-canvas');
    this.historyCanvas = document.getElementById('history-canvas');

    // Visualizers
    this.brainViz = new BrainVisualizer(this.brainCanvas);
    this.analytics = new AnalyticsDashboard(this.phaseCanvas, this.historyCanvas);

    // 3D Rendering Engine
    this.renderer = new RenderEngine(this.container, this.ecosystem, (agent) => {
      this.brainViz.setAgent(agent);
      this.audio.resume();
    });

    // Auto select first predator or boid for immediate HUD feedback
    if (this.ecosystem.predators.length > 0) {
      this.renderer.setSelectedAgent(this.ecosystem.predators[0]);
    } else if (this.ecosystem.boids.length > 0) {
      this.renderer.setSelectedAgent(this.ecosystem.boids[0]);
    }

    this._bindUI();
    this._bindHotkeys();

    // Animation Loop
    this.lastTime = performance.now();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  _bindUI() {
    // 1. EXHIBITION THEMES
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const theme = e.target.dataset.theme;
        this.renderer.setTheme(theme);
        this._flashBanner(`EXHIBITION PALETTE: ${theme.toUpperCase()}`);
      });
    });

    // 2. ECOLOGICAL SCENARIOS
    document.querySelectorAll('.scenario-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const scenario = e.target.dataset.scenario;
        this._loadScenario(scenario);
      });
    });

    // 3. PLAY / PAUSE
    const btnPause = document.getElementById('btn-pause');
    if (btnPause) {
      btnPause.addEventListener('click', () => {
        this.isPaused = !this.isPaused;
        btnPause.textContent = this.isPaused ? "Resume" : "Pause";
        btnPause.classList.toggle('active', !this.isPaused);
        this.audio.resume();
      });
    }

    // 4. TURBO SPEED
    const btnTurbo = document.getElementById('btn-turbo');
    if (btnTurbo) {
      btnTurbo.addEventListener('click', () => {
        this.isTurbo = !this.isTurbo;
        btnTurbo.textContent = `Turbo: ${this.isTurbo ? "ON" : "OFF"}`;
        btnTurbo.classList.toggle('active', this.isTurbo);
        this.audio.resume();
      });
    }

    // 5. TEMPO BUTTONS
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.timeScale = parseFloat(e.target.dataset.speed || 1.0);
        this.audio.resume();
      });
    });

    // 6. CAMERA DIRECTOR
    document.querySelectorAll('.cam-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const mode = e.target.dataset.cam;
        this.renderer.setCameraMode(mode);
        this.audio.resume();
      });
    });

    // 7. INSTINCT VS NEURAL BLEND
    const sliderInstinct = document.getElementById('slider-instinct');
    const labelInstinct = document.getElementById('val-instinct');
    if (sliderInstinct && labelInstinct) {
      sliderInstinct.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this.ecosystem.instinctBlend = val;
        labelInstinct.textContent = `${Math.round(val * 100)}%`;
      });
    }

    // 8. INTERVENTIONS
    const btnAddBoids = document.getElementById('btn-add-boids');
    if (btnAddBoids) {
      btnAddBoids.addEventListener('click', () => {
        for (let i = 0; i < 20; i++) {
          this.ecosystem.boids.push(new (this.ecosystem.boids[0]?.constructor || Object)(
            (Math.random() - 0.5) * 160,
            (Math.random() - 0.5) * 80 + 20,
            (Math.random() - 0.5) * 160
          ));
        }
      });
    }

    const btnAddPred = document.getElementById('btn-add-pred');
    if (btnAddPred) {
      btnAddPred.addEventListener('click', () => {
        for (let i = 0; i < 2; i++) {
          this.ecosystem.predators.push(new (this.ecosystem.predators[0]?.constructor || Object)(
            (Math.random() - 0.5) * 200,
            (Math.random() - 0.5) * 80 + 20,
            (Math.random() - 0.5) * 200
          ));
        }
      });
    }

    const btnSeedFood = document.getElementById('btn-seed-food');
    if (btnSeedFood) {
      btnSeedFood.addEventListener('click', () => {
        const cx = (Math.random() - 0.5) * 150;
        const cy = (Math.random() - 0.5) * 80 + 20;
        const cz = (Math.random() - 0.5) * 150;
        for (let i = 0; i < 25; i++) {
          this.ecosystem.spawnFood(cx, cy, cz);
        }
        this._flashBanner("✦ BIOLUMINESCENT SPORE BLOOM SEEDED");
      });
    }

    const btnMutate = document.getElementById('btn-mutate');
    if (btnMutate) {
      btnMutate.addEventListener('click', () => {
        this.ecosystem.triggerRadiationPulse(0.35, 0.7);
        this._flashBanner("⚡ GENETIC RADIATION PULSE INDUCED");
      });
    }

    const btnVortex = document.getElementById('btn-vortex');
    if (btnVortex) {
      btnVortex.addEventListener('click', () => {
        const x = (Math.random() - 0.5) * 120;
        const y = 20;
        const z = (Math.random() - 0.5) * 120;
        this.ecosystem.stirWindVortex(x, y, z, 2.2, 100, 7.0);
        this._flashBanner("🌀 ATMOSPHERIC VORTEX STIRRED");
      });
    }

    const btnReset = document.getElementById('btn-reset');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this.ecosystem.initPopulations();
        this._flashBanner("♻️ BIOSPHERE SEEDED ANEW");
      });
    }

    // 9. SOUND TOGGLE
    const btnSound = document.getElementById('btn-sound');
    if (btnSound) {
      btnSound.addEventListener('click', () => {
        this.audio.init();
        const muted = this.audio.toggleMute();
        btnSound.textContent = muted ? "🔔 Ambient Sound: OFF" : "🔔 Ambient Sound: ON";
        btnSound.classList.toggle('active', !muted);
      });
    }

    // 10. HIGH-RES FRAME SNAPSHOT
    const btnSnapshot = document.getElementById('btn-snapshot');
    if (btnSnapshot) {
      btnSnapshot.addEventListener('click', () => {
        const dataUrl = this.renderer.renderer.domElement.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `apexflock_exhibition_frame_${Date.now()}.png`;
        a.click();
        this._flashBanner("📸 HIGH-RESOLUTION FRAME EXPORTED");
      });
    }

    // 11. EXPORT GENOMES
    const btnExport = document.getElementById('btn-export');
    if (btnExport) {
      btnExport.addEventListener('click', () => {
        const data = this.ecosystem.exportChampions();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `apexflock_champions_${Math.round(this.ecosystem.time)}s.json`;
        a.click();
        URL.revokeObjectURL(url);
      });
    }
  }

  _loadScenario(name) {
    if (name === 'murmuration') {
      this.ecosystem.targetBoids = 175;
      this.ecosystem.targetPredators = 2;
      this.ecosystem.maxFoods = 110;
      this.ecosystem.instinctBlend = 0.35;
      this.ecosystem.initPopulations();
      this._flashBanner("SCENARIO: CELESTIAL STARLING MURMURATION");
    } else if (name === 'hunt') {
      this.ecosystem.targetBoids = 80;
      this.ecosystem.targetPredators = 8;
      this.ecosystem.maxFoods = 60;
      this.ecosystem.instinctBlend = 0.15;
      this.ecosystem.initPopulations();
      this._flashBanner("SCENARIO: PACK AMBUSH & PINCH HUNT");
    } else if (name === 'crucible') {
      this.ecosystem.targetBoids = 110;
      this.ecosystem.targetPredators = 6;
      this.ecosystem.maxFoods = 80;
      this.ecosystem.instinctBlend = 0.05; // High neural reliance
      this.ecosystem.initPopulations();
      this.ecosystem.triggerRadiationPulse(0.45, 0.9);
      this._flashBanner("SCENARIO: EVOLUTIONARY CRUCIBLE");
    } else if (name === 'equilibrium') {
      this.ecosystem.targetBoids = 120;
      this.ecosystem.targetPredators = 5;
      this.ecosystem.maxFoods = 90;
      this.ecosystem.instinctBlend = 0.25;
      this.ecosystem.initPopulations();
      this._flashBanner("SCENARIO: LOTKA-VOLTERRA EQUILIBRIUM");
    }

    const slider = document.getElementById('slider-instinct');
    const label = document.getElementById('val-instinct');
    if (slider && label) {
      slider.value = this.ecosystem.instinctBlend;
      label.textContent = `${Math.round(this.ecosystem.instinctBlend * 100)}%`;
    }
  }

  _bindHotkeys() {
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        const btn = document.getElementById('btn-pause');
        if (btn) btn.click();
      } else if (e.code === 'KeyC') {
        const modes = ['orbit', 'chase', 'firstPerson', 'swarm'];
        const curIdx = modes.indexOf(this.renderer.cameraMode);
        const nextMode = modes[(curIdx + 1) % modes.length];
        this.renderer.setCameraMode(nextMode);
        document.querySelectorAll('.cam-btn').forEach(b => {
          b.classList.toggle('active', b.dataset.cam === nextMode);
        });
      } else if (e.code === 'KeyT') {
        const btn = document.getElementById('btn-turbo');
        if (btn) btn.click();
      } else if (e.code === 'KeyP') {
        const btn = document.getElementById('btn-mutate');
        if (btn) btn.click();
      } else if (e.code === 'KeyV') {
        const btn = document.getElementById('btn-vortex');
        if (btn) btn.click();
      } else if (e.code === 'KeyM') {
        const btn = document.getElementById('btn-sound');
        if (btn) btn.click();
      }
    });
  }

  _flashBanner(text) {
    const banner = document.getElementById('status-banner');
    if (!banner) return;
    banner.textContent = text;
    banner.style.opacity = '1';
    clearTimeout(this._bannerTimeout);
    this._bannerTimeout = setTimeout(() => {
      banner.style.opacity = '0';
    }, 2800);
  }

  _updateStatsHUD() {
    const elPrey = document.getElementById('stat-prey');
    const elPred = document.getElementById('stat-pred');
    const elFood = document.getElementById('stat-food');
    const elKills = document.getElementById('stat-kills');
    const elTime = document.getElementById('stat-time');

    if (elPrey) elPrey.textContent = this.ecosystem.boids.length;
    if (elPred) elPred.textContent = this.ecosystem.predators.length;
    if (elFood) elFood.textContent = this.ecosystem.foods.length;
    if (elKills) elKills.textContent = this.ecosystem.totalKills;
    if (elTime) elTime.textContent = `${Math.floor(this.ecosystem.time)}s`;
  }

  _loop(now) {
    requestAnimationFrame(this._loop);

    const deltaMs = Math.min(64, now - this.lastTime);
    this.lastTime = now;
    let dt = (deltaMs / 1000.0) * this.timeScale;

    if (!this.isPaused) {
      if (this.isTurbo) {
        const subSteps = 6;
        const subDt = (1.0 / 60.0) * this.timeScale;
        for (let s = 0; s < subSteps; s++) {
          this.ecosystem.update(subDt);
        }
      } else {
        this.ecosystem.update(dt);
      }
    }

    // Update 3D render
    this.renderer.render(dt);

    // Update 2D telemetry HUDs
    this.brainViz.render();
    this.analytics.render(this.ecosystem);
    this._updateStatsHUD();

    // Update generative soundscape parameters
    if (this.audio && this.audio.initialized) {
      let totalSpeed = 0;
      for (let i = 0; i < this.ecosystem.boids.length; i++) {
        const b = this.ecosystem.boids[i];
        totalSpeed += Math.hypot(b.velocity.x, b.velocity.y, b.velocity.z);
      }
      const avgSpeed = this.ecosystem.boids.length > 0 ? totalSpeed / this.ecosystem.boids.length : 15;
      this.audio.updateAerodynamics(avgSpeed);
    }
  }
}

// Auto bootstrap on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.app = new SimulationApp();
});

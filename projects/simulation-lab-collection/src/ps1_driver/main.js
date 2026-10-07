// Main Application Loop & Game State Controller

import { RoadTrack, TRACK_THEMES } from './road.js';
import { PlayerCar, TrafficSystem } from './car.js';
import { SpriteManager } from './sprites.js';
import { AudioManager } from './audio.js';
import { PS1Renderer } from './renderer.js';

export class PS1DriverApp {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.minimapCanvas = document.getElementById('minimap-canvas');
    this.mCtx = this.minimapCanvas ? this.minimapCanvas.getContext('2d') : null;

    // Subsystems
    this.audio = new AudioManager();
    this.sprites = new SpriteManager();
    this.road = new RoadTrack('touge');
    this.player = new PlayerCar();
    this.traffic = new TrafficSystem(this.road.totalLength, 24);
    this.renderer = new PS1Renderer(this.canvas, this.road, this.player, this.traffic, this.sprites);

    // Input state
    this.keys = {
      up: false,
      down: false,
      left: false,
      right: false,
      space: false
    };

    // Arcade Game State
    this.timeLeft = 45.0; // Countdown timer
    this.score = 0;
    this.lastCheckpointIndex = -1;
    this.isGameOver = false;
    this.lapStartTime = Date.now();
    this.lap = 1;

    // DOM UI elements
    this.uiTime = document.getElementById('ui-time');
    this.uiScore = document.getElementById('ui-score');
    this.uiSpeed = document.getElementById('ui-speed');
    this.uiGear = document.getElementById('ui-gear');
    this.tachoNeedle = document.getElementById('tacho-needle');
    this.checkpointAlert = document.getElementById('checkpoint-alert');
    this.driftBanner = document.getElementById('drift-banner');
    this.driftPts = document.getElementById('drift-pts');
    this.btnTheme = document.getElementById('btn-theme');
    this.btnCam = document.getElementById('btn-cam');
    this.btnDither = document.getElementById('btn-dither');
    this.btnCrt = document.getElementById('btn-crt');
    this.btnAudio = document.getElementById('btn-audio');

    // Timing loop
    this.lastTime = performance.now();
    this.bindEvents();
    this.initMinimap();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  bindEvents() {
    // Keyboard inputs
    window.addEventListener('keydown', (e) => {
      this.audio.init();
      this.audio.resume();

      switch (e.code) {
        case 'ArrowUp':
        case 'KeyW':
          this.keys.up = true;
          break;
        case 'ArrowDown':
        case 'KeyS':
          this.keys.down = true;
          break;
        case 'ArrowLeft':
        case 'KeyA':
          this.keys.left = true;
          break;
        case 'ArrowRight':
        case 'KeyD':
          this.keys.right = true;
          break;
        case 'Space':
          this.keys.space = true;
          e.preventDefault();
          break;
        case 'KeyC':
          this.cycleCamera();
          break;
        case 'KeyV':
          this.toggleDither();
          break;
        case 'KeyT':
          this.cycleTheme();
          break;
        case 'KeyM':
          this.toggleAudio();
          break;
        case 'KeyR':
          this.resetCar();
          break;
      }
    });

    window.addEventListener('keyup', (e) => {
      switch (e.code) {
        case 'ArrowUp':
        case 'KeyW':
          this.keys.up = false;
          break;
        case 'ArrowDown':
        case 'KeyS':
          this.keys.down = false;
          break;
        case 'ArrowLeft':
        case 'KeyA':
          this.keys.left = false;
          break;
        case 'ArrowRight':
        case 'KeyD':
          this.keys.right = false;
          break;
        case 'Space':
          this.keys.space = false;
          break;
      }
    });

    // Control buttons
    if (this.btnTheme) {
      this.btnTheme.addEventListener('click', () => this.cycleTheme());
    }
    if (this.btnCam) {
      this.btnCam.addEventListener('click', () => this.cycleCamera());
    }
    if (this.btnDither) {
      this.btnDither.addEventListener('click', () => this.toggleDither());
    }
    if (this.btnCrt) {
      this.btnCrt.addEventListener('click', () => this.toggleCRT());
    }
    if (this.btnAudio) {
      this.btnAudio.addEventListener('click', () => {
        this.audio.init();
        this.audio.resume();
        this.toggleAudio();
      });
    }

    // Touch controls
    const bindTouch = (id, key) => {
      const el = document.getElementById(id);
      if (!el) return;
      const onStart = (e) => {
        e.preventDefault();
        this.audio.init();
        this.audio.resume();
        this.keys[key] = true;
      };
      const onEnd = (e) => {
        e.preventDefault();
        this.keys[key] = false;
      };
      el.addEventListener('touchstart', onStart, { passive: false });
      el.addEventListener('touchend', onEnd, { passive: false });
      el.addEventListener('mousedown', onStart);
      el.addEventListener('mouseup', onEnd);
      el.addEventListener('mouseleave', onEnd);
    };

    bindTouch('touch-left', 'left');
    bindTouch('touch-right', 'right');
    bindTouch('touch-gas', 'up');
    bindTouch('touch-brake', 'down');
    bindTouch('touch-drift', 'space');
  }

  cycleCamera() {
    const mode = this.player.cycleCamera();
    if (this.btnCam) {
      this.btnCam.textContent = `Cam: ${mode.toUpperCase()}`;
    }
  }

  cycleTheme() {
    const themeKeys = Object.keys(TRACK_THEMES);
    const currentIdx = themeKeys.indexOf(this.road.theme.id);
    const nextKey = themeKeys[(currentIdx + 1) % themeKeys.length];
    this.road.setTheme(nextKey);
    if (this.btnTheme) {
      this.btnTheme.textContent = `Track: ${this.road.theme.name}`;
    }
  }

  toggleDither() {
    const active = this.renderer.toggleDither();
    if (this.btnDither) {
      this.btnDither.classList.toggle('active', active);
      this.btnDither.textContent = `Dither: ${active ? 'ON' : 'OFF'}`;
    }
  }

  toggleCRT() {
    const overlay = document.getElementById('crt-overlay');
    if (overlay) {
      const isDisabled = overlay.classList.toggle('disabled');
      if (this.btnCrt) {
        this.btnCrt.classList.toggle('active', !isDisabled);
        this.btnCrt.textContent = `CRT: ${!isDisabled ? 'ON' : 'OFF'}`;
      }
    }
  }

  toggleAudio() {
    const isMuted = this.audio.toggleMute();
    if (this.btnAudio) {
      this.btnAudio.classList.toggle('active', !isMuted);
      this.btnAudio.textContent = `Audio: ${!isMuted ? 'ON' : 'MUTED'}`;
    }
  }

  resetCar() {
    this.player.reset();
  }

  triggerCheckpoint() {
    this.timeLeft += 20.0;
    this.score += 5000;
    this.audio.playCheckpointChime();

    if (this.checkpointAlert) {
      this.checkpointAlert.classList.add('active');
      clearTimeout(this.checkpointTimer);
      this.checkpointTimer = setTimeout(() => {
        this.checkpointAlert.classList.remove('active');
      }, 1800);
    }
  }

  initMinimap() {
    if (!this.minimapCanvas) return;
    this.minimapCanvas.width = 120;
    this.minimapCanvas.height = 70;
  }

  drawMinimap() {
    if (!this.mCtx) return;
    const ctx = this.mCtx;
    const w = this.minimapCanvas.width;
    const h = this.minimapCanvas.height;

    ctx.fillStyle = 'rgba(10, 12, 16, 0.9)';
    ctx.fillRect(0, 0, w, h);

    // Draw upcoming road curvature
    const currentSeg = this.road.findSegment(this.player.z);
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let sx = w / 2;
    let sy = h - 10;
    ctx.moveTo(sx, sy);

    const lookAhead = 40;
    for (let i = 0; i < lookAhead; i++) {
      const seg = this.road.segments[(currentSeg.index + i) % this.road.segments.length];
      sx += (seg.curve * 0.8) - (this.player.steer * 0.1);
      sy -= (h - 20) / lookAhead;
      ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Draw player dot
    ctx.fillStyle = '#fffa38';
    ctx.beginPath();
    ctx.arc(w / 2 + (this.player.x * 12), h - 10, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  updateHUD() {
    const kmh = this.player.getSpeedKmh();
    if (this.uiSpeed) {
      this.uiSpeed.textContent = kmh.toString().padStart(3, '0');
    }

    if (this.uiGear) {
      this.uiGear.textContent = `GEAR ${this.player.gear}`;
    }

    // Tachometer Needle (-120 deg to +120 deg)
    if (this.tachoNeedle) {
      const rpmRatio = Math.min(1, Math.max(0, this.player.rpm / 8500));
      const deg = -120 + (rpmRatio * 240);
      this.tachoNeedle.setAttribute('transform', `rotate(${deg} 70 70)`);
    }

    // Time countdown
    if (this.uiTime) {
      const t = Math.max(0, this.timeLeft);
      const secs = Math.floor(t);
      const ms = Math.floor((t - secs) * 100);
      this.uiTime.textContent = `${secs.toString().padStart(2, '0')}:${ms.toString().padStart(2, '0')}`;
      if (this.timeLeft <= 10) {
        this.uiTime.style.color = '#ff3344';
      } else {
        this.uiTime.style.color = 'var(--crt-amber)';
      }
    }

    // Drift banner
    if (this.driftBanner && this.driftPts) {
      if (this.player.isDrifting) {
        this.driftBanner.classList.add('active');
        this.driftPts.textContent = `+${this.player.driftScore} x${this.player.driftMultiplier.toFixed(1)}`;
      } else {
        this.driftBanner.classList.remove('active');
      }
    }

    // Total score
    if (this.uiScore) {
      const totalScore = this.score + this.player.driftScore + Math.floor(this.player.z * 0.1);
      this.uiScore.textContent = totalScore.toLocaleString();
    }
  }

  loop(currentTime) {
    const rawDt = (currentTime - this.lastTime) / 1000;
    this.lastTime = currentTime;
    const dt = Math.min(0.05, Math.max(0.001, rawDt));

    // Update countdown
    if (this.player.speed > 100) {
      this.timeLeft -= dt;
      if (this.timeLeft <= 0) {
        this.timeLeft = 0;
        this.isGameOver = true;
      }
    }

    // Previous gear for audio shift detection
    const oldGear = this.player.gear;

    // Update physics
    const currentSegment = this.road.findSegment(this.player.z);
    this.player.update(dt, this.keys, currentSegment, this.road.totalLength);
    this.traffic.update(dt, this.player);

    // Audio gear shift chime
    if (this.player.gear !== oldGear && this.player.speed > 500) {
      this.audio.playGearShift();
    }

    // Update engine and screech audio
    const speedRatio = this.player.speed / this.player.maxSpeed;
    this.audio.updateEngine(
      this.player.rpm,
      speedRatio,
      this.keys.up,
      this.player.isDrifting
    );

    // Checkpoint detection
    if (currentSegment.isCheckpoint && currentSegment.index !== this.lastCheckpointIndex) {
      this.lastCheckpointIndex = currentSegment.index;
      this.triggerCheckpoint();
    }

    // Render frame
    this.renderer.render(dt);
    this.drawMinimap();
    this.updateHUD();

    requestAnimationFrame(this.loop);
  }
}

// Auto-boot on load
window.addEventListener('DOMContentLoaded', () => {
  window.app = new PS1DriverApp();
});

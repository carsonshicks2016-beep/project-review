/**
 * ApexFlock - Generative Ambient Soundscape & Harmonic Resonance
 * High-fidelity generative procedural audio (Web Audio API):
 * - Resonant singing bowl / celestial glass drone pad
 * - Aerodynamic flock wind shimmer modulated by collective kinetic energy
 * - Ethereal pentatonic glass bells on feeding events
 * - Deep resonant sub-harmonic sweep on predatory strikes
 */

export class AudioSynth {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.isMuted = false;
    this.initialized = false;
    this.volume = 0.32;

    this.droneGain = null;
    this.flockGain = null;
    this.flockFilter = null;

    // Pentatonic frequencies for organic feeding chimes (A major pentatonic)
    this.chimeNotes = [880, 987.77, 1108.73, 1318.51, 1479.98, 1760.0];
  }

  init() {
    if (this.initialized) return;
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      this.ctx = new AudioContext();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.setValueAtTime(this.volume, this.ctx.currentTime);
      this.masterGain.connect(this.ctx.destination);

      this._initSingingBowlDrone();
      this._initFlockAerodynamics();
      this.initialized = true;
    } catch (e) {
      console.warn("Web Audio API not supported or blocked", e);
    }
  }

  resume() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  _initSingingBowlDrone() {
    if (!this.ctx) return;
    // Resonant fundamental + gentle overtone
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const osc3 = this.ctx.createOscillator();
    const filter = this.ctx.createBiquadFilter();

    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(55, this.ctx.currentTime);     // A1 fundamental
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(110.2, this.ctx.currentTime);  // A2 with gentle micro-chorus
    osc3.type = 'sine';
    osc3.frequency.setValueAtTime(165.1, this.ctx.currentTime);  // E3 harmonic fifth

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(220, this.ctx.currentTime);
    filter.Q.setValueAtTime(2.5, this.ctx.currentTime);

    this.droneGain = this.ctx.createGain();
    this.droneGain.gain.setValueAtTime(0.16, this.ctx.currentTime);

    osc1.connect(filter);
    osc2.connect(filter);
    osc3.connect(filter);
    filter.connect(this.droneGain);
    this.droneGain.connect(this.masterGain);

    osc1.start();
    osc2.start();
    osc3.start();
  }

  _initFlockAerodynamics() {
    if (!this.ctx) return;
    const bufferSize = this.ctx.sampleRate * 2;
    const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    // Pink noise approximation for natural wind shimmers
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      output[i] = (b0 + b1 + b2 + white * 0.5362) * 0.15;
    }

    const noiseSource = this.ctx.createBufferSource();
    noiseSource.buffer = noiseBuffer;
    noiseSource.loop = true;

    this.flockFilter = this.ctx.createBiquadFilter();
    this.flockFilter.type = 'bandpass';
    this.flockFilter.frequency.setValueAtTime(320, this.ctx.currentTime);
    this.flockFilter.Q.setValueAtTime(2.0, this.ctx.currentTime);

    this.flockGain = this.ctx.createGain();
    this.flockGain.gain.setValueAtTime(0.035, this.ctx.currentTime);

    noiseSource.connect(this.flockFilter);
    this.flockFilter.connect(this.flockGain);
    this.flockGain.connect(this.masterGain);

    noiseSource.start();
  }

  updateAerodynamics(meanSpeed) {
    if (!this.ctx || !this.flockFilter) return;
    const freq = 200 + Math.min(800, meanSpeed * 18);
    this.flockFilter.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.15);
  }

  playAlarmChirp(pitch = 1480) {
    if (!this.ctx || this.isMuted) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(pitch, now);
    osc.frequency.exponentialRampToValueAtTime(pitch * 1.35, now + 0.12);

    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(now);
    osc.stop(now + 0.15);
  }

  playPredatorStrike() {
    if (!this.ctx || this.isMuted) return;
    const now = this.ctx.currentTime;
    // Resonant sub-harmonic dive sweep
    const osc = this.ctx.createOscillator();
    const filter = this.ctx.createBiquadFilter();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(220, now);
    osc.frequency.exponentialRampToValueAtTime(55, now + 0.35); // Deep sub-bass drop

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(350, now);

    gain.gain.setValueAtTime(0.24, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    osc.start(now);
    osc.stop(now + 0.4);
  }

  playFoodChime() {
    if (!this.ctx || this.isMuted) return;
    const now = this.ctx.currentTime;
    const note = this.chimeNotes[Math.floor(Math.random() * this.chimeNotes.length)];

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(note, now);

    gain.gain.setValueAtTime(0.06, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(now);
    osc.stop(now + 0.32);
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setValueAtTime(this.isMuted ? 0 : this.volume, this.ctx.currentTime);
    }
    return this.isMuted;
  }
}

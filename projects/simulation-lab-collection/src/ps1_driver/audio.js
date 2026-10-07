// Procedural Web Audio API Sound Engine & Retro 90s Chiptune Synthesizer

export class AudioManager {
  constructor() {
    this.ctx = null;
    this.isMuted = false;
    this.isMusicMuted = false;
    this.isInitialized = false;

    // Engine sound nodes
    this.engineOsc1 = null;
    this.engineOsc2 = null;
    this.engineFilter = null;
    this.engineGain = null;

    // Tire screech nodes
    this.screechNoise = null;
    this.screechFilter = null;
    this.screechGain = null;

    // Music sequencer state
    this.musicTimer = null;
    this.musicStep = 0;
    this.bpm = 128;
  }

  init() {
    if (this.isInitialized) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
      this.setupEngineSynth();
      this.setupTireScreech();
      this.startMusicSequencer();
      this.isInitialized = true;
    } catch (e) {
      console.warn('Web Audio could not be initialized:', e);
    }
  }

  resume() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  setupEngineSynth() {
    if (!this.ctx) return;

    // Osc 1: Sawtooth (aggressive engine rasp)
    this.engineOsc1 = this.ctx.createOscillator();
    this.engineOsc1.type = 'sawtooth';
    this.engineOsc1.frequency.setValueAtTime(45, this.ctx.currentTime);

    // Osc 2: Triangle (sub low rumble)
    this.engineOsc2 = this.ctx.createOscillator();
    this.engineOsc2.type = 'triangle';
    this.engineOsc2.frequency.setValueAtTime(22.5, this.ctx.currentTime);

    // Lowpass filter for engine body
    this.engineFilter = this.ctx.createBiquadFilter();
    this.engineFilter.type = 'lowpass';
    this.engineFilter.frequency.setValueAtTime(280, this.ctx.currentTime);
    this.engineFilter.Q.setValueAtTime(3.5, this.ctx.currentTime);

    // Master engine gain
    this.engineGain = this.ctx.createGain();
    this.engineGain.gain.setValueAtTime(0.18, this.ctx.currentTime);

    this.engineOsc1.connect(this.engineFilter);
    this.engineOsc2.connect(this.engineFilter);
    this.engineFilter.connect(this.engineGain);
    this.engineGain.connect(this.ctx.destination);

    this.engineOsc1.start();
    this.engineOsc2.start();
  }

  setupTireScreech() {
    if (!this.ctx) return;

    // Create 2-second white noise buffer
    const bufferSize = this.ctx.sampleRate * 2;
    const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }

    this.screechNoise = this.ctx.createBufferSource();
    this.screechNoise.buffer = noiseBuffer;
    this.screechNoise.loop = true;

    // Bandpass filter centered at tire friction frequencies
    this.screechFilter = this.ctx.createBiquadFilter();
    this.screechFilter.type = 'bandpass';
    this.screechFilter.frequency.setValueAtTime(1100, this.ctx.currentTime);
    this.screechFilter.Q.setValueAtTime(4.0, this.ctx.currentTime);

    this.screechGain = this.ctx.createGain();
    this.screechGain.gain.setValueAtTime(0, this.ctx.currentTime);

    this.screechNoise.connect(this.screechFilter);
    this.screechFilter.connect(this.screechGain);
    this.screechGain.connect(this.ctx.destination);

    this.screechNoise.start();
  }

  updateEngine(rpm, speedRatio, isAccelerating, isDrifting) {
    if (!this.ctx || !this.engineOsc1 || this.isMuted) return;

    const t = this.ctx.currentTime;
    // Map RPM (900 - 8200) to engine fundamental freq (40 Hz - 320 Hz)
    const baseFreq = 35 + (rpm / 8000) * 280;
    this.engineOsc1.frequency.setTargetAtTime(baseFreq, t, 0.04);
    this.engineOsc2.frequency.setTargetAtTime(baseFreq * 0.5, t, 0.04);

    // Filter frequency opens up with throttle
    const filterCutoff = 250 + (isAccelerating ? 900 : 200) + speedRatio * 800;
    this.engineFilter.frequency.setTargetAtTime(filterCutoff, t, 0.06);

    // Tire Screech Volume
    if (this.screechGain) {
      const screechVol = isDrifting ? Math.min(0.25, 0.08 + speedRatio * 0.18) : 0;
      this.screechGain.gain.setTargetAtTime(this.isMuted ? 0 : screechVol, t, 0.05);
      if (isDrifting) {
        this.screechFilter.frequency.setTargetAtTime(950 + speedRatio * 500, t, 0.05);
      }
    }
  }

  playGearShift() {
    if (!this.ctx || this.isMuted) return;
    const t = this.ctx.currentTime;
    // Low frequency thunk
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(90, t);
    osc.frequency.exponentialRampToValueAtTime(30, t + 0.08);

    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.08);

    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(t);
    osc.stop(t + 0.09);
  }

  playCheckpointChime() {
    if (!this.ctx || this.isMuted) return;
    const t = this.ctx.currentTime;
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    notes.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t + idx * 0.08);

      gain.gain.setValueAtTime(0.2, t + idx * 0.08);
      gain.gain.exponentialRampToValueAtTime(0.001, t + idx * 0.08 + 0.25);

      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t + idx * 0.08);
      osc.stop(t + idx * 0.08 + 0.26);
    });
  }

  playCollisionSound() {
    if (!this.ctx || this.isMuted) return;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(140, t);
    osc.frequency.exponentialRampToValueAtTime(40, t + 0.15);

    gain.gain.setValueAtTime(0.4, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.15);

    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(t);
    osc.stop(t + 0.16);
  }

  startMusicSequencer() {
    // 90s Ridge Racer style bassline & synth chords
    const bassline = [
      110.00, 110.00, 130.81, 110.00, // A2, A2, C3, A2
      146.83, 110.00, 130.81, 98.00,  // D3, A2, C3, G2
      87.31,  87.31,  110.00, 87.31,   // F2, F2, A2, F2
      98.00,  98.00,  123.47, 130.81   // G2, G2, B2, C3
    ];

    const stepDuration = 60 / this.bpm / 2; // 16th notes
    let nextNoteTime = this.ctx.currentTime + 0.1;

    const schedule = () => {
      while (nextNoteTime < this.ctx.currentTime + 0.25) {
        if (!this.isMusicMuted && !this.isMuted) {
          const noteFreq = bassline[this.musicStep % bassline.length];
          this.triggerBassNote(noteFreq, nextNoteTime, stepDuration * 0.85);

          // Percussion / Hi-hat on 16ths, snare on steps 4, 12
          const isSnare = (this.musicStep % 8 === 4);
          this.triggerHat(nextNoteTime, isSnare);
        }
        nextNoteTime += stepDuration;
        this.musicStep++;
      }
      this.musicTimer = setTimeout(schedule, 50);
    };

    schedule();
  }

  triggerBassNote(freq, time, dur) {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(freq, time);

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(800, time);
    filter.frequency.exponentialRampToValueAtTime(160, time + dur);

    gain.gain.setValueAtTime(0.12, time);
    gain.gain.exponentialRampToValueAtTime(0.001, time + dur);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(time);
    osc.stop(time + dur + 0.02);
  }

  triggerHat(time, isSnare) {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = isSnare ? 'triangle' : 'square';
    osc.frequency.setValueAtTime(isSnare ? 220 : 8000, time);

    const dur = isSnare ? 0.08 : 0.02;
    gain.gain.setValueAtTime(isSnare ? 0.09 : 0.025, time);
    gain.gain.exponentialRampToValueAtTime(0.001, time + dur);

    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(time);
    osc.stop(time + dur + 0.01);
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.engineGain) {
      this.engineGain.gain.setValueAtTime(this.isMuted ? 0 : 0.18, this.ctx.currentTime);
    }
    return this.isMuted;
  }

  toggleMusic() {
    this.isMusicMuted = !this.isMusicMuted;
    return this.isMusicMuted;
  }
}

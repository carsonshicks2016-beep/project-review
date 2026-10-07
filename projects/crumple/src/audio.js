// Fully synthesized audio — no asset files. Engine is a detuned saw/square
// pair driven by wheel speed; impacts are enveloped noise bursts; tire squeal
// is a bandpassed noise loop gated on slip.

export class Sound {
  constructor() {
    this.ready = false;
    this.ctx = null;
    this.lastCrash = 0;
  }

  start() {
    if (this.ready) return;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    this.ctx = ctx;

    this.master = ctx.createGain();
    this.master.gain.value = 0.55;
    this.master.connect(ctx.destination);

    // Engine
    this.engineGain = ctx.createGain();
    this.engineGain.gain.value = 0;
    this.engineFilter = ctx.createBiquadFilter();
    this.engineFilter.type = 'lowpass';
    this.engineFilter.frequency.value = 900;
    this.engineFilter.Q.value = 3;
    this.engineGain.connect(this.engineFilter);
    this.engineFilter.connect(this.master);

    this.osc1 = ctx.createOscillator();
    this.osc1.type = 'sawtooth';
    this.osc2 = ctx.createOscillator();
    this.osc2.type = 'square';
    const sub = ctx.createGain();
    sub.gain.value = 0.45;
    this.osc1.connect(this.engineGain);
    this.osc2.connect(sub);
    sub.connect(this.engineGain);
    this.osc1.start();
    this.osc2.start();

    // Reusable noise buffer
    const len = ctx.sampleRate * 2;
    this.noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = this.noise.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;

    // Tire squeal loop
    this.squealGain = ctx.createGain();
    this.squealGain.gain.value = 0;
    const sq = ctx.createBiquadFilter();
    sq.type = 'bandpass';
    sq.frequency.value = 1750;
    sq.Q.value = 7;
    this.squealSrc = ctx.createBufferSource();
    this.squealSrc.buffer = this.noise;
    this.squealSrc.loop = true;
    this.squealSrc.connect(sq);
    sq.connect(this.squealGain);
    this.squealGain.connect(this.master);
    this.squealSrc.start();

    this.ready = true;
  }

  engine(rpm, throttle, alive) {
    if (!this.ready) return;
    const t = this.ctx.currentTime;
    // Fake gearing: pitch ramps then drops, so acceleration has texture.
    const gear = Math.min(5, Math.floor(rpm * 5));
    const within = rpm * 5 - gear;
    const f = 46 + within * 105 + gear * 9;
    this.osc1.frequency.setTargetAtTime(f, t, 0.05);
    this.osc2.frequency.setTargetAtTime(f * 0.5, t, 0.05);
    this.engineFilter.frequency.setTargetAtTime(500 + rpm * 2200, t, 0.08);
    const g = alive ? 0.05 + Math.abs(throttle) * 0.10 + rpm * 0.07 : 0;
    this.engineGain.gain.setTargetAtTime(g, t, 0.1);
  }

  squeal(amount) {
    if (!this.ready) return;
    this.squealGain.gain.setTargetAtTime(Math.min(amount, 1) * 0.09, this.ctx.currentTime, 0.05);
  }

  crash(intensity) {
    if (!this.ready) return;
    const t = this.ctx.currentTime;
    if (t - this.lastCrash < 0.045) return; // avoid stacking a whole pile-up
    this.lastCrash = t;

    const amp = Math.min(intensity / 18, 1);
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 0.55 + Math.random() * 0.5;

    const filt = this.ctx.createBiquadFilter();
    filt.type = 'bandpass';
    filt.frequency.value = 260 + Math.random() * 900;
    filt.Q.value = 0.9;

    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.35 * amp + 0.02, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.18 + amp * 0.4);

    // Low thud under the crunch.
    const thud = this.ctx.createOscillator();
    thud.type = 'sine';
    thud.frequency.setValueAtTime(120, t);
    thud.frequency.exponentialRampToValueAtTime(38, t + 0.22);
    const tg = this.ctx.createGain();
    tg.gain.setValueAtTime(0.4 * amp, t);
    tg.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);

    src.connect(filt); filt.connect(g); g.connect(this.master);
    thud.connect(tg); tg.connect(this.master);
    src.start(t); src.stop(t + 0.7);
    thud.start(t); thud.stop(t + 0.35);
  }
}

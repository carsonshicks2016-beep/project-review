import { extractEvents, KEYFRAME_HZ } from './raceEvents.js';
import { templateVoice } from './voices/template.js';

// Schedules commentary against playback time and renders it into a feed.
//
// Voices are pluggable and async, so a local WebLLM or a hosted model can be dropped in
// without touching the event layer or the UI.

const VOICES = { template: templateVoice };

// Higher wins when two events come due inside the same cooldown window.
const PRIORITY = {
  record: 100, finish: 90, wallHit: 85, groundHit: 85, timeout: 80,
  nearMiss: 60, gate: 50, stall: 40, topSpeed: 30, bank: 20, launch: 10
};

export class Commentator {
  constructor({ voice = 'template', maxLines = 40, cooldown = 1.1, speak = false } = {}) {
    this.voice = VOICES[voice] || templateVoice;
    this.maxLines = maxLines;
    this.cooldown = cooldown;   // Minimum seconds between spoken lines
    this.speak = speak;
    this.lines = [];
    this.listeners = [];
    this.reset();
  }

  setVoice(name) { this.voice = VOICES[name] || templateVoice; }
  onLine(fn) { this.listeners.push(fn); }

  reset() {
    this.events = [];
    this.cursor = 0;
    this.lastEmit = -999;
    this.lastTime = 0;
  }

  // Called whenever a new run starts playing.
  loadRun(run, opts) {
    this.events = extractEvents(run, opts);
    this.cursor = 0;
    this.lastEmit = -999;
    this.lastTime = 0;
    return this.events.length;
  }

  // Drive from playback position (in keyframes, matching replayTime).
  update(frame) {
    const t = frame / KEYFRAME_HZ;
    // Scrubbing backwards should re-arm rather than stay silent for the rest of the run.
    if (t < this.lastTime - 0.5) {
      this.cursor = 0;
      this.lastEmit = -999;
      while (this.cursor < this.events.length && this.events[this.cursor].frame <= frame) this.cursor++;
    }
    this.lastTime = t;

    // Collect everything now due, then say only the most interesting one: a gate pass and
    // a near miss often land on the same frame, and reading both is noise.
    let best = null;
    while (this.cursor < this.events.length && this.events[this.cursor].frame <= frame) {
      const e = this.events[this.cursor++];
      if (!best || (PRIORITY[e.type] || 0) > (PRIORITY[best.type] || 0)) best = e;
    }
    if (!best) return;
    if (t - this.lastEmit < this.cooldown && (PRIORITY[best.type] || 0) < 80) return;

    this.lastEmit = t;
    this._emit(best, t);
  }

  async _emit(event, t) {
    const text = await this.voice.say(event);
    if (!text) return;
    const line = { t, type: event.type, text };
    this.lines.push(line);
    if (this.lines.length > this.maxLines) this.lines.shift();
    this.listeners.forEach(fn => fn(line));
    if (this.speak) this._speak(text);
  }

  _speak(text) {
    if (typeof speechSynthesis === 'undefined') return;
    // Never let commentary queue up behind itself; the newest line is the relevant one.
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.15;
    u.pitch = 0.9;
    speechSynthesis.speak(u);
  }
}

/* Signed little-endian int16 stereo PCM, fixed at 48 kHz. */
class PcmJitterProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.channels = config.channels || 2;
    this.capacityFrames = 48000;
    // ~171 ms of startup headroom absorbs Python DSP and WebSocket scheduling
    // jitter — including the bursty stalls that appear while the box is also
    // running training — without drifting far from the engine's live pitch.
    this.targetFrames = 8192;
    // After a dropout, resume as soon as ~50 ms is rebuffered instead of waiting
    // for a full re-prime, so a recovered gap is short rather than a long mute.
    this.resumeFrames = 2400;
    this.everPrimed = false;
    this.buffer = new Int16Array(this.capacityFrames * this.channels);
    this.readFrame = 0;
    this.writeFrame = 0;
    this.availableFrames = 0;
    this.primed = false;
    this.underruns = 0;
    this.degraded = false;
    this.outputGain = 0;
    this.lastSample = new Float32Array(this.channels);
    this.port.onmessage = (event) => this.receive(event.data);
  }

  receive(message) {
    if (message?.type === "reset") {
      this.readFrame = 0;
      this.writeFrame = 0;
      this.availableFrames = 0;
      this.primed = false;
      this.everPrimed = false;
      this.degraded = false;
      this.outputGain = 0;
      this.lastSample.fill(0);
      return;
    }
    if (message?.type !== "pcm" || !(message.buffer instanceof ArrayBuffer)) return;
    const bytes = new DataView(message.buffer);
    const incomingFrames = Math.floor(bytes.byteLength / (2 * this.channels));
    if (incomingFrames <= 0) return;
    const overflow = Math.max(0, this.availableFrames + incomingFrames - this.capacityFrames);
    if (overflow) {
      this.readFrame = (this.readFrame + overflow) % this.capacityFrames;
      this.availableFrames -= overflow;
    }
    for (let frame = 0; frame < incomingFrames; frame += 1) {
      for (let channel = 0; channel < this.channels; channel += 1) {
        const source = (frame * this.channels + channel) * 2;
        this.buffer[this.writeFrame * this.channels + channel] = bytes.getInt16(source, true);
      }
      this.writeFrame = (this.writeFrame + 1) % this.capacityFrames;
    }
    this.availableFrames = Math.min(this.capacityFrames, this.availableFrames + incomingFrames);
    const threshold = this.everPrimed ? this.resumeFrames : this.targetFrames;
    if (this.availableFrames >= threshold) {
      this.primed = true;
      this.everPrimed = true;
      if (this.degraded) {
        this.degraded = false;
        this.port.postMessage({ type: "recovered" });
      }
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const block = output[0]?.length || 128;
    for (let frame = 0; frame < block; frame += 1) {
      const playable = this.primed && this.availableFrames > 0;
      const targetGain = playable ? 1 : 0;
      // Asymmetric smoothing: fast fade-out on underrun (no droning held
      // sample), gentle fade-in on recovery (no pop).
      const alpha = targetGain > this.outputGain ? 0.008 : 0.035;
      this.outputGain += (targetGain - this.outputGain) * alpha;
      for (let channel = 0; channel < output.length; channel += 1) {
        const sourceChannel = Math.min(channel, this.channels - 1);
        if (playable) {
          this.lastSample[sourceChannel] = this.buffer[this.readFrame * this.channels + sourceChannel] / 32768;
        }
        output[channel][frame] = this.lastSample[sourceChannel] * this.outputGain;
      }
      if (playable) {
        this.readFrame = (this.readFrame + 1) % this.capacityFrames;
        this.availableFrames -= 1;
      } else if (this.primed) {
        this.primed = false;
        this.degraded = true;
        this.underruns += 1;
        // Clear held sample so recovery doesn't start from a stale DC offset
        this.lastSample.fill(0);
        this.port.postMessage({ type: "underrun", count: this.underruns });
      }
    }
    return true;
  }
}

registerProcessor("pcm-jitter-processor", PcmJitterProcessor);

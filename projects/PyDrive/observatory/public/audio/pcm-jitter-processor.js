/* Fresh FOA1 PCM jitter buffer — signed int16 LE stereo @ 48 kHz. */
class PcmJitterProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.channels = config.channels || 2;
    this.capacityFrames = 48000;
    this.targetFrames = 8192;
    this.resumeFrames = 2400;
    this.buffer = new Int16Array(this.capacityFrames * this.channels);
    this.readFrame = 0;
    this.writeFrame = 0;
    this.availableFrames = 0;
    this.primed = false;
    this.everPrimed = false;
    this.degraded = false;
    this.underruns = 0;
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
    const incoming = Math.floor(bytes.byteLength / (2 * this.channels));
    if (incoming <= 0) return;
    const overflow = Math.max(0, this.availableFrames + incoming - this.capacityFrames);
    if (overflow) {
      this.readFrame = (this.readFrame + overflow) % this.capacityFrames;
      this.availableFrames -= overflow;
    }
    for (let frame = 0; frame < incoming; frame += 1) {
      for (let ch = 0; ch < this.channels; ch += 1) {
        this.buffer[this.writeFrame * this.channels + ch] =
          bytes.getInt16((frame * this.channels + ch) * 2, true);
      }
      this.writeFrame = (this.writeFrame + 1) % this.capacityFrames;
    }
    this.availableFrames = Math.min(this.capacityFrames, this.availableFrames + incoming);
    const need = this.everPrimed ? this.resumeFrames : this.targetFrames;
    if (this.availableFrames >= need) {
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
    for (let i = 0; i < block; i += 1) {
      const playable = this.primed && this.availableFrames > 0;
      const target = playable ? 1 : 0;
      this.outputGain += (target - this.outputGain) * (target > this.outputGain ? 0.008 : 0.035);
      for (let ch = 0; ch < output.length; ch += 1) {
        const src = Math.min(ch, this.channels - 1);
        if (playable) {
          this.lastSample[src] = this.buffer[this.readFrame * this.channels + src] / 32768;
        }
        output[ch][i] = this.lastSample[src] * this.outputGain;
      }
      if (playable) {
        this.readFrame = (this.readFrame + 1) % this.capacityFrames;
        this.availableFrames -= 1;
      } else if (this.primed) {
        this.primed = false;
        this.degraded = true;
        this.underruns += 1;
        this.lastSample.fill(0);
        this.port.postMessage({ type: "underrun", count: this.underruns });
      }
    }
    return true;
  }
}

registerProcessor("pcm-jitter-processor", PcmJitterProcessor);

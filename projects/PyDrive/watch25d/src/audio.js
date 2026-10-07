/** FOA1 header: magic, version, channels, rate, frames, sequence, sim_time. */
export const FOA1_HEADER = {
  MAGIC: "FOA1",
  BYTES: 32,
  // <4s H H I I Q d  → 4+2+2+4+4+8+8 = 32
};

export function parseFoa1(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < FOA1_HEADER.BYTES) {
    throw new Error("FOA1 chunk too short");
  }
  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3),
  );
  if (magic !== FOA1_HEADER.MAGIC) throw new Error(`bad FOA1 magic ${magic}`);
  const version = view.getUint16(4, true);
  const channels = view.getUint16(6, true);
  const sampleRate = view.getUint32(8, true);
  const frames = view.getUint32(12, true);
  const sequence = Number(view.getBigUint64(16, true));
  const simTime = view.getFloat64(24, true);
  const pcm = buffer.slice(FOA1_HEADER.BYTES);
  const expected = frames * channels * 2;
  if (pcm.byteLength < expected) throw new Error("FOA1 PCM truncated");
  return { version, channels, sampleRate, frames, sequence, simTime, pcm: pcm.slice(0, expected) };
}

/**
 * Browser PCM bridge. Python owns synthesis + Doppler; this only buffers
 * FOA1 chunks into the AudioWorklet and reports mute / resume / underrun.
 */
export class AudioBridge {
  constructor({ workletUrl = "./audio/pcm-jitter-processor.js" } = {}) {
    this.workletUrl = workletUrl;
    this.socket = null;
    this.context = null;
    this.node = null;
    this.gain = null;
    this.enabled = false;
    this.degraded = false;
    this.status = "off";
    this.onStatus = null;
  }

  _emit(status, detail = "") {
    this.status = status;
    this.onStatus?.(status, detail);
  }

  async enable(audioUrl) {
    if (this.enabled) return this.status;
    if (!audioUrl) {
      this.degraded = true;
      this._emit("unavailable", "audio unavailable");
      return this.status;
    }
    try {
      this.context = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
      await this.context.audioWorklet.addModule(this.workletUrl);
      this.node = new AudioWorkletNode(this.context, "pcm-jitter-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        processorOptions: { channels: 2 },
      });
      this.gain = this.context.createGain();
      this.gain.gain.value = 1;
      this.node.connect(this.gain).connect(this.context.destination);
      this.node.port.onmessage = (event) => {
        const type = event.data?.type;
        if (type === "underrun") {
          this.degraded = true;
          this._emit("buffering", "audio buffering");
        } else if (type === "recovered") {
          this.degraded = false;
          this._emit("live");
        }
      };
      if (this.context.state === "suspended") await this.context.resume();
      this.socket = new WebSocket(audioUrl);
      this.socket.binaryType = "arraybuffer";
      this.socket.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer)) return;
        try {
          const chunk = parseFoa1(event.data);
          this.node?.port.postMessage({ type: "pcm", buffer: chunk.pcm }, [chunk.pcm]);
        } catch {
          /* drop malformed chunks; worklet underrun path reports status */
        }
      };
      this.socket.onerror = () => {
        this.degraded = true;
        this._emit("offline", "audio stream offline");
      };
      this.socket.onclose = () => {
        if (this.enabled) {
          this.degraded = true;
          this._emit("muted", "audio muted");
        }
      };
      this.enabled = true;
      this.degraded = false;
      this._emit("buffering", "audio buffering");
      return this.status;
    } catch (error) {
      this.degraded = true;
      this._emit("unavailable", "audio unavailable");
      await this.disable();
      throw error;
    }
  }

  async mute() {
    this._sendControl({ type: "mute" });
    if (this.gain) this.gain.gain.value = 0;
    this._emit("muted");
  }

  async resumeStream() {
    this._sendControl({ type: "resume" });
    if (this.gain) this.gain.gain.value = 1;
    if (this.context?.state === "suspended") await this.context.resume();
    this._emit(this.degraded ? "buffering" : "live");
  }

  async disable() {
    this.enabled = false;
    try { this.socket?.close(); } catch { /* ignore */ }
    this.socket = null;
    try { this.node?.port.postMessage({ type: "reset" }); } catch { /* ignore */ }
    try { this.node?.disconnect(); } catch { /* ignore */ }
    try { this.gain?.disconnect(); } catch { /* ignore */ }
    this.node = null;
    this.gain = null;
    if (this.context) {
      try { await this.context.close(); } catch { /* ignore */ }
    }
    this.context = null;
    this.degraded = false;
    this._emit("off");
  }

  _sendControl(message) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }
}

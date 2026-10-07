import { socketUrl } from "./protocol.js";

export function audioSocketPath(sessionId, returnedUrl = "") {
  return returnedUrl || `/api/observatory/audio/${encodeURIComponent(sessionId)}`;
}

export class EngineAudioStream extends EventTarget {
  constructor(sessionId, audioUrl = "") {
    super();
    this.sessionId = sessionId;
    this.audioUrl = audioUrl;
    this.context = null;
    this.node = null;
    this.socket = null;
    this.enabled = false;
    this.underrunTimes = [];
  }

  async enable() {
    if (this.enabled) return true;
    const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Context || !globalThis.AudioWorkletNode || !globalThis.WebSocket) {
      this.emit("status", { enabled: false, label: "Audio unavailable" });
      return false;
    }
    try {
      this.context ||= new Context({ sampleRate: 48000, latencyHint: "interactive" });
      await this.context.audioWorklet.addModule(`${import.meta.env.BASE_URL}audio/pcm-jitter-processor.js`);
      this.node ||= new AudioWorkletNode(this.context, "pcm-jitter-processor", {
        outputChannelCount: [2],
        processorOptions: { sampleRate: 48000, channels: 2 },
      });
      this.node.port.onmessage = (event) => this.handleWorkletStatus(event.data);
      this.node.connect(this.context.destination);
      await this.context.resume();
      if (this.socket?.readyState === 1) {
        this.socket.send(JSON.stringify({ type: "resume" }));
      } else if (!this.socket || this.socket.readyState >= 2) {
        this.openSocket();
      }
      this.enabled = true;
      this.emit("status", { enabled: true, label: "Audio live" });
      return true;
    } catch (error) {
      this.enabled = false;
      this.emit("status", { enabled: false, label: `Audio muted: ${error.message}` });
      return false;
    }
  }

  disable() {
    this.enabled = false;
    this.underrunTimes.length = 0;
    const socket = this.socket;
    const requestMute = () => {
      if (socket?.readyState !== 1) return;
      socket.send(JSON.stringify({ type: "mute" }));
    };
    if (socket?.readyState === 0) socket.addEventListener("open", requestMute, { once: true });
    else requestMute();
    this.node?.port.postMessage({ type: "reset" });
    this.context?.suspend();
    this.emit("status", { enabled: false, label: "Audio muted" });
  }

  shutdown(timeoutMs = 750) {
    this.enabled = false;
    this.underrunTimes.length = 0;
    const socket = this.socket;
    this.socket = null;
    this.node?.port.postMessage({ type: "reset" });
    this.context?.suspend();
    if (!socket || socket.readyState === 3) return Promise.resolve();
    return new Promise((resolve) => {
      let timer = 0;
      const finished = () => {
        clearTimeout(timer);
        resolve();
      };
      socket.addEventListener("close", finished, { once: true });
      timer = setTimeout(finished, timeoutMs);
      try {
        socket.close(1000, "session released");
      } catch {
        finished();
      }
    });
  }

  toggle() {
    return this.enabled ? (this.disable(), Promise.resolve(false)) : this.enable();
  }

  openSocket() {
    this.socket?.close();
    const socket = new WebSocket(socketUrl(audioSocketPath(this.sessionId, this.audioUrl)));
    this.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.addEventListener("message", (event) => {
      if (!this.enabled || !(event.data instanceof ArrayBuffer)) return;
      let pcm = event.data;
      if (pcm.byteLength >= 32) {
        const view = new DataView(pcm);
        const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
        if (magic === "FOA1") {
          const channels = view.getUint16(6, true);
          const sampleRate = view.getUint32(8, true);
          const frames = view.getUint32(12, true);
          if (channels !== 2 || sampleRate !== 48000 || pcm.byteLength !== 32 + frames * channels * 2) {
            this.emit("status", { enabled: false, label: "Audio muted: invalid PCM contract" });
            return;
          }
          pcm = pcm.slice(32);
        }
      }
      this.node?.port.postMessage({ type: "pcm", buffer: pcm }, [pcm]);
    });
    socket.addEventListener("close", () => {
      if (this.enabled && socket === this.socket) {
        this.enabled = false;
        this.emit("status", { enabled: false, label: "Audio stream offline" });
      }
    });
    socket.addEventListener("error", () => this.emit("status", { enabled: false, label: "Audio muted: stream error" }));
  }

  handleWorkletStatus(message) {
    if (!this.enabled) return;
    if (message?.type === "underrun") {
      const now = performance.now();
      this.underrunTimes.push(now);
      this.underrunTimes = this.underrunTimes.filter((stamp) => now - stamp <= 8000);
      if (this.underrunTimes.length >= 3) {
        this.emit("status", { enabled: true, degraded: true, label: "Audio buffering — playback continues muted" });
      }
    } else if (message?.type === "recovered") {
      this.underrunTimes.length = 0;
      this.emit("status", { enabled: true, degraded: false, label: "Audio live" });
    }
  }

  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

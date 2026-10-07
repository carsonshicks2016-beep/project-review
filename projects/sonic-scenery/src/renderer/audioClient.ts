/**
 * AudioClient — connects to the native Swift helper's WebSocket (AUDIO_WS_URL),
 * parses AudioFrame messages, and exposes the latest frame to the render loop.
 *
 * MUST run gracefully when the helper is offline: it auto-reconnects with
 * backoff and, while disconnected, synthesizes a gentle self-animating frame so
 * the scene keeps breathing instead of freezing or crashing.
 */
import { AUDIO_FRAME_BANDS, type AudioFrame } from "../contracts";

export type ConnectionState = "connecting" | "open" | "offline";

export class AudioClient {
  private ws: WebSocket | null = null;
  private url: string;
  private latest: AudioFrame | null = null;
  private lastMessageAt = 0;
  private reconnectMs = 1000;
  private readonly maxReconnectMs = 8000;
  private disposed = false;
  private startedAt = performance.now();

  state: ConnectionState = "connecting";

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    if (this.disposed) return;
    this.state = "connecting";
    let sock: WebSocket;
    try {
      sock = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = sock;

    sock.onopen = () => {
      this.state = "open";
      this.reconnectMs = 1000;
    };
    sock.onmessage = (ev) => {
      const frame = this.parse(ev.data);
      if (frame) {
        this.latest = frame;
        this.lastMessageAt = performance.now();
      }
    };
    sock.onerror = () => {
      // onclose handles the reconnect; swallow to avoid console spam.
    };
    sock.onclose = () => {
      this.ws = null;
      this.state = "offline";
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.disposed) return;
    this.state = "offline";
    setTimeout(() => this.connect(), this.reconnectMs);
    this.reconnectMs = Math.min(this.reconnectMs * 1.6, this.maxReconnectMs);
  }

  private parse(data: unknown): AudioFrame | null {
    if (typeof data !== "string") return null;
    try {
      const obj = JSON.parse(data) as Partial<AudioFrame>;
      if (typeof obj.rms !== "number" || !Array.isArray(obj.bands)) return null;
      const bands = obj.bands.slice(0, AUDIO_FRAME_BANDS).map((b) => +b || 0);
      while (bands.length < AUDIO_FRAME_BANDS) bands.push(0);
      return {
        t: +(obj.t ?? 0),
        rms: clamp01(obj.rms),
        bands,
        bass: clamp01(obj.bass ?? bands[0] ?? 0),
        mid: clamp01(obj.mid ?? bands[Math.floor(AUDIO_FRAME_BANDS / 2)] ?? 0),
        treble: clamp01(obj.treble ?? bands[AUDIO_FRAME_BANDS - 1] ?? 0),
        centroid: clamp01(obj.centroid ?? 0.5),
        onset: Boolean(obj.onset),
        beatConfidence: clamp01(obj.beatConfidence ?? 0),
      };
    } catch {
      return null;
    }
  }

  /**
   * The frame to drive visuals with this update. Returns the live frame when
   * fresh; otherwise a synthetic self-animating frame so the scene stays alive.
   * `now` is in ms (performance.now()).
   */
  frame(now: number): AudioFrame {
    const fresh = this.latest && now - this.lastMessageAt < 250;
    if (fresh && this.latest) return this.latest;
    return this.syntheticFrame(now);
  }

  /** True when a live frame arrived recently. */
  get live(): boolean {
    return Boolean(this.latest && performance.now() - this.lastMessageAt < 250);
  }

  /** Gentle, deterministic idle animation so an offline scene still moves. */
  private syntheticFrame(now: number): AudioFrame {
    const tt = (now - this.startedAt) / 1000;
    const slow = 0.5 + 0.5 * Math.sin(tt * 0.6);
    const med = 0.5 + 0.5 * Math.sin(tt * 1.3 + 1.0);
    const fast = 0.5 + 0.5 * Math.sin(tt * 2.7 + 2.0);
    const bands: number[] = [];
    for (let i = 0; i < AUDIO_FRAME_BANDS; i++) {
      bands.push(
        0.18 + 0.12 * Math.sin(tt * (0.7 + i * 0.35) + i * 0.9) ** 2,
      );
    }
    // Occasional soft "beat" so lightning/onset hooks fire while idle.
    const onset = Math.sin(tt * 1.05) > 0.985;
    return {
      t: now,
      rms: 0.25 + 0.12 * slow,
      bands,
      bass: 0.3 + 0.25 * slow,
      mid: 0.3 + 0.2 * med,
      treble: 0.25 + 0.2 * fast,
      centroid: 0.45 + 0.2 * med,
      onset,
      beatConfidence: 0.2,
    };
  }

  dispose(): void {
    this.disposed = true;
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
  }
}

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

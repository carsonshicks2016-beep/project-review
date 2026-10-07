/**
 * F4 live human drive — browser keyboard → control room at 30 Hz.
 *
 * POST /api/drive → WS /api/drive/{id}/stream
 * Sends [steer, throttle, brake, handbrake]; receives F3-style replay packets
 * (header → frame + sense → end / stream_end).
 *
 * Keys: WASD / arrows · Space handbrake · R reset · P save replay
 *
 * Port-F owns packages/sim/rallyai/control/drive.py; this client is ready when
 * that API lands. RallyAI2 actions stay 4-dim — Space is handbrake, not brake.
 */

import type { Frame, Stage } from "../replay";
import { senseFromPacket, type SenseDebug } from "./sensors";
import { DEFAULT_CONTROL_BASE } from "./types";

export type LiveHandlers = {
  onStage: (stage: Stage) => void;
  onFrame: (frame: Frame) => void;
  onSense?: (sense: SenseDebug | null) => void;
  onEnd?: (info: Record<string, unknown>) => void;
  onStatus?: (msg: string) => void;
  onSaved?: (name: string, path?: string) => void;
  onConnection?: (connected: boolean) => void;
};

const CONTROL_HZ = 30;
const CONTROL_MS = 1000 / CONTROL_HZ;

function toWsBase(httpBase: string): string {
  if (!httpBase) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}`;
  }
  if (httpBase.startsWith("https://")) return "wss://" + httpBase.slice("https://".length);
  if (httpBase.startsWith("http://")) return "ws://" + httpBase.slice("http://".length);
  return httpBase;
}

export type StartDriveOpts = {
  seed?: number;
  tier?: number;
  procedural?: boolean;
  stage?: string | null;
  max_steps?: number;
  loop?: boolean;
  save_replay?: boolean;
  replay_name?: string | null;
  baseUrl?: string;
};

export class LiveClient {
  private ws: WebSocket | null = null;
  private keys = {
    up: false,
    down: false,
    left: false,
    right: false,
    handbrake: false,
  };
  private handlers: LiveHandlers;
  private sendTimer: number | null = null;
  private keysBound = false;
  private baseUrl = DEFAULT_CONTROL_BASE;
  private sessionId: string | null = null;
  private cursor = 0;

  constructor(handlers: LiveHandlers) {
    this.handlers = handlers;
  }

  get connected(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  get activeSessionId(): string | null {
    return this.sessionId;
  }

  setBaseUrl(url: string): void {
    this.baseUrl = url.trim().replace(/\/+$/, "");
  }

  /** Start a drive session and open the duplex stream. */
  async connect(opts: StartDriveOpts = {}): Promise<void> {
    this.disconnect();
    if (opts.baseUrl) this.setBaseUrl(opts.baseUrl);

    this.handlers.onStatus?.("Starting human-drive session…");
    const res = await fetch(this.url("/api/drive"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        seed: opts.seed ?? 0,
        tier: opts.tier ?? 0,
        procedural: opts.procedural ?? true,
        stage: opts.stage ?? null,
        max_steps: opts.max_steps ?? 4500,
        loop: opts.loop ?? true,
        save_replay: opts.save_replay ?? true,
        replay_name: opts.replay_name ?? null,
      }),
    });
    if (!res.ok) {
      const msg = await readErr(res);
      this.handlers.onStatus?.(msg);
      this.handlers.onConnection?.(false);
      throw new Error(msg);
    }
    const session = (await res.json()) as {
      session_id: string;
      state: string;
      error?: string | null;
    };
    if (session.state === "error") {
      const msg = session.error || "drive session error";
      this.handlers.onStatus?.(msg);
      this.handlers.onConnection?.(false);
      throw new Error(msg);
    }
    this.sessionId = session.session_id;
    this.cursor = 0;
    this.openWs();
  }

  disconnect(): void {
    if (this.sendTimer) window.clearInterval(this.sendTimer);
    this.sendTimer = null;
    const id = this.sessionId;
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      if (
        this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING
      ) {
        try {
          this.ws.send(JSON.stringify({ type: "stop" }));
        } catch {
          /* ignore */
        }
        this.ws.close();
      }
      this.ws = null;
    }
    if (id) {
      void fetch(this.url(`/api/drive/${encodeURIComponent(id)}/stop`), {
        method: "POST",
      }).catch(() => undefined);
    }
    this.sessionId = null;
    this.handlers.onConnection?.(false);
  }

  save(name: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "save", name }));
  }

  bindKeys(): void {
    if (this.keysBound) return;
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    this.keysBound = true;
  }

  unbindKeys(): void {
    if (!this.keysBound) return;
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    this.keysBound = false;
    this.keys = {
      up: false,
      down: false,
      left: false,
      right: false,
      handbrake: false,
    };
  }

  private url(path: string): string {
    const base = this.baseUrl;
    if (!base) return path;
    return `${base}${path}`;
  }

  private openWs(): void {
    if (!this.sessionId) return;
    const wsBase = toWsBase(this.baseUrl);
    const url = `${wsBase}/api/drive/${encodeURIComponent(this.sessionId)}/stream?since=${this.cursor}`;
    this.handlers.onStatus?.(`Connecting ${url}…`);
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen = () => {
      this.handlers.onStatus?.(
        "Connected — WASD/arrows drive · Space handbrake · R reset · P save",
      );
      this.handlers.onConnection?.(true);
      if (this.sendTimer) window.clearInterval(this.sendTimer);
      this.sendTimer = window.setInterval(() => this.sendInput(), CONTROL_MS);
    };
    ws.onclose = () => {
      this.handlers.onStatus?.("Disconnected from drive session");
      this.handlers.onConnection?.(false);
      if (this.sendTimer) window.clearInterval(this.sendTimer);
      this.sendTimer = null;
    };
    ws.onerror = () => {
      this.handlers.onStatus?.(
        "WebSocket error — is rallyai.control running on :8765?",
      );
      this.handlers.onConnection?.(false);
    };
    ws.onmessage = (ev) => {
      try {
        const packet = JSON.parse(ev.data as string) as Record<string, unknown>;
        this.handlePacket(packet);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        this.handlers.onStatus?.(message);
      }
    };
  }

  private handlePacket(packet: Record<string, unknown>): void {
    const type = packet.type as string | undefined;
    if (type === "header") {
      const stage = packet.stage as Stage | undefined;
      if (stage) this.handlers.onStage(stage);
      return;
    }
    if (type === "frame") {
      const i = typeof packet.i === "number" ? packet.i : this.cursor;
      this.cursor = Math.max(this.cursor, i + 1);
      const frame = packet.frame as Frame;
      const sense = senseFromPacket(
        packet as { sense?: unknown; frame?: { sense?: unknown } },
      );
      this.handlers.onFrame(frame);
      this.handlers.onSense?.(sense);
      return;
    }
    if (type === "end") {
      this.handlers.onEnd?.(packet);
      const term = String(packet.termination ?? "end");
      const progress = typeof packet.progress === "number" ? packet.progress : 0;
      this.handlers.onStatus?.(
        `Episode end: ${term} (${(progress * 100).toFixed(0)}%) — press R to reset`,
      );
      return;
    }
    if (type === "saved") {
      const name = String(packet.name ?? "replay.json");
      const path = typeof packet.path === "string" ? packet.path : undefined;
      this.handlers.onStatus?.(path ? `Saved ${name} → ${path}` : `Saved ${name}`);
      this.handlers.onSaved?.(name, path);
      return;
    }
    if (type === "stream_end") {
      this.handlers.onStatus?.("Drive session ended");
      this.handlers.onConnection?.(false);
      return;
    }
    if (type === "error") {
      this.handlers.onStatus?.(String(packet.message ?? "drive error"));
      return;
    }
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.repeat) return;
    switch (e.code) {
      case "KeyW":
      case "ArrowUp":
        this.keys.up = true;
        e.preventDefault();
        break;
      case "KeyS":
      case "ArrowDown":
        this.keys.down = true;
        e.preventDefault();
        break;
      case "KeyA":
      case "ArrowLeft":
        this.keys.left = true;
        e.preventDefault();
        break;
      case "KeyD":
      case "ArrowRight":
        this.keys.right = true;
        e.preventDefault();
        break;
      case "Space":
        this.keys.handbrake = true;
        e.preventDefault();
        break;
      case "KeyR":
        this.ws?.send(JSON.stringify({ type: "reset" }));
        e.preventDefault();
        break;
      case "KeyP":
        this.save(`human_${Date.now()}.json`);
        e.preventDefault();
        break;
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    switch (e.code) {
      case "KeyW":
      case "ArrowUp":
        this.keys.up = false;
        break;
      case "KeyS":
      case "ArrowDown":
        this.keys.down = false;
        break;
      case "KeyA":
      case "ArrowLeft":
        this.keys.left = false;
        break;
      case "KeyD":
      case "ArrowRight":
        this.keys.right = false;
        break;
      case "Space":
        this.keys.handbrake = false;
        break;
    }
  };

  private sendInput(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const steer = (this.keys.left ? -1 : 0) + (this.keys.right ? 1 : 0);
    const throttle = this.keys.up ? 1 : 0;
    const brake = this.keys.down ? 1 : 0;
    const handbrake = this.keys.handbrake ? 1 : 0;
    this.ws.send(
      JSON.stringify({
        type: "input",
        steer,
        throttle,
        brake,
        handbrake,
        action: [steer, throttle, brake, handbrake],
      }),
    );
  }
}

async function readErr(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string; message?: string };
    return body.detail || body.message || `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

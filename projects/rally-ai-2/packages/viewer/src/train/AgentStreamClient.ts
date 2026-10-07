/**
 * F3 live agent stream client — a replay that has not finished being written.
 *
 * POST /api/live → WS /api/live/{id}/stream?since=N
 * Packets: header → frame (+ sense) → end / stream_end
 *
 * Port-F owns the server; this client matches the planned RallyAI2 contract
 * (84-dim Observation debug sidecar, 4-dim actions, evo_rally stage docs).
 */

import type { Frame, Stage } from "../replay";
import { senseFromPacket, type SenseDebug } from "./sensors";
import { DEFAULT_CONTROL_BASE, TRAIN_BASE_URL_KEY } from "./types";

export type LiveSessionState =
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "error";

export interface LiveSessionDetail {
  session_id: string;
  state: LiveSessionState;
  error?: string | null;
  source?: string;
  checkpoint?: string | null;
  seed?: number;
  tier?: number;
  procedural?: boolean;
  frames?: number;
  steps?: number;
  termination?: string | null;
  stream?: string;
  dt?: number;
  hz?: number;
}

export interface StartLiveOpts {
  seed?: number;
  tier?: number;
  procedural?: boolean;
  stage?: string | null;
  checkpoint?: string | null;
  max_steps?: number;
  loop?: boolean;
}

export type AgentStreamHandlers = {
  onHeader?: (header: {
    stage: Stage;
    stage_id?: string;
    source?: string;
    meta?: Record<string, unknown>;
    session_id?: string;
  }) => void;
  onFrame?: (frame: Frame, sense: SenseDebug | null, i: number) => void;
  onEnd?: (info: Record<string, unknown>) => void;
  onStreamEnd?: (info: Record<string, unknown>) => void;
  onSession?: (session: LiveSessionDetail) => void;
  onError?: (message: string) => void;
  onConnection?: (connected: boolean) => void;
};

function normalizeBase(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

function toWsBase(httpBase: string): string {
  if (!httpBase) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}`;
  }
  if (httpBase.startsWith("https://")) return "wss://" + httpBase.slice("https://".length);
  if (httpBase.startsWith("http://")) return "ws://" + httpBase.slice("http://".length);
  return httpBase;
}

function loadBaseUrl(): string {
  try {
    const v = localStorage.getItem(TRAIN_BASE_URL_KEY);
    if (v != null) return normalizeBase(v);
  } catch {
    /* ignore */
  }
  return DEFAULT_CONTROL_BASE;
}

export class AgentStreamClient {
  private handlers: AgentStreamHandlers;
  private baseUrl = DEFAULT_CONTROL_BASE;
  private ws: WebSocket | null = null;
  private pollTimer: number | null = null;
  private sessionId: string | null = null;
  private cursor = 0;
  private streaming = false;
  private _session: LiveSessionDetail | null = null;

  constructor(handlers: AgentStreamHandlers = {}) {
    this.handlers = handlers;
    this.baseUrl = loadBaseUrl();
  }

  get session(): LiveSessionDetail | null {
    return this._session;
  }

  get activeSessionId(): string | null {
    return this.sessionId;
  }

  get connected(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  setBaseUrl(url: string): void {
    this.baseUrl = normalizeBase(url);
  }

  private url(path: string): string {
    const base = this.baseUrl;
    if (!base) return path;
    return `${base}${path}`;
  }

  async listSessions(): Promise<LiveSessionDetail[]> {
    const res = await fetch(this.url("/api/live"));
    if (!res.ok) throw new Error(await readErr(res));
    const data = (await res.json()) as { sessions: LiveSessionDetail[] };
    return data.sessions ?? [];
  }

  async getSession(sessionId: string): Promise<LiveSessionDetail> {
    const res = await fetch(this.url(`/api/live/${encodeURIComponent(sessionId)}`));
    if (!res.ok) throw new Error(await readErr(res));
    this._session = (await res.json()) as LiveSessionDetail;
    this.handlers.onSession?.(this._session);
    return this._session;
  }

  async start(opts: StartLiveOpts = {}): Promise<LiveSessionDetail> {
    const res = await fetch(this.url("/api/live"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        seed: opts.seed ?? 7,
        tier: opts.tier ?? 0,
        procedural: opts.procedural ?? true,
        stage: opts.stage ?? null,
        checkpoint: opts.checkpoint ?? null,
        max_steps: opts.max_steps ?? 4500,
        loop: opts.loop ?? true,
      }),
    });
    if (!res.ok) throw new Error(await readErr(res));
    this._session = (await res.json()) as LiveSessionDetail;
    this.sessionId = this._session.session_id;
    this.cursor = 0;
    this.handlers.onSession?.(this._session);
    return this._session;
  }

  async stop(): Promise<LiveSessionDetail | null> {
    const id = this.sessionId;
    this.disconnect();
    if (!id) return null;
    const res = await fetch(this.url(`/api/live/${encodeURIComponent(id)}/stop`), {
      method: "POST",
    });
    if (!res.ok) throw new Error(await readErr(res));
    this._session = (await res.json()) as LiveSessionDetail;
    this.handlers.onSession?.(this._session);
    return this._session;
  }

  /** Open the WS stream (polling fallback if the socket drops). */
  connect(sessionId: string, since = 0): void {
    this.disconnect();
    this.sessionId = sessionId;
    this.cursor = Math.max(0, since);
    this.streaming = true;
    this.openWs();
  }

  disconnect(): void {
    this.streaming = false;
    if (this.pollTimer != null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      if (
        this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING
      ) {
        this.ws.close();
      }
      this.ws = null;
    }
    this.handlers.onConnection?.(false);
  }

  private openWs(): void {
    if (!this.sessionId || !this.streaming) return;
    const wsBase = toWsBase(this.baseUrl);
    const url = `${wsBase}/api/live/${encodeURIComponent(this.sessionId)}/stream?since=${this.cursor}`;
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen = () => {
      this.handlers.onConnection?.(true);
    };
    ws.onmessage = (ev) => {
      try {
        const packet = JSON.parse(ev.data as string) as Record<string, unknown>;
        this.handlePacket(packet);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        this.handlers.onError?.(message);
      }
    };
    ws.onerror = () => {
      this.handlers.onError?.("live stream WebSocket error");
    };
    ws.onclose = () => {
      this.handlers.onConnection?.(false);
      if (this.streaming && this.sessionId) {
        this.startPollFallback();
      }
    };
  }

  private startPollFallback(): void {
    if (this.pollTimer != null) return;
    this.pollTimer = window.setInterval(() => {
      void this.pollOnce();
    }, 100);
    void this.pollOnce();
  }

  private async pollOnce(): Promise<void> {
    if (!this.streaming || !this.sessionId) return;
    try {
      const res = await fetch(
        this.url(
          `/api/live/${encodeURIComponent(this.sessionId)}/frames?since=${this.cursor}`,
        ),
      );
      if (!res.ok) throw new Error(await readErr(res));
      const batch = (await res.json()) as {
        packets: Record<string, unknown>[];
        next: number;
        state: LiveSessionState;
        termination?: string | null;
        header?: Record<string, unknown> | null;
      };
      if (batch.header && batch.header.type === "header") {
        this.handlePacket(batch.header);
      }
      for (const packet of batch.packets) {
        this.handlePacket(packet);
      }
      this.cursor = batch.next;
      if (
        (batch.state === "stopped" || batch.state === "error") &&
        batch.packets.length === 0
      ) {
        this.streaming = false;
        if (this.pollTimer != null) {
          window.clearInterval(this.pollTimer);
          this.pollTimer = null;
        }
        this.handlers.onStreamEnd?.({
          type: "stream_end",
          session_id: this.sessionId,
          state: batch.state,
          termination: batch.termination,
          next: this.cursor,
        });
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.handlers.onError?.(message);
    }
  }

  private handlePacket(packet: Record<string, unknown>): void {
    const type = packet.type as string | undefined;
    if (type === "header") {
      const stage = packet.stage as Stage | undefined;
      if (stage) {
        const header: {
          stage: Stage;
          stage_id?: string;
          source?: string;
          meta?: Record<string, unknown>;
          session_id?: string;
        } = { stage };
        if (typeof packet.stage_id === "string") header.stage_id = packet.stage_id;
        if (typeof packet.source === "string") header.source = packet.source;
        if (packet.meta && typeof packet.meta === "object") {
          header.meta = packet.meta as Record<string, unknown>;
        }
        if (typeof packet.session_id === "string") {
          header.session_id = packet.session_id;
        }
        this.handlers.onHeader?.(header);
      }
      return;
    }
    if (type === "frame") {
      const i = typeof packet.i === "number" ? packet.i : this.cursor;
      this.cursor = Math.max(this.cursor, i + 1);
      const frame = packet.frame as Frame;
      const sense = senseFromPacket(
        packet as { sense?: unknown; frame?: { sense?: unknown } },
      );
      this.handlers.onFrame?.(frame, sense, i);
      return;
    }
    if (type === "end") {
      this.handlers.onEnd?.(packet);
      return;
    }
    if (type === "stream_end") {
      this.streaming = false;
      this.handlers.onStreamEnd?.(packet);
      this.disconnect();
      return;
    }
    if (type === "error") {
      this.handlers.onError?.(String(packet.message ?? "live stream error"));
      return;
    }
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

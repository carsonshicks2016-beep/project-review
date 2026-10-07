import {
  DEFAULT_CONTROL_BASE,
  TRAIN_BASE_URL_KEY,
  TRAIN_SESSION_KEY,
  type MetricsBatch,
  type MetricsLine,
  type RunConfig,
  type RunDetail,
  type TrainSession,
} from "./types";

export type TrainClientHandlers = {
  onMetric?: (line: MetricsLine) => void;
  onRun?: (run: RunDetail) => void;
  onStreamEnd?: (run: RunDetail | null, next: number) => void;
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

/**
 * F1/F2 control-room client.
 *
 * POST/GET /api/runs · GET /api/runs/{id}/metrics?since=N · WS .../stream?since=N
 * Falls back to metrics polling when the WebSocket drops.
 */
export class TrainClient {
  private handlers: TrainClientHandlers;
  private baseUrl = DEFAULT_CONTROL_BASE;
  private ws: WebSocket | null = null;
  private pollTimer: number | null = null;
  private cursor = 0;
  private runId: string | null = null;
  private streaming = false;
  private _run: RunDetail | null = null;

  constructor(handlers: TrainClientHandlers = {}) {
    this.handlers = handlers;
    this.baseUrl = loadBaseUrl();
  }

  get run(): RunDetail | null {
    return this._run;
  }

  get since(): number {
    return this.cursor;
  }

  get activeRunId(): string | null {
    return this.runId;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  setBaseUrl(url: string): void {
    this.baseUrl = normalizeBase(url);
    try {
      localStorage.setItem(TRAIN_BASE_URL_KEY, this.baseUrl);
    } catch {
      /* ignore */
    }
  }

  private url(path: string): string {
    const base = this.baseUrl;
    if (!base) return path;
    return `${base}${path}`;
  }

  async health(): Promise<{ ok: boolean; service?: string }> {
    const res = await fetch(this.url("/api/health"));
    if (!res.ok) throw new Error(`health ${res.status}`);
    return (await res.json()) as { ok: boolean; service?: string };
  }

  async listRuns(): Promise<RunDetail[]> {
    const res = await fetch(this.url("/api/runs"));
    if (!res.ok) throw new Error(await readErr(res));
    const data = (await res.json()) as { runs: RunDetail[] };
    return data.runs ?? [];
  }

  async getRun(runId: string): Promise<RunDetail> {
    const res = await fetch(this.url(`/api/runs/${encodeURIComponent(runId)}`));
    if (!res.ok) throw new Error(await readErr(res));
    this._run = (await res.json()) as RunDetail;
    this.handlers.onRun?.(this._run);
    return this._run;
  }

  async startRun(config: RunConfig): Promise<RunDetail> {
    const res = await fetch(this.url("/api/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stage: config.stage,
        workers: config.workers,
        timesteps: config.timesteps,
        tier: config.tier,
      }),
    });
    if (!res.ok) throw new Error(await readErr(res));
    this._run = (await res.json()) as RunDetail;
    this.runId = this._run.run_id;
    this.cursor = 0;
    saveSession({ runId: this.runId, since: 0 });
    this.handlers.onRun?.(this._run);
    return this._run;
  }

  async stopRun(runId?: string): Promise<RunDetail> {
    const id = runId ?? this.runId;
    if (!id) throw new Error("no active run");
    const res = await fetch(this.url(`/api/runs/${encodeURIComponent(id)}/stop`), {
      method: "POST",
    });
    if (!res.ok) throw new Error(await readErr(res));
    this._run = (await res.json()) as RunDetail;
    this.handlers.onRun?.(this._run);
    return this._run;
  }

  async readMetrics(runId: string, since: number, limit = 500): Promise<MetricsBatch> {
    const q = new URLSearchParams({
      since: String(since),
      limit: String(limit),
    });
    const res = await fetch(
      this.url(`/api/runs/${encodeURIComponent(runId)}/metrics?${q}`),
    );
    if (!res.ok) throw new Error(await readErr(res));
    return (await res.json()) as MetricsBatch;
  }

  /**
   * Attach to a run and stream metrics from ``since``.
   * Prefers WebSocket; falls back to GET polling with the same cursor.
   */
  connect(runId: string, since = 0): void {
    this.disconnect();
    this.runId = runId;
    this.cursor = Math.max(0, since);
    this.streaming = true;
    saveSession({ runId, since: this.cursor });
    this.openWebSocket();
  }

  disconnect(): void {
    this.streaming = false;
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
    if (this.pollTimer != null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.handlers.onConnection?.(false);
  }

  private openWebSocket(): void {
    if (!this.runId || !this.streaming) return;
    const wsBase = toWsBase(this.baseUrl);
    const url = `${wsBase}/api/runs/${encodeURIComponent(this.runId)}/stream?since=${this.cursor}`;
    try {
      this.ws = new WebSocket(url);
    } catch {
      this.startPolling();
      return;
    }

    this.ws.onopen = () => {
      this.handlers.onConnection?.(true);
    };
    this.ws.onmessage = (ev) => {
      let msg: MetricsLine;
      try {
        msg = JSON.parse(ev.data as string) as MetricsLine;
      } catch {
        return;
      }
      this.handleStreamMessage(msg);
    };
    this.ws.onerror = () => {
      /* onclose handles fallback */
    };
    this.ws.onclose = () => {
      this.ws = null;
      this.handlers.onConnection?.(false);
      if (!this.streaming || !this.runId) return;
      this.startPolling();
    };
  }

  private startPolling(): void {
    if (this.pollTimer != null || !this.runId || !this.streaming) return;
    this.handlers.onConnection?.(true);
    const tick = async () => {
      if (!this.runId || !this.streaming) return;
      try {
        const batch = await this.readMetrics(this.runId, this.cursor);
        for (const line of batch.lines) {
          this.emitMetric(line);
        }
        this.cursor = batch.next;
        saveSession({ runId: this.runId, since: this.cursor });

        const run = await this.getRun(this.runId);
        if (
          (run.state === "stopped" || run.state === "error") &&
          batch.lines.length === 0
        ) {
          this.streaming = false;
          if (this.pollTimer != null) {
            window.clearInterval(this.pollTimer);
            this.pollTimer = null;
          }
          this.handlers.onStreamEnd?.(run, this.cursor);
          this.handlers.onConnection?.(false);
        }
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        this.handlers.onError?.(message);
      }
    };
    void tick();
    this.pollTimer = window.setInterval(() => void tick(), 500);
  }

  private handleStreamMessage(msg: MetricsLine): void {
    if (msg.type === "error") {
      this.handlers.onError?.(msg.message ?? "stream error");
      return;
    }
    if (msg.type === "stream_end") {
      this.cursor = typeof msg.next === "number" ? msg.next : this.cursor;
      if (this.runId) saveSession({ runId: this.runId, since: this.cursor });
      this.streaming = false;
      void this.getRun(this.runId!).then((run) => {
        this.handlers.onStreamEnd?.(run, this.cursor);
      });
      this.handlers.onConnection?.(false);
      return;
    }
    this.emitMetric(msg);
    this.cursor += 1;
    if (this.runId) saveSession({ runId: this.runId, since: this.cursor });
  }

  private emitMetric(line: MetricsLine): void {
    if (line.kind == null && line.type != null) return;
    this.handlers.onMetric?.(line);
  }
}

export function loadBaseUrl(): string {
  try {
    const v = localStorage.getItem(TRAIN_BASE_URL_KEY);
    if (v != null) return normalizeBase(v);
  } catch {
    /* ignore */
  }
  return DEFAULT_CONTROL_BASE;
}

export function loadSession(): TrainSession | null {
  try {
    const raw = localStorage.getItem(TRAIN_SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as TrainSession;
    if (!parsed?.runId || typeof parsed.since !== "number") return null;
    return { runId: parsed.runId, since: Math.max(0, parsed.since) };
  } catch {
    return null;
  }
}

export function saveSession(session: TrainSession): void {
  try {
    localStorage.setItem(TRAIN_SESSION_KEY, JSON.stringify(session));
  } catch {
    /* ignore */
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(TRAIN_SESSION_KEY);
  } catch {
    /* ignore */
  }
}

async function readErr(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: string | { msg?: string } };
    if (typeof j.detail === "string") return j.detail;
    if (j.detail && typeof j.detail === "object" && j.detail.msg) return j.detail.msg;
    return res.statusText || `HTTP ${res.status}`;
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

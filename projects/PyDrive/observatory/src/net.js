// Session + telemetry transport for fable-observatory-v1.
// REST creates the session; one WS owns frames; controls are JSON text.
// The browser never invents physics, policy actions, or checkpoint paths.

export const PROTOCOL = "fable-observatory-v1";

function wsUrl(path) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}`;
}

export function launchParams(search = location.search) {
  const q = new URLSearchParams(search);
  return {
    // 919 Evo is the active program car; 787b stays reachable via ?edition=787b.
    edition: q.get("edition") || "919",
    // Car query is decorative / legacy; the registry edition wins server-side.
    car: q.get("car") || "",
    checkpoint: q.get("checkpoint") || "active-best",
    mode: q.get("mode") === "follow-active-best" ? "follow-active-best" : "replay",
    seed: Number.parseInt(q.get("seed") || "7", 10) || 7,
    audio: q.get("audio") !== "0",
  };
}

export async function openSession(params = launchParams()) {
  const response = await fetch("/api/observatory/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({
      edition: params.edition,
      checkpoint: params.checkpoint,
      mode: params.mode,
      seed: params.seed,
      audio: params.audio,
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `session create failed (${response.status})`);
  }
  if (body.protocol !== PROTOCOL) {
    throw new Error(`unexpected protocol ${body.protocol}`);
  }
  return body;
}

export async function releaseSession(sessionId) {
  if (!sessionId) return;
  try {
    await fetch(`/api/observatory/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
      credentials: "same-origin",
    });
  } catch {
    /* pagehide / unload best-effort */
  }
}

export class TelemetryLink {
  constructor(session) {
    this.session = session;
    this.socket = null;
    this.hello = session.hello || null;
    this.onHello = null;
    this.onFrame = null;
    this.onEvent = null;
    this.onWarning = null;
    this.onError = null;
    this.onState = null;
    this._state = "connecting";
  }

  get state() {
    return this._state;
  }

  _setState(state) {
    this._state = state;
    this.onState?.(state);
  }

  connect() {
    const url = wsUrl(this.session.telemetry_socket);
    this.socket = new WebSocket(url);
    this._setState("connecting");
    this.socket.addEventListener("open", () => {
      // Stay "connecting" until hello or first frame — matches live gate.
    });
    this.socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") return;
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      const type = String(message.type || "");
      if (type === "hello") {
        this.hello = message;
        this.onHello?.(message);
        this._setState("live");
        return;
      }
      if (type === "frame") {
        if (this._state !== "live") this._setState("live");
        this.onFrame?.(message);
        this.send({ type: "frame_ack", sequence: message.sequence });
        return;
      }
      if (type === "warning") {
        this.onWarning?.(message);
        return;
      }
      if (type === "error") {
        this._setState("error");
        this.onError?.(message);
        return;
      }
      this.onEvent?.(message);
    });
    this.socket.addEventListener("close", () => {
      if (this._state !== "error") this._setState("error");
    });
    this.socket.addEventListener("error", () => {
      this._setState("error");
      this.onError?.({ code: "socket_error", message: "telemetry socket failed" });
    });
  }

  send(payload) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }

  control(type, extra = {}) {
    this.send({ type, ...extra });
  }

  /** Listener pose in Python/sim axes (x,y horizontal; z up). */
  sendListener(listener) {
    this.send({ type: "listener", ...listener });
  }

  disconnect() {
    try {
      this.send({ type: "disconnect" });
    } catch { /* ignore */ }
    try {
      this.socket?.close();
    } catch { /* ignore */ }
    this.socket = null;
  }

  get audioUrl() {
    return this.session.audio_socket ? wsUrl(this.session.audio_socket) : null;
  }
}

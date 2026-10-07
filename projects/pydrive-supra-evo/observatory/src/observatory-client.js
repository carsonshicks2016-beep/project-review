import { classifyMessage, encodeControl, socketUrl } from "./protocol.js";

const TOKEN_KEY = "fable-observatory-browser-token-v1";

export function browserToken(storage = globalThis.localStorage, cryptoLike = globalThis.crypto) {
  let value;
  try { value = storage?.getItem(TOKEN_KEY); } catch { value = ""; }
  if (!value) {
    value = cryptoLike?.randomUUID?.() || `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    try { storage?.setItem(TOKEN_KEY, value); } catch { /* private-mode fallback stays valid for this page */ }
  }
  return value;
}

export async function createObservatorySession(locationLike = globalThis.location, fetchImpl = globalThis.fetch) {
  const query = new URLSearchParams(locationLike?.search || "");
  const existing = query.get("session") || query.get("id");
  const token = browserToken();
  if (existing) return { sessionId: existing, browserToken: token };
  const payload = {
    edition: query.get("edition") || undefined,
    checkpoint: query.get("checkpoint") || undefined,
    mode: query.get("mode") || "follow-active-best",
  };
  Object.keys(payload).forEach((key) => payload[key] === undefined && delete payload[key]);
  const response = await fetchImpl("/api/observatory/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Observatory-Browser": token },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || body.message || `session request failed (${response.status})`);
  const sessionId = body.session_id || body.sessionId || body.id;
  if (!sessionId) throw new Error("session response did not include a session ID");
  return {
    sessionId,
    browserToken: token,
    trackHash: body.track_hash || body.hello?.track?.hash || "",
    wsUrl: body.telemetry_socket || body.telemetry_ws_url || body.ws_url || body.websocket_url || body.urls?.telemetry || body.urls?.websocket,
    audioUrl: body.audio_socket || body.audio_ws_url || body.audio_url || body.urls?.audio,
    hello: body.hello || body.session?.hello || (body.resolved_checkpoint ? {
      protocol: "fable-observatory-v1",
      checkpoint: body.resolved_checkpoint,
      track: { hash: body.track_hash || "" },
    } : undefined),
  };
}

export function releaseObservatorySession(session, fetchImpl = globalThis.fetch) {
  if (!session?.sessionId || !fetchImpl) return;
  fetchImpl(`/api/observatory/sessions/${encodeURIComponent(session.sessionId)}`, {
    method: "DELETE",
    headers: { "X-Observatory-Browser": session.browserToken || browserToken() },
    keepalive: true,
  }).catch(() => {});
}

export class ObservatoryClient extends EventTarget {
  constructor(session, { WebSocketImpl = globalThis.WebSocket } = {}) {
    super();
    this.sessionId = typeof session === "string" ? session : session.sessionId;
    this.wsUrl = typeof session === "object" ? session.wsUrl : "";
    this.WebSocketImpl = WebSocketImpl;
    this.socket = null;
    this.connected = false;
    this.closed = false;
    this.retry = 0;
    this.retryTimer = 0;
    this.pendingControls = [];
  }

  connect() {
    if (this.closed || !this.WebSocketImpl) {
      this.emit("status", { state: "offline", label: "NO WEBSOCKET" });
      return;
    }
    clearTimeout(this.retryTimer);
    this.emit("status", { state: "connecting", label: "CONNECTING" });
    const url = socketUrl(this.wsUrl || `/api/observatory/ws/${encodeURIComponent(this.sessionId)}`);
    const socket = new this.WebSocketImpl(url);
    this.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", () => {
      this.connected = true;
      this.retry = 0;
      for (const payload of this.pendingControls.splice(0)) socket.send(payload);
      this.emit("status", { state: "live", label: "LIVE" });
    });
    socket.addEventListener("message", (event) => {
      try {
        const message = classifyMessage(event.data);
        this.emit(message.type, message.value);
        if (message.type === "frame" && socket.readyState === this.WebSocketImpl.OPEN) {
          socket.send(JSON.stringify({ type: "frame_ack", sequence: message.value.seq }));
        }
      } catch (error) {
        this.emit("warning", { message: `Ignored malformed Observatory message: ${error.message}` });
      }
    });
    socket.addEventListener("error", () => this.emit("status", { state: "error", label: "LINK ERROR" }));
    socket.addEventListener("close", () => {
      this.connected = false;
      if (this.closed || socket !== this.socket) return;
      this.emit("status", { state: "offline", label: "RECONNECTING" });
      const delay = Math.min(10000, 500 * 2 ** this.retry++);
      this.retryTimer = setTimeout(() => this.connect(), delay);
    });
  }

  send(command, value) {
    if (!this.socket || this.socket.readyState !== this.WebSocketImpl.OPEN) {
      // Playback commands are user intent, so preserve them across a brief
      // reconnect. Listener/camera samples are high-rate ephemera and may be
      // dropped safely; the next sample replaces them within 250 ms.
      if (!["listener", "camera"].includes(command) && !this.closed) {
        this.pendingControls.push(encodeControl(command, value));
        if (this.pendingControls.length > 16) this.pendingControls.shift();
      }
      this.emit("warning", { message: "Playback control is unavailable while the session link is offline." });
      return false;
    }
    this.socket.send(encodeControl(command, value));
    return true;
  }

  close(timeoutMs = 750) {
    this.closed = true;
    this.connected = false;
    this.pendingControls.length = 0;
    clearTimeout(this.retryTimer);
    const socket = this.socket;
    this.socket = null;
    if (!socket || socket.readyState === 3) return Promise.resolve();
    return new Promise((resolve) => {
      let timer = 0;
      let closeTimer = 0;
      const finished = () => {
        clearTimeout(timer);
        clearTimeout(closeTimer);
        resolve();
      };
      socket.addEventListener("close", finished, { once: true });
      timer = setTimeout(finished, timeoutMs);
      const closeSocket = () => {
        try { socket.close(1000, "session released"); }
        catch { finished(); }
      };
      if (socket.readyState === this.WebSocketImpl.OPEN) {
        try {
          socket.send(JSON.stringify({ type: "disconnect" }));
          // The server stops its telemetry loop and owns the normal close.
          // Keep a fallback for a dead peer, but do not race queued frames by
          // starting a second close handshake almost immediately.
          closeTimer = setTimeout(closeSocket, Math.max(250, timeoutMs - 100));
        } catch {
          closeSocket();
        }
      } else closeSocket();
    });
  }

  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

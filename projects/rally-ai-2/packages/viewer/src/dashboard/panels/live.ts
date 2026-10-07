/**
 * Dashboard LIVE panel — F3 agent stream in the program hub.
 *
 * Embeds the existing TRAIN `LiveViewport` (same Three.js scene stack) and
 * drives it with `AgentStreamClient`. Section id: `live`.
 *
 * Checkpoint picker uses `api/checkpointsClient` when the control room exposes
 * `GET /api/runs/{id}/checkpoints`; otherwise path input + Latest / Reference pilot.
 */

import {
  discoverRunIds,
  fetchRunCheckpoints,
  indexBySlot,
  readLiveCheckpoint,
  setLiveCheckpoint,
  type CheckpointEntry,
} from "../api/checkpointsClient";
import { registerPanel } from "../registry";
import type { PanelContext, PanelMount, SectionId } from "../types";
import { btn, el, panelHeader } from "../ui";
import {
  AgentStreamClient,
  type LiveSessionDetail,
  type StartLiveOpts,
} from "../../train/AgentStreamClient";
import { LiveViewport } from "../../train/LiveViewport";
import type { SenseDebug } from "../../train/sensors";
import type { Frame, Stage } from "../../replay";
import "./live.css";

export const LIVE_SECTION_ID: SectionId = "live";

const LATEST_SENTINEL = "__latest__";
const PILOT_SENTINEL = "";
const PATH_SENTINEL = "__path__";
const DEFAULT_LATEST_PATH = "runs/foundation_01_latest.pt";

export type CheckpointOption = {
  /** Value sent as `checkpoint` (empty → reference pilot). */
  value: string;
  label: string;
};

/** Build picker options from control-room HoF listings; null if none available. */
export async function fetchCheckpointOptions(
  apiBase: string,
  signal?: AbortSignal,
): Promise<CheckpointOption[] | null> {
  try {
    const runIds = await discoverRunIds(apiBase, signal);
    if (!runIds.length) return null;

    const options: CheckpointOption[] = [];
    const seen = new Set<string>();

    // Prefer first few runs so the picker stays short.
    for (const runId of runIds.slice(0, 6)) {
      try {
        const { checkpoints } = await fetchRunCheckpoints(apiBase, runId, signal);
        pushCheckpointOptions(checkpoints, options, seen);
      } catch {
        /* run may lack the endpoint */
      }
    }

    return options.length ? options : null;
  } catch {
    return null;
  }
}

function pushCheckpointOptions(
  checkpoints: CheckpointEntry[],
  out: CheckpointOption[],
  seen: Set<string>,
): void {
  const bySlot = indexBySlot(checkpoints);
  for (const slot of ["latest", "best", "cleanest", "fastest", "furthest"] as const) {
    const ck = bySlot.get(slot);
    if (!ck || seen.has(ck.path)) continue;
    seen.add(ck.path);
    const steps =
      ck.timesteps != null ? ` · ${formatSteps(ck.timesteps)}` : "";
    out.push({
      value: ck.path,
      label: `${ck.run_id}/${slot}${steps}`,
    });
  }
  for (const ck of checkpoints) {
    if (seen.has(ck.path)) continue;
    seen.add(ck.path);
    out.push({
      value: ck.path,
      label: ck.slot ? `${ck.run_id}/${ck.slot}` : ck.name || ck.path,
    });
  }
}

function formatSteps(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

/**
 * Mount the LIVE section into the dashboard content region.
 * Returns a disposer invoked when the section is left.
 */
export const mountLive: PanelMount = (host, ctx) => {
  host.replaceChildren();
  host.classList.add("dash-live-host");

  const header = panelHeader("Live", "idle");
  const root = el("div", { className: "dash-live" });

  const seedInput = el("input", {
    attrs: {
      type: "number",
      min: 0,
      step: 1,
      value: "7",
      spellcheck: "false",
    },
  }) as HTMLInputElement;

  const tierInput = el("input", {
    attrs: {
      type: "number",
      min: 0,
      max: 32,
      step: 1,
      value: "0",
      spellcheck: "false",
    },
  }) as HTMLInputElement;

  const ckptSelect = el("select", {
    attrs: { "aria-label": "Checkpoint" },
  }) as HTMLSelectElement;

  const stored = readLiveCheckpoint() ?? "";
  const ckptPath = el("input", {
    className: "wide",
    attrs: {
      type: "text",
      spellcheck: "false",
      placeholder: "runs/…_latest.pt",
      value: stored,
    },
  }) as HTMLInputElement;

  const sensorsToggle = el("input", {
    attrs: { type: "checkbox" },
  }) as HTMLInputElement;
  sensorsToggle.checked = true;

  const noteEl = el("div", {
    className: "dash-live-note",
    text: "No checkpoint selected — live uses ReferencePilot (demo source) until a policy path is set.",
  });

  const errorEl = el("div", {
    className: "dash-live-error",
    attrs: { hidden: true },
  });

  const stateVal = el("b", { text: "idle", attrs: { "data-state": "idle" } });
  const sourceVal = el("b", { text: "—" });
  const sessionVal = el("b", { text: "—" });
  const paceVal = el("b", { text: "—" });
  const ckptVal = el("b", { text: "reference pilot" });

  const viewportHost = el("div", {
    className: "dash-live-viewport",
    id: "dash-live-viewport",
  });

  let sensorsOn = true;
  let fullscreenLocal = false;
  let active = true;
  let latestPath = DEFAULT_LATEST_PATH;
  const abort = new AbortController();

  const setError = (message: string | null): void => {
    if (!message) {
      errorEl.hidden = true;
      errorEl.textContent = "";
      return;
    }
    errorEl.hidden = false;
    errorEl.textContent = message;
  };

  const resolveSelectedCheckpoint = (): string | null => {
    const fromSelect = ckptSelect.value;
    if (fromSelect === PILOT_SENTINEL) return null;
    if (fromSelect === LATEST_SENTINEL) {
      return ckptPath.value.trim() || latestPath;
    }
    if (fromSelect === PATH_SENTINEL) {
      const typed = ckptPath.value.trim();
      return typed || null;
    }
    if (fromSelect) return fromSelect;
    const typed = ckptPath.value.trim();
    return typed || null;
  };

  const syncPilotNote = (): void => {
    const ckpt = resolveSelectedCheckpoint();
    noteEl.hidden = ckpt != null;
    ckptVal.textContent = ckpt ?? "reference pilot";
  };

  const fillCheckpointSelect = (options: CheckpointOption[] | null): void => {
    ckptSelect.replaceChildren();
    const add = (value: string, label: string): void => {
      ckptSelect.appendChild(el("option", { text: label, attrs: { value } }));
    };
    add(PILOT_SENTINEL, "Reference pilot");
    add(LATEST_SENTINEL, "Latest");
    if (options?.length) {
      for (const opt of options) add(opt.value, opt.label);
      const latestOpt = options.find((o) => /\/latest$/.test(o.label) || o.label.includes("/latest ·"));
      if (latestOpt) latestPath = latestOpt.value;
    } else {
      add(PATH_SENTINEL, "Custom path…");
    }

    const pref = readLiveCheckpoint();
    if (pref && options?.some((o) => o.value === pref)) {
      ckptSelect.value = pref;
      ckptPath.value = pref;
    } else if (pref) {
      ckptSelect.value = options?.length ? LATEST_SENTINEL : PATH_SENTINEL;
      ckptPath.value = pref;
    } else {
      ckptSelect.value = PILOT_SENTINEL;
    }
    syncPilotNote();
  };

  const startBtn = btn({
    label: "Start",
    onClick: () => void startLive(),
  });
  const stopBtn = btn({
    label: "Stop",
    variant: "danger",
    disabled: true,
    onClick: () => void stopLive(),
  });
  const fsBtn = btn({
    label: "Fullscreen",
    onClick: () => toggleFullscreen(),
  });
  const refreshCkptBtn = btn({
    label: "Refresh",
    onClick: () => void loadCheckpoints(),
  });

  const setLiveState = (
    state: string,
    opts: {
      sessionId?: string | null;
      pace?: string | null;
      source?: string | null;
    } = {},
  ): void => {
    stateVal.textContent = state;
    stateVal.dataset.state = state;
    header.setMeta(state);
    if (opts.sessionId !== undefined) {
      sessionVal.textContent = opts.sessionId
        ? truncate(opts.sessionId, 18)
        : "—";
      sessionVal.title = opts.sessionId ?? "";
    }
    if (opts.pace !== undefined) paceVal.textContent = opts.pace || "—";
    if (opts.source !== undefined) sourceVal.textContent = opts.source || "—";
    const live =
      state === "starting" ||
      state === "running" ||
      state === "stopping" ||
      state === "streaming";
    startBtn.disabled = live;
    stopBtn.disabled = !live && state !== "error";
  };

  const viewport = new LiveViewport(viewportHost);
  viewport.start();
  viewport.setEmpty(true);

  const stream = new AgentStreamClient({
    onHeader: (h: { stage: Stage; source?: string }) => {
      if (!active) return;
      viewport.setStage(h.stage);
      if (h.source) sourceVal.textContent = h.source;
    },
    onFrame: (frame: Frame, sense: SenseDebug | null) => {
      if (!active) return;
      viewport.applyFrame(frame, sensorsOn ? sense : null);
      if (!sensorsOn) viewport.clearSense();
      const pace =
        sense?.pace_note?.text ||
        sense?.pace_note?.dir ||
        frame.note ||
        "—";
      setLiveState("streaming", { pace: String(pace) });
    },
    onEnd: (info) => {
      setLiveState("ended", { pace: String(info.termination ?? "end") });
    },
    onStreamEnd: () => {
      viewport.clearSense();
      viewport.setEmpty(true);
      setLiveState("idle", { sessionId: null, pace: "—" });
    },
    onSession: (s: LiveSessionDetail) => {
      setLiveState(s.state, {
        sessionId: s.session_id,
        source: s.source ?? null,
      });
      if (s.checkpoint) {
        ckptVal.textContent = s.checkpoint;
        noteEl.hidden = true;
      } else if (s.source === "demo") {
        ckptVal.textContent = "reference pilot";
        noteEl.hidden = false;
      }
      if (s.error) setError(s.error);
    },
    onError: (message) => {
      setError(message);
      setLiveState("error");
    },
    onConnection: (on) => {
      if (on) setLiveState("streaming");
    },
  });
  stream.setBaseUrl(ctx.apiBase);

  async function startLive(): Promise<void> {
    try {
      setError(null);
      setLiveState("starting");
      viewport.clearSense();
      stream.setBaseUrl(ctx.apiBase);

      const checkpoint = resolveSelectedCheckpoint();
      if (checkpoint) setLiveCheckpoint(checkpoint);
      syncPilotNote();

      const opts: StartLiveOpts = {
        seed: Math.max(0, Number(seedInput.value) || 0),
        tier: Math.max(0, Number(tierInput.value) || 0),
        procedural: true,
        loop: true,
        checkpoint,
      };

      const session = await stream.start(opts);
      if (session.state === "error") {
        setLiveState("error", { sessionId: session.session_id });
        setError(session.error ?? "live session failed");
        return;
      }
      stream.connect(session.session_id, 0);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setLiveState("error");
      setError(message);
    }
  }

  async function stopLive(): Promise<void> {
    try {
      await stream.stop();
      viewport.clearSense();
      viewport.setEmpty(true);
      setLiveState("idle", { sessionId: null, pace: "—" });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
    }
  }

  function toggleFullscreen(): void {
    const node = viewportHost;
    if (document.fullscreenElement === node) {
      void document.exitFullscreen();
      return;
    }
    if (fullscreenLocal) {
      exitLocalFullscreen();
      return;
    }
    if (typeof node.requestFullscreen === "function") {
      void node.requestFullscreen().catch(() => enterLocalFullscreen());
      return;
    }
    enterLocalFullscreen();
  }

  function enterLocalFullscreen(): void {
    fullscreenLocal = true;
    viewportHost.classList.add("is-fs-local");
    fsBtn.textContent = "Exit full";
    requestAnimationFrame(() => viewport.resize());
  }

  function exitLocalFullscreen(): void {
    fullscreenLocal = false;
    viewportHost.classList.remove("is-fs-local");
    fsBtn.textContent = "Fullscreen";
    requestAnimationFrame(() => viewport.resize());
  }

  const onFsChange = (): void => {
    const on = document.fullscreenElement === viewportHost;
    fsBtn.textContent = on || fullscreenLocal ? "Exit full" : "Fullscreen";
    requestAnimationFrame(() => viewport.resize());
  };

  const onResize = (): void => viewport.resize();
  const ro = new ResizeObserver(() => viewport.resize());
  ro.observe(viewportHost);

  sensorsToggle.addEventListener("change", () => {
    sensorsOn = sensorsToggle.checked;
    if (!sensorsOn) viewport.clearSense();
  });
  ckptSelect.addEventListener("change", () => {
    if (
      ckptSelect.value === PATH_SENTINEL ||
      ckptSelect.value === LATEST_SENTINEL
    ) {
      if (ckptSelect.value === LATEST_SENTINEL && !ckptPath.value.trim()) {
        ckptPath.value = latestPath;
      }
      ckptPath.focus();
    } else if (ckptSelect.value && ckptSelect.value !== PILOT_SENTINEL) {
      ckptPath.value = ckptSelect.value;
    }
    syncPilotNote();
  });
  ckptPath.addEventListener("change", () => {
    const v = ckptPath.value.trim();
    if (v) setLiveCheckpoint(v);
    syncPilotNote();
  });

  async function loadCheckpoints(): Promise<void> {
    const options = await fetchCheckpointOptions(ctx.apiBase, abort.signal);
    if (!active) return;
    fillCheckpointSelect(options);
  }

  const toolbar = el(
    "div",
    { className: "dash-live-toolbar" },
    el("div", { className: "dash-live-field" }, el("label", { text: "Seed" }), seedInput),
    el("div", { className: "dash-live-field" }, el("label", { text: "Tier" }), tierInput),
    el(
      "div",
      { className: "dash-live-field" },
      el("label", { text: "Checkpoint" }),
      ckptSelect,
    ),
    el(
      "div",
      { className: "dash-live-field" },
      el("label", { text: "Path" }),
      ckptPath,
    ),
    el(
      "div",
      { className: "dash-live-actions" },
      startBtn,
      stopBtn,
      refreshCkptBtn,
      fsBtn,
      el(
        "label",
        { className: "dash-live-toggle" },
        sensorsToggle,
        document.createTextNode("Sensors"),
      ),
    ),
  );

  const status = el(
    "div",
    { className: "dash-live-status" },
    el("span", null, "State ", stateVal),
    el("span", null, "Source ", sourceVal),
    el("span", null, "Session ", sessionVal),
    el("span", null, "Pace ", paceVal),
    el("span", null, "Policy ", ckptVal),
  );

  root.append(header.root, toolbar, noteEl, errorEl, viewportHost, status);
  host.append(root);

  fillCheckpointSelect(null);
  void loadCheckpoints();
  window.addEventListener("resize", onResize);
  document.addEventListener("fullscreenchange", onFsChange);
  requestAnimationFrame(() => viewport.resize());

  void (async () => {
    try {
      const sessions = await stream.listSessions();
      const live =
        sessions.find((s) => s.state === "running" || s.state === "starting") ??
        null;
      if (!live || !active) return;
      setLiveState(live.state, {
        sessionId: live.session_id,
        source: live.source ?? null,
      });
      if (live.checkpoint) {
        ckptPath.value = live.checkpoint;
        ckptVal.textContent = live.checkpoint;
        noteEl.hidden = true;
      }
      stream.connect(live.session_id, 0);
    } catch {
      /* control room may be down */
    }
  })();

  return () => {
    active = false;
    abort.abort();
    window.removeEventListener("resize", onResize);
    document.removeEventListener("fullscreenchange", onFsChange);
    ro.disconnect();
    if (document.fullscreenElement === viewportHost) {
      void document.exitFullscreen();
    }
    exitLocalFullscreen();
    stream.disconnect();
    void stream.stop().catch(() => {
      /* ignore */
    });
    viewport.destroy();
    host.classList.remove("dash-live-host");
    host.replaceChildren();
  };
};

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

/** Thin register helper — importing this module also self-registers. */
export function registerLivePanel(
  register: (id: SectionId, mount: PanelMount) => void = registerPanel,
): void {
  register(LIVE_SECTION_ID, mountLive);
}

registerLivePanel();

/** Convenience for tests / manual mounts that already have a context. */
export function mountLivePanel(
  host: HTMLElement,
  ctx: PanelContext,
): () => void {
  const dispose = mountLive(host, ctx);
  return typeof dispose === "function" ? dispose : () => undefined;
}

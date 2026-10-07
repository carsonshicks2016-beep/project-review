/**
 * Dashboard DRIVE / HUMAN BASELINE panel — section `drive`.
 *
 * Two jobs, one panel:
 *
 * 1. **Record.** Launch an F4 human-drive session (`POST /api/drive` → WS
 *    `/api/drive/{id}/stream`) through the TRAIN `LiveClient` + `LiveViewport`
 *    stack, so a browser drive is the same env at the same 30 Hz control rate
 *    the agent trains at.
 * 2. **Account.** Show the held-out seed set (`seed % 10 == 7`), the current
 *    `docs/baselines/human/times.json` table, and what still has to happen
 *    before any of it is claimable.
 *
 * The panel never invents a baseline: an empty `seeds[]` renders as "not
 * claimable", never as a time of zero.
 */

import { registerPanel } from "../registry";
import type { PanelContext, PanelMount, SectionId } from "../types";
import { btn, el, panelHeader, stat, type StatHandle } from "../ui";
import { LiveClient } from "../../train/LiveClient";
import { LiveViewport } from "../../train/LiveViewport";
import type { SenseDebug } from "../../train/sensors";
import type { Frame, Stage } from "../../replay";
import "./drive.css";

export const DRIVE_SECTION_ID: SectionId = "drive";

/** Control rate shared by training, eval and every human recorder. */
const CONTROL_HZ = 30;

/** Held-out rule used until `times.json` states its own. */
const HELD_OUT_FALLBACK = { modulus: 10, remainder: 7 };

/** How many held-out seeds to offer as chips. */
const HELD_OUT_CHIPS = 12;

/** Ceiling on the in-tab frame mirror (~33 min at 30 Hz). */
const MAX_CAPTURE_FRAMES = 60_000;

/** Where the control room writes browser replays, unless health says otherwise. */
const DEFAULT_REPLAY_DIR = "docs/baselines/human/";

/* ------------------------------------------------------------------ types */

export interface BaselineAttempt {
  attempt?: number;
  time_s?: number;
  replay?: string;
  clean?: boolean;
  termination?: string;
}

export interface BaselineRow {
  seed?: number;
  tier?: number;
  best_time_s?: number;
  median_time_s?: number;
  attempts?: BaselineAttempt[];
}

/** Mirror of `docs/baselines/human/times.schema.json` — all fields defensive. */
export interface HumanBaseline {
  schema_version?: number;
  control_hz?: number;
  physics_version?: string;
  car?: string;
  held_out?: { modulus?: number; remainder?: number; note?: string };
  attempts_per_seed?: number;
  notes?: string;
  seeds?: BaselineRow[];
}

interface BaselineLoad {
  doc: HumanBaseline;
  source: string;
  /** True when the doc came from a copy that can lag the file on disk. */
  mirrored: boolean;
}

interface DriveHealth {
  ok?: boolean;
  drive?: { replay_dir?: string; controls?: string };
}

/** Candidate URLs for the aggregate table, best source first. */
export function baselineUrls(apiBase: string): string[] {
  return [
    `${apiBase}/api/baselines/human`,
    `${apiBase}/api/baselines/human/times.json`,
    "/dashboard/drive/times.json",
  ];
}

/** `true` when the seed belongs to the human/eval half of the split. */
export function isHeldOutSeed(
  seed: number,
  rule: { modulus: number; remainder: number } = HELD_OUT_FALLBACK,
): boolean {
  const m = Math.max(2, rule.modulus);
  return ((seed % m) + m) % m === rule.remainder;
}

/* ------------------------------------------------------------------ mount */

export const mountDrive: PanelMount = (host, ctx) => {
  host.replaceChildren();
  host.classList.add("dash-drive-host");

  const header = panelHeader("Drive / Human baseline", "checking…");
  const root = el("div", { className: "dash-drive" });

  /* ------------------------------------------------------------- state */

  let baseline: BaselineLoad | null = null;
  let health: DriveHealth | null = null;
  let viewport: LiveViewport | null = null;
  let driving = false;
  let dirty = false;
  let active = true;
  let paintTimer: number | null = null;
  const abort = new AbortController();

  /** Everything this tab saw on the wire for the current session. */
  const capture = {
    stage: null as Stage | null,
    frames: [] as Frame[],
    truncated: false,
    lastFrame: null as Frame | null,
    termination: null as string | null,
    sessionId: null as string | null,
    savedPaths: [] as string[],
  };

  /* ------------------------------------------------------------ controls */

  const seedInput = el("input", {
    attrs: { type: "number", min: 0, step: 1, value: "7", spellcheck: "false" },
  });
  const tierInput = el("input", {
    attrs: { type: "number", min: 0, max: 5, step: 1, value: "0", spellcheck: "false" },
  });
  const attemptInput = el("input", {
    attrs: { type: "number", min: 1, step: 1, value: "1", spellcheck: "false" },
  });

  const heldPill = el("span", { className: "dash-drive-pill", text: "—" });
  const replayNameEl = el("code", { className: "dash-drive-name", text: "—" });
  const healthPill = el("span", { className: "dash-drive-pill", text: "checking…" });
  const apiEl = el("code", { text: ctx.apiBase || "(proxied /api → 127.0.0.1:8765)" });

  const statusEl = el("p", {
    className: "dash-drive-status",
    text: "Idle. A session takes over WASD in this tab until you end it.",
  });

  const startBtn = btn({ label: "Drive", onClick: () => void startDrive() });
  const saveBtn = btn({
    label: "Save now",
    disabled: true,
    onClick: () => {
      client.save(replayName());
      setStatus(`Save requested as ${replayName()}`);
    },
  });
  const stopBtn = btn({
    label: "End drive",
    variant: "danger",
    disabled: true,
    onClick: () => void stopDrive(),
  });

  const stats: Record<string, StatHandle> = {
    state: stat({ label: "state", value: "idle" }),
    session: stat({ label: "session" }),
    frames: stat({ label: "frames", value: "0" }),
    time: stat({ label: "stage time" }),
    speed: stat({ label: "speed" }),
    distance: stat({ label: "distance" }),
    surface: stat({ label: "surface" }),
    termination: stat({ label: "termination" }),
  };

  const viewportHost = el("div", { className: "dash-drive-viewport" });

  /* --------------------------------------------------------- drive client */

  const client = new LiveClient({
    onStage: (stage) => {
      capture.stage = stage;
      capture.frames = [];
      capture.truncated = false;
      capture.termination = null;
      viewport?.setStage(stage);
      dirty = true;
    },
    onFrame: (frame) => {
      capture.lastFrame = frame;
      if (capture.frames.length < MAX_CAPTURE_FRAMES) capture.frames.push(frame);
      else capture.truncated = true;
      viewport?.applyFrame(frame, null);
      dirty = true;
    },
    onSense: (sense: SenseDebug | null) => {
      const frame = capture.lastFrame;
      if (frame) viewport?.applyFrame(frame, sense);
    },
    onEnd: (info) => {
      capture.termination = String(info.termination ?? "end");
      dirty = true;
    },
    onStatus: (msg) => setStatus(msg, /error|unreachable|failed/i.test(msg)),
    onSaved: (name, path) => {
      capture.savedPaths.push(path ?? name);
      renderExport();
    },
    onConnection: (on) => {
      driving = on;
      capture.sessionId = client.activeSessionId;
      syncControls();
      dirty = true;
      if (!on) {
        viewport?.clearSense();
        // Keep the last frame after a finished session — blanking it hides the
        // only visual proof that the drive happened. Show the empty overlay
        // only when we never received a stage.
        if (!capture.stage) viewport?.setEmpty(true);
      }
    },
  });
  client.setBaseUrl(ctx.apiBase);

  /* ------------------------------------------------------------- helpers */

  const seed = (): number => Math.max(0, Math.trunc(Number(seedInput.value) || 0));
  const tier = (): number =>
    Math.min(5, Math.max(0, Math.trunc(Number(tierInput.value) || 0)));
  const attempt = (): number =>
    Math.max(1, Math.trunc(Number(attemptInput.value) || 1));

  function heldOutRule(): { modulus: number; remainder: number } {
    const rule = baseline?.doc.held_out;
    return {
      modulus: Math.max(2, intOr(rule?.modulus, HELD_OUT_FALLBACK.modulus)),
      remainder: intOr(rule?.remainder, HELD_OUT_FALLBACK.remainder),
    };
  }

  const heldOut = (value: number): boolean => isHeldOutSeed(value, heldOutRule());

  const replayName = (): string =>
    `human_seed${seed()}_tier${tier()}_a${attempt()}.json`;

  function setStatus(message: string, isError = false): void {
    statusEl.textContent = message;
    statusEl.classList.toggle("bad", isError);
  }

  function syncControls(): void {
    startBtn.disabled = driving;
    stopBtn.disabled = !driving;
    saveBtn.disabled = !driving;
    downloadBtn.disabled = !capture.stage || capture.frames.length === 0;
    for (const input of [seedInput, tierInput, attemptInput]) {
      input.disabled = driving;
    }
  }

  /* ------------------------------------------------------------- renders */

  const chipsEl = el("div", { className: "dash-drive-chips" });
  const tableHost = el("div", { className: "dash-drive-table-host" });
  const metaEl = el("div", { className: "dash-drive-meta" });
  const sourceEl = el("span", { className: "dash-muted", text: "source: not loaded" });
  const notesEl = el("p", { className: "dash-drive-note", attrs: { hidden: true } });
  const cmdEl = el("pre", { className: "dash-code dash-drive-cmd" });
  const exportHost = el("div", { className: "dash-drive-export" });

  function renderSelection(): void {
    const rule = heldOutRule();
    const ok = heldOut(seed());
    heldPill.textContent = ok
      ? `held out · seed % ${rule.modulus} == ${rule.remainder}`
      : "TRAIN seed — not held out";
    heldPill.classList.toggle("good", ok);
    heldPill.classList.toggle("bad", !ok);
    heldPill.title = ok
      ? "Eval and the human baseline both draw from this set."
      : `Training draws seed % ${rule.modulus} != ${rule.remainder}. A time here is practice — the agent has trained on this stage.`;
    replayNameEl.textContent = replayName();
    renderChips();
    renderCommand();
  }

  function renderChips(): void {
    const { modulus, remainder } = heldOutRule();
    const recorded = recordedKeys();
    const current = seed();
    chipsEl.replaceChildren();
    for (let i = 0; i < HELD_OUT_CHIPS; i += 1) {
      const value = remainder + i * modulus;
      const done = recorded.has(`${value}:${tier()}`);
      const chip = el("button", {
        className: [
          "dash-drive-chip",
          done ? "done" : "",
          value === current ? "current" : "",
        ]
          .filter(Boolean)
          .join(" "),
        text: String(value),
        attrs: {
          type: "button",
          title: done
            ? `seed ${value} tier ${tier()} — has rows in times.json`
            : `seed ${value} tier ${tier()} — nothing recorded yet`,
        },
        on: {
          click: () => {
            if (driving) return;
            seedInput.value = String(value);
            renderSelection();
          },
        },
      });
      chipsEl.appendChild(chip);
    }
  }

  function recordedKeys(): Set<string> {
    const keys = new Set<string>();
    for (const row of baseline?.doc.seeds ?? []) {
      if (row.seed == null) continue;
      keys.add(`${row.seed}:${row.tier ?? 0}`);
    }
    return keys;
  }

  function renderClaim(): void {
    const rows = baseline?.doc.seeds ?? [];
    if (rows.length === 0) {
      header.setMeta("not claimable · 0 held-out seeds recorded");
      header.root.classList.add("is-bad");
      header.root.classList.remove("is-warn", "is-good");
      return;
    }
    const attempts = rows.reduce((n, row) => n + (row.attempts?.length ?? 0), 0);
    const target = intOr(baseline?.doc.attempts_per_seed, 5);
    const short = rows.filter((row) => (row.attempts?.length ?? 0) < target).length;
    header.setMeta(
      short > 0
        ? `${rows.length} seeds · ${attempts} attempts · ${short} below ${target}`
        : `${rows.length} seeds · ${attempts} attempts recorded`,
    );
    header.root.classList.remove("is-bad");
    header.root.classList.toggle("is-warn", short > 0);
    header.root.classList.toggle("is-good", short === 0);
  }

  function renderBaseline(): void {
    const doc = baseline?.doc;
    const rows = [...(doc?.seeds ?? [])].sort(
      (a, b) => (a.tier ?? 0) - (b.tier ?? 0) || (a.seed ?? 0) - (b.seed ?? 0),
    );

    metaEl.replaceChildren();
    if (doc) {
      const hz = intOr(doc.control_hz, CONTROL_HZ);
      const rule = heldOutRule();
      metaEl.append(
        metaItem("control", `${hz} Hz`, hz !== CONTROL_HZ),
        metaItem("physics", doc.physics_version ?? "—"),
        metaItem("car", doc.car ?? "—"),
        metaItem("attempts/seed", String(intOr(doc.attempts_per_seed, 5))),
        metaItem("held out", `seed % ${rule.modulus} == ${rule.remainder}`),
      );
    }

    sourceEl.textContent = baseline
      ? `source: ${baseline.source}${baseline.mirrored ? " · static mirror, may lag the file on disk" : ""}`
      : "source: not loaded";
    sourceEl.classList.toggle("bad", !baseline);

    notesEl.hidden = !doc?.notes;
    notesEl.textContent = doc?.notes ?? "";

    tableHost.replaceChildren();
    if (!doc) {
      tableHost.append(
        emptyBox(
          "times.json is not reachable from the browser. Read it on disk at docs/baselines/human/times.json, or load it with the file picker above.",
        ),
      );
      return;
    }
    if (rows.length === 0) {
      tableHost.append(
        emptyBox(
          "No rows. No human drives have been recorded yet — an empty table, not a time of zero. Eval scores time_vs_human as null until finished held-out replays land here.",
        ),
      );
      return;
    }

    const body = el("tbody");
    for (const row of rows) {
      const attempts = row.attempts ?? [];
      const clean = attempts.filter((a) => a.clean).length;
      const held = row.seed != null && heldOut(row.seed);
      const tr = el(
        "tr",
        attempts.length
          ? {
              attrs: {
                title: attempts
                  .map(
                    (a) =>
                      `#${a.attempt ?? "?"} ${fmtTime(a.time_s)} ${a.clean ? "clean" : "contact"} — ${a.replay ?? "?"}`,
                  )
                  .join("\n"),
              },
            }
          : null,
        cell(String(row.seed ?? "—")),
        cell(String(row.tier ?? 0)),
        cell(fmtTime(row.best_time_s), "num"),
        cell(fmtTime(row.median_time_s), "num"),
        cell(String(attempts.length), "num"),
        cell(`${clean}/${attempts.length}`, "num"),
        cell(held ? "yes" : "NO", held ? "" : "bad"),
      );
      body.appendChild(tr);
    }

    tableHost.append(
      el(
        "table",
        { className: "dash-drive-table" },
        el(
          "thead",
          null,
          el(
            "tr",
            null,
            ...["seed", "tier", "best", "median", "attempts", "clean", "held out"].map(
              (label) => el("th", { text: label }),
            ),
          ),
        ),
        body,
      ),
    );
  }

  function renderCommand(): void {
    cmdEl.textContent = [
      "cd packages/sim",
      `.venv/bin/python -m rallyai.env.drive --seed ${seed()} --tier ${tier()} --attempt ${attempt()}`,
    ].join("\n");
  }

  function renderExport(): void {
    exportHost.replaceChildren();
    const dir = health?.drive?.replay_dir ?? DEFAULT_REPLAY_DIR;

    if (capture.savedPaths.length === 0) {
      exportHost.append(
        emptyBox(
          `Nothing saved this session. The control room writes the replay into ${dir} when the session stops, or on Save now.`,
        ),
      );
    } else {
      const list = el("ul", { className: "dash-drive-paths" });
      for (const path of capture.savedPaths) {
        const copy = btn({ label: "copy", className: "dash-drive-mini" });
        copy.addEventListener("click", () => void copyText(path, copy));
        list.append(el("li", null, el("code", { text: path }), copy));
      }
      exportHost.append(list);
    }

    exportHost.append(
      noteLine(
        capture.truncated
          ? `This tab's mirror stopped at ${MAX_CAPTURE_FRAMES} frames; the server-side file is complete.`
          : `${capture.frames.length} frames mirrored in this tab. The server file is the canonical, contract-stamped replay — the download is a convenience copy and carries no hash.`,
      ),
    );
    syncControls();
  }

  function paint(): void {
    if (!dirty) return;
    dirty = false;
    const frame = capture.lastFrame;
    stats.state?.set(driving ? "driving" : "idle");
    stats.state?.root.classList.toggle("on", driving);
    stats.session?.set(capture.sessionId ? truncate(capture.sessionId, 18) : "—");
    if (stats.session) stats.session.root.title = capture.sessionId ?? "";
    stats.frames?.set(String(capture.frames.length));
    stats.time?.set(frame ? fmtTime(frame.t) : "—");
    stats.speed?.set(frame ? `${(frame.v * 3.6).toFixed(0)} km/h` : "—");
    stats.distance?.set(frame ? `${frame.s.toFixed(0)} m` : "—");
    stats.surface?.set(frame?.surf ?? "—");
    stats.termination?.set(capture.termination ?? "—");
  }

  /* ------------------------------------------------------------- actions */

  async function startDrive(): Promise<void> {
    if (driving) return;
    try {
      setStatus("Starting human-drive session…");
      capture.savedPaths = [];
      capture.frames = [];
      capture.lastFrame = null;
      capture.termination = null;
      capture.truncated = false;
      renderExport();

      if (!viewport) {
        viewport = new LiveViewport(viewportHost);
        viewport.start();
        viewport.setEmpty(true);
        // LiveViewport's placeholder is written for the agent stream.
        const placeholder = viewportHost.querySelector(".train-viewport-empty");
        if (placeholder) placeholder.textContent = "Human drive — you have the wheel";
        resizeObserver.observe(viewportHost);
      }
      viewportHost.classList.add("is-live");
      viewport.clearSense();

      client.setBaseUrl(ctx.apiBase);
      client.bindKeys();
      await client.connect({
        seed: seed(),
        tier: tier(),
        procedural: true,
        loop: true,
        save_replay: true,
        replay_name: replayName(),
        baseUrl: ctx.apiBase,
      });
      capture.sessionId = client.activeSessionId;
      requestAnimationFrame(() => viewport?.resize());
      if (!heldOut(seed())) {
        setStatus(
          `Driving seed ${seed()} — a TRAIN seed, so this run is practice, not a baseline.`,
          true,
        );
      }
    } catch (e) {
      client.unbindKeys();
      driving = false;
      // A session that never opened should not leave a dead viewport behind.
      teardownViewport();
      syncControls();
      setStatus(errText(e), true);
    }
  }

  function teardownViewport(): void {
    resizeObserver.disconnect();
    viewport?.destroy();
    viewport = null;
    viewportHost.classList.remove("is-live");
    viewportHost.replaceChildren();
  }

  async function stopDrive(): Promise<void> {
    const id = client.activeSessionId;
    client.unbindKeys();
    client.disconnect();
    driving = false;
    syncControls();
    dirty = true;
    paint();
    if (!id) return;

    setStatus("Stopping session — waiting for the replay to be written…");
    const detail = await pollReplayPath(id);
    if (!active) return;
    if (detail?.replay_path) {
      if (!capture.savedPaths.includes(detail.replay_path)) {
        capture.savedPaths.push(detail.replay_path);
      }
      capture.termination = detail.termination ?? capture.termination;
      renderExport();
      dirty = true;
      paint();
      setStatus(`Session stopped — replay written to ${detail.replay_path}`);
    } else {
      setStatus(
        "Session stopped. The control room reported no replay path — check its log.",
        true,
      );
    }
  }

  /** The F4 session writes its replay as it shuts down; give it a moment. */
  async function pollReplayPath(
    id: string,
  ): Promise<{ replay_path?: string | null; termination?: string | null } | null> {
    for (let i = 0; i < 6; i += 1) {
      await sleep(400);
      if (!active) return null;
      try {
        const res = await fetch(`${ctx.apiBase}/api/drive/${encodeURIComponent(id)}`, {
          signal: abort.signal,
        });
        if (!res.ok) continue;
        const detail = (await res.json()) as {
          replay_path?: string | null;
          termination?: string | null;
          state?: string;
        };
        if (detail.replay_path || detail.state === "error") return detail;
      } catch {
        return null;
      }
    }
    return null;
  }

  function downloadCapture(): void {
    if (!capture.stage || capture.frames.length === 0) return;
    const doc = {
      schema_version: 1,
      dt: 1 / CONTROL_HZ,
      source: "human",
      meta: {
        termination: capture.termination ?? "aborted",
        time_s: capture.lastFrame?.t ?? 0,
        car: baseline?.doc.car ?? "unknown",
        physics_version: baseline?.doc.physics_version ?? "unknown",
        seed: seed(),
        tier: tier(),
        attempt: attempt(),
        note: "browser mirror of the F4 stream — not contract-stamped",
      },
      frames: capture.frames,
      stage: capture.stage,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(doc)], { type: "application/json" }),
    );
    const a = el("a", { attrs: { href: url, download: replayName() } });
    a.click();
    URL.revokeObjectURL(url);
  }

  async function loadBaseline(): Promise<void> {
    for (const url of baselineUrls(ctx.apiBase)) {
      try {
        const res = await fetch(url, { signal: abort.signal });
        if (!res.ok) continue;
        const doc = (await res.json()) as HumanBaseline;
        if (!looksLikeBaseline(doc)) continue;
        baseline = { doc, source: url, mirrored: url.startsWith("/dashboard/") };
        break;
      } catch {
        /* try the next candidate */
      }
    }
    if (!active) return;
    renderClaim();
    renderBaseline();
    renderSelection();
  }

  async function loadHealth(): Promise<void> {
    try {
      const res = await fetch(`${ctx.apiBase}/api/health`, { signal: abort.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      health = (await res.json()) as DriveHealth;
      if (!active) return;
      healthPill.textContent = "control room up";
      healthPill.className = "dash-drive-pill good";
    } catch {
      if (!active) return;
      health = null;
      healthPill.textContent = "unreachable — start rallyai.control on :8765";
      healthPill.className = "dash-drive-pill bad";
    }
    renderExport();
  }

  function loadFile(file: File): void {
    void file
      .text()
      .then((text) => {
        const doc = JSON.parse(text) as HumanBaseline;
        if (!looksLikeBaseline(doc)) throw new Error("not a human-baseline times.json");
        baseline = { doc, source: `local file · ${file.name}`, mirrored: false };
        renderClaim();
        renderBaseline();
        renderSelection();
      })
      .catch((e: unknown) => {
        sourceEl.textContent = `source: ${file.name} rejected — ${errText(e)}`;
        sourceEl.classList.add("bad");
      });
  }

  /* ---------------------------------------------------------------- tree */

  const downloadBtn = btn({
    label: "Download this tab's capture",
    disabled: true,
    onClick: () => downloadCapture(),
  });
  const watchBtn = btn({ label: "Open WATCH", onClick: () => ctx.navigate("watch") });
  const reloadBtn = btn({
    label: "reload",
    className: "dash-drive-mini",
    onClick: () => {
      baseline = null;
      void loadBaseline();
      void loadHealth();
    },
  });
  const copyCmdBtn = btn({ label: "copy", className: "dash-drive-mini" });
  copyCmdBtn.addEventListener("click", () => void copyText(cmdEl.textContent ?? "", copyCmdBtn));

  const fileInput = el("input", {
    attrs: { type: "file", accept: "application/json,.json", hidden: true },
    on: {
      change: () => {
        const file = fileInput.files?.[0];
        if (file) loadFile(file);
      },
    },
  });

  for (const input of [seedInput, tierInput, attemptInput]) {
    input.addEventListener("input", () => renderSelection());
  }

  const sessionSection = section(
    "Live drive",
    "F4 · POST /api/drive → WS stream · 30 Hz",
    el(
      "div",
      { className: "dash-drive-toolbar" },
      field("Seed", seedInput),
      field("Tier", tierInput),
      field("Attempt", attemptInput),
      el(
        "div",
        { className: "dash-drive-selection" },
        heldPill,
        el("span", { className: "dash-muted" }, "writes ", replayNameEl),
      ),
    ),
    el("div", { className: "dash-btn-row" }, startBtn, saveBtn, stopBtn),
    el("div", { className: "dash-row dash-drive-api" }, el("span", { className: "dash-label", text: "control room" }), apiEl, healthPill),
    statusEl,
    el("div", { className: "dash-drive-stats" }, ...Object.values(stats).map((s) => s.root)),
    viewportHost,
    el("p", {
      className: "dash-drive-note",
      text:
        "W/↑ throttle · S/↓ brake · A/D steer · Space handbrake · R reset · P save (timestamped name; the Save now button uses the attempt name). " +
        "The server paces the loop at 30 Hz and never double-steps to catch up, so network jitter cannot change the physics.",
    }),
  );

  const exportSection = section(
    "Replay export",
    "server file is canonical",
    exportHost,
    el("div", { className: "dash-btn-row" }, downloadBtn, watchBtn),
  );

  const heldOutSection = section(
    "Held-out seeds",
    `seed % ${HELD_OUT_FALLBACK.modulus} == ${HELD_OUT_FALLBACK.remainder}`,
    el("p", {
      className: "dash-drive-body",
      text:
        "Training draws the complement of this set, so these seeds are the only honest place to compare human and agent. Click a seed to load it.",
    }),
    chipsEl,
    el("p", {
      className: "dash-drive-note",
      text:
        "Filled chips already have rows in times.json at the selected tier. Start at tier 0 — harder tiers come after an easy-tier bar exists.",
    }),
  );

  const baselineSection = section(
    "docs/baselines/human/times.json",
    "best · median on held-out seeds",
    metaEl,
    el(
      "div",
      { className: "dash-row dash-drive-source" },
      sourceEl,
      reloadBtn,
      el("label", { className: "dash-btn dash-drive-mini" }, "load file…", fileInput),
    ),
    notesEl,
    tableHost,
  );

  const protocolSection = section(
    "Recording protocol",
    "same physics, same rate, finished runs only",
    el(
      "div",
      { className: "dash-grid-2 dash-drive-protocol" },
      el(
        "div",
        { className: "dash-stack-tight" },
        el("h4", { className: "dash-label", text: "Invariants (non-negotiable)" }),
        invariantsTable(),
        el("p", {
          className: "dash-drive-note",
          text:
            "A human at 60 Hz against an agent at 30 Hz is not a comparison. Both the terminal recorder and this browser session sleep to the control rate on purpose.",
        }),
      ),
      el(
        "div",
        { className: "dash-stack-tight" },
        el("h4", { className: "dash-label", text: "Terminal recorder (curses — needs a real TTY)" }),
        cmdEl,
        el("div", { className: "dash-btn-row" }, copyCmdBtn),
        el(
          "ol",
          { className: "dash-drive-steps" },
          el("li", { text: "Five finished attempts per (seed, tier) before summarising." }),
          el("li", { text: 'Only meta.termination == "finish" enters the baseline.' }),
          el("li", { text: "Discarded a run? Delete that replay and re-record the same attempt." }),
          el("li", { text: "best_time_s = min finished; median_time_s = median of the five." }),
          el("li", { text: "Never hand-edit a replay — hashes catch it. Re-record instead." }),
        ),
        el("p", {
          className: "dash-drive-note",
          text: `Terminal replays land in replays/human/; browser sessions land in ${DEFAULT_REPLAY_DIR}. Both are valid — the aggregate table is filled by hand from finished runs.`,
        }),
      ),
    ),
  );

  root.append(
    header.root,
    el("p", {
      className: "dash-drive-lede",
      text:
        "The North Star claim is faster than a competent human on a keyboard. A time only counts once Carson has driven the held-out seeds at 30 Hz on the physics the agent trains on.",
    }),
    sessionSection,
    exportSection,
    heldOutSection,
    baselineSection,
    protocolSection,
  );
  host.append(root);

  const resizeObserver = new ResizeObserver(() => viewport?.resize());
  const onResize = (): void => viewport?.resize();
  window.addEventListener("resize", onResize);

  renderSelection();
  renderClaim();
  renderBaseline();
  renderExport();
  syncControls();
  dirty = true;
  paint();
  paintTimer = window.setInterval(paint, 100);
  void loadBaseline();
  void loadHealth();

  return () => {
    active = false;
    if (paintTimer != null) window.clearInterval(paintTimer);
    paintTimer = null;
    abort.abort();
    window.removeEventListener("resize", onResize);
    client.unbindKeys();
    client.disconnect();
    teardownViewport();
    host.classList.remove("dash-drive-host");
    host.replaceChildren();
  };
};

/* ------------------------------------------------------------ DOM helpers */

function section(
  title: string,
  meta: string,
  ...children: (Node | null)[]
): HTMLElement {
  const head = panelHeader(title, meta);
  const body = el("div", { className: "dash-section-body" });
  for (const child of children) if (child) body.appendChild(child);
  return el("section", { className: "dash-section" }, head.root, body);
}

function field(label: string, input: HTMLInputElement): HTMLElement {
  return el("div", { className: "dash-drive-field" }, el("label", { text: label }), input);
}

function cell(text: string, cls = ""): HTMLTableCellElement {
  return el("td", cls ? { className: cls, text } : { text });
}

function metaItem(label: string, value: string, warn = false): HTMLElement {
  return el(
    "div",
    { className: "dash-stat" },
    el("div", { className: "dash-stat-label", text: label }),
    el("div", {
      className: warn ? "dash-stat-value warn" : "dash-stat-value",
      text: value,
    }),
  );
}

function emptyBox(text: string): HTMLElement {
  return el("p", { className: "dash-drive-empty", text });
}

function noteLine(text: string): HTMLElement {
  return el("p", { className: "dash-drive-note", text });
}

function invariantsTable(): HTMLElement {
  const rows: [string, string][] = [
    ["physics", "same RallyEnv / vendored dynamics as training"],
    ["stages", "same generate(seed, tier) as eval"],
    ["control rate", "30 Hz — CONTROL_DT = 1/30, wall-clock paced"],
    ["replay", 'contracts.write_json(kind="replay"), source: "human"'],
    ["held out", "seed % 10 == 7"],
  ];
  return el(
    "table",
    { className: "dash-drive-table dash-drive-invariants" },
    el(
      "tbody",
      null,
      ...rows.map(([k, v]) => el("tr", null, el("td", { text: k }), el("td", { text: v }))),
    ),
  );
}

/* ---------------------------------------------------------------- format */

function intOr(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.trunc(value)
    : fallback;
}

function looksLikeBaseline(doc: unknown): doc is HumanBaseline {
  if (!doc || typeof doc !== "object") return false;
  const d = doc as HumanBaseline;
  return Array.isArray(d.seeds) || typeof d.held_out === "object";
}

export function fmtTime(seconds: number | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return m > 0 ? `${m}:${s.toFixed(2).padStart(5, "0")}` : `${s.toFixed(2)}s`;
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function copyText(text: string, button: HTMLButtonElement): Promise<void> {
  const label = button.dataset.label ?? button.textContent ?? "copy";
  button.dataset.label = label;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "copied";
  } catch {
    button.textContent = "copy failed";
  }
  window.setTimeout(() => {
    button.textContent = label;
  }, 1400);
}

/* -------------------------------------------------------------- registration */

/** Thin register helper — importing this module also self-registers. */
export function registerDrivePanel(
  register: (id: SectionId, mount: PanelMount) => void = registerPanel,
): void {
  register(DRIVE_SECTION_ID, mountDrive);
}

registerDrivePanel();

/** Convenience for tests / manual mounts that already have a context. */
export function mountDrivePanel(host: HTMLElement, ctx: PanelContext): () => void {
  const dispose = mountDrive(host, ctx);
  return typeof dispose === "function" ? dispose : () => undefined;
}

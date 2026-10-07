/**
 * Dashboard PIPELINE panel — foundation → flow → fast → finish → frontier.
 *
 * Stage RewardConfig weights come from `public/dashboard/stages.json`
 * (synced from `packages/sim/rallyai/train/stages.py`). Curriculum mastery
 * gates match `CurriculumConfig`. Live tier occupancy is derived from the
 * active run's metrics stream when the control room is up.
 */

import { TrainClient, loadSession } from "../../train/TrainClient";
import type { MetricsLine, RunDetail } from "../../train/types";
import { registerPanel } from "../DashboardApp";
import type { PanelContext, PanelMount, SectionId } from "../types";
import "./pipeline.css";

export const PIPELINE_SECTION: SectionId = "pipeline";

export type StageName =
  | "foundation"
  | "flow"
  | "fast"
  | "finish"
  | "frontier";

export interface StageWeights {
  progress: number;
  speed: number;
  speed_exp: number;
  throttle_commit: number;
  time_cost: number;
  smooth: number;
  finish: number;
  finish_pace: number;
}

export interface StageConfig {
  intent: string;
  tier_min: number;
  tier_max: number;
  weights: StageWeights;
}

export interface StagesDocument {
  source: string;
  stages: StageName[];
  weight_keys: (keyof StageWeights)[];
  curriculum: {
    window: number;
    promote_at: number;
    demote_at: number;
  };
  hof_feed: {
    slot: string;
    note: string;
    cli: string;
  };
  configs: Record<StageName, StageConfig>;
}

/** Embedded fallback identical to public/dashboard/stages.json. */
export const EMBEDDED_STAGES: StagesDocument = {
  source: "packages/sim/rallyai/train/stages.py",
  stages: ["foundation", "flow", "fast", "finish", "frontier"],
  weight_keys: [
    "progress",
    "speed",
    "speed_exp",
    "throttle_commit",
    "time_cost",
    "smooth",
    "finish",
    "finish_pace",
  ],
  curriculum: {
    window: 100,
    promote_at: 0.75,
    demote_at: 0.35,
  },
  hof_feed: {
    slot: "best",
    note: "Next stage resumes from the previous stage run's HoF best slot via --resume.",
    cli: "python -m rallyai.train --stage <next> --resume <prev_run_dir>/<run_id>_best.pt",
  },
  configs: {
    foundation: {
      intent: "Survive and complete: progress dominant, pace terms near zero.",
      tier_min: 0,
      tier_max: 2,
      weights: {
        progress: 1.5,
        speed: 0.5,
        speed_exp: 1.0,
        throttle_commit: 0.0,
        time_cost: 0.15,
        smooth: 0.2,
        finish: 120.0,
        finish_pace: 10.0,
      },
    },
    flow: {
      intent: "Carry speed through the stage.",
      tier_min: 0,
      tier_max: 3,
      weights: {
        progress: 1.0,
        speed: 18.0,
        speed_exp: 2.0,
        throttle_commit: 0.8,
        time_cost: 0.4,
        smooth: 0.4,
        finish: 100.0,
        finish_pace: 20.0,
      },
    },
    fast: {
      intent: "Commit: throttle and time pressure up.",
      tier_min: 1,
      tier_max: 4,
      weights: {
        progress: 0.8,
        speed: 22.0,
        speed_exp: 2.2,
        throttle_commit: 3.2,
        time_cost: 0.9,
        smooth: 0.3,
        finish: 100.0,
        finish_pace: 30.0,
      },
    },
    finish: {
      intent: "Optimise the last seconds: finish_pace / time up, progress down.",
      tier_min: 2,
      tier_max: 5,
      weights: {
        progress: 0.4,
        speed: 20.0,
        speed_exp: 2.0,
        throttle_commit: 2.4,
        time_cost: 1.2,
        smooth: 0.3,
        finish: 80.0,
        finish_pace: 50.0,
      },
    },
    frontier: {
      intent: "Full weights, intended for hardest tiers only.",
      tier_min: 4,
      tier_max: 5,
      weights: {
        progress: 1.0,
        speed: 18.0,
        speed_exp: 2.0,
        throttle_commit: 2.4,
        time_cost: 0.7,
        smooth: 0.3,
        finish: 100.0,
        finish_pace: 35.0,
      },
    },
  },
};

interface CurriculumEventView {
  wall_t: number;
  timesteps?: number;
  old_tier: number;
  new_tier: number;
  rate?: number;
  reason: "promote" | "demote" | string;
}

interface OccupancyState {
  run: RunDetail | null;
  currentTier: number | null;
  currentStage: StageName | null;
  /** Update-line counts per tier 0–5 (proxy for time spent). */
  tierCounts: number[];
  events: CurriculumEventView[];
  bestCkpt: string | null;
  status: "idle" | "loading" | "live" | "offline" | "error";
  message: string;
}

const MAX_TIER = 5;
const STAGES_URL = "/dashboard/stages.json";

/**
 * Mount the PIPELINE panel into `el`. Section id: `pipeline`.
 * Returns a disposer that stops metrics polling / WS.
 */
export const mountPipeline: PanelMount = (el, ctx) => {
  el.replaceChildren();
  el.id = "pipeline";
  el.dataset.section = "pipeline";

  const root = document.createElement("div");
  root.className = "pipeline-panel";
  root.innerHTML = `
    <div class="pl-head">
      <div>
        <div class="pl-kicker">Staged pipeline</div>
        <div class="pl-status" data-pl="status">Loading stage config…</div>
      </div>
      <div class="pl-actions">
        <button type="button" data-pl="refresh">Refresh metrics</button>
        <button type="button" data-pl="goto-train">Open Train</button>
      </div>
    </div>
    <div class="pl-rail" data-pl="rail" role="tablist" aria-label="Pipeline stages"></div>
    <div class="pl-grid">
      <div class="pl-block">
        <h3>Reward weighting</h3>
        <p class="pl-intent" data-pl="intent"></p>
        <div class="pl-weights" data-pl="weights"></div>
      </div>
      <div class="pl-block">
        <h3>Curriculum mastery</h3>
        <div class="pl-rules" data-pl="rules"></div>
      </div>
      <div class="pl-block">
        <h3>Tier occupancy</h3>
        <div data-pl="occupancy"></div>
      </div>
      <div class="pl-block">
        <h3>HoF → next stage</h3>
        <div data-pl="hof"></div>
      </div>
    </div>
  `;
  el.appendChild(root);

  let doc = EMBEDDED_STAGES;
  let selected: StageName = "foundation";
  const occupancy: OccupancyState = {
    run: null,
    currentTier: null,
    currentStage: null,
    tierCounts: zeros(MAX_TIER + 1),
    events: [],
    bestCkpt: null,
    status: "loading",
    message: "Connecting…",
  };

  const client = new TrainClient({
    onMetric: (line) => ingestMetric(line, occupancy, () => renderOccupancy()),
    onRun: (run) => {
      occupancy.run = run;
      occupancy.currentStage = normalizeStage(run.config?.stage);
      renderOccupancy();
      renderHof();
      renderRail();
    },
    onError: (message) => {
      occupancy.status = "error";
      occupancy.message = message;
      setStatus(message, "err");
    },
    onConnection: (connected) => {
      if (connected && occupancy.run) {
        occupancy.status = "live";
        setStatus(`Live · ${occupancy.run.run_id}`, "live");
      }
    },
  });
  if (ctx.apiBase) client.setBaseUrl(ctx.apiBase);

  const statusEl = qs(root, "[data-pl=status]");
  const railEl = qs(root, "[data-pl=rail]");
  const intentEl = qs(root, "[data-pl=intent]");
  const weightsEl = qs(root, "[data-pl=weights]");
  const rulesEl = qs(root, "[data-pl=rules]");
  const occupancyEl = qs(root, "[data-pl=occupancy]");
  const hofEl = qs(root, "[data-pl=hof]");

  qs(root, "[data-pl=goto-train]").addEventListener("click", () => {
    ctx.navigate("train");
  });
  qs(root, "[data-pl=refresh]").addEventListener("click", () => {
    void refreshMetrics();
  });

  void bootstrap();

  return () => {
    client.disconnect();
  };

  async function bootstrap(): Promise<void> {
    doc = await loadStagesDoc();
    if (occupancy.currentStage && doc.stages.includes(occupancy.currentStage)) {
      selected = occupancy.currentStage;
    }
    renderRules();
    renderRail();
    renderWeights();
    renderHof();
    await refreshMetrics();
  }

  function setStatus(text: string, kind?: "live" | "err"): void {
    statusEl.textContent = text;
    statusEl.classList.toggle("live", kind === "live");
    statusEl.classList.toggle("err", kind === "err");
  }

  function renderRail(): void {
    railEl.replaceChildren();
    doc.stages.forEach((name, i) => {
      const cfg = doc.configs[name];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pl-stage";
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", String(name === selected));
      if (name === selected) btn.classList.add("active");
      if (name === occupancy.currentStage) btn.classList.add("current-run");
      btn.innerHTML = `
        <span class="pl-stage-idx">0${i + 1}</span>
        <span class="pl-stage-name">${name}</span>
        <span class="pl-stage-tiers">T${cfg.tier_min}–T${cfg.tier_max}</span>
      `;
      btn.addEventListener("click", () => {
        selected = name;
        renderRail();
        renderWeights();
        renderHof();
        renderOccupancy();
      });
      railEl.appendChild(btn);
    });
  }

  function renderWeights(): void {
    const cfg = doc.configs[selected];
    const prevName = previousStage(doc.stages, selected);
    const prev = prevName ? doc.configs[prevName].weights : null;
    intentEl.textContent = cfg.intent;

    const maxAbs = Math.max(
      ...doc.weight_keys.map((k) =>
        Math.max(...doc.stages.map((s) => Math.abs(doc.configs[s].weights[k]))),
      ),
      1e-9,
    );

    weightsEl.replaceChildren();
    for (const key of doc.weight_keys) {
      const value = cfg.weights[key];
      const row = document.createElement("div");
      row.className = "pl-weight-row";
      const pct = Math.min(100, (Math.abs(value) / maxAbs) * 100);
      let deltaClass = "";
      if (prev) {
        if (value > prev[key]) deltaClass = "delta-up";
        else if (value < prev[key]) deltaClass = "delta-down";
      }
      row.innerHTML = `
        <span class="key">${key}</span>
        <div class="pl-bar-track"><div class="pl-bar-fill ${deltaClass}" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="val">${formatNum(value)}</span>
      `;
      weightsEl.appendChild(row);
    }
  }

  function renderRules(): void {
    const c = doc.curriculum;
    rulesEl.innerHTML = `
      <div class="pl-rule promote">
        <div class="label">Promote</div>
        <div class="numeral">${(c.promote_at * 100).toFixed(0)}%</div>
        <div class="hint">Rolling clean completion ≥ threshold</div>
      </div>
      <div class="pl-rule demote">
        <div class="label">Demote</div>
        <div class="numeral">${(c.demote_at * 100).toFixed(0)}%</div>
        <div class="hint">Rolling clean completion ≤ threshold</div>
      </div>
      <div class="pl-rule">
        <div class="label">Window</div>
        <div class="numeral">${c.window}</div>
        <div class="hint">Episodes before a gate can fire; cooldown = same</div>
      </div>
    `;
  }

  function renderOccupancy(): void {
    const cfg = doc.configs[selected];
    const total = occupancy.tierCounts.reduce((a, b) => a + b, 0);
    const maxCount = Math.max(...occupancy.tierCounts, 1);

    if (occupancy.status === "offline" || occupancy.status === "error") {
      occupancyEl.innerHTML = `<p class="pl-empty">${escapeHtml(occupancy.message)}</p>`;
      return;
    }

    if (!occupancy.run && total === 0) {
      occupancyEl.innerHTML = `
        <p class="pl-empty">No active run metrics yet. Start a train job or attach a session — curriculum events and update lines fill the tier histogram.</p>
      `;
      return;
    }

    const tiers = document.createElement("div");
    tiers.className = "pl-tiers";
    for (let t = 0; t <= MAX_TIER; t++) {
      const count = occupancy.tierCounts[t] ?? 0;
      const pct = total > 0 ? (count / maxCount) * 100 : 0;
      const share = total > 0 ? ((count / total) * 100).toFixed(0) : "0";
      const row = document.createElement("div");
      row.className = "pl-tier-row";
      if (t >= cfg.tier_min && t <= cfg.tier_max) row.classList.add("in-bounds");
      if (occupancy.currentTier === t) row.classList.add("current");
      row.innerHTML = `
        <span class="key">T${t}</span>
        <div class="pl-bar-track"><div class="pl-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="val">${share}%</span>
      `;
      tiers.appendChild(row);
    }
    occupancyEl.replaceChildren(tiers);

    const meta = document.createElement("p");
    meta.className = "pl-empty";
    meta.style.marginTop = "0.55rem";
    const runLabel = occupancy.run?.run_id ?? "—";
    const tierLabel =
      occupancy.currentTier != null ? `T${occupancy.currentTier}` : "—";
    const stageLabel = occupancy.currentStage ?? "—";
    meta.textContent = `Run ${runLabel} · stage ${stageLabel} · tier ${tierLabel} · ${total} update samples`;
    occupancyEl.appendChild(meta);

    if (occupancy.events.length) {
      const list = document.createElement("ul");
      list.className = "pl-events";
      for (const ev of occupancy.events.slice(-12).reverse()) {
        const li = document.createElement("li");
        li.className = ev.reason;
        const steps =
          ev.timesteps != null ? ` @ ${formatInt(ev.timesteps)}` : "";
        const rate =
          ev.rate != null ? ` · ${(ev.rate * 100).toFixed(0)}%` : "";
        li.textContent = `${ev.reason} T${ev.old_tier}→T${ev.new_tier}${rate}${steps}`;
        list.appendChild(li);
      }
      occupancyEl.appendChild(list);
    }
  }

  function renderHof(): void {
    const stages = doc.stages;
    const slot = doc.hof_feed.slot;
    const chain = document.createElement("div");
    chain.className = "pl-hof-chain";

    stages.forEach((name, i) => {
      const node = document.createElement("span");
      node.className = "pl-hof-node";
      if (name === selected) node.classList.add("active");
      node.textContent = name;
      chain.appendChild(node);
      if (i < stages.length - 1) {
        const edge = document.createElement("span");
        edge.className = "pl-hof-edge";
        edge.textContent = `→ ${slot}.pt →`;
        chain.appendChild(edge);
      }
    });

    hofEl.replaceChildren(chain);

    const note = document.createElement("p");
    note.className = "pl-empty";
    note.style.marginTop = "0.55rem";
    const prev = previousStage(stages, selected);
    if (!prev) {
      note.textContent =
        "Foundation is the pipeline head — start from scratch (no --resume), then promote its HoF best into flow.";
    } else {
      const feedPath =
        occupancy.bestCkpt && occupancy.currentStage === prev
          ? occupancy.bestCkpt
          : `<${prev}_run>/${prev}_run_id_${slot}.pt`;
      note.textContent = `${selected} resumes from ${prev}'s HoF ${slot} checkpoint: ${feedPath}`;
    }
    hofEl.appendChild(note);

    const next = nextStage(stages, selected);
    const cli = document.createElement("div");
    cli.className = "pl-cli";
    const code = document.createElement("code");
    if (next) {
      const resumeHint =
        occupancy.bestCkpt && occupancy.currentStage === selected
          ? occupancy.bestCkpt
          : `<${selected}_run_dir>/<run_id>_${slot}.pt`;
      code.textContent = `python -m rallyai.train --stage ${next} --resume ${resumeHint}`;
    } else {
      code.textContent =
        "frontier is the terminal stage — keep training / evaluating; no further HoF handoff.";
    }
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy";
    copy.disabled = !next;
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent ?? "");
        copy.textContent = "Copied";
        window.setTimeout(() => {
          copy.textContent = "Copy";
        }, 1200);
      } catch {
        copy.textContent = "Fail";
      }
    });
    cli.append(code, copy);
    hofEl.appendChild(cli);
  }

  async function refreshMetrics(): Promise<void> {
    occupancy.status = "loading";
    setStatus("Fetching run metrics…");
    try {
      const run = await pickRun(client);
      if (!run) {
        occupancy.run = null;
        occupancy.status = "offline";
        occupancy.message =
          "Control room reachable but no runs yet — start training to see tier occupancy.";
        setStatus(occupancy.message);
        renderOccupancy();
        renderHof();
        return;
      }

      occupancy.run = run;
      occupancy.currentStage = normalizeStage(run.config?.stage);
      if (
        occupancy.currentStage &&
        doc.stages.includes(occupancy.currentStage)
      ) {
        selected = occupancy.currentStage;
      }

      occupancy.tierCounts = zeros(MAX_TIER + 1);
      occupancy.events = [];
      occupancy.bestCkpt = null;
      occupancy.currentTier =
        typeof run.config?.tier === "number" ? run.config.tier : null;

      let since = 0;
      // Page through metrics (control returns batches).
      for (let page = 0; page < 40; page++) {
        const batch = await client.readMetrics(run.run_id, since, 500);
        for (const line of batch.lines) {
          ingestMetric(line, occupancy);
        }
        since = batch.next;
        if (batch.lines.length === 0) break;
      }

      // Prefer live stream when the run is still going.
      if (run.state === "running" || run.state === "starting") {
        const session = loadSession();
        const cursor =
          session?.runId === run.run_id ? session.since : since;
        client.connect(run.run_id, cursor);
        occupancy.status = "live";
        setStatus(`Live · ${run.run_id}`, "live");
      } else {
        client.disconnect();
        occupancy.status = "idle";
        setStatus(`Run ${run.run_id} · ${run.state}`);
      }

      renderRail();
      renderWeights();
      renderOccupancy();
      renderHof();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      occupancy.status = "offline";
      occupancy.message = `Control offline (${message}). Stage weights still load from static config.`;
      setStatus(occupancy.message, "err");
      renderOccupancy();
    }
  }
};

/** Soft-register with an external registry (tests / alternate shells). */
export function registerPipelinePanel(registry: {
  registerPanel: (id: SectionId, mount: PanelMount) => void;
}): void {
  registry.registerPanel(PIPELINE_SECTION, mountPipeline);
}

// Side-effect registration for the program dashboard shell.
registerPanel(PIPELINE_SECTION, mountPipeline);

// --- helpers ---------------------------------------------------------------

async function loadStagesDoc(): Promise<StagesDocument> {
  try {
    const res = await fetch(STAGES_URL);
    if (!res.ok) return EMBEDDED_STAGES;
    const raw = (await res.json()) as StagesDocument;
    if (!raw?.stages?.length || !raw.configs) return EMBEDDED_STAGES;
    return {
      ...EMBEDDED_STAGES,
      ...raw,
      curriculum: { ...EMBEDDED_STAGES.curriculum, ...raw.curriculum },
      hof_feed: { ...EMBEDDED_STAGES.hof_feed, ...raw.hof_feed },
      configs: { ...EMBEDDED_STAGES.configs, ...raw.configs },
    };
  } catch {
    return EMBEDDED_STAGES;
  }
}

async function pickRun(client: TrainClient): Promise<RunDetail | null> {
  const session = loadSession();
  if (session?.runId) {
    try {
      return await client.getRun(session.runId);
    } catch {
      /* fall through to list */
    }
  }
  const runs = await client.listRuns();
  if (!runs.length) return null;
  const active = runs.find(
    (r) => r.state === "running" || r.state === "starting",
  );
  if (active) return active;
  return [...runs].sort((a, b) => {
    const ta = a.updated_at ?? a.started_at ?? a.created_at ?? 0;
    const tb = b.updated_at ?? b.started_at ?? b.created_at ?? 0;
    return tb - ta;
  })[0]!;
}

function ingestMetric(
  line: MetricsLine,
  state: OccupancyState,
  onChange?: () => void,
): void {
  if (line.kind == null) return;

  if (typeof line.tier === "number") {
    state.currentTier = line.tier;
  }

  if (line.kind === "update" && typeof line.tier === "number") {
    const t = Math.max(0, Math.min(MAX_TIER, Math.floor(line.tier)));
    state.tierCounts[t] = (state.tierCounts[t] ?? 0) + 1;
  }

  if (line.kind === "curriculum") {
    const parsed = parseCurriculumMsg(line.msg ?? "");
    const newTier =
      typeof line.tier === "number" ? line.tier : (parsed?.new_tier ?? 0);
    const oldTier = parsed?.old_tier ?? Math.max(0, newTier - 1);
    const reason = parsed?.reason ?? guessReason(line.msg);
    const event: CurriculumEventView = {
      wall_t: line.wall_t,
      old_tier: oldTier,
      new_tier: newTier,
      reason,
    };
    if (line.timesteps != null) event.timesteps = line.timesteps;
    if (line.completion?.rate != null) event.rate = line.completion.rate;
    state.events.push(event);
    state.currentTier = newTier;
  }

  if (line.kind === "checkpoint" && line.checkpoint?.slot === "best") {
    state.bestCkpt = line.checkpoint.path ?? state.bestCkpt;
  }

  if (line.kind === "run_start" && line.msg) {
    const m = /stage=(\w+)/.exec(line.msg);
    if (m) state.currentStage = normalizeStage(m[1]);
  }

  onChange?.();
}

function parseCurriculumMsg(
  msg: string,
): { reason: string; old_tier: number; new_tier: number } | null {
  // "promote 0->1" / "demote 2->1"
  const m = /^(promote|demote)\s+(\d+)\s*->\s*(\d+)/i.exec(msg.trim());
  if (!m) return null;
  return {
    reason: m[1]!.toLowerCase(),
    old_tier: Number(m[2]),
    new_tier: Number(m[3]),
  };
}

function guessReason(msg?: string): string {
  if (!msg) return "curriculum";
  if (/demote/i.test(msg)) return "demote";
  if (/promote/i.test(msg)) return "promote";
  return "curriculum";
}

function normalizeStage(raw?: string | null): StageName | null {
  if (!raw) return null;
  const name = raw.trim().toLowerCase();
  if (
    name === "foundation" ||
    name === "flow" ||
    name === "fast" ||
    name === "finish" ||
    name === "frontier"
  ) {
    return name;
  }
  return null;
}

function previousStage(
  stages: StageName[],
  current: StageName,
): StageName | null {
  const i = stages.indexOf(current);
  return i > 0 ? stages[i - 1]! : null;
}

function nextStage(stages: StageName[], current: StageName): StageName | null {
  const i = stages.indexOf(current);
  return i >= 0 && i < stages.length - 1 ? stages[i + 1]! : null;
}

function zeros(n: number): number[] {
  return Array.from({ length: n }, () => 0);
}

function formatNum(n: number): string {
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(Math.abs(n) >= 10 ? 1 : 2);
}

function formatInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function qs(root: ParentNode, sel: string): HTMLElement {
  const el = root.querySelector<HTMLElement>(sel);
  if (!el) throw new Error(`pipeline panel missing ${sel}`);
  return el;
}

// Satisfy unused-import lint if PanelContext is only used via PanelMount.
export type { PanelContext };

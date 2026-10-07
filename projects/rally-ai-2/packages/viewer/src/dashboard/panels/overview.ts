/**
 * Overview / Home panel — RallyAI identity, run status, measured standing
 * numbers, quick actions, and phase strip A–J.
 *
 * Section id: `overview`
 */

import {
  loadMeasured,
  loadRoadmapStatus,
  type MeasuredSnapshot,
  type PhaseStatus,
  type RoadmapPhase,
} from "../data/measured";
import { registerPanel } from "../registry";
import { btn, codeBlock, el, panelHeader } from "../ui";
import type { PanelContext, PanelMount } from "../types";
import type { RunDetail } from "../../train/types";
import "./overview.css";

const CONTROL_HINT =
  "cd packages/sim && .venv/bin/python -m rallyai.control --host 127.0.0.1 --port 8765";

const NORTH_STAR =
  "A reinforcement-learning agent teaches itself to drive a rally car flat-out, rendered as a PlayStation-1 rally game.";

function apiUrl(apiBase: string, path: string): string {
  const base = apiBase.replace(/\/$/, "");
  if (!base) return path;
  return `${base}${path}`;
}

type RunFetch =
  | { ok: true; runs: RunDetail[] }
  | { ok: false; reason: string };

async function fetchRuns(apiBase: string): Promise<RunFetch> {
  try {
    const res = await fetch(apiUrl(apiBase, "/api/runs"), {
      signal: AbortSignal.timeout(2500),
    });
    if (!res.ok) {
      return { ok: false, reason: `control responded ${res.status}` };
    }
    const data = (await res.json()) as { runs?: RunDetail[] };
    return { ok: true, runs: data.runs ?? [] };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "unreachable";
    return { ok: false, reason: msg };
  }
}

function pickActiveRun(runs: RunDetail[]): RunDetail | null {
  const order = ["running", "starting", "stopping", "error", "stopped"] as const;
  for (const state of order) {
    const hit = runs.find((r) => r.state === state);
    if (hit) return hit;
  }
  return runs[0] ?? null;
}

function stateClass(state: string | null): string {
  if (!state) return "muted";
  if (state === "running" || state === "starting") return "ok";
  if (state === "stopping") return "warn";
  if (state === "error") return "bad";
  return "muted";
}

function formatStatusLabel(fetch: RunFetch, run: RunDetail | null): string {
  if (!fetch.ok) return "Offline";
  if (!run) return "Idle";
  return run.state;
}

function formatRunMeta(fetch: RunFetch, run: RunDetail | null): string {
  if (!fetch.ok) {
    return `Control room down — ${fetch.reason}. Start it, then refresh.`;
  }
  if (!run) return `${fetch.runs.length} run(s) · none active`;
  const cfg = run.config;
  const bits = [
    run.run_id,
    cfg ? `tier ${cfg.tier}` : null,
    cfg ? `${cfg.workers}w` : null,
    cfg?.stage ?? null,
  ].filter(Boolean);
  return bits.join(" · ");
}

function statusLabel(status: PhaseStatus): string {
  switch (status) {
    case "done":
      return "done";
    case "partial":
      return "partial";
    case "blocked":
      return "blocked";
    default:
      return "idle";
  }
}

function buildHero(): HTMLElement {
  return el(
    "div",
    { className: "ov-hero" },
    el("div", { className: "ov-hero-brand", text: "RallyAI" }),
    el("p", { className: "ov-hero-line", text: NORTH_STAR }),
  );
}

function buildRunBlock(): {
  root: HTMLElement;
  set: (fetch: RunFetch, run: RunDetail | null) => void;
} {
  const stateEl = el("div", {
    className: "ov-run-state muted",
    text: "…",
  });
  const metaEl = el("div", { className: "ov-run-meta", text: "Checking control…" });
  const root = el(
    "section",
    { className: "dash-section" },
    panelHeader("Current run", "GET /api/runs").root,
    el("div", { className: "ov-run" }, stateEl, metaEl),
  );
  return {
    root,
    set(fetch, run) {
      const label = formatStatusLabel(fetch, run);
      stateEl.textContent = label;
      stateEl.className = `ov-run-state ${stateClass(fetch.ok ? run?.state ?? null : null)}`;
      if (!fetch.ok) stateEl.className = "ov-run-state muted";
      metaEl.textContent = formatRunMeta(fetch, run);
    },
  };
}

function buildMetrics(m: MeasuredSnapshot): HTMLElement {
  const tarmac = m.evo_rally.surfaces.find((s) => s.name === "tarmac");
  const tier0 = m.curriculum_pilot.tiers[0];
  const tier5 = m.curriculum_pilot.tiers[5];

  const cells = [
    {
      label: "Throughput",
      value: String(m.throughput.async_workers_8_steps_per_s),
      sub: `async/8 steps/s · ${m.throughput.async_scale_vs_sync1}× sync/1`,
    },
    {
      label: "Pace share",
      value: `${m.pace_share.mean_pct}%`,
      sub: `mean across tiers · target ~${m.pace_share.target_band_pct}%`,
    },
    {
      label: "evo_rally",
      value: tarmac ? `${tarmac.lateral_g} g` : "—",
      sub: tarmac
        ? `tarmac · ${tarmac.max_sustained_kmh} km/h · 0–100 ${m.evo_rally.zero_to_100_s}s`
        : "skidpad",
    },
    {
      label: "Curriculum pilot",
      value: tier0 && tier5 ? `${tier0.completed}→${tier5.completed}` : "—",
      sub: "tiers 0→5 completion · reference pilot",
    },
  ];

  const grid = el("div", { className: "ov-metrics" });
  for (const cell of cells) {
    grid.appendChild(
      el(
        "div",
        { className: "ov-metric" },
        el("div", { className: "dash-stat-label", text: cell.label }),
        el("div", { className: "ov-metric-value", text: cell.value }),
        el("div", { className: "ov-metric-sub", text: cell.sub }),
      ),
    );
  }

  return el(
    "section",
    { className: "dash-section" },
    panelHeader("Standing measured", `snapshot ${m.updated}`).root,
    grid,
    el("p", {
      className: "ov-hint",
      text: `From ${m.source}. Re-derive before citing — see public/dashboard/measured.json.`,
    }),
  );
}

function buildActions(ctx: PanelContext): HTMLElement {
  const row = el("div", { className: "ov-actions" });

  row.appendChild(
    btn({
      label: "Train",
      onClick: () => ctx.navigate("train"),
    }),
  );
  row.appendChild(
    btn({
      label: "Live",
      onClick: () => ctx.navigate("live"),
    }),
  );
  row.appendChild(
    btn({
      label: "Watch",
      onClick: () => {
        location.href = "?";
      },
    }),
  );
  row.appendChild(
    btn({
      label: "Roadmap",
      onClick: () => ctx.navigate("roadmap"),
    }),
  );

  const openTrain = el("a", {
    className: "dash-btn",
    text: "TRAIN mode",
    attrs: { href: "?mode=train" },
  });
  row.appendChild(openTrain);

  return el(
    "section",
    { className: "dash-section" },
    panelHeader("Quick links", "studio").root,
    row,
    el("p", {
      className: "ov-hint",
      text: "Control room must be up for Train / Live / Drive. Hint:",
    }),
    codeBlock({ code: CONTROL_HINT, lang: "bash" }),
  );
}

function buildPhaseStrip(
  phases: RoadmapPhase[],
  onRoadmap: () => void,
): { root: HTMLElement; tip: HTMLElement } {
  const strip = el("div", { className: "ov-phases" });
  const tip = el("p", {
    className: "ov-phase-tip",
    text: "Hover a phase for evidence. Edit public/dashboard/roadmap-status.json to update.",
  });

  const shown = phases.filter((p) => /^[A-J]$/i.test(p.id)).slice(0, 10);
  for (const phase of shown) {
    const status = (phase.status ?? "not-started") as PhaseStatus;
    const cell = el(
      "button",
      {
        className: "ov-phase",
        attrs: {
          type: "button",
          "data-status": status,
          title: phase.summary || phase.name,
        },
        on: {
          click: () => onRoadmap(),
          mouseenter: () => {
            tip.textContent = `${phase.id} · ${phase.name} — ${phase.summary || statusLabel(status)}`;
          },
        },
      },
      el("div", { className: "ov-phase-id", text: phase.id }),
      el("div", { className: "ov-phase-mark" }),
      el("div", {
        className: "ov-phase-status",
        text: statusLabel(status),
      }),
    );
    strip.appendChild(cell);
  }

  const root = el(
    "section",
    { className: "dash-section" },
    panelHeader("Phases A–J", "roadmap-status.json").root,
    strip,
    tip,
  );
  return { root, tip };
}

/**
 * Mount the Overview / Home panel into `host`.
 * Returns a disposer that cancels in-flight fetches.
 */
export const mountOverview: PanelMount = (host, ctx) => {
  const ac = new AbortController();
  host.replaceChildren();
  host.classList.add("ov");

  const run = buildRunBlock();
  const metricsHost = el("div");
  const phaseHost = el("div");

  host.append(
    buildHero(),
    run.root,
    metricsHost,
    buildActions(ctx),
    phaseHost,
  );

  // Placeholder metrics while JSON loads.
  metricsHost.appendChild(
    el(
      "section",
      { className: "dash-section" },
      panelHeader("Standing measured", "loading").root,
      el("div", { className: "dash-empty", text: "Loading measured snapshot…" }),
    ),
  );
  phaseHost.appendChild(
    el(
      "section",
      { className: "dash-section" },
      panelHeader("Phases A–J", "loading").root,
      el("div", { className: "dash-empty", text: "Loading roadmap status…" }),
    ),
  );

  void (async () => {
    const [runs, measured, roadmap] = await Promise.all([
      fetchRuns(ctx.apiBase),
      loadMeasured(),
      loadRoadmapStatus(),
    ]);
    if (ac.signal.aborted) return;

    run.set(runs, runs.ok ? pickActiveRun(runs.runs) : null);

    metricsHost.replaceChildren(buildMetrics(measured));

    const strip = buildPhaseStrip(roadmap.phases, () => ctx.navigate("roadmap"));
    phaseHost.replaceChildren(strip.root);
  })();

  return () => {
    ac.abort();
    host.classList.remove("ov");
  };
};

/**
 * Thin register helper — call once at dashboard boot.
 * Importing this module also self-registers (see main.ts panel imports).
 */
export function registerOverviewPanel(): void {
  registerPanel("overview", mountOverview);
}

registerOverviewPanel();
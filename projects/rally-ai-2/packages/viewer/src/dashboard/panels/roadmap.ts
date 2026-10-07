/**
 * Dashboard ROADMAP panel — phases A–K dependency graph + work-item status.
 *
 * Source of truth: `public/dashboard/roadmap-status.json`
 * (edit that file to update the panel; see `update_path` in the JSON).
 */

import { registerPanel } from "../registry";
import type { PanelContext, PanelMount, SectionId } from "../types";
import "./roadmap.css";

export const ROADMAP_SECTION: SectionId = "roadmap";

const STATUS_URL = "/dashboard/roadmap-status.json";

export type WorkItemStatus = "done" | "partial" | "blocked" | "not-started";

export interface RoadmapWorkItem {
  id: string;
  phase: string;
  title: string;
  status: WorkItemStatus;
  measure: string;
  verify: string;
  note?: string;
  waiver?: {
    criterion: string;
    measured: string;
    tracked_metric: string;
  };
}

export interface RoadmapPhase {
  id: string;
  title: string;
  status: WorkItemStatus;
  depends_on: string[];
  work_items: string[];
  note?: string;
}

export interface RoadmapStatusDoc {
  updated?: string;
  update_path?: string[];
  session_prefill?: Record<string, string>;
  graph: {
    ascii?: string;
    edges: { from: string; to: string }[];
    notes: string[];
  };
  phases: RoadmapPhase[];
  work_items: RoadmapWorkItem[];
}

/** Layout for §3 dependency graph (viewBox 0 0 640 300). */
const NODE_LAYOUT: Record<string, { x: number; y: number; label: string }> = {
  A: { x: 520, y: 28, label: "Viewer" },
  B: { x: 40, y: 88, label: "Vec" },
  C: { x: 180, y: 88, label: "Trainer" },
  D: { x: 180, y: 168, label: "Eval" },
  E: { x: 320, y: 88, label: "Result" },
  F: { x: 40, y: 168, label: "Control" },
  G: { x: 180, y: 248, label: "TRAIN" },
  H: { x: 320, y: 168, label: "Spectacle" },
  I: { x: 460, y: 168, label: "Long run" },
  J: { x: 580, y: 168, label: "Done" },
  K: { x: 520, y: 248, label: "Rules" },
};

const NODE_W = 72;
const NODE_H = 28;

function isStatus(value: unknown): value is WorkItemStatus {
  return (
    value === "done" ||
    value === "partial" ||
    value === "blocked" ||
    value === "not-started"
  );
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function statusLabel(status: WorkItemStatus): string {
  switch (status) {
    case "done":
      return "done";
    case "partial":
      return "partial";
    case "blocked":
      return "blocked";
    case "not-started":
      return "not started";
  }
}

function phaseStatusMap(doc: RoadmapStatusDoc): Map<string, WorkItemStatus> {
  const map = new Map<string, WorkItemStatus>();
  for (const phase of doc.phases) {
    map.set(phase.id, isStatus(phase.status) ? phase.status : "not-started");
  }
  return map;
}

function edgePath(
  from: string,
  to: string,
): string | null {
  const a = NODE_LAYOUT[from];
  const b = NODE_LAYOUT[to];
  if (!a || !b) return null;
  const x1 = a.x + NODE_W;
  const y1 = a.y + NODE_H / 2;
  const x2 = b.x;
  const y2 = b.y + NODE_H / 2;
  // Vertical-ish edges leave from bottom/top.
  if (Math.abs(a.x - b.x) < 20) {
    const yStart = a.y < b.y ? a.y + NODE_H : a.y;
    const yEnd = a.y < b.y ? b.y : b.y + NODE_H;
    const x = a.x + NODE_W / 2;
    return `M ${x} ${yStart} L ${x} ${yEnd}`;
  }
  if (Math.abs(a.y - b.y) < 20) {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
  // Elbow: horizontal then vertical into target.
  const midX = (x1 + x2) / 2;
  return `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`;
}

function renderGraphSvg(
  doc: RoadmapStatusDoc,
  activePhase: string | "all",
): string {
  const statuses = phaseStatusMap(doc);
  const edges = doc.graph.edges
    .map((e) => {
      const d = edgePath(e.from, e.to);
      if (!d) return "";
      return `<path class="rm-edge" d="${d}" />`;
    })
    .join("");

  // Independent A → G dashed edge already in edges; also show K as floating.
  const nodes = Object.entries(NODE_LAYOUT)
    .map(([id, pos]) => {
      const st = statuses.get(id) ?? "not-started";
      const active = activePhase === id ? " active" : "";
      return `
        <g class="rm-node ${st}${active}" data-phase="${id}" transform="translate(${pos.x},${pos.y})">
          <rect width="${NODE_W}" height="${NODE_H}" rx="1" ry="1" />
          <text x="8" y="18">${id}</text>
          <text class="rm-node-label" x="26" y="18">${escapeHtml(pos.label)}</text>
        </g>`;
    })
    .join("");

  return `
    <svg class="rm-graph" viewBox="0 0 660 300" role="img" aria-label="Roadmap phase dependency graph">
      ${edges}
      ${nodes}
    </svg>`;
}

function renderItems(
  items: RoadmapWorkItem[],
  phaseFilter: string | "all",
): string {
  const filtered =
    phaseFilter === "all"
      ? items
      : items.filter((item) => item.phase === phaseFilter);

  if (filtered.length === 0) {
    return `<div class="rm-empty">No work items for this filter.</div>`;
  }

  return filtered
    .map((item) => {
      const waiver = item.waiver
        ? `<div class="rm-item-note">Waiver: ${escapeHtml(item.waiver.criterion)} — measured ${escapeHtml(item.waiver.measured)}; track ${escapeHtml(item.waiver.tracked_metric)}.</div>`
        : "";
      const note = item.note
        ? `<div class="rm-item-note">${escapeHtml(item.note)}</div>`
        : "";
      return `
        <article class="rm-item" data-id="${escapeHtml(item.id)}">
          <div class="rm-item-id">${escapeHtml(item.id)}</div>
          <div class="rm-item-status ${item.status}">${statusLabel(item.status)}</div>
          <div class="rm-item-body">
            <div class="rm-item-title">${escapeHtml(item.title)}</div>
            <dl class="rm-item-field">
              <dt>Measure</dt>
              <dd>${escapeHtml(item.measure)}</dd>
            </dl>
            <dl class="rm-item-field">
              <dt>Verify</dt>
              <dd class="verify"><code>${escapeHtml(item.verify)}</code></dd>
            </dl>
            ${waiver}
            ${note}
            <div class="rm-item-actions">
              <button type="button" class="rm-copy" data-copy="${escapeHtml(item.verify)}">Copy verify</button>
            </div>
          </div>
        </article>`;
    })
    .join("");
}

function countsByStatus(items: RoadmapWorkItem[]): string {
  const tally: Record<WorkItemStatus, number> = {
    done: 0,
    partial: 0,
    blocked: 0,
    "not-started": 0,
  };
  for (const item of items) {
    if (isStatus(item.status)) tally[item.status] += 1;
  }
  return `${tally.done} done · ${tally.partial} partial · ${tally.blocked} blocked · ${tally["not-started"]} not started`;
}

/**
 * Mount the ROADMAP section. Loads status JSON; filter by phase; graph matches
 * docs/roadmap.md §3.
 */
export const mountRoadmap: PanelMount = (el, _ctx: PanelContext) => {
  el.id = "roadmap";
  el.classList.add("roadmap-panel");
  el.innerHTML = `
    <header class="rm-head">
      <div>
        <div class="rm-kicker">Execution plan</div>
        <h2 class="rm-title">Roadmap</h2>
      </div>
      <div class="rm-meta" data-rm="meta">Loading status…</div>
    </header>
    <div class="rm-filters" data-rm="filters" role="toolbar" aria-label="Filter by phase"></div>
    <section class="rm-section">
      <div class="rm-section-hd">
        <h3>Phase dependency graph · §3</h3>
        <span class="rm-count" data-rm="graph-meta"></span>
      </div>
      <div class="rm-graph-wrap" data-rm="graph"></div>
      <ul class="rm-graph-notes" data-rm="notes"></ul>
    </section>
    <section class="rm-section">
      <div class="rm-section-hd">
        <h3>Phases</h3>
        <span class="rm-count" data-rm="phase-meta"></span>
      </div>
      <div class="rm-phase-strip" data-rm="phases"></div>
    </section>
    <section class="rm-section">
      <div class="rm-section-hd">
        <h3>Work items</h3>
        <span class="rm-count" data-rm="item-meta"></span>
      </div>
      <div class="rm-items" data-rm="items"></div>
    </section>
    <section class="rm-section">
      <div class="rm-section-hd"><h3>Updating status</h3></div>
      <div class="rm-update" data-rm="update"></div>
    </section>
  `;

  const metaEl = el.querySelector<HTMLElement>("[data-rm=meta]")!;
  const filtersEl = el.querySelector<HTMLElement>("[data-rm=filters]")!;
  const graphEl = el.querySelector<HTMLElement>("[data-rm=graph]")!;
  const notesEl = el.querySelector<HTMLElement>("[data-rm=notes]")!;
  const phasesEl = el.querySelector<HTMLElement>("[data-rm=phases]")!;
  const itemsEl = el.querySelector<HTMLElement>("[data-rm=items]")!;
  const itemMetaEl = el.querySelector<HTMLElement>("[data-rm=item-meta]")!;
  const phaseMetaEl = el.querySelector<HTMLElement>("[data-rm=phase-meta]")!;
  const graphMetaEl = el.querySelector<HTMLElement>("[data-rm=graph-meta]")!;
  const updateEl = el.querySelector<HTMLElement>("[data-rm=update]")!;

  let doc: RoadmapStatusDoc | null = null;
  let phaseFilter: string | "all" = "all";
  let alive = true;

  const setFilter = (next: string | "all"): void => {
    phaseFilter = next;
    render();
  };

  const render = (): void => {
    if (!doc || !alive) return;

    const phaseIds = doc.phases.map((p) => p.id);
    filtersEl.innerHTML = [
      `<button type="button" class="rm-filter${phaseFilter === "all" ? " active" : ""}" data-phase="all">All</button>`,
      ...phaseIds.map(
        (id) =>
          `<button type="button" class="rm-filter${phaseFilter === id ? " active" : ""}" data-phase="${id}">${id}</button>`,
      ),
    ].join("");

    graphEl.innerHTML = renderGraphSvg(doc, phaseFilter);
    notesEl.innerHTML = doc.graph.notes
      .map((n) => `<li>${escapeHtml(n)}</li>`)
      .join("");
    graphMetaEl.textContent = `${doc.graph.edges.length} edges · K throughout`;

    phasesEl.innerHTML = doc.phases
      .map((p) => {
        const active = phaseFilter === p.id ? " active" : "";
        return `<button type="button" class="rm-phase-chip ${p.status}${active}" data-phase="${p.id}" title="${escapeHtml(p.title)}">
          <span class="rm-phase-id">${p.id}</span>
          <span class="rm-phase-st">${statusLabel(p.status)}</span>
        </button>`;
      })
      .join("");
    phaseMetaEl.textContent = `${doc.phases.length} phases A–K`;

    const visible =
      phaseFilter === "all"
        ? doc.work_items
        : doc.work_items.filter((w) => w.phase === phaseFilter);
    itemsEl.innerHTML = renderItems(doc.work_items, phaseFilter);
    itemMetaEl.textContent =
      phaseFilter === "all"
        ? countsByStatus(doc.work_items)
        : `${visible.length} items · phase ${phaseFilter}`;

    const path = doc.update_path?.length
      ? doc.update_path
      : [
          "Edit packages/viewer/public/dashboard/roadmap-status.json",
          "Panel reloads the JSON on next mount (refresh the section).",
        ];
    updateEl.innerHTML = `
      <p><strong>JSON is the source of truth</strong> for this panel — not hard-coded TS.
      Path: <code>packages/viewer/public/dashboard/roadmap-status.json</code>
      (served as <code>${STATUS_URL}</code>).</p>
      <ol>${path.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol>
    `;

    const pref = doc.session_prefill
      ? Object.entries(doc.session_prefill)
          .map(([k, v]) => `${k}: ${v}`)
          .join(" · ")
      : "";
    metaEl.textContent = [
      doc.updated ? `Updated ${doc.updated}` : null,
      pref || null,
    ]
      .filter(Boolean)
      .join(" — ");
  };

  const onClick = (ev: Event): void => {
    const target = ev.target as HTMLElement | null;
    if (!target) return;

    const copyBtn = target.closest<HTMLButtonElement>(".rm-copy");
    if (copyBtn?.dataset.copy) {
      void navigator.clipboard?.writeText(copyBtn.dataset.copy).catch(() => {
        /* ignore */
      });
      copyBtn.textContent = "Copied";
      window.setTimeout(() => {
        if (alive) copyBtn.textContent = "Copy verify";
      }, 900);
      return;
    }

    const phaseBtn = target.closest<HTMLElement>("[data-phase]");
    if (phaseBtn?.dataset.phase) {
      const next = phaseBtn.dataset.phase;
      setFilter(next === "all" ? "all" : next);
    }
  };

  el.addEventListener("click", onClick);

  void (async () => {
    try {
      const res = await fetch(STATUS_URL);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const raw = (await res.json()) as RoadmapStatusDoc;
      if (!raw?.phases?.length || !raw?.work_items?.length) {
        throw new Error("roadmap-status.json missing phases or work_items");
      }
      if (!alive) return;
      doc = raw;
      render();
    } catch (err) {
      if (!alive) return;
      const message = err instanceof Error ? err.message : String(err);
      metaEl.textContent = "Status load failed";
      graphEl.innerHTML = `<div class="rm-error">Could not load ${STATUS_URL}: ${escapeHtml(message)}</div>`;
      updateEl.innerHTML = `
        <p>Create or fix <code>packages/viewer/public/dashboard/roadmap-status.json</code>,
        then remount this section.</p>`;
    }
  })();

  return () => {
    alive = false;
    el.removeEventListener("click", onClick);
  };
};

/** Soft-register with the shell registry when available. */
export function registerRoadmapPanel(registry: {
  registerPanel: (id: SectionId, mount: PanelMount) => void;
}): void {
  registry.registerPanel(ROADMAP_SECTION, mountRoadmap);
}

registerPanel(ROADMAP_SECTION, mountRoadmap);

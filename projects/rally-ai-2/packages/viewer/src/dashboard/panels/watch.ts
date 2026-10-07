/**
 * Dashboard WATCH / replay browser.
 *
 * Lists demos + proving-ground + exported policy replays, shows seed / tier /
 * duration / frames, and deep-links into the existing WATCH player (`?r=`).
 * Does not clone the Three.js player — optional mini path sketch only.
 *
 * Catalog sources (merged, later wins on same id):
 *   1. Static manifest: `public/dashboard/replays.json`
 *   2. Control room (optional): `GET {apiBase}/api/replays`
 *   3. Local file picker (session-only entries)
 *
 * Control-agent note — expected list endpoint (not implemented here):
 *   GET /api/replays → { replays: ReplayEntry[] }
 *   Same fields as the static manifest entries (id, url, watchParam, …).
 *   Generated / gitignored exports under viewer/public/replays/ should appear
 *   here so the browser stays current without regenerating the JSON catalog.
 */

import { registerPanel } from "../registry";
import type { PanelContext } from "../types";
import { btn, el, panelHeader, stat } from "../ui";
import "./watch.css";

const MANIFEST_URL = "dashboard/replays.json";
const API_LIST_PATH = "/api/replays";

export type ReplayKind =
  | "demo"
  | "proving_ground"
  | "policy"
  | "human"
  | "eval"
  | "other";

export interface PathPoint {
  x: number;
  y: number;
}

/** One browsable replay — shared by static manifest + control API. */
export interface ReplayEntry {
  id: string;
  label?: string;
  /** Fetchable JSON URL (relative to viewer origin or absolute). */
  url: string;
  /** Maps to WATCH `?r=<watchParam>` (stem under public/replays/). */
  watchParam?: string;
  kind?: ReplayKind | string;
  source?: string;
  stageId?: string;
  seed?: number | null;
  tier?: number | null;
  duration_s?: number | null;
  frames?: number | null;
  termination?: string | null;
  car?: string | null;
  clean?: boolean | null;
  checkpoint?: string | null;
  /** Optional downsampled centerline for the mini sketch. */
  path?: PathPoint[];
}

interface ReplaysManifest {
  schema_version?: number;
  note?: string;
  control_api?: unknown;
  replays?: ReplayEntry[];
}

interface ApiReplaysResponse {
  replays?: ReplayEntry[];
}

type FilterId = "all" | ReplayKind;

const FILTERS: { id: FilterId; label: string }[] = [
  { id: "all", label: "All" },
  { id: "demo", label: "Demo" },
  { id: "proving_ground", label: "Proving" },
  { id: "policy", label: "Policy" },
  { id: "human", label: "Human" },
  { id: "eval", label: "Eval" },
];

function kindOf(entry: ReplayEntry): ReplayKind {
  const k = (entry.kind ?? "").toLowerCase();
  if (
    k === "demo" ||
    k === "proving_ground" ||
    k === "policy" ||
    k === "human" ||
    k === "eval"
  ) {
    return k;
  }
  if (entry.source === "human") return "human";
  if (entry.source === "agent" || entry.checkpoint) return "policy";
  if (entry.source === "eval") return "eval";
  if (entry.id === "proving_ground" || entry.stageId === "proving_ground") {
    return "proving_ground";
  }
  if (entry.id === "demo" || entry.source === "replay_test") return "demo";
  return "other";
}

function watchHref(entry: ReplayEntry): string {
  const param = entry.watchParam ?? entry.id;
  // Prefer the existing WATCH contract (`?r=stem`). `?replay=` is accepted as
  // an alias by the viewer when it points at a full path/URL.
  if (entry.watchParam || !entry.url.startsWith("blob:")) {
    return `/?r=${encodeURIComponent(param)}`;
  }
  return `/?replay=${encodeURIComponent(entry.url)}`;
}

function formatDuration(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return `${m}m ${rem.toFixed(0)}s`;
}

function formatInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

function formatSeed(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return String(n);
}

function formatTier(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return `T${n}`;
}

function mergeById(...lists: ReplayEntry[][]): ReplayEntry[] {
  const map = new Map<string, ReplayEntry>();
  for (const list of lists) {
    for (const entry of list) {
      if (!entry?.id) continue;
      const prev = map.get(entry.id);
      map.set(entry.id, prev ? { ...prev, ...entry } : entry);
    }
  }
  return [...map.values()].sort((a, b) => {
    const ka = kindOf(a);
    const kb = kindOf(b);
    if (ka !== kb) return ka.localeCompare(kb);
    return (a.label ?? a.id).localeCompare(b.label ?? b.id);
  });
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

function apiUrl(apiBase: string, path: string): string {
  const base = apiBase.replace(/\/$/, "");
  return `${base}${path}`;
}

/** Pull listing fields from a full replay document (local file / enrich). */
export function entryFromReplayDoc(
  doc: Record<string, unknown>,
  opts: { id: string; url: string; watchParam?: string; kind?: ReplayKind },
): ReplayEntry {
  const meta = (doc.meta ?? {}) as Record<string, unknown>;
  const stage = (doc.stage ?? {}) as Record<string, unknown>;
  const gen = (stage.generator ?? {}) as Record<string, unknown>;
  const frames = Array.isArray(doc.frames) ? doc.frames.length : null;
  const cl = Array.isArray(stage.centerline)
    ? (stage.centerline as { x?: number; y?: number }[])
    : [];
  const step = Math.max(1, Math.floor(cl.length / 48));
  const path: PathPoint[] = [];
  for (let i = 0; i < cl.length; i += step) {
    const p = cl[i]!;
    path.push({ x: Number(p.x) || 0, y: Number(p.y) || 0 });
  }
  if (cl.length && path.length) {
    const last = cl[cl.length - 1]!;
    const tip = { x: Number(last.x) || 0, y: Number(last.y) || 0 };
    const prev = path[path.length - 1]!;
    if (prev.x !== tip.x || prev.y !== tip.y) path.push(tip);
  }
  const entry: ReplayEntry = {
    id: opts.id,
    label: typeof stage.name === "string" ? stage.name : opts.id,
    url: opts.url,
    kind: opts.kind ?? "other",
    seed:
      typeof gen.seed === "number"
        ? gen.seed
        : typeof meta.seed === "number"
          ? meta.seed
          : null,
    tier: typeof gen.tier === "number" ? gen.tier : null,
    duration_s: typeof meta.time_s === "number" ? meta.time_s : null,
    frames,
    termination: typeof meta.termination === "string" ? meta.termination : null,
    car: typeof meta.car === "string" ? meta.car : null,
    clean: typeof meta.clean === "boolean" ? meta.clean : null,
  };
  if (opts.watchParam !== undefined) entry.watchParam = opts.watchParam;
  if (typeof doc.source === "string") entry.source = doc.source;
  if (typeof stage.id === "string") entry.stageId = stage.id;
  if (typeof meta.checkpoint === "string") entry.checkpoint = meta.checkpoint;
  if (path.length >= 2) entry.path = path;
  return entry;
}

function drawPath(canvas: HTMLCanvasElement, points: PathPoint[] | undefined): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 320;
  const cssH = canvas.clientHeight || 160;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#040608";
  ctx.fillRect(0, 0, cssW, cssH);

  if (!points || points.length < 2) {
    canvas.classList.add("empty");
    ctx.fillStyle = "#6f7b88";
    ctx.font = "11px ui-monospace, SF Mono, Menlo, monospace";
    ctx.fillText("No path preview", 12, 24);
    return;
  }
  canvas.classList.remove("empty");

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const pad = 14;
  const spanX = Math.max(1e-3, maxX - minX);
  const spanY = Math.max(1e-3, maxY - minY);
  const scale = Math.min((cssW - pad * 2) / spanX, (cssH - pad * 2) / spanY);

  const map = (p: PathPoint): [number, number] => {
    const x = pad + (p.x - minX) * scale + ((cssW - pad * 2) - spanX * scale) / 2;
    // Stage Y → screen (flip so +Y reads "up" the page).
    const y =
      cssH -
      (pad + (p.y - minY) * scale + ((cssH - pad * 2) - spanY * scale) / 2);
    return [x, y];
  };

  ctx.strokeStyle = "rgba(246, 244, 233, 0.55)";
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  points.forEach((p, i) => {
    const [x, y] = map(p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const [sx, sy] = map(points[0]!);
  const [ex, ey] = map(points[points.length - 1]!);
  ctx.fillStyle = "#47d978";
  ctx.beginPath();
  ctx.arc(sx, sy, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#f2d21a";
  ctx.beginPath();
  ctx.arc(ex, ey, 3, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Mount the WATCH / replay browser into a dashboard content region.
 * Section id: `watch`.
 */
export function mountWatch(
  elHost: HTMLElement,
  ctx: PanelContext,
): () => void {
  elHost.replaceChildren();
  elHost.classList.add("watch-panel");

  const header = panelHeader("Replay browser", "loading…");
  const status = el("div", { className: "watch-status", text: "" });

  const filterRow = el("div", { className: "watch-filters" });
  const filterBtns = new Map<FilterId, HTMLButtonElement>();
  let filter: FilterId = "all";

  for (const f of FILTERS) {
    const b = el("button", {
      className: `watch-filter${f.id === filter ? " active" : ""}`,
      text: f.label,
      attrs: { type: "button", "data-filter": f.id },
    });
    b.addEventListener("click", () => {
      filter = f.id;
      for (const [id, btnEl] of filterBtns) {
        btnEl.classList.toggle("active", id === filter);
      }
      renderList();
    });
    filterBtns.set(f.id, b);
    filterRow.appendChild(b);
  }

  const refreshBtn = btn({
    label: "Refresh",
    onClick: () => {
      void loadCatalog();
    },
  });

  const fileLabel = el("label", { className: "dash-btn watch-file", text: "Open file…" });
  const fileInput = el("input", {
    attrs: { type: "file", accept: "application/json,.json" },
  });
  fileLabel.appendChild(fileInput);

  const toolbar = el(
    "div",
    { className: "watch-toolbar" },
    filterRow,
    refreshBtn,
    fileLabel,
  );

  const listEl = el("div", {
    className: "watch-list",
    attrs: { role: "listbox", "aria-label": "Available replays" },
  });
  const detailEl = el("div", { className: "watch-detail" });
  const layout = el("div", { className: "watch-layout" }, listEl, detailEl);

  const note = el("p", {
    className: "watch-note",
    text:
      "Open in WATCH uses the live player (`?r=`). Policy exports that land in " +
      "public/replays/ appear after Refresh when listed in the manifest or via " +
      "GET /api/replays (control room).",
  });

  elHost.append(header.root, toolbar, status, layout, note);

  let entries: ReplayEntry[] = [];
  let selectedId: string | null = null;
  let localBlobUrls: string[] = [];
  let disposed = false;

  const seedStat = stat({ label: "Seed", value: "—" });
  const tierStat = stat({ label: "Tier", value: "—" });
  const durStat = stat({ label: "Duration", value: "—" });
  const framesStat = stat({ label: "Frames", value: "—" });

  const sketch = el("canvas", {
    className: "watch-sketch empty",
    attrs: { "aria-label": "Stage path preview" },
  }) as HTMLCanvasElement;

  function selected(): ReplayEntry | undefined {
    return entries.find((e) => e.id === selectedId);
  }

  function filtered(): ReplayEntry[] {
    if (filter === "all") return entries;
    return entries.filter((e) => kindOf(e) === filter);
  }

  function renderList(): void {
    const rows = filtered();
    listEl.replaceChildren();
    if (rows.length === 0) {
      listEl.appendChild(
        el("div", {
          className: "watch-list-empty",
          text:
            entries.length === 0
              ? "No replays found. Export one into public/replays/ or open a JSON file."
              : "No replays match this filter.",
        }),
      );
      return;
    }
    for (const entry of rows) {
      const kind = kindOf(entry);
      const item = el("button", {
        className: `watch-item${entry.id === selectedId ? " active" : ""}`,
        attrs: {
          type: "button",
          role: "option",
          "aria-selected": entry.id === selectedId ? "true" : "false",
        },
      });
      item.append(
        el("div", {
          className: "watch-item-title",
          text: entry.label ?? entry.id,
        }),
        el("div", { className: "watch-item-kind", text: kind }),
        el("div", {
          className: "watch-item-meta",
          text: [
            formatTier(entry.tier),
            `seed ${formatSeed(entry.seed)}`,
            formatDuration(entry.duration_s),
            `${formatInt(entry.frames)} fr`,
          ].join(" · "),
        }),
      );
      item.addEventListener("click", () => {
        selectedId = entry.id;
        renderList();
        renderDetail();
      });
      listEl.appendChild(item);
    }
  }

  function renderDetail(): void {
    detailEl.replaceChildren();
    const entry = selected();
    if (!entry) {
      detailEl.appendChild(
        el("div", {
          className: "watch-detail-empty",
          text: "Select a replay to inspect metadata and open WATCH.",
        }),
      );
      return;
    }

    seedStat.set(formatSeed(entry.seed));
    tierStat.set(formatTier(entry.tier));
    durStat.set(formatDuration(entry.duration_s));
    framesStat.set(formatInt(entry.frames));

    const href = watchHref(entry);
    const openLink = el("a", {
      className: "watch-link primary",
      text: "Open in WATCH",
      attrs: { href, target: "_blank", rel: "noopener" },
    });
    const sameTab = el("a", {
      className: "watch-link",
      text: "Open here",
      attrs: { href },
    });
    const copyBtn = btn({
      label: "Copy link",
      onClick: async () => {
        const abs = new URL(href, location.origin).href;
        try {
          await navigator.clipboard.writeText(abs);
          status.textContent = `Copied ${abs}`;
          status.classList.remove("warn");
        } catch {
          status.textContent = abs;
          status.classList.remove("warn");
        }
      },
    });

    const metaBits = [
      entry.stageId ? `stage ${entry.stageId}` : null,
      entry.source ? `source ${entry.source}` : null,
      entry.termination ? entry.termination : null,
      entry.car ? entry.car : null,
      entry.clean === true ? "clean" : entry.clean === false ? "dirty" : null,
      entry.checkpoint ? `ckpt ${entry.checkpoint}` : null,
    ]
      .filter(Boolean)
      .join(" · ");

    detailEl.append(
      el(
        "div",
        { className: "watch-detail-hd" },
        el("h2", {
          className: "watch-detail-name",
          text: entry.label ?? entry.id,
        }),
        el("div", {
          className: "watch-detail-sub",
          text: metaBits || entry.id,
        }),
      ),
      el(
        "div",
        { className: "watch-stats" },
        seedStat.root,
        tierStat.root,
        durStat.root,
        framesStat.root,
      ),
      sketch,
      el("div", { className: "watch-actions" }, openLink, sameTab, copyBtn),
      el("div", {
        className: "watch-note",
        text: `Deep link ${href} · ${entry.url}`,
      }),
    );

    requestAnimationFrame(() => drawPath(sketch, entry.path));
  }

  async function loadCatalog(): Promise<void> {
    if (disposed) return;
    header.setMeta("loading…");
    status.textContent = "";
    status.classList.remove("warn");

    const manifest = await fetchJson<ReplaysManifest>(MANIFEST_URL);
    const fromManifest = manifest?.replays ?? [];

    const api = await fetchJson<ApiReplaysResponse>(
      apiUrl(ctx.apiBase, API_LIST_PATH),
    );
    const fromApi = api?.replays ?? [];

    const locals = entries.filter((e) => e.url.startsWith("blob:"));
    entries = mergeById(fromManifest, fromApi, locals);

    const sources: string[] = [];
    if (fromManifest.length) sources.push(`manifest ${fromManifest.length}`);
    if (api && fromApi.length) sources.push(`api ${fromApi.length}`);
    else if (api) sources.push("api 0");
    if (locals.length) sources.push(`local ${locals.length}`);
    if (!fromManifest.length && !fromApi.length && !locals.length) {
      sources.push("empty");
    }

    header.setMeta(sources.join(" · "));
    if (!api) {
      // Soft note only — control endpoint is optional.
      status.textContent =
        fromManifest.length > 0
          ? "Control GET /api/replays unavailable — using static manifest."
          : "No catalog: static manifest missing and GET /api/replays unavailable.";
      if (!fromManifest.length) status.classList.add("warn");
    }

    if (!selectedId || !entries.some((e) => e.id === selectedId)) {
      selectedId = entries[0]?.id ?? null;
    }
    renderList();
    renderDetail();
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (!file) return;
    const blobUrl = URL.createObjectURL(file);
    localBlobUrls.push(blobUrl);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const doc = JSON.parse(String(reader.result)) as Record<string, unknown>;
        const stem = file.name.replace(/\.json$/i, "") || `local-${Date.now()}`;
        const entry = entryFromReplayDoc(doc, {
          id: `local:${stem}`,
          url: blobUrl,
          kind: "other",
        });
        // Local blobs cannot use `?r=` (no public file). Keep for inspection;
        // deep-link still points at stem so a matching public file would open.
        entry.watchParam = stem;
        entries = mergeById(entries, [entry]);
        selectedId = entry.id;
        header.setMeta(`${entries.length} replays · local file`);
        status.textContent =
          "Local file loaded for metadata. Open in WATCH only works if the same " +
          `stem exists at public/replays/${stem}.json — copy it there to play.`;
        status.classList.remove("warn");
        renderList();
        renderDetail();
      } catch (err) {
        status.textContent = `Could not parse replay: ${(err as Error).message}`;
        status.classList.add("warn");
        URL.revokeObjectURL(blobUrl);
        localBlobUrls = localBlobUrls.filter((u) => u !== blobUrl);
      }
    };
    reader.readAsText(file);
  });

  const onResize = (): void => {
    const entry = selected();
    if (entry) drawPath(sketch, entry.path);
  };
  window.addEventListener("resize", onResize);

  void loadCatalog();

  return () => {
    disposed = true;
    window.removeEventListener("resize", onResize);
    for (const u of localBlobUrls) URL.revokeObjectURL(u);
    localBlobUrls = [];
    elHost.classList.remove("watch-panel");
    elHost.replaceChildren();
  };
}

registerPanel("watch", mountWatch);

import type { PanelContext } from "../types";

interface SectorResult {
  name: string;
  time_s: number | null;
  optimal_time_s: number | null;
  time_vs_optimal: number | null;
}

interface EpisodeResult {
  seed: number;
  tier: number;
  termination: string;
  time_s: number;
  finished: boolean;
  time_vs_optimal?: number | null;
  sectors?: SectorResult[];
}

interface EvalReport {
  schema_version?: number;
  kind?: string;
  checkpoint?: string | null;
  actor?: string;
  tier: number;
  seeds?: number[];
  completion_rate: number;
  mean_time_s: number | null;
  time_vs_optimal: number | null;
  terminations: Record<string, number>;
  episodes: EpisodeResult[];
  json_path?: string;
}

interface LoadedEval {
  id: string;
  label: string;
  report: EvalReport;
}

interface TierSummary {
  tier: number;
  total: number;
  finished: number;
  completion: number;
  meanTime: number | null;
  timeVsOptimal: number | null;
}

const TERMINATION_ORDER = [
  "finish",
  "crash",
  "off_course",
  "spun",
  "stuck",
  "timeout",
  "other",
] as const;

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseReport(value: unknown): EvalReport {
  if (!isRecord(value)) throw new Error("eval JSON must contain an object");
  if (value.kind !== undefined && value.kind !== "eval") {
    throw new Error(`expected kind "eval", got ${String(value.kind)}`);
  }
  const tier = finite(value.tier);
  const completion = finite(value.completion_rate);
  if (tier === null || completion === null || !Array.isArray(value.episodes)) {
    throw new Error("missing tier, completion_rate, or episodes");
  }

  const episodes: EpisodeResult[] = value.episodes.map((raw, index) => {
    if (!isRecord(raw)) throw new Error(`episode ${index + 1} is not an object`);
    const seed = finite(raw.seed);
    const episodeTier = finite(raw.tier);
    const time = finite(raw.time_s);
    if (seed === null || episodeTier === null || time === null) {
      throw new Error(`episode ${index + 1} has invalid seed, tier, or time_s`);
    }
    const sectors = Array.isArray(raw.sectors)
      ? raw.sectors.flatMap((sector): SectorResult[] => {
          if (!isRecord(sector) || typeof sector.name !== "string") return [];
          return [{
            name: sector.name,
            time_s: finite(sector.time_s),
            optimal_time_s: finite(sector.optimal_time_s),
            time_vs_optimal: finite(sector.time_vs_optimal),
          }];
        })
      : undefined;
    return {
      seed,
      tier: episodeTier,
      termination:
        typeof raw.termination === "string" ? raw.termination : "other",
      time_s: time,
      finished: raw.finished === true,
      time_vs_optimal: finite(raw.time_vs_optimal),
      ...(sectors ? { sectors } : {}),
    };
  });

  const terminations: Record<string, number> = {};
  if (isRecord(value.terminations)) {
    for (const [key, count] of Object.entries(value.terminations)) {
      const parsed = finite(count);
      if (parsed !== null) terminations[key] = parsed;
    }
  }

  return {
    tier,
    completion_rate: completion,
    mean_time_s: finite(value.mean_time_s),
    time_vs_optimal: finite(value.time_vs_optimal),
    terminations,
    episodes,
    ...(typeof value.schema_version === "number"
      ? { schema_version: value.schema_version }
      : {}),
    ...(typeof value.kind === "string" ? { kind: value.kind } : {}),
    ...(typeof value.checkpoint === "string" || value.checkpoint === null
      ? { checkpoint: value.checkpoint }
      : {}),
    ...(typeof value.actor === "string" ? { actor: value.actor } : {}),
    ...(Array.isArray(value.seeds)
      ? { seeds: value.seeds.filter((seed): seed is number => finite(seed) !== null) }
      : {}),
    ...(typeof value.json_path === "string" ? { json_path: value.json_path } : {}),
  };
}

function labelFor(report: EvalReport, fallback: string): string {
  if (report.checkpoint) {
    const parts = report.checkpoint.split(/[\\/]/);
    return `${parts.at(-1) ?? report.checkpoint} · T${report.tier}`;
  }
  if (report.actor) return `${report.actor} · T${report.tier}`;
  return `${fallback} · T${report.tier}`;
}

function percent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function seconds(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(2)}s`;
}

function ratio(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(2)}×`;
}

function average(values: Array<number | null>): number | null {
  const usable = values.filter((value): value is number => value !== null);
  return usable.length
    ? usable.reduce((sum, value) => sum + value, 0) / usable.length
    : null;
}

function tiers(report: EvalReport): TierSummary[] {
  const grouped = new Map<number, EpisodeResult[]>();
  for (const episode of report.episodes) {
    const rows = grouped.get(episode.tier) ?? [];
    rows.push(episode);
    grouped.set(episode.tier, rows);
  }
  if (grouped.size === 0) grouped.set(report.tier, []);

  return [...grouped.entries()]
    .sort(([a], [b]) => a - b)
    .map(([tier, episodes]) => {
      const finished = episodes.filter((episode) => episode.finished);
      const ratios = finished.map((episode) => finite(episode.time_vs_optimal));
      const singleReport = grouped.size === 1 && tier === report.tier;
      return {
        tier,
        total: episodes.length,
        finished: finished.length,
        completion: episodes.length
          ? finished.length / episodes.length
          : report.completion_rate,
        meanTime: finished.length
          ? average(finished.map((episode) => episode.time_s))
          : singleReport
            ? report.mean_time_s
            : null,
        timeVsOptimal:
          average(ratios) ?? (singleReport ? report.time_vs_optimal : null),
      };
    });
}

function terminationCounts(report: EvalReport): Record<string, number> {
  if (Object.keys(report.terminations).length) return report.terminations;
  const counts: Record<string, number> = {};
  for (const episode of report.episodes) {
    counts[episode.termination] = (counts[episode.termination] ?? 0) + 1;
  }
  return counts;
}

function setText(parent: Element, selector: string, text: string): void {
  const node = parent.querySelector(selector);
  if (node) node.textContent = text;
}

function td(text: string, className = ""): HTMLTableCellElement {
  const cell = document.createElement("td");
  cell.textContent = text;
  cell.className = className;
  return cell;
}

function delta(current: number | null, comparison: number | null, scale = 1): string {
  if (current === null || comparison === null) return "";
  const value = (current - comparison) * scale;
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

/**
 * Mount held-out evaluation evidence. Data can come from `/api/evals`, the
 * static `/dashboard/evals/index.json` manifest, a URL, or local JSON files.
 */
export function mountEvals(
  el: HTMLElement,
  ctx: PanelContext,
): () => void {
  el.id = "evals";
  el.classList.add("evals-panel");
  el.innerHTML = `
    <style>
      .evals-panel{--ev-ink:#e8eadf;--ev-muted:#92998d;--ev-line:#30372f;--ev-green:#a7d46f;--ev-amber:#efb55f;color:var(--ev-ink);font-variant-numeric:tabular-nums}
      .evals-panel *{box-sizing:border-box}.ev-head{display:flex;gap:20px;align-items:end;justify-content:space-between;margin-bottom:24px}.ev-kicker{color:var(--ev-green);font:700 11px/1.2 monospace;letter-spacing:.18em;text-transform:uppercase}.ev-title{margin:5px 0 0;font-size:28px;line-height:1}.ev-source{color:var(--ev-muted);font:12px/1.4 monospace;max-width:52ch;text-align:right}
      .ev-loader{border:1px dashed #4a5348;padding:15px;display:grid;grid-template-columns:minmax(170px,1fr) auto auto;gap:8px;align-items:center;background:#171b17;margin-bottom:18px}.ev-loader.drag{border-color:var(--ev-green);background:#1d261a}.ev-loader input,.ev-loader select,.ev-loader button{height:34px;border:1px solid #3e473d;background:#101310;color:var(--ev-ink);font:12px monospace;padding:0 10px}.ev-loader button{cursor:pointer;text-transform:uppercase;letter-spacing:.08em}.ev-loader button:hover{border-color:var(--ev-green)}.ev-file{position:absolute;inline-size:1px;block-size:1px;opacity:0}
      .ev-message{grid-column:1/-1;color:var(--ev-muted);font:11px/1.4 monospace}.ev-message.error{color:#ef8c74}.ev-choices{display:flex;gap:10px;align-items:center;margin-bottom:18px}.ev-choices label{color:var(--ev-muted);font:10px monospace;text-transform:uppercase;letter-spacing:.12em}.ev-choices select{min-width:180px;background:#101310;border:1px solid var(--ev-line);color:var(--ev-ink);padding:7px;font:12px monospace}
      .ev-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--ev-line);border:1px solid var(--ev-line);margin-bottom:20px}.ev-stat{background:#151915;padding:16px}.ev-stat-label{color:var(--ev-muted);font:10px monospace;letter-spacing:.12em;text-transform:uppercase}.ev-stat-value{font:600 27px/1.15 monospace;margin-top:8px}.ev-stat-delta{color:var(--ev-amber);font:11px monospace;min-height:14px;margin-top:4px}
      .ev-grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(260px,.9fr);gap:18px}.ev-card{border-top:2px solid #4b5549;background:#151915;padding:15px;min-width:0}.ev-card h3{font:700 11px monospace;letter-spacing:.14em;text-transform:uppercase;margin:0 0 14px;color:#bbc1b5}.ev-table{border-collapse:collapse;width:100%;font:12px monospace}.ev-table th{text-align:left;color:var(--ev-muted);font-weight:400;padding:7px 8px;border-bottom:1px solid var(--ev-line)}.ev-table td{padding:8px;border-bottom:1px solid #252b25}.ev-table td:not(:first-child),.ev-table th:not(:first-child){text-align:right}.ev-positive{color:var(--ev-green)}.ev-bars{display:grid;gap:9px}.ev-bar-row{display:grid;grid-template-columns:88px minmax(0,1fr) 32px;align-items:center;gap:9px;font:11px monospace}.ev-bar-label{color:var(--ev-muted);overflow:hidden;text-overflow:ellipsis}.ev-bar-track{height:8px;background:#252b25}.ev-bar-fill{display:block;height:100%;min-width:2px;background:var(--ev-amber)}.ev-bar-row:first-child .ev-bar-fill{background:var(--ev-green)}.ev-empty{padding:32px 15px;border:1px solid var(--ev-line);color:var(--ev-muted);text-align:center;font:12px/1.6 monospace}.ev-sectors{grid-column:1/-1;overflow:auto}
      @media(max-width:760px){.ev-head{display:block}.ev-source{text-align:left;margin-top:10px}.ev-loader{grid-template-columns:1fr 1fr}.ev-loader input{grid-column:1/-1}.ev-grid{grid-template-columns:1fr}.ev-stats{grid-template-columns:1fr}.ev-sectors{grid-column:auto}.ev-choices{align-items:stretch;flex-direction:column}.ev-choices select{width:100%}}
    </style>
    <header class="ev-head">
      <div><div class="ev-kicker">Held-out evidence</div><h2 class="ev-title">Evals</h2></div>
      <div class="ev-source">Mean-action policy · frozen normaliser · reserved seeds</div>
    </header>
    <div class="ev-loader" tabindex="0">
      <input class="ev-url" aria-label="Eval JSON URL" value="/api/evals" />
      <button class="ev-load" type="button">Load URL</button>
      <button class="ev-pick" type="button">Open JSON</button>
      <input class="ev-file" type="file" accept=".json,application/json" multiple />
      <div class="ev-message">Drop one or more <code>*.eval.json</code> files here to compare checkpoints.</div>
    </div>
    <div class="ev-content"><div class="ev-empty">No eval loaded.<br>Choose a Phase D eval JSON or drop it anywhere above.</div></div>
  `;

  const abort = new AbortController();
  const loaded = new Map<string, LoadedEval>();
  const loader = el.querySelector<HTMLElement>(".ev-loader")!;
  const fileInput = el.querySelector<HTMLInputElement>(".ev-file")!;
  const urlInput = el.querySelector<HTMLInputElement>(".ev-url")!;
  const content = el.querySelector<HTMLElement>(".ev-content")!;
  let primaryId = "";
  let compareId = "";
  let alive = true;

  const message = (text: string, error = false): void => {
    const node = el.querySelector<HTMLElement>(".ev-message");
    if (!node) return;
    node.textContent = text;
    node.classList.toggle("error", error);
  };

  const add = (value: unknown, source: string): void => {
    const report = parseReport(value);
    const base = `${source}:${report.checkpoint ?? report.actor ?? "eval"}:${report.tier}`;
    const id = `${base}:${loaded.size}`;
    loaded.set(id, { id, label: labelFor(report, source), report });
    primaryId = id;
  };

  const render = (): void => {
    const primary = loaded.get(primaryId) ?? [...loaded.values()].at(-1);
    if (!primary) return;
    primaryId = primary.id;
    const comparison = compareId ? loaded.get(compareId) : undefined;
    const report = primary.report;
    const tierRows = tiers(report);
    const comparisonTiers = comparison ? tiers(comparison.report) : [];
    const total = report.episodes.length || report.seeds?.length || 0;

    content.innerHTML = `
      <div class="ev-choices">
        <label>Report <select class="ev-primary"></select></label>
        <label>Compare <select class="ev-compare"><option value="">None</option></select></label>
      </div>
      <div class="ev-stats">
        <div class="ev-stat"><div class="ev-stat-label">Completion</div><div class="ev-stat-value ev-completion"></div><div class="ev-stat-delta ev-completion-delta"></div></div>
        <div class="ev-stat"><div class="ev-stat-label">Mean finish time</div><div class="ev-stat-value ev-time"></div><div class="ev-stat-delta ev-time-delta"></div></div>
        <div class="ev-stat"><div class="ev-stat-label">vs theoretical min</div><div class="ev-stat-value ev-optimal"></div><div class="ev-stat-delta ev-optimal-delta"></div></div>
      </div>
      <div class="ev-grid">
        <section class="ev-card"><h3>Completion by tier</h3><table class="ev-table ev-tier-table"><thead><tr><th>Tier</th><th>Finished</th><th>Rate</th><th>Mean</th><th>Optimal</th></tr></thead><tbody></tbody></table></section>
        <section class="ev-card"><h3>Terminations · ${total} episodes</h3><div class="ev-bars"></div></section>
        <section class="ev-card ev-sectors"><h3>Sector analysis</h3><div class="ev-sector-body"></div></section>
      </div>
    `;

    setText(content, ".ev-completion", percent(report.completion_rate));
    setText(content, ".ev-time", seconds(report.mean_time_s));
    setText(content, ".ev-optimal", ratio(report.time_vs_optimal));
    if (comparison) {
      setText(
        content,
        ".ev-completion-delta",
        `${delta(report.completion_rate, comparison.report.completion_rate, 100)} pp vs compare`,
      );
      setText(
        content,
        ".ev-time-delta",
        `${delta(report.mean_time_s, comparison.report.mean_time_s)}s vs compare`,
      );
      setText(
        content,
        ".ev-optimal-delta",
        `${delta(report.time_vs_optimal, comparison.report.time_vs_optimal)}× vs compare`,
      );
    }

    const primarySelect = content.querySelector<HTMLSelectElement>(".ev-primary")!;
    const compareSelect = content.querySelector<HTMLSelectElement>(".ev-compare")!;
    for (const item of loaded.values()) {
      const option = new Option(item.label, item.id, false, item.id === primary.id);
      primarySelect.add(option);
      if (item.id !== primary.id) {
        compareSelect.add(new Option(item.label, item.id, false, item.id === compareId));
      }
    }
    primarySelect.addEventListener("change", () => {
      primaryId = primarySelect.value;
      if (compareId === primaryId) compareId = "";
      render();
    });
    compareSelect.addEventListener("change", () => {
      compareId = compareSelect.value;
      render();
    });

    const tierBody = content.querySelector<HTMLTableSectionElement>(".ev-tier-table tbody")!;
    for (const row of tierRows) {
      const tr = document.createElement("tr");
      const other = comparisonTiers.find((candidate) => candidate.tier === row.tier);
      tr.append(
        td(`T${row.tier}`),
        td(`${row.finished}/${row.total}`),
        td(
          `${percent(row.completion)}${other ? ` (${delta(row.completion, other.completion, 100)}pp)` : ""}`,
          "ev-positive",
        ),
        td(seconds(row.meanTime)),
        td(ratio(row.timeVsOptimal)),
      );
      tierBody.append(tr);
    }

    const counts = terminationCounts(report);
    const entries = [
      ...TERMINATION_ORDER,
      ...Object.keys(counts).filter(
        (key) => !TERMINATION_ORDER.includes(key as (typeof TERMINATION_ORDER)[number]),
      ),
    ].filter((key, index, all) => all.indexOf(key) === index && (counts[key] ?? 0) > 0);
    const maximum = Math.max(1, ...entries.map((key) => counts[key] ?? 0));
    const bars = content.querySelector<HTMLElement>(".ev-bars")!;
    for (const key of entries) {
      const count = counts[key] ?? 0;
      const row = document.createElement("div");
      row.className = "ev-bar-row";
      row.innerHTML = `<span class="ev-bar-label"></span><span class="ev-bar-track"><span class="ev-bar-fill"></span></span><span></span>`;
      setText(row, ".ev-bar-label", key.replaceAll("_", " "));
      const fill = row.querySelector<HTMLElement>(".ev-bar-fill")!;
      fill.style.width = `${(count / maximum) * 100}%`;
      row.lastElementChild!.textContent = String(count);
      bars.append(row);
    }

    const sectorBody = content.querySelector<HTMLElement>(".ev-sector-body")!;
    const sectorNames = [...new Set(
      report.episodes.flatMap((episode) => episode.sectors?.map((sector) => sector.name) ?? []),
    )];
    if (!sectorNames.length) {
      sectorBody.className = "ev-empty";
      sectorBody.textContent = "No per-sector rows in this eval.";
    } else {
      const table = document.createElement("table");
      table.className = "ev-table";
      table.innerHTML = "<thead><tr><th>Sector</th><th>Samples</th><th>Mean time</th><th>Theoretical</th><th>Ratio</th></tr></thead><tbody></tbody>";
      const body = table.querySelector("tbody")!;
      for (const name of sectorNames) {
        const sectors = report.episodes.flatMap(
          (episode) => episode.sectors?.filter((sector) => sector.name === name) ?? [],
        );
        const times = sectors.map((sector) => sector.time_s);
        const optimals = sectors.map((sector) => sector.optimal_time_s);
        const ratios = sectors.map((sector) => sector.time_vs_optimal);
        const tr = document.createElement("tr");
        tr.append(
          td(name),
          td(String(times.filter((value) => value !== null).length)),
          td(seconds(average(times))),
          td(seconds(average(optimals))),
          td(ratio(average(ratios)), "ev-positive"),
        );
        body.append(tr);
      }
      sectorBody.append(table);
    }
  };

  const loadPayload = async (value: unknown, source: string): Promise<number> => {
    if (Array.isArray(value)) {
      for (const [index, item] of value.entries()) add(item, `${source} #${index + 1}`);
      return value.length;
    }
    if (isRecord(value) && Array.isArray(value.evals)) {
      let count = 0;
      for (const [index, item] of value.evals.entries()) {
        if (typeof item === "string") {
          count += await loadUrl(new URL(item, location.href).toString(), false);
        } else {
          add(item, `${source} #${index + 1}`);
          count++;
        }
      }
      return count;
    }
    if (isRecord(value) && Array.isArray(value.files)) {
      let count = 0;
      for (const item of value.files) {
        if (typeof item !== "string") continue;
        count += await loadUrl(
          new URL(item, new URL("/dashboard/evals/", location.href)).toString(),
          false,
        );
      }
      return count;
    }
    add(value, source);
    return 1;
  };

  const loadUrl = async (url: string, announce = true): Promise<number> => {
    if (announce) message(`Loading ${url}…`);
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      signal: abort.signal,
    });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const count = await loadPayload(await response.json(), url);
    if (alive) {
      render();
      if (announce) message(`Loaded ${count} eval${count === 1 ? "" : "s"} from ${url}.`);
    }
    return count;
  };

  const loadFiles = async (files: FileList | File[]): Promise<void> => {
    let count = 0;
    for (const file of files) {
      try {
        count += await loadPayload(JSON.parse(await file.text()) as unknown, file.name);
      } catch (error) {
        message(`${file.name}: ${(error as Error).message}`, true);
        return;
      }
    }
    render();
    message(`Loaded ${count} eval${count === 1 ? "" : "s"} from disk.`);
  };

  el.querySelector(".ev-pick")?.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files) void loadFiles(fileInput.files);
    fileInput.value = "";
  });
  el.querySelector(".ev-load")?.addEventListener("click", () => {
    void loadUrl(urlInput.value.trim()).catch((error: unknown) => {
      message(`Could not load URL: ${(error as Error).message}`, true);
    });
  });
  urlInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      (el.querySelector(".ev-load") as HTMLButtonElement | null)?.click();
    }
  });
  loader.addEventListener("dragover", (event) => {
    event.preventDefault();
    loader.classList.add("drag");
  });
  loader.addEventListener("dragleave", () => loader.classList.remove("drag"));
  loader.addEventListener("drop", (event) => {
    event.preventDefault();
    loader.classList.remove("drag");
    if (event.dataTransfer?.files.length) void loadFiles(event.dataTransfer.files);
  });

  // Static copies are listed here for production builds. Development also
  // probes the control service endpoint, which can return one eval, an array,
  // or `{ evals: [...] }`.
  void loadUrl("/dashboard/evals/index.json", false)
    .then((count) => {
      if (count) message(`Loaded ${count} bundled eval${count === 1 ? "" : "s"}.`);
    })
    .catch(() => undefined);
  const apiUrl = `${ctx.apiBase}/api/evals`;
  void loadUrl(apiUrl, false)
    .then((count) => {
      if (count) message(`Loaded ${count} eval${count === 1 ? "" : "s"} from control.`);
    })
    .catch(() => undefined);

  return () => {
    alive = false;
    abort.abort();
  };
}

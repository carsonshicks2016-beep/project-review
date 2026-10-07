/**
 * Viewer-side client for hall-of-fame / checkpoint listings.
 *
 * Control contract (packages/sim/rallyai/control):
 *
 *   GET /api/runs/{id}/checkpoints
 *   → {
 *       run_id: string,
 *       checkpoints: Array<{
 *         run_id: string,
 *         name: string,
 *         path: string,
 *         slot: "latest" | "best" | "cleanest" | "fastest" | "furthest" | null,
 *         mtime: number | null,   // unix seconds
 *         size: number | null,    // bytes
 *         // Cheap torch meta (from checkpoint.py payload; soft-fail if unloadable):
 *         timesteps?: number,
 *         tier?: number,
 *         stage?: string,
 *         run_id?: string,
 *         policy_sha256?: string,
 *         obs_dim?: number,
 *         act_dim?: number,
 *         obs_layout_version?: number,
 *         // Optional extras when present on the payload:
 *         has_normaliser?: boolean,
 *         extra?: Record<string, unknown>,
 *         mean_return?: number,
 *         clean_rate?: number,
 *         mean_finish_time?: number,
 *         max_progress?: number,
 *       }>
 *     }
 *
 *   GET /api/runs → { runs: RunDetail[] }  (control-room jobs)
 *   GET /api/dashboard/summary → includes checkpoints.by_run for disk discovery
 *
 * HoF slot names match `rallyai.train.checkpoint.HOF_SLOTS`.
 */

import type { RunDetail } from "../../train/types";

/** Canonical hall-of-fame slots (order = display order). */
export const HOF_SLOTS = [
  "latest",
  "best",
  "cleanest",
  "fastest",
  "furthest",
] as const;

export type HofSlot = (typeof HOF_SLOTS)[number];

/** Shared with LIVE panel — path selected via "Use for live eval". */
export const DASH_LIVE_CHECKPOINT_KEY = "rallyai.dashboard.liveCheckpoint";

export interface CheckpointEntry {
  run_id: string;
  name: string;
  path: string;
  slot: HofSlot | string | null;
  mtime: number | null;
  size: number | null;
  timesteps?: number | null;
  tier?: number | null;
  stage?: string | null;
  policy_sha256?: string | null;
  obs_dim?: number | null;
  act_dim?: number | null;
  obs_layout_version?: number | null;
  has_normaliser?: boolean | null;
  extra?: Record<string, unknown> | null;
  mean_return?: number | null;
  clean_rate?: number | null;
  mean_finish_time?: number | null;
  max_progress?: number | null;
}

export interface RunCheckpointsResponse {
  run_id: string;
  checkpoints: CheckpointEntry[];
}

export interface DashboardCheckpointCounts {
  total: number;
  by_slot: Record<string, number>;
  by_run: Record<string, number>;
}

export interface DashboardSummary {
  runs_dir?: string;
  runs_total?: number;
  checkpoints?: DashboardCheckpointCounts;
  active_runs?: RunDetail[];
}

function normalizeBase(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

function apiUrl(apiBase: string, path: string): string {
  const base = normalizeBase(apiBase);
  if (!base) return path;
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
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

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asString(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

function parseEntry(raw: unknown, fallbackRunId: string): CheckpointEntry | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const path = asString(r.path);
  const name = asString(r.name) ?? (path ? path.split(/[/\\]/).pop() ?? path : null);
  if (!path || !name) return null;

  const slotRaw = r.slot;
  const slot =
    typeof slotRaw === "string" && slotRaw.length > 0
      ? slotRaw
      : slotRaw === null
        ? null
        : null;

  const extra =
    r.extra && typeof r.extra === "object" && !Array.isArray(r.extra)
      ? (r.extra as Record<string, unknown>)
      : null;

  return {
    run_id: asString(r.run_id) ?? fallbackRunId,
    name,
    path,
    slot,
    mtime: asNumber(r.mtime),
    size: asNumber(r.size) ?? asNumber(r.size_bytes),
    timesteps: asNumber(r.timesteps),
    tier: asNumber(r.tier),
    stage: asString(r.stage),
    policy_sha256: asString(r.policy_sha256) ?? asString(r.policy_hash),
    obs_dim: asNumber(r.obs_dim),
    act_dim: asNumber(r.act_dim),
    obs_layout_version: asNumber(r.obs_layout_version),
    has_normaliser:
      typeof r.has_normaliser === "boolean" ? r.has_normaliser : null,
    extra,
    mean_return:
      asNumber(r.mean_return) ??
      (extra ? asNumber(extra.mean_return) : null),
    clean_rate:
      asNumber(r.clean_rate) ?? (extra ? asNumber(extra.clean_rate) : null),
    mean_finish_time:
      asNumber(r.mean_finish_time) ??
      (extra ? asNumber(extra.mean_finish_time) : null),
    max_progress:
      asNumber(r.max_progress) ??
      (extra ? asNumber(extra.max_progress) : null),
  };
}

/** Fetch HoF / checkpoint files for one run. */
export async function fetchRunCheckpoints(
  apiBase: string,
  runId: string,
  signal?: AbortSignal,
): Promise<RunCheckpointsResponse> {
  const res = await fetch(
    apiUrl(apiBase, `/api/runs/${encodeURIComponent(runId)}/checkpoints`),
    signal ? { signal } : {},
  );
  if (!res.ok) throw new Error(await readErr(res));
  const data = (await res.json()) as unknown;

  if (Array.isArray(data)) {
    return {
      run_id: runId,
      checkpoints: data
        .map((row) => parseEntry(row, runId))
        .filter((x): x is CheckpointEntry => x != null),
    };
  }

  if (!data || typeof data !== "object") {
    throw new Error("checkpoints response is not an object");
  }
  const obj = data as Record<string, unknown>;
  const id = asString(obj.run_id) ?? runId;
  const list = Array.isArray(obj.checkpoints)
    ? obj.checkpoints
    : Array.isArray(obj.slots)
      ? obj.slots
      : Array.isArray(obj.files)
        ? obj.files
        : [];

  return {
    run_id: id,
    checkpoints: list
      .map((row) => parseEntry(row, id))
      .filter((x): x is CheckpointEntry => x != null),
  };
}

/** Control-room job list (may miss flat CLI runs that only have HoF files). */
export async function fetchControlRuns(
  apiBase: string,
  signal?: AbortSignal,
): Promise<RunDetail[]> {
  const res = await fetch(apiUrl(apiBase, "/api/runs"), signal ? { signal } : {});
  if (!res.ok) throw new Error(await readErr(res));
  const data = (await res.json()) as { runs?: RunDetail[] };
  return data.runs ?? [];
}

/** Dashboard summary — use `checkpoints.by_run` to discover disk runs. */
export async function fetchDashboardSummary(
  apiBase: string,
  signal?: AbortSignal,
): Promise<DashboardSummary | null> {
  const res = await fetch(
    apiUrl(apiBase, "/api/dashboard/summary"),
    signal ? { signal } : {},
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(await readErr(res));
  return (await res.json()) as DashboardSummary;
}

/**
 * Union of control-room runs + runs that have checkpoint files on disk.
 * Sorted newest-ish first (active / control runs ahead of bare stems).
 */
export async function discoverRunIds(
  apiBase: string,
  signal?: AbortSignal,
): Promise<string[]> {
  const ids = new Set<string>();
  const control: string[] = [];

  try {
    const runs = await fetchControlRuns(apiBase, signal);
    for (const r of runs) {
      if (r.run_id) {
        ids.add(r.run_id);
        control.push(r.run_id);
      }
    }
  } catch {
    /* control may be down */
  }

  try {
    const summary = await fetchDashboardSummary(apiBase, signal);
    const byRun = summary?.checkpoints?.by_run;
    if (byRun) {
      for (const id of Object.keys(byRun)) {
        if (id && id !== "_other") ids.add(id);
      }
    }
  } catch {
    /* summary endpoint optional during rollout */
  }

  const rest = [...ids].filter((id) => !control.includes(id)).sort().reverse();
  return [...control, ...rest];
}

export function evaluateCommand(checkpointPath: string): string {
  return (
    `python -m rallyai.train.eval_cli ` +
    `--checkpoint ${shellQuote(checkpointPath)} --tiers 0 1 2`
  );
}

function shellQuote(path: string): string {
  if (/^[A-Za-z0-9_./:@+-]+$/.test(path)) return path;
  return `'${path.replace(/'/g, `'\\''`)}'`;
}

export function setLiveCheckpoint(path: string): void {
  try {
    localStorage.setItem(DASH_LIVE_CHECKPOINT_KEY, path);
  } catch {
    /* ignore */
  }
}

export function readLiveCheckpoint(): string | null {
  try {
    return localStorage.getItem(DASH_LIVE_CHECKPOINT_KEY);
  } catch {
    return null;
  }
}

export function isHofSlot(slot: string | null | undefined): slot is HofSlot {
  return !!slot && (HOF_SLOTS as readonly string[]).includes(slot);
}

/** Map slot → entry; missing slots stay absent. */
export function indexBySlot(
  checkpoints: CheckpointEntry[],
): Map<HofSlot, CheckpointEntry> {
  const map = new Map<HofSlot, CheckpointEntry>();
  for (const ck of checkpoints) {
    if (isHofSlot(ck.slot)) map.set(ck.slot, ck);
  }
  return map;
}

export const SLOT_BLURB: Record<HofSlot, string> = {
  latest: "Most recent write",
  best: "Highest mean return",
  cleanest: "Highest clean finish rate",
  fastest: "Lowest mean finish time",
  furthest: "Highest progress",
};

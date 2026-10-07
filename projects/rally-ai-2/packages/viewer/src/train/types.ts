/**
 * Control-room types for TRAIN (Phase G1).
 *
 * Matches the planned F1/F2 API in docs/roadmap.md and docs/MIGRATION_FROM_V1.md.
 * Port-F lands the FastAPI server under packages/sim/rallyai/control/; these
 * types are the client contract so G can ship before F is wired.
 */

export type RunState = "starting" | "running" | "stopping" | "stopped" | "error";

/** Local UI state: idle/armed live only in the tab; the rest mirror the server. */
export type UiJobState = "idle" | "armed" | RunState;

export interface RunConfig {
  stage: string;
  workers: number;
  timesteps: number;
  /** Curriculum tier 0–5 on RallyAI2. */
  tier: number;
}

export const DEFAULT_RUN_CONFIG: RunConfig = {
  stage: "proving_ground",
  workers: 1,
  timesteps: 50_000,
  tier: 0,
};

export interface RunDetail {
  run_id: string;
  state: RunState;
  created_at?: number;
  started_at?: number;
  ended_at?: number | null;
  updated_at?: number;
  config?: RunConfig & { train_module?: string };
  error?: string | null;
  exit_code?: number | null;
  pid?: number | null;
  metrics_path?: string;
}

export type MetricKind =
  | "run_start"
  | "update"
  | "eval"
  | "checkpoint"
  | "curriculum"
  | "worker"
  | "run_end";

export interface MetricsLine {
  schema_version: 1;
  kind: MetricKind | string;
  run_id: string;
  wall_t: number;
  timesteps?: number;
  reward?: {
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    n?: number;
  };
  terms?: Record<string, number>;
  completion?: {
    rate?: number;
    finish?: number;
    crash?: number;
    off_course?: number;
    stuck?: number;
    spun?: number;
    timeout?: number;
  };
  tier?: number;
  ppo?: {
    policy_loss?: number;
    value_loss?: number;
    entropy?: number;
    approx_kl?: number;
    clip_frac?: number;
    explained_var?: number;
    lr?: number;
    grad_norm?: number;
    rejected?: boolean;
  };
  throughput?: {
    steps_per_s?: number;
    n_workers?: number;
    rollout_s?: number;
    update_s?: number;
  };
  eval?: {
    seeds?: number[];
    completion_rate?: number;
    mean_time_s?: number;
    time_vs_optimal?: number;
    clean_rate?: number;
    json_path?: string;
  };
  checkpoint?: {
    path?: string;
    slot?: string;
    policy_hash?: string;
    has_normaliser?: boolean;
  };
  worker?: {
    index?: number;
    event?: string;
    detail?: string;
  };
  msg?: string;
  /** Stream control messages from the WS helper (not schema lines). */
  type?: "error" | "stream_end";
  message?: string;
  state?: RunState;
  next?: number;
}

export interface MetricsBatch {
  run_id: string;
  since: number;
  next: number;
  lines: MetricsLine[];
}

/**
 * Empty base = Vite proxies `/api` to the control room on :8765.
 * Absolute URL still works when the control room exposes CORS.
 */
export const DEFAULT_CONTROL_BASE = "";

export const TRAIN_SESSION_KEY = "rallyai.train.session";
export const TRAIN_BASE_URL_KEY = "rallyai.train.baseUrl";

export interface TrainSession {
  runId: string;
  since: number;
}

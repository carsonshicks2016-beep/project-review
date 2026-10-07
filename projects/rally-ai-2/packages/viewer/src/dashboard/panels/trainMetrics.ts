/**
 * Dashboard TRAIN / METRICS panel — section id `train`.
 *
 * Wraps existing G1 pieces (TrainClient, ControlRail job-only, Sparkline,
 * MetricsPanel, CurriculumNumerals) into the dashboard shell content region.
 */

import { ControlRail } from "../../train/ControlRail";
import { CurriculumNumerals } from "../../train/CurriculumNumerals";
import { MetricsPanel } from "../../train/MetricsPanel";
import { Sparkline } from "../../train/Sparkline";
import {
  clearSession,
  loadSession,
  TrainClient,
} from "../../train/TrainClient";
import {
  DEFAULT_RUN_CONFIG,
  type MetricsLine,
  type RunConfig,
  type RunDetail,
  type UiJobState,
} from "../../train/types";
import type { PanelContext, PanelMount, SectionId } from "../types";
import { registerPanel } from "../DashboardApp";
import "../../train/train.css";
import "./trainMetrics.css";

export const TRAIN_SECTION_ID: SectionId = "train";

/**
 * Mount TRAIN metrics into the dashboard main region.
 * Returns a disposer for the shell to call on section leave.
 */
export function mountTrainMetrics(
  el: HTMLElement,
  ctx: PanelContext,
): () => void {
  const panel = new TrainMetricsPanel(el, ctx);
  void panel.enter();
  return () => panel.destroy();
}

/** Register with the shell registry when available. */
export function registerTrainMetrics(
  register: (id: SectionId, mount: PanelMount) => void = registerPanel,
): void {
  register(TRAIN_SECTION_ID, mountTrainMetrics);
}

registerPanel(TRAIN_SECTION_ID, mountTrainMetrics);

class TrainMetricsPanel {
  private host: HTMLElement;
  private ctx: PanelContext;
  private root: HTMLElement | null = null;

  private client: TrainClient;
  private rail: ControlRail | null = null;
  private rewardSpark: Sparkline | null = null;
  private completionSpark: Sparkline | null = null;
  private numerals: CurriculumNumerals | null = null;
  private metrics: MetricsPanel | null = null;

  private bannerEl: HTMLElement | null = null;
  private captionEl: HTMLElement | null = null;
  private runIdEl: HTMLElement | null = null;
  private pathEl: HTMLElement | null = null;
  private sinceEl: HTMLElement | null = null;
  private connEl: HTMLElement | null = null;

  private config: RunConfig = { ...DEFAULT_RUN_CONFIG };
  private active = false;
  private statusPoll: number | null = null;
  private onResizeBound = (): void => this.onResize();

  constructor(host: HTMLElement, ctx: PanelContext) {
    this.host = host;
    this.ctx = ctx;
    this.client = new TrainClient({
      onMetric: (m) => this.onMetric(m),
      onRun: (run) => this.onRun(run),
      onStreamEnd: (run) => {
        if (run) this.onRun(run);
      },
      onError: (message) => {
        this.rail?.setError(message);
        console.error("[dash/train]", message);
      },
      onConnection: (on) => this.setConnected(on),
    });
    if (ctx.apiBase) this.client.setBaseUrl(ctx.apiBase);
  }

  async enter(): Promise<void> {
    this.active = true;
    this.host.replaceChildren();

    this.root = document.createElement("div");
    this.root.className = "dash-train";
    this.root.innerHTML = `
      <aside class="dash-train-rail" id="dt-rail"></aside>
      <section class="dash-train-stage">
        <div class="dash-train-charts">
          <div class="train-banner" id="dt-banner" hidden></div>
          <div class="dash-train-chart">
            <div class="chart-label">Reward</div>
            <canvas id="dt-reward"></canvas>
          </div>
          <div class="dash-train-chart">
            <div class="chart-label">Completion</div>
            <canvas id="dt-completion"></canvas>
          </div>
          <div class="train-caption" id="dt-caption">reward · completion · steps/s</div>
          <div class="dash-train-meta">
            <span class="k">Run</span>
            <span class="v" id="dt-run-id">—</span>
            <span class="k">Metrics</span>
            <span class="v path" id="dt-path" title="">—</span>
            <span class="k">Since</span>
            <span class="v" id="dt-since">0</span>
            <span class="k">Stream</span>
            <span class="v dash-train-conn" id="dt-conn" data-on="0">idle</span>
          </div>
          <div class="dash-train-links">
            <button type="button" class="dash-train-nav" id="dt-to-live">Open Live →</button>
          </div>
        </div>
        <div class="dash-train-side">
          <div class="curriculum-block" id="dt-curr"></div>
          <div id="dt-metrics"></div>
        </div>
      </section>
    `;
    this.host.appendChild(this.root);

    const railSlot = this.root.querySelector("#dt-rail") as HTMLElement;
    this.rail = new ControlRail(
      railSlot,
      {
        onArm: () => this.arm(),
        onStart: () => void this.start(),
        onStop: () => void this.stop(),
        onConfigChange: (c) => {
          this.config = c;
        },
        onBaseUrlChange: (url) => {
          this.client.setBaseUrl(url);
          void this.probeAndResume();
        },
      },
      { jobOnly: true },
    );
    this.rail.setBaseUrl(this.client.getBaseUrl());
    this.rail.setConfig(this.config);

    this.bannerEl = this.root.querySelector("#dt-banner");
    this.captionEl = this.root.querySelector("#dt-caption");
    this.runIdEl = this.root.querySelector("#dt-run-id");
    this.pathEl = this.root.querySelector("#dt-path");
    this.sinceEl = this.root.querySelector("#dt-since");
    this.connEl = this.root.querySelector("#dt-conn");

    this.rewardSpark = new Sparkline(
      this.root.querySelector("#dt-reward") as HTMLCanvasElement,
      { stroke: "#C8FF3D" },
    );
    this.completionSpark = new Sparkline(
      this.root.querySelector("#dt-completion") as HTMLCanvasElement,
      { stroke: "#F6F4E9" },
    );
    this.numerals = new CurriculumNumerals(
      this.root.querySelector("#dt-curr") as HTMLElement,
    );
    this.metrics = new MetricsPanel(
      this.root.querySelector("#dt-metrics") as HTMLElement,
    );
    this.numerals.setLevel(this.config.tier);

    this.root.querySelector("#dt-to-live")?.addEventListener("click", () => {
      this.ctx.navigate("live");
    });

    window.addEventListener("resize", this.onResizeBound);
    requestAnimationFrame(() => this.onResize());

    await this.probeAndResume();
  }

  destroy(): void {
    this.active = false;
    window.removeEventListener("resize", this.onResizeBound);
    this.stopStatusPoll();
    this.client.disconnect();
    this.rail?.destroy();
    this.rail = null;
    this.rewardSpark = null;
    this.completionSpark = null;
    this.numerals = null;
    this.metrics?.destroy();
    this.metrics = null;
    this.bannerEl = null;
    this.captionEl = null;
    this.runIdEl = null;
    this.pathEl = null;
    this.sinceEl = null;
    this.connEl = null;
    this.host.replaceChildren();
    this.root = null;
  }

  private onResize(): void {
    this.rewardSpark?.resize();
    this.completionSpark?.resize();
  }

  private async probeAndResume(): Promise<void> {
    this.stopStatusPoll();
    this.client.disconnect();
    this.setBanner(null);
    this.rail?.setBaseUrl(this.client.getBaseUrl());
    this.setConnected(false);

    try {
      await this.client.health();
    } catch {
      this.setUiState("idle", {
        error: "control room unreachable — start rallyai.control on :8765",
      });
      return;
    }

    const session = loadSession();
    let run: RunDetail | null = null;

    if (session?.runId) {
      try {
        run = await this.client.getRun(session.runId);
      } catch {
        clearSession();
      }
    }

    if (!run) {
      try {
        const runs = await this.client.listRuns();
        run =
          runs.find(
            (r) =>
              r.state === "running" ||
              r.state === "starting" ||
              r.state === "stopping",
          ) ??
          runs[0] ??
          null;
      } catch {
        run = null;
      }
    }

    if (!run) {
      this.setUiState("idle");
      this.setRunMeta(null);
      return;
    }

    if (run.config) {
      this.config = {
        stage: run.config.stage ?? this.config.stage,
        workers: run.config.workers ?? this.config.workers,
        timesteps: run.config.timesteps ?? this.config.timesteps,
        tier: run.config.tier ?? this.config.tier,
      };
      this.rail?.setConfig(this.config);
      this.numerals?.setLevel(this.config.tier);
    }

    this.onRun(run);

    const live =
      run.state === "starting" ||
      run.state === "running" ||
      run.state === "stopping";
    const since = session?.runId === run.run_id ? session.since : 0;

    this.rewardSpark?.clear();
    this.completionSpark?.clear();
    this.metrics?.clear();

    try {
      // Replay from 0 so sparklines rebuild after refresh; stream from max(since, next).
      const batch = await this.client.readMetrics(run.run_id, 0);
      for (const line of batch.lines) this.onMetric(line);
      this.setSince(Math.max(since, batch.next));
      if (live) {
        this.client.connect(run.run_id, Math.max(since, batch.next));
        this.startStatusPoll(run.run_id);
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.rail?.setError(message);
    }
  }

  private arm(): void {
    this.config = this.rail?.getConfig() ?? this.config;
    this.client.setBaseUrl(this.rail?.getBaseUrl() ?? this.client.getBaseUrl());
    this.setUiState("armed", { error: null });
  }

  private async start(): Promise<void> {
    try {
      this.config = this.rail?.getConfig() ?? this.config;
      this.client.setBaseUrl(this.rail?.getBaseUrl() ?? this.client.getBaseUrl());
      this.rewardSpark?.clear();
      this.completionSpark?.clear();
      this.metrics?.clear();
      this.setBanner(null);
      this.setSince(0);

      const run = await this.client.startRun(this.config);
      this.onRun(run);

      if (run.state === "error") {
        this.setBanner(run.error ?? "trainer failed to start");
        return;
      }

      this.client.connect(run.run_id, 0);
      this.startStatusPoll(run.run_id);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.setUiState("error", { error: message });
      this.setBanner(message);
    }
  }

  private async stop(): Promise<void> {
    try {
      const run = await this.client.stopRun();
      this.onRun(run);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.rail?.setError(message);
    }
  }

  private onRun(run: RunDetail): void {
    if (!this.active) return;
    this.setUiState(run.state as UiJobState, {
      runId: run.run_id,
      error: run.error ?? null,
    });
    this.setRunMeta(run);
    if (run.state === "error") {
      this.setBanner(run.error ?? "run error");
    } else if (run.state === "stopped") {
      this.setBanner(null);
      this.stopStatusPoll();
    } else if (
      run.state === "starting" ||
      run.state === "running" ||
      run.state === "stopping"
    ) {
      this.setBanner(null);
    }
  }

  private onMetric(m: MetricsLine): void {
    if (!this.active) return;
    if (
      m.kind === "update" ||
      m.kind === "run_start" ||
      m.kind === "curriculum" ||
      m.kind === "eval"
    ) {
      if (m.reward?.mean != null && m.timesteps != null) {
        this.rewardSpark?.push(m.timesteps, m.reward.mean);
      }
      if (m.completion?.rate != null && m.timesteps != null) {
        this.completionSpark?.push(m.timesteps, m.completion.rate);
      }
      if (m.tier != null) this.numerals?.setLevel(m.tier);
      this.metrics?.apply(m);
      this.setSince(this.client.since);

      const reward = m.reward?.mean != null ? m.reward.mean.toFixed(2) : "—";
      const rate =
        m.completion?.rate != null
          ? `${(m.completion.rate * 100).toFixed(0)}%`
          : "—";
      const sps =
        m.throughput?.steps_per_s != null
          ? m.throughput.steps_per_s.toFixed(0)
          : "—";
      if (this.captionEl) {
        this.captionEl.textContent = `reward ${reward} · completion ${rate} · ${sps} steps/s`;
      }
    }
  }

  private setUiState(
    state: UiJobState,
    opts: { runId?: string | null; error?: string | null } = {},
  ): void {
    this.rail?.setJobState(state, opts);
  }

  private setRunMeta(run: RunDetail | null): void {
    if (this.runIdEl) {
      this.runIdEl.textContent = run?.run_id ?? "—";
      this.runIdEl.title = run?.run_id ?? "";
    }
    const path =
      run?.metrics_path ??
      (run?.run_id ? `packages/sim/runs/${run.run_id}/metrics.jsonl` : null);
    if (this.pathEl) {
      this.pathEl.textContent = path ?? "—";
      this.pathEl.title = path ?? "";
    }
  }

  private setSince(n: number): void {
    if (this.sinceEl) this.sinceEl.textContent = String(Math.max(0, n));
  }

  private setConnected(on: boolean): void {
    if (!this.connEl) return;
    this.connEl.dataset.on = on ? "1" : "0";
    this.connEl.textContent = on ? "live" : "idle";
  }

  private setBanner(message: string | null): void {
    if (!this.bannerEl) return;
    if (!message) {
      this.bannerEl.hidden = true;
      this.bannerEl.textContent = "";
      return;
    }
    this.bannerEl.hidden = false;
    this.bannerEl.textContent = message;
  }

  private startStatusPoll(runId: string): void {
    this.stopStatusPoll();
    this.statusPoll = window.setInterval(() => {
      void this.client.getRun(runId).catch(() => {
        /* ignore transient */
      });
    }, 2000);
  }

  private stopStatusPoll(): void {
    if (this.statusPoll != null) {
      window.clearInterval(this.statusPoll);
      this.statusPoll = null;
    }
  }
}

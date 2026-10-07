/**
 * TRAIN — G1 metrics UI + G2 sensor overlay + G3 live viewport.
 *
 * Job control talks to F1/F2 (`/api/runs`). Live agent frames come from F3
 * (`/api/live`). Human drive is F4 (`/api/drive`) and reuses the same viewport.
 */

import { AgentStreamClient } from "./AgentStreamClient";
import { ControlRail } from "./ControlRail";
import { CurriculumNumerals } from "./CurriculumNumerals";
import { LiveClient } from "./LiveClient";
import { LiveViewport } from "./LiveViewport";
import { MetricsPanel } from "./MetricsPanel";
import { Sparkline } from "./Sparkline";
import {
  clearSession,
  loadSession,
  TrainClient,
} from "./TrainClient";
import type { SenseDebug } from "./sensors";
import {
  DEFAULT_RUN_CONFIG,
  type MetricsLine,
  type RunConfig,
  type RunDetail,
  type UiJobState,
} from "./types";
import type { Frame, Stage } from "../replay";
import "./train.css";

/**
 * Mount TRAIN into the page shell (replaces WATCH's #game contents).
 */
export async function bootTrain(host: HTMLElement): Promise<TrainMode> {
  const mode = new TrainMode(host);
  await mode.enter();
  return mode;
}

export class TrainMode {
  private host: HTMLElement;
  private root: HTMLElement | null = null;
  private railParent: HTMLElement | null = null;
  private stageParent: HTMLElement | null = null;

  private client: TrainClient;
  private stream: AgentStreamClient;
  private drive: LiveClient;

  private rail: ControlRail | null = null;
  private sparkline: Sparkline | null = null;
  private numerals: CurriculumNumerals | null = null;
  private metrics: MetricsPanel | null = null;
  private viewport: LiveViewport | null = null;
  private captionEl: HTMLElement | null = null;
  private bannerEl: HTMLElement | null = null;

  private config: RunConfig = { ...DEFAULT_RUN_CONFIG };
  private active = false;
  private statusPoll: number | null = null;
  private liveSeed = 7;
  private onResizeBound = (): void => this.onResize();

  constructor(host: HTMLElement) {
    this.host = host;
    this.client = new TrainClient({
      onMetric: (m) => this.onMetric(m),
      onRun: (run) => this.onRun(run),
      onStreamEnd: (run) => {
        if (run) this.onRun(run);
      },
      onError: (message) => {
        this.rail?.setError(message);
        console.error("[train]", message);
      },
    });
    this.stream = new AgentStreamClient({
      onHeader: (h) => this.onLiveHeader(h),
      onFrame: (frame, sense) => this.onLiveFrame(frame, sense),
      onEnd: (info) => {
        const term = String(info.termination ?? "end");
        this.rail?.setLiveState("ended", { pace: term });
      },
      onStreamEnd: () => {
        this.rail?.setLiveState("idle");
        this.viewport?.clearSense();
        this.viewport?.setEmpty(true);
      },
      onSession: (s) => {
        this.rail?.setLiveState(s.state, { sessionId: s.session_id });
        if (s.seed != null) {
          this.liveSeed = s.seed;
          this.rail?.setLiveSeed(s.seed);
        }
      },
      onError: (message) => {
        this.rail?.setError(message);
        this.rail?.setLiveState("error");
        console.error("[train/live]", message);
      },
      onConnection: (on) => {
        if (on) this.rail?.setLiveState("streaming");
      },
    });
    this.drive = new LiveClient({
      onStage: (stage) => this.onLiveHeader({ stage }),
      onFrame: (frame) => {
        this.lastDriveFrame = frame;
        // Sense may arrive in the same packet via onSense; apply pose immediately
        // so the car does not wait on a missing sidecar.
        this.viewport?.applyFrame(frame, null);
      },
      onSense: (sense) => {
        const frame = this.lastDriveFrame;
        if (!frame) return;
        this.viewport?.applyFrame(frame, sense);
        const pace =
          sense?.pace_note?.text ||
          sense?.pace_note?.dir ||
          frame.note ||
          "—";
        this.rail?.setLiveState("streaming", { pace: String(pace) });
      },
      onEnd: (info) => {
        const term = String(info.termination ?? "end");
        this.rail?.setLiveState("ended", { pace: term });
      },
      onStatus: (msg) => {
        if (msg.toLowerCase().includes("error")) this.rail?.setError(msg);
      },
      onSaved: (name) => {
        this.rail?.setError(null);
        this.rail?.setLiveState("streaming", { pace: `saved ${name}` });
      },
      onConnection: (on) => {
        this.rail?.setDriving(on);
        if (!on) {
          this.viewport?.clearSense();
          this.viewport?.setEmpty(true);
        }
      },
    });
  }

  private lastDriveFrame: Frame | null = null;

  async enter(): Promise<void> {
    this.active = true;
    this.host.innerHTML = "";
    this.host.classList.add("train-host");

    this.root = document.createElement("div");
    this.root.className = "train-app";
    this.root.innerHTML = `
      <aside class="train-rail-slot" id="train-rail"></aside>
      <section class="train-stage-slot" id="train-stage"></section>
    `;
    this.host.appendChild(this.root);
    this.railParent = this.root.querySelector("#train-rail")!;
    this.stageParent = this.root.querySelector("#train-stage")!;

    this.rail = new ControlRail(this.railParent, {
      onArm: () => this.arm(),
      onStart: () => void this.start(),
      onStop: () => void this.stop(),
      onConfigChange: (c) => {
        this.config = c;
      },
      onBaseUrlChange: (url) => {
        this.client.setBaseUrl(url);
        this.stream.setBaseUrl(url);
        this.drive.setBaseUrl(url);
        void this.probeAndResume();
      },
      onLiveStart: () => void this.startLive(),
      onLiveStop: () => void this.stopLive(),
      onLiveSeedChange: (seed) => {
        this.liveSeed = seed;
      },
      onDriveStart: () => void this.startDrive(),
      onDriveStop: () => this.stopDrive(),
    });
    this.rail.setBaseUrl(this.client.getBaseUrl());
    this.rail.setConfig(this.config);
    this.rail.setLiveSeed(this.liveSeed);

    const stage = document.createElement("div");
    stage.className = "train-stage";
    stage.innerHTML = `
      <div class="train-main">
        <div id="train-viewport-slot"></div>
        <div class="train-charts">
          <div class="train-banner" id="train-banner" hidden></div>
          <canvas id="spark-canvas"></canvas>
          <div class="train-caption" id="train-caption">reward · completion · steps/s</div>
        </div>
      </div>
      <div class="train-side">
        <div class="curriculum-block" id="curr-block"></div>
        <div id="metrics-block"></div>
      </div>
    `;
    this.stageParent.appendChild(stage);

    const viewportSlot = stage.querySelector("#train-viewport-slot") as HTMLElement;
    this.viewport = new LiveViewport(viewportSlot);
    this.viewport.start();
    this.viewport.setEmpty(true);

    const canvas = stage.querySelector("#spark-canvas") as HTMLCanvasElement;
    this.sparkline = new Sparkline(canvas);
    this.numerals = new CurriculumNumerals(stage.querySelector("#curr-block")!);
    this.metrics = new MetricsPanel(stage.querySelector("#metrics-block")!);
    this.captionEl = stage.querySelector("#train-caption");
    this.bannerEl = stage.querySelector("#train-banner");
    this.numerals.setLevel(this.config.tier);

    window.addEventListener("resize", this.onResizeBound);
    requestAnimationFrame(() => {
      this.sparkline?.resize();
      this.onResize();
    });

    await this.probeAndResume();
  }

  exit(): void {
    this.active = false;
    window.removeEventListener("resize", this.onResizeBound);
    this.stopStatusPoll();
    this.stopDrive();
    this.stream.disconnect();
    this.client.disconnect();
    this.viewport?.destroy();
    this.viewport = null;
    this.rail?.destroy();
    this.rail = null;
    this.sparkline = null;
    this.numerals = null;
    this.metrics?.destroy();
    this.metrics = null;
    this.captionEl = null;
    this.bannerEl = null;
    this.host.innerHTML = "";
    this.host.classList.remove("train-host");
  }

  onResize(): void {
    this.sparkline?.resize();
    this.viewport?.resize();
  }

  private onLiveHeader(h: { stage: Stage }): void {
    if (!this.active || !this.viewport) return;
    this.viewport.setStage(h.stage);
  }

  private onLiveFrame(frame: Frame, sense: SenseDebug | null): void {
    if (!this.active) return;
    this.lastDriveFrame = frame;
    // Prefer envelope sense; drive path may call applyFrame twice (frame then sense).
    this.viewport?.applyFrame(frame, sense);
    const pace =
      sense?.pace_note?.text ||
      sense?.pace_note?.dir ||
      frame.note ||
      "—";
    this.rail?.setLiveState("streaming", { pace: String(pace) });
  }

  private async startLive(): Promise<void> {
    try {
      this.stopDrive();
      this.stream.setBaseUrl(this.rail?.getBaseUrl() ?? this.client.getBaseUrl());
      this.liveSeed = this.rail?.getLiveSeed() ?? this.liveSeed;
      this.rail?.setError(null);
      this.rail?.setLiveState("starting");
      this.viewport?.clearSense();

      const session = await this.stream.start({
        seed: this.liveSeed,
        tier: this.config.tier,
        procedural: true,
        loop: true,
      });

      if (session.state === "error") {
        this.rail?.setLiveState("error", { sessionId: session.session_id });
        this.rail?.setError(session.error ?? "live session failed");
        return;
      }

      this.stream.connect(session.session_id, 0);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.rail?.setLiveState("error");
      this.rail?.setError(message);
    }
  }

  private async stopLive(): Promise<void> {
    try {
      await this.stream.stop();
      this.viewport?.clearSense();
      this.rail?.setLiveState("idle", { sessionId: null, pace: "—" });
      this.viewport?.setEmpty(true);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.rail?.setError(message);
    }
  }

  private async startDrive(): Promise<void> {
    try {
      await this.stopLive();
      const base = this.rail?.getBaseUrl() ?? this.client.getBaseUrl();
      this.drive.setBaseUrl(base);
      this.liveSeed = this.rail?.getLiveSeed() ?? this.liveSeed;
      this.rail?.setError(null);
      this.rail?.setDriving(true);
      this.viewport?.clearSense();
      this.drive.bindKeys();
      await this.drive.connect({
        seed: this.liveSeed,
        tier: this.config.tier,
        procedural: true,
        loop: true,
        save_replay: true,
        baseUrl: base,
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.drive.unbindKeys();
      this.rail?.setDriving(false);
      this.rail?.setError(message);
    }
  }

  private stopDrive(): void {
    this.drive.unbindKeys();
    this.drive.disconnect();
    this.rail?.setDriving(false);
    this.viewport?.clearSense();
    this.rail?.setLiveState("idle", { sessionId: null, pace: "—" });
    this.viewport?.setEmpty(true);
  }

  private async probeAndResume(): Promise<void> {
    this.stopStatusPoll();
    this.client.disconnect();
    this.stream.disconnect();
    this.setBanner(null);
    this.rail?.setBaseUrl(this.client.getBaseUrl());
    this.stream.setBaseUrl(this.client.getBaseUrl());

    try {
      await this.client.health();
    } catch {
      this.setUiState("idle", {
        error: "control room unreachable — start rallyai.control on :8765",
      });
      return;
    }

    try {
      const sessions = await this.stream.listSessions();
      const live =
        sessions.find((s) => s.state === "running" || s.state === "starting") ??
        null;
      if (live) {
        this.rail?.setLiveState(live.state, { sessionId: live.session_id });
        if (live.seed != null) {
          this.liveSeed = live.seed;
          this.rail?.setLiveSeed(live.seed);
        }
        // since=0 so the header (stage) always arrives.
        this.stream.connect(live.session_id, 0);
      } else {
        this.rail?.setLiveState("idle");
        this.viewport?.setEmpty(true);
      }
    } catch {
      this.rail?.setLiveState("idle");
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

    this.sparkline?.clear();
    this.metrics?.clear();
    try {
      const batch = await this.client.readMetrics(run.run_id, 0);
      for (const line of batch.lines) this.onMetric(line);
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
      this.sparkline?.clear();
      this.metrics?.clear();
      this.setBanner(null);

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
    const ui = run.state as UiJobState;
    this.setUiState(ui, { runId: run.run_id, error: run.error ?? null });
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
        this.sparkline?.push(m.timesteps, m.reward.mean);
      }
      if (m.tier != null) this.numerals?.setLevel(m.tier);
      this.metrics?.apply(m);

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

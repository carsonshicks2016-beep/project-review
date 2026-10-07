import {
  DEFAULT_CONTROL_BASE,
  DEFAULT_RUN_CONFIG,
  type RunConfig,
  type UiJobState,
} from "./types";

export type ControlRailHandlers = {
  onArm: () => void;
  onStart: () => void;
  onStop: () => void;
  onConfigChange: (config: RunConfig) => void;
  onBaseUrlChange: (url: string) => void;
  onLiveStart?: () => void;
  onLiveStop?: () => void;
  onLiveSeedChange?: (seed: number) => void;
  onDriveStart?: () => void;
  onDriveStop?: () => void;
};

export type ControlRailOptions = {
  /**
   * Job arm/start/stop + run config only — hide live agent / human-drive
   * (dashboard TRAIN metrics panel; full TRAIN mode keeps the default).
   */
  jobOnly?: boolean;
};

/** Left rail: job arm/start/stop + live agent / human-drive controls. */
export class ControlRail {
  root: HTMLElement;
  private config: RunConfig = { ...DEFAULT_RUN_CONFIG };
  private baseUrl = DEFAULT_CONTROL_BASE;
  private locked = false;
  private liveSeed = 7;
  private handlers: ControlRailHandlers;
  private jobOnly: boolean;

  private baseEl!: HTMLInputElement;
  private stageEl!: HTMLInputElement;
  private tierEl!: HTMLInputElement;
  private stepsEl!: HTMLInputElement;
  private workersEl!: HTMLInputElement;
  private runEl!: HTMLElement;
  private stateEl!: HTMLElement;
  private errEl!: HTMLElement;
  private armBtn!: HTMLButtonElement;
  private startBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private seedEl: HTMLInputElement | null = null;
  private liveStateEl: HTMLElement | null = null;
  private liveSessionEl: HTMLElement | null = null;
  private liveStartBtn: HTMLButtonElement | null = null;
  private liveStopBtn: HTMLButtonElement | null = null;
  private driveStartBtn: HTMLButtonElement | null = null;
  private driveStopBtn: HTMLButtonElement | null = null;
  private paceEl: HTMLElement | null = null;
  private driving = false;

  constructor(
    parent: HTMLElement,
    handlers: ControlRailHandlers,
    options: ControlRailOptions = {},
  ) {
    this.handlers = handlers;
    this.jobOnly = options.jobOnly === true;
    this.root = document.createElement("div");
    this.root.className = this.jobOnly ? "train-rail train-rail--job" : "train-rail";
    parent.appendChild(this.root);
    this.render();
  }

  setBaseUrl(url: string): void {
    this.baseUrl = url;
    if (this.baseEl) this.baseEl.value = url;
  }

  setConfig(config: RunConfig): void {
    this.config = { ...config };
    this.syncFields();
  }

  getConfig(): RunConfig {
    return {
      stage: this.stageEl.value.trim() || DEFAULT_RUN_CONFIG.stage,
      tier: Math.max(0, Number(this.tierEl.value) || 0),
      timesteps: Math.max(1, Number(this.stepsEl.value) || DEFAULT_RUN_CONFIG.timesteps),
      workers: Math.max(1, Math.min(64, Number(this.workersEl.value) || 1)),
    };
  }

  getBaseUrl(): string {
    return this.baseEl.value.trim() || DEFAULT_CONTROL_BASE;
  }

  getLiveSeed(): number {
    if (!this.seedEl) return this.liveSeed;
    return Math.max(0, Number(this.seedEl.value) || 0);
  }

  setLiveSeed(seed: number): void {
    this.liveSeed = seed;
    if (this.seedEl) this.seedEl.value = String(seed);
  }

  setLiveState(
    state: string,
    opts: { sessionId?: string | null; pace?: string | null } = {},
  ): void {
    if (this.jobOnly || !this.liveStateEl) return;
    this.liveStateEl.textContent = state;
    this.liveStateEl.dataset.state = state;
    const live =
      state === "starting" ||
      state === "running" ||
      state === "stopping" ||
      state === "streaming";
    if (this.liveStartBtn) this.liveStartBtn.disabled = live || this.driving;
    if (this.liveStopBtn) {
      this.liveStopBtn.disabled = (!live && state !== "error") || this.driving;
    }
    if (this.driveStartBtn) this.driveStartBtn.disabled = live || this.driving;
    if (this.driveStopBtn) this.driveStopBtn.disabled = !this.driving;
    if (opts.sessionId !== undefined && this.liveSessionEl) {
      this.liveSessionEl.textContent = opts.sessionId
        ? truncate(opts.sessionId, 22)
        : "—";
      this.liveSessionEl.title = opts.sessionId ?? "";
    }
    if (opts.pace !== undefined && this.paceEl) {
      this.paceEl.textContent = opts.pace || "—";
    }
  }

  setDriving(on: boolean): void {
    if (this.jobOnly || !this.driveStartBtn) return;
    this.driving = on;
    this.driveStartBtn.disabled = on;
    if (this.driveStopBtn) this.driveStopBtn.disabled = !on;
    if (on) {
      if (this.liveStartBtn) this.liveStartBtn.disabled = true;
      if (this.liveStopBtn) this.liveStopBtn.disabled = true;
      if (this.liveStateEl) {
        this.liveStateEl.textContent = "human";
        this.liveStateEl.dataset.state = "streaming";
      }
    }
  }

  setJobState(
    state: UiJobState,
    opts: { runId?: string | null; error?: string | null } = {},
  ): void {
    this.locked =
      state === "starting" || state === "running" || state === "stopping";
    this.syncFields();
    this.stateEl.textContent = state;
    this.stateEl.dataset.state = state;
    if (opts.runId !== undefined) {
      this.runEl.textContent = opts.runId ? truncate(opts.runId, 22) : "—";
      this.runEl.title = opts.runId ?? "";
    }
    if (opts.error !== undefined) {
      this.errEl.textContent = opts.error ? truncate(opts.error, 120) : "";
      this.errEl.title = opts.error ?? "";
      this.errEl.hidden = !opts.error;
    }

    this.armBtn.classList.toggle("armed", state === "armed");
    this.armBtn.disabled = this.locked || state === "armed";
    this.startBtn.disabled = this.locked;
    this.stopBtn.disabled =
      state !== "starting" && state !== "running" && state !== "stopping";

    [this.baseEl, this.stageEl, this.tierEl, this.stepsEl, this.workersEl].forEach(
      (el) => {
        el.disabled = this.locked;
      },
    );
  }

  setError(message: string | null): void {
    this.errEl.textContent = message ? truncate(message, 120) : "";
    this.errEl.title = message ?? "";
    this.errEl.hidden = !message;
  }

  destroy(): void {
    this.root.remove();
  }

  private render(): void {
    const liveBlock = this.jobOnly
      ? ""
      : `
      <div class="train-row train-live-heading">
        <label>Live agent</label>
        <div class="value muted">F3 · G2 sensors</div>
      </div>
      <div class="train-row">
        <label>Seed</label>
        <input id="tr-seed" type="number" min="0" step="1" />
      </div>
      <div class="train-row">
        <label>Live</label>
        <div class="value" id="tr-live-state" data-state="idle">idle</div>
      </div>
      <div class="train-row">
        <label>Session</label>
        <div class="value" id="tr-live-session">—</div>
      </div>
      <div class="train-row">
        <label>Pace</label>
        <div class="value" id="tr-pace">—</div>
      </div>
      <div class="train-actions">
        <button type="button" id="tr-live-start">Watch live</button>
        <button type="button" id="tr-live-stop" class="danger" disabled>End live</button>
      </div>
      <div class="train-row train-live-heading">
        <label>Human drive</label>
        <div class="value muted" title="WASD/arrows · Space handbrake · R reset · P save">
          F4 · WASD · Space
        </div>
      </div>
      <div class="train-actions">
        <button type="button" id="tr-drive-start">Drive</button>
        <button type="button" id="tr-drive-stop" class="danger" disabled>End drive</button>
      </div>
      <a class="train-watch-link" href="?">← WATCH</a>
    `;

    this.root.innerHTML = `
      <div class="train-rail-brand">TRAIN</div>
      <div class="train-row">
        <label>Control</label>
        <input id="tr-base" type="text" spellcheck="false" placeholder="(proxied /api)" />
      </div>
      <div class="train-row">
        <label>Stage</label>
        <input id="tr-stage" type="text" spellcheck="false" placeholder="proving_ground" />
      </div>
      <div class="train-row">
        <label>Tier</label>
        <input id="tr-tier" type="number" min="0" max="32" step="1" />
      </div>
      <div class="train-row">
        <label>Steps</label>
        <input id="tr-steps" type="number" min="1" step="1000" />
      </div>
      <div class="train-row">
        <label>Workers</label>
        <input id="tr-workers" type="number" min="1" max="64" step="1" />
      </div>
      <div class="train-row">
        <label>Run</label>
        <div class="value" id="tr-run">—</div>
      </div>
      <div class="train-row">
        <label>State</label>
        <div class="value" id="tr-state" data-state="idle">idle</div>
      </div>
      <div class="train-error" id="tr-err" hidden></div>
      <div class="train-actions">
        <button type="button" id="tr-arm">Arm</button>
        <button type="button" id="tr-start">Start</button>
        <button type="button" id="tr-stop" class="danger" disabled>Stop</button>
      </div>
      ${liveBlock}
    `;
    this.baseEl = this.root.querySelector("#tr-base")!;
    this.stageEl = this.root.querySelector("#tr-stage")!;
    this.tierEl = this.root.querySelector("#tr-tier")!;
    this.stepsEl = this.root.querySelector("#tr-steps")!;
    this.workersEl = this.root.querySelector("#tr-workers")!;
    this.runEl = this.root.querySelector("#tr-run")!;
    this.stateEl = this.root.querySelector("#tr-state")!;
    this.errEl = this.root.querySelector("#tr-err")!;
    this.armBtn = this.root.querySelector("#tr-arm")!;
    this.startBtn = this.root.querySelector("#tr-start")!;
    this.stopBtn = this.root.querySelector("#tr-stop")!;
    this.seedEl = this.root.querySelector("#tr-seed");
    this.liveStateEl = this.root.querySelector("#tr-live-state");
    this.liveSessionEl = this.root.querySelector("#tr-live-session");
    this.paceEl = this.root.querySelector("#tr-pace");
    this.liveStartBtn = this.root.querySelector("#tr-live-start");
    this.liveStopBtn = this.root.querySelector("#tr-live-stop");
    this.driveStartBtn = this.root.querySelector("#tr-drive-start");
    this.driveStopBtn = this.root.querySelector("#tr-drive-stop");

    const emitConfig = () => this.handlers.onConfigChange(this.getConfig());
    [this.stageEl, this.tierEl, this.stepsEl, this.workersEl].forEach((el) => {
      el.addEventListener("change", emitConfig);
    });
    this.baseEl.addEventListener("change", () => {
      this.handlers.onBaseUrlChange(this.getBaseUrl());
    });
    this.seedEl?.addEventListener("change", () => {
      this.liveSeed = this.getLiveSeed();
      this.handlers.onLiveSeedChange?.(this.liveSeed);
    });
    this.armBtn.addEventListener("click", () => this.handlers.onArm());
    this.startBtn.addEventListener("click", () => this.handlers.onStart());
    this.stopBtn.addEventListener("click", () => this.handlers.onStop());
    this.liveStartBtn?.addEventListener("click", () => this.handlers.onLiveStart?.());
    this.liveStopBtn?.addEventListener("click", () => this.handlers.onLiveStop?.());
    this.driveStartBtn?.addEventListener("click", () => this.handlers.onDriveStart?.());
    this.driveStopBtn?.addEventListener("click", () => this.handlers.onDriveStop?.());
    this.syncFields();
    this.setLiveSeed(this.liveSeed);
    this.setLiveState("idle");
    this.setDriving(false);
  }

  private syncFields(): void {
    if (!this.stageEl) return;
    this.baseEl.value = this.baseUrl;
    this.stageEl.value = this.config.stage;
    this.tierEl.value = String(this.config.tier);
    this.stepsEl.value = String(this.config.timesteps);
    this.workersEl.value = String(this.config.workers);
    if (this.seedEl) this.seedEl.value = String(this.liveSeed);
  }
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}

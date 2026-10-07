import type { MetricsLine } from "./types";

/** Completion / throughput / PPO diagnostics beside the sparkline. */
export class MetricsPanel {
  private root: HTMLElement;
  private completionEl: HTMLElement;
  private stepsEl: HTMLElement;
  private timestepsEl: HTMLElement;
  private klEl: HTMLElement;
  private policyEl: HTMLElement;
  private valueEl: HTMLElement;
  private entropyEl: HTMLElement;
  private clipEl: HTMLElement;
  private explainedEl: HTMLElement;
  private gradEl: HTMLElement;
  private rejectedEl: HTMLElement;
  private outcomesEl: HTMLElement;
  private rewardEl: HTMLElement;

  constructor(parent: HTMLElement) {
    this.root = document.createElement("div");
    this.root.className = "train-metrics";
    this.root.innerHTML = `
      <div class="train-metric-block">
        <div class="label">Reward</div>
        <div class="numeral" id="tm-reward">—</div>
      </div>
      <div class="train-metric-block">
        <div class="label">Completion</div>
        <div class="numeral" id="tm-completion">—</div>
      </div>
      <div class="train-metric-grid">
        <div class="train-metric-row"><span>Steps/s</span><span id="tm-steps">—</span></div>
        <div class="train-metric-row"><span>Timesteps</span><span id="tm-timesteps">—</span></div>
        <div class="train-metric-row"><span>Finish/Crash</span><span id="tm-outcomes">—</span></div>
      </div>
      <div class="train-ppo">
        <div class="label">PPO</div>
        <div class="train-metric-row"><span>approx_kl</span><span id="tm-kl">—</span></div>
        <div class="train-metric-row"><span>policy</span><span id="tm-policy">—</span></div>
        <div class="train-metric-row"><span>value</span><span id="tm-value">—</span></div>
        <div class="train-metric-row"><span>entropy</span><span id="tm-entropy">—</span></div>
        <div class="train-metric-row"><span>clip_frac</span><span id="tm-clip">—</span></div>
        <div class="train-metric-row"><span>explained</span><span id="tm-explained">—</span></div>
        <div class="train-metric-row"><span>grad_norm</span><span id="tm-grad">—</span></div>
        <div class="train-metric-row"><span>rejected</span><span id="tm-rejected">—</span></div>
      </div>
    `;
    parent.appendChild(this.root);
    this.rewardEl = this.root.querySelector("#tm-reward")!;
    this.completionEl = this.root.querySelector("#tm-completion")!;
    this.stepsEl = this.root.querySelector("#tm-steps")!;
    this.timestepsEl = this.root.querySelector("#tm-timesteps")!;
    this.outcomesEl = this.root.querySelector("#tm-outcomes")!;
    this.klEl = this.root.querySelector("#tm-kl")!;
    this.policyEl = this.root.querySelector("#tm-policy")!;
    this.valueEl = this.root.querySelector("#tm-value")!;
    this.entropyEl = this.root.querySelector("#tm-entropy")!;
    this.clipEl = this.root.querySelector("#tm-clip")!;
    this.explainedEl = this.root.querySelector("#tm-explained")!;
    this.gradEl = this.root.querySelector("#tm-grad")!;
    this.rejectedEl = this.root.querySelector("#tm-rejected")!;
  }

  clear(): void {
    this.rewardEl.textContent = "—";
    this.completionEl.textContent = "—";
    this.stepsEl.textContent = "—";
    this.timestepsEl.textContent = "—";
    this.outcomesEl.textContent = "—";
    this.klEl.textContent = "—";
    this.policyEl.textContent = "—";
    this.valueEl.textContent = "—";
    this.entropyEl.textContent = "—";
    this.clipEl.textContent = "—";
    this.explainedEl.textContent = "—";
    this.gradEl.textContent = "—";
    this.rejectedEl.textContent = "—";
    this.rejectedEl.classList.remove("warn");
  }

  apply(line: MetricsLine): void {
    if (line.timesteps != null) {
      this.timestepsEl.textContent = formatInt(line.timesteps);
    }
    if (line.reward?.mean != null) {
      this.rewardEl.textContent = fmt(line.reward.mean, 2);
    }
    if (line.completion?.rate != null) {
      this.completionEl.textContent = `${(line.completion.rate * 100).toFixed(0)}%`;
    }
    if (line.completion) {
      const f = line.completion.finish ?? 0;
      const c = line.completion.crash ?? 0;
      this.outcomesEl.textContent = `${f}/${c}`;
    }
    if (line.throughput?.steps_per_s != null) {
      this.stepsEl.textContent = line.throughput.steps_per_s.toFixed(0);
    }
    const ppo = line.ppo;
    if (!ppo) return;
    if (ppo.approx_kl != null) this.klEl.textContent = fmt(ppo.approx_kl, 4);
    if (ppo.policy_loss != null) this.policyEl.textContent = fmt(ppo.policy_loss, 3);
    if (ppo.value_loss != null) this.valueEl.textContent = fmt(ppo.value_loss, 3);
    if (ppo.entropy != null) this.entropyEl.textContent = fmt(ppo.entropy, 3);
    if (ppo.clip_frac != null) this.clipEl.textContent = fmt(ppo.clip_frac, 3);
    if (ppo.explained_var != null) this.explainedEl.textContent = fmt(ppo.explained_var, 3);
    if (ppo.grad_norm != null) this.gradEl.textContent = fmt(ppo.grad_norm, 3);
    if (ppo.rejected != null) {
      this.rejectedEl.textContent = ppo.rejected ? "yes" : "no";
      this.rejectedEl.classList.toggle("warn", ppo.rejected);
    }
  }

  destroy(): void {
    this.root.remove();
  }
}

function fmt(n: number, digits: number): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function formatInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

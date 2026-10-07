/**
 * DecisionLogPanel component — displays supervisor decisions.
 */
import { el, panelHeader } from "../ui";
import type { PanelContext } from "../types";

export interface DecisionLogPanelOptions {
  runId: string;
}

export class DecisionLogPanel {
  private root: HTMLElement;
  private ctx: PanelContext;
  private listEl: HTMLElement;
  private ac: AbortController | null = null;
  private runId: string;

  constructor(ctx: PanelContext, options: DecisionLogPanelOptions) {
    this.ctx = ctx;
    this.runId = options.runId;

    this.listEl = el("div", {
      className: "decision-list",
      style: "max-height: 400px; overflow-y: auto; padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);",
    });

    this.root = el(
      "section",
      { className: "dash-section" },
      panelHeader("Decision Log", `${this.runId}`).root,
      this.listEl,
    );
  }

  mount(host: HTMLElement): void {
    host.appendChild(this.root);
    this.fetchData();
  }

  destroy(): void {
    this.ac?.abort();
    this.root.remove();
  }

  private async fetchData(): Promise<void> {
    this.ac = new AbortController();
    try {
      const base = this.ctx.apiBase.replace(/\/+$/, "");
      const res = await fetch(`${base}/api/runs/${encodeURIComponent(this.runId)}/decisions`, {
        signal: this.ac.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.renderList(data.decisions || []);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      this.listEl.textContent = "Failed to load decisions: " + (e instanceof Error ? e.message : String(e));
    }
  }

  private renderList(decisions: Array<{ type: string; message: string; timestamp: number }>): void {
    this.listEl.replaceChildren();
    if (decisions.length === 0) {
      this.listEl.appendChild(el("div", { text: "No decisions found.", className: "dash-faint" }));
      return;
    }
    for (const d of decisions) {
      let color = "gray";
      if (d.type === "rollback") color = "var(--bad, #ff4444)";
      else if (d.type === "plateau_reseed") color = "var(--warn, #ffbb33)";
      else if (d.type === "probe_warning") color = "var(--warn-alt, #ff8800)";
      else if (d.type === "probe_ok") color = "var(--ok, #00C851)";

      this.listEl.appendChild(
        el("div", {
          style: `border-left: 4px solid ${color}; padding-left: 0.5rem; margin-bottom: 0.5rem;`,
        }, 
        el("strong", { text: d.type }),
        el("span", { text: ` - ${d.message}`, className: "dash-faint" }))
      );
    }
  }
}

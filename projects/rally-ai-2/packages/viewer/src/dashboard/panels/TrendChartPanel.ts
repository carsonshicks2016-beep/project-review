/**
 * TrendChartPanel component — displays completion_rate over time.
 */
import { el, panelHeader } from "../ui";
import type { PanelContext } from "../types";

export interface TrendChartPanelOptions {
  runId: string;
}

export class TrendChartPanel {
  private root: HTMLElement;
  private ctx: PanelContext;
  private runId: string;
  private canvas: HTMLCanvasElement;
  private ac: AbortController | null = null;
  private dpr = window.devicePixelRatio || 1;

  constructor(ctx: PanelContext, options: TrendChartPanelOptions) {
    this.ctx = ctx;
    this.runId = options.runId;

    this.canvas = el("canvas", {
      style: "width: 100%; height: 200px; display: block; background: rgba(0,0,0,0.2); border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);",
    });

    this.root = el(
      "section",
      { className: "dash-section" },
      panelHeader("Trend: Completion Rate", `${this.runId}`).root,
      this.canvas,
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
      const res = await fetch(`${base}/api/runs/${encodeURIComponent(this.runId)}/metrics?since=0`, {
        signal: this.ac.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      
      const text = await res.text();
      const lines = text.trim().split("\n").filter(Boolean);
      const dataPoints: { step: number; rate: number }[] = [];
      
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.timesteps != null && parsed.completion?.rate != null) {
            dataPoints.push({ step: parsed.timesteps, rate: parsed.completion.rate });
          }
        } catch (e) {
          // ignore parse errors
        }
      }
      
      this.drawChart(dataPoints);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      console.error("TrendChartPanel error:", e);
    }
  }

  private drawChart(data: { step: number; rate: number }[]): void {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    
    ctx.scale(this.dpr, this.dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    
    if (data.length === 0) {
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "14px sans-serif";
      ctx.fillText("No completion rate data", 10, 20);
      return;
    }
    
    let maxStep = 0;
    for (const d of data) {
      if (d.step > maxStep) maxStep = d.step;
    }
    
    ctx.beginPath();
    ctx.strokeStyle = "var(--ok, #00C851)";
    ctx.lineWidth = 2;
    
    for (let i = 0; i < data.length; i++) {
      const d = data[i]!;
      const x = maxStep > 0 ? (d.step / maxStep) * rect.width : 0;
      const y = rect.height - (d.rate * rect.height); // rate is 0 to 1
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

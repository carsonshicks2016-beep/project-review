import { TrendChartPanel } from "./TrendChartPanel";
import { registerPanel } from "../registry";
import type { PanelMount } from "../types";

export const mountTrendChart: PanelMount = (host, ctx) => {
  host.replaceChildren();
  const panel = new TrendChartPanel(ctx, { runId: "foundation_01" });
  panel.mount(host);
  return () => panel.destroy();
};

export function registerTrendChartPanel(): void {
  registerPanel("trend", mountTrendChart);
}

registerTrendChartPanel();

import { DecisionLogPanel } from "./DecisionLogPanel";
import { registerPanel } from "../registry";
import type { PanelMount } from "../types";

export const mountDecisionLog: PanelMount = (host, ctx) => {
  host.replaceChildren();
  // Using a fallback runId, or ideally finding the current active run.
  // We'll default to foundation_01 or the active run if we could pass it.
  const panel = new DecisionLogPanel(ctx, { runId: "foundation_01" });
  panel.mount(host);
  return () => panel.destroy();
};

export function registerDecisionLogPanel(): void {
  registerPanel("decisions", mountDecisionLog);
}

registerDecisionLogPanel();

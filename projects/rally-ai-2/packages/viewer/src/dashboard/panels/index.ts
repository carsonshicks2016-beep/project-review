/**
 * Panel barrel — the one place every dashboard section is wired to a mount.
 *
 * Panels are written by different hands and register themselves in slightly
 * different ways (some call `registerPanel` at import, some only export a
 * mount). Rather than police that, this module does two things:
 *
 * 1. Imports every panel module, so any self-registration runs.
 * 2. Fills in whatever is still unregistered from `PANEL_MOUNTS`.
 *
 * `PANEL_MOUNTS` is typed `Record<SectionId, PanelMount>`, so adding a section
 * to `SectionId` without wiring a panel is a compile error rather than a route
 * that quietly falls through to a stub at runtime. That type is the reason the
 * shell can promise it never 404s a nav item.
 */

import { getPanel, registerPanel } from "../registry";
import type { PanelMount, SectionId } from "../types";

import { mountCheckpoints } from "./checkpoints";
import { mountDrive } from "./drive";
import { mountEvals } from "./evals";
import { mountLive } from "./live";
import { mountOverview } from "./overview";
import { mountPipeline } from "./pipeline";
import { mountRoadmap } from "./roadmap";
import { mountThroughput } from "./throughput";
import { mountTrainMetrics } from "./trainMetrics";
import { mountWatch } from "./watch";
import { mountDecisionLog } from "./decisions";
import { mountTrendChart } from "./trend";

/**
 * Section → mount. Exhaustive by construction.
 *
 * Note `train` is served by `trainMetrics.ts`: the module is named for what it
 * shows, the route is named for what it is.
 */
export const PANEL_MOUNTS: Record<SectionId, PanelMount> = {
  overview: mountOverview,
  roadmap: mountRoadmap,
  pipeline: mountPipeline,
  train: mountTrainMetrics,
  live: mountLive,
  drive: mountDrive,
  watch: mountWatch,
  evals: mountEvals,
  checkpoints: mountCheckpoints,
  throughput: mountThroughput,
  decisions: mountDecisionLog,
  trend: mountTrendChart,
};

/**
 * Register any section a panel module did not claim for itself.
 *
 * Self-registration wins: a panel that registered a different mount than its
 * headline export did so on purpose, and clobbering it here would silently
 * undo that choice. Safe to call more than once.
 */
export function registerAllPanels(): void {
  for (const [id, mount] of Object.entries(PANEL_MOUNTS) as [
    SectionId,
    PanelMount,
  ][]) {
    if (getPanel(id)) continue;
    registerPanel(id, mount);
  }
}

/** Sections this barrel knows how to mount — used by the check script. */
export const WIRED_SECTIONS = Object.keys(PANEL_MOUNTS) as SectionId[];

registerAllPanels();

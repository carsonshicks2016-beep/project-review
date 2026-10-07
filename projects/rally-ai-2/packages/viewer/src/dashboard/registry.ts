/**
 * Panel registry — shared by the shell and sibling panel modules.
 *
 * Kept separate from `DashboardApp` so panels can `registerPanel` without a
 * circular import when the app dynamically loads them at boot.
 */

import type { PanelMount, SectionId } from "./types";

const panelRegistry = new Map<SectionId, PanelMount>();

/** Register (or replace) the mount function for a section. */
export function registerPanel(id: SectionId, mount: PanelMount): void {
  panelRegistry.set(id, mount);
}

/** Remove a panel registration (tests / hot reload). */
export function unregisterPanel(id: SectionId): void {
  panelRegistry.delete(id);
}

export function getPanel(id: SectionId): PanelMount | undefined {
  return panelRegistry.get(id);
}

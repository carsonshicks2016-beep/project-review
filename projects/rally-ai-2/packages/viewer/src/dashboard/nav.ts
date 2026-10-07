/**
 * Dashboard navigation — structural sections, not a card grid of everything.
 */

import type { SectionId } from "./types";

export interface NavSection {
  id: SectionId;
  /** Short HUD label shown in the left rail. */
  label: string;
  /** Optional group heading above this item (only first of a group sets it). */
  group?: string;
}

/**
 * Canonical section list. Order = nav order. Labels stay short and monospace-
 * friendly so the rail reads like a stage HUD, not a product menu.
 */
export const NAV_SECTIONS: readonly NavSection[] = [
  { id: "overview", label: "Overview", group: "Program" },
  { id: "roadmap", label: "Roadmap" },
  { id: "pipeline", label: "Pipeline" },
  { id: "train", label: "Train", group: "Studio" },
  { id: "live", label: "Live" },
  { id: "drive", label: "Drive" },
  { id: "watch", label: "Watch" },
  { id: "evals", label: "Evals", group: "Evidence" },
  { id: "checkpoints", label: "Checkpoints" },
  { id: "checkpoints", label: "Checkpoints" },
  { id: "throughput", label: "Throughput" },
  { id: "decisions", label: "Decisions" },
  { id: "trend", label: "Trend" }
] as const;

const BY_ID = new Map<SectionId, NavSection>(
  NAV_SECTIONS.map((s) => [s.id, s]),
);

export function sectionById(id: SectionId): NavSection {
  return BY_ID.get(id) ?? NAV_SECTIONS[0]!;
}

export function isSectionId(value: string): value is SectionId {
  return BY_ID.has(value as SectionId);
}

export function titleFor(id: SectionId): string {
  return sectionById(id).label;
}

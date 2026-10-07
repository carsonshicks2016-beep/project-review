/**
 * Hash router for dashboard mode.
 *
 * Routes live under the page hash so `?mode=dashboard` stays put:
 * `?mode=dashboard#/train`, `?mode=dashboard#/evals`, …
 */

import { isSectionId } from "./nav";
import type { SectionId } from "./types";

export const DEFAULT_SECTION: SectionId = "overview";

export type RouteListener = (section: SectionId) => void;

/** Parse `#/train` / `#train` / empty → section id. */
export function parseHash(hash: string = location.hash): SectionId {
  const raw = hash.replace(/^#\/?/, "").trim();
  const segment = (raw.split(/[/?#]/)[0] ?? "").toLowerCase();
  if (segment && isSectionId(segment)) return segment;
  return DEFAULT_SECTION;
}

/** Write the hash without forcing a full navigation. */
export function sectionHash(section: SectionId): string {
  return `#/${section}`;
}

export class DashboardRouter {
  private listeners = new Set<RouteListener>();
  private current: SectionId;
  private onHash = (): void => this.syncFromLocation();

  constructor() {
    this.current = parseHash();
    // Normalise bare `#`, `#train`, or missing hash to `#/<section>`.
    if (location.hash !== sectionHash(this.current)) {
      history.replaceState(null, "", sectionHash(this.current));
    }
    window.addEventListener("hashchange", this.onHash);
  }

  get section(): SectionId {
    return this.current;
  }

  navigate(section: SectionId): void {
    if (section === this.current && location.hash === sectionHash(section)) {
      return;
    }
    location.hash = sectionHash(section);
  }

  subscribe(listener: RouteListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  dispose(): void {
    window.removeEventListener("hashchange", this.onHash);
    this.listeners.clear();
  }

  private syncFromLocation(): void {
    const next = parseHash();
    if (next === this.current) return;
    this.current = next;
    for (const listener of this.listeners) listener(next);
  }
}

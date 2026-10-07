/**
 * Dashboard layout shell: left nav · main stage · optional right rail.
 *
 * One composition per view — the shell is structural chrome, not a dashboard
 * of cards. Sibling panels mount into `#dash-main` (and optionally the rail).
 */

import { NAV_SECTIONS, titleFor } from "./nav";
import type { SectionId } from "./types";
import "./theme.css";

export interface ShellElements {
  root: HTMLElement;
  nav: HTMLElement;
  main: HTMLElement;
  rail: HTMLElement;
  title: HTMLElement;
  route: HTMLElement;
}

export interface ShellOptions {
  onNavigate: (section: SectionId) => void;
  /** When true, the right rail column is visible. */
  railVisible?: boolean;
}

export function createShell(host: HTMLElement, options: ShellOptions): ShellElements {
  host.replaceChildren();
  host.classList.add("dash-host");

  const root = document.createElement("div");
  root.className = "dash-app";
  root.innerHTML = `
    <header class="dash-header">
      <div class="dash-brand">
        <b>RALLYAI</b>
        <span>program</span>
      </div>
      <div class="dash-header-title">
        <h1 class="dash-title"></h1>
        <span class="dash-route" aria-hidden="true"></span>
      </div>
      <div class="dash-nav-foot">
        <a class="dash-ext" href="?">WATCH</a>
        <a class="dash-ext" href="?mode=train">TRAIN</a>
      </div>
    </header>
    <nav class="dash-nav tabs" aria-label="Program sections">
      <div class="dash-nav-list"></div>
    </nav>
    <div class="dash-body content">
      <section class="dash-main" id="dash-main" aria-live="polite"></section>
      <aside class="dash-rail" id="dash-rail" hidden aria-label="Context rail"></aside>
    </div>
  `;

  host.appendChild(root);

  const navList = root.querySelector<HTMLElement>(".dash-nav-list")!;
  const main = root.querySelector<HTMLElement>("#dash-main")!;
  const rail = root.querySelector<HTMLElement>("#dash-rail")!;
  const title = root.querySelector<HTMLElement>(".dash-title")!;
  const route = root.querySelector<HTMLElement>(".dash-route")!;
  const navAside = root.querySelector<HTMLElement>(".dash-nav")!;

  let lastGroup: string | undefined;
  for (const section of NAV_SECTIONS) {
    if (section.group && section.group !== lastGroup) {
      lastGroup = section.group;
      const heading = document.createElement("div");
      heading.className = "dash-nav-group";
      heading.textContent = section.group;
      navList.appendChild(heading);
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dash-nav-item";
    btn.dataset.section = section.id;
    btn.textContent = section.label;
    btn.addEventListener("click", () => options.onNavigate(section.id));
    navList.appendChild(btn);
  }

  setRailVisible(rail, options.railVisible ?? false);

  return { root, nav: navAside, main, rail, title, route };
}

/** Sync active nav item + header copy for the current section. */
export function syncShell(
  els: ShellElements,
  section: SectionId,
  opts?: { railVisible?: boolean },
): void {
  const items = els.nav.querySelectorAll<HTMLElement>(".dash-nav-item");
  for (const item of items) {
    item.classList.toggle("active", item.dataset.section === section);
  }
  els.title.textContent = titleFor(section);
  els.route.textContent = `#/${section}`;
  if (opts?.railVisible !== undefined) {
    setRailVisible(els.rail, opts.railVisible);
  }
}

export function setRailVisible(rail: HTMLElement, visible: boolean): void {
  rail.hidden = !visible;
  rail.parentElement?.classList.toggle("has-rail", visible);
}

/**
 * Empty a mount host and restore the identity the shell gave it.
 *
 * Panels treat the host as their own root — several set `el.id` and add their
 * own class to it so their stylesheet can be scoped. Both outlive the panel
 * unless something puts them back: navigate roadmap → evals and the host ends
 * up carrying `roadmap-panel evals-panel` with an id of `evals`, so one
 * panel's CSS starts painting another's markup and `#dash-main` no longer
 * exists. Resetting here keeps that a panel-local liberty instead of a
 * cross-section bug, and costs one assignment per navigation.
 */
export function clearMount(el: HTMLElement, identity?: MountIdentity): void {
  el.replaceChildren();
  if (!identity) return;
  el.id = identity.id;
  el.className = identity.className;
}

export interface MountIdentity {
  id: string;
  className: string;
}

/** The identities `createShell` hands out, for `clearMount` to restore. */
export const MAIN_IDENTITY: MountIdentity = {
  id: "dash-main",
  className: "dash-main",
};
export const RAIL_IDENTITY: MountIdentity = {
  id: "dash-rail",
  className: "dash-rail",
};

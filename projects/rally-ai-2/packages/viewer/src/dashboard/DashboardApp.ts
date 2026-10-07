/**
 * Dashboard app — root mount, panel registry, route → panel wiring.
 *
 * Other agents fill sections by calling `registerPanel(id, mount)` before or
 * after boot. Unregistered sections get a titled stub so the shell is usable
 * as a navigation skeleton immediately.
 *
 * Panel registry lives in `./registry` so panels can register without a
 * circular import through this module.
 */

import { titleFor } from "./nav";
import { getPanel } from "./registry";
import { DashboardRouter } from "./router";
import {
  clearMount,
  createShell,
  MAIN_IDENTITY,
  RAIL_IDENTITY,
  setRailVisible,
  syncShell,
  type ShellElements,
} from "./shell";
import type { PanelContext, PanelMount, SectionId } from "./types";
import "./dashboard.css";

export { registerPanel, unregisterPanel } from "./registry";

/** Resolve control-room base URL. Empty string → Vite `/api` proxy. */
export function resolveApiBase(search: string = location.search): string {
  const override = new URLSearchParams(search).get("api");
  if (override != null && override !== "") {
    return override.replace(/\/$/, "");
  }
  // Dev server proxies `/api` → :8765; production / file open needs absolute.
  if (import.meta.env.DEV) return "";
  return "http://127.0.0.1:8765";
}

function stubMount(id: SectionId): PanelMount {
  return (el) => {
    el.classList.add("dash-stub");
    const kicker = document.createElement("div");
    kicker.className = "dash-stub-kicker";
    kicker.textContent = "Panel stub";
    const heading = document.createElement("h2");
    heading.className = "dash-stub-title";
    heading.textContent = titleFor(id);
    const note = document.createElement("p");
    note.className = "dash-stub-note";
    note.textContent =
      `registerPanel("${id}", mount) — awaiting panel implementation.`;
    el.append(kicker, heading, note);
  };
}

export interface DashboardAppOptions {
  /** Override api base; defaults to `resolveApiBase()`. */
  apiBase?: string;
}

export class DashboardApp {
  private host: HTMLElement;
  private apiBase: string;
  private router: DashboardRouter;
  private shell: ShellElements | null = null;
  private disposePanel: (() => void) | null = null;
  private unsub: (() => void) | null = null;

  constructor(host: HTMLElement, options: DashboardAppOptions = {}) {
    this.host = host;
    this.apiBase = options.apiBase ?? resolveApiBase();
    this.router = new DashboardRouter();
  }

  /** Build the shell and show the current hash section. */
  mount(): void {
    document.body.dataset.mode = "dashboard";
    this.shell = createShell(this.host, {
      onNavigate: (section) => this.router.navigate(section),
    });
    this.unsub = this.router.subscribe((section) => this.show(section));
    this.show(this.router.section);
  }

  navigate(section: SectionId): void {
    this.router.navigate(section);
  }

  get section(): SectionId {
    return this.router.section;
  }

  /** Show or hide the optional right rail. */
  setRailVisible(visible: boolean): void {
    if (!this.shell) return;
    setRailVisible(this.shell.rail, visible);
  }

  dispose(): void {
    this.disposePanel?.();
    this.disposePanel = null;
    this.unsub?.();
    this.unsub = null;
    this.router.dispose();
    this.host.replaceChildren();
    this.host.classList.remove("dash-host");
    delete document.body.dataset.mode;
    this.shell = null;
  }

  private show(section: SectionId): void {
    if (!this.shell) return;
    this.disposePanel?.();
    this.disposePanel = null;
    clearMount(this.shell.main, MAIN_IDENTITY);
    clearMount(this.shell.rail, RAIL_IDENTITY);
    setRailVisible(this.shell.rail, false);
    syncShell(this.shell, section);

    const mount = getPanel(section) ?? stubMount(section);
    const ctx: PanelContext = {
      apiBase: this.apiBase,
      root: this.shell.main,
      navigate: (id) => this.router.navigate(id),
    };
    const disposer = mount(this.shell.main, ctx);
    if (typeof disposer === "function") this.disposePanel = disposer;
  }
}

/**
 * Load every panel and make sure each section ends up with a mount.
 *
 * The import is dynamic on purpose. Some panels reach back into this module
 * for `registerPanel`, so a static import here would form a cycle and run
 * panel bodies while this module was still initialising — the registry would
 * be in its temporal dead zone and every registration would throw. Importing
 * at boot means this module is fully evaluated before any panel runs.
 *
 * `packages/viewer/src/dashboard/panels/index.ts` owns the section → mount
 * table and is exhaustive over `SectionId`.
 */
async function loadRegisteredPanels(): Promise<void> {
  const { registerAllPanels } = await import("./panels");
  registerAllPanels();
}

/**
 * Boot dashboard into the page shell (replaces WATCH's #game contents).
 * Hides the replay transport bar.
 */
export async function bootDashboard(
  host: HTMLElement,
  options?: DashboardAppOptions,
): Promise<DashboardApp> {
  const transport = document.querySelector<HTMLElement>(".transport");
  if (transport) transport.hidden = true;
  const status = document.getElementById("status");
  status?.remove();

  await loadRegisteredPanels();

  const app = new DashboardApp(host, options);
  app.mount();
  return app;
}

export type { PanelContext, PanelMount, SectionId } from "./types";
export { NAV_SECTIONS, titleFor } from "./nav";
export { parseHash, sectionHash, DEFAULT_SECTION } from "./router";

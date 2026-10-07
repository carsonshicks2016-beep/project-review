/**
 * Shared dashboard types — section ids, panel mount contract, shell context.
 *
 * Sibling agents register panels via `registerPanel(id, mount)`; the shell
 * owns layout and routing. Panel contents are deliberately out of scope here.
 */

/** Hash routes under `?mode=dashboard` — `#/overview`, `#/train`, … */
export type SectionId =
  | "overview"
  | "train"
  | "live"
  | "watch"
  | "evals"
  | "checkpoints"
  | "throughput"
  | "drive"
  | "pipeline"
  | "roadmap"
  | "decisions"
  | "trend";

/** Context passed to every panel mount. */
export interface PanelContext {
  /** Control-room origin. `''` uses the Vite `/api` proxy; else e.g. `http://127.0.0.1:8765`. */
  apiBase: string;
  /** Main content host for this panel. */
  root: HTMLElement;
  /** Navigate to another dashboard section (updates the hash). */
  navigate: (section: SectionId) => void;
}

/**
 * Mount a panel into `root`. Return an optional disposer invoked when the
 * section is left or the app unmounts.
 *
 * Prefer naming the host `root` or `host` — not `el` — so it does not shadow
 * the `el()` DOM helper from `./ui`.
 */
export type PanelMount = (
  root: HTMLElement,
  ctx: PanelContext,
) => void | (() => void);

/** Props every stub / real panel can rely on. */
export interface PanelProps {
  id: SectionId;
  title: string;
  ctx: PanelContext;
}

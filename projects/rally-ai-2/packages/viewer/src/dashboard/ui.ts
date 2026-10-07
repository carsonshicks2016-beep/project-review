/**
 * Dashboard UI helpers — vanilla DOM, no React.
 *
 * Pair with `theme.css` classes (`.dash-*`). Panels should prefer these over
 * ad-hoc markup so metrics and controls stay consistent with TRAIN.
 *
 * Tip: name the PanelMount host `root` or `host`, not `el`, so it does not
 * shadow this module’s `el()` helper.
 */

import { Sparkline } from "../train/Sparkline";

type Child = Node | string | number | null | undefined | false;

export type ElProps = {
  className?: string;
  id?: string;
  text?: string;
  html?: string;
  attrs?: Record<string, string | number | boolean | null | undefined>;
  on?: Partial<{
    [K in keyof HTMLElementEventMap]: (ev: HTMLElementEventMap[K]) => void;
  }>;
  style?: Partial<CSSStyleDeclaration> | string;
};

/** Create an element with optional props and children. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props?: ElProps | null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (props) {
    if (props.className) node.className = props.className;
    if (props.id) node.id = props.id;
    if (props.text != null) node.textContent = props.text;
    else if (props.html != null) node.innerHTML = props.html;
    if (props.style) {
      if (typeof props.style === "string") node.setAttribute("style", props.style);
      else Object.assign(node.style, props.style);
    }
    if (props.attrs) {
      for (const [k, v] of Object.entries(props.attrs)) {
        if (v == null || v === false) continue;
        if (v === true) node.setAttribute(k, "");
        else node.setAttribute(k, String(v));
      }
    }
    if (props.on) {
      for (const [type, handler] of Object.entries(props.on)) {
        if (handler) node.addEventListener(type, handler as EventListener);
      }
    }
  }
  for (const child of children) {
    if (child == null || child === false) continue;
    if (typeof child === "string" || typeof child === "number") {
      node.appendChild(document.createTextNode(String(child)));
    } else {
      node.appendChild(child);
    }
  }
  return node;
}

export type PanelHeaderHandle = {
  root: HTMLElement;
  setMeta: (meta: string) => void;
};

/** Section header: uppercase title + optional meta (timesteps, status, …). */
export function panelHeader(
  title: string,
  meta?: string,
): PanelHeaderHandle {
  const metaEl = el("span", { className: "dash-meta", text: meta ?? "" });
  const root = el("header", { className: "dash-section-hd" },
    el("span", { className: "dash-section-title", text: title }),
    metaEl,
  );
  return {
    root,
    setMeta(next: string) {
      metaEl.textContent = next;
    },
  };
}

export type StatHandle = {
  root: HTMLElement;
  set: (value: string) => void;
  setWarn: (on: boolean) => void;
};

export type StatOptions = {
  label: string;
  value?: string;
  /** Heavy display numeral (curriculum / completion style). */
  large?: boolean;
  className?: string;
};

/** Label + tabular value. Use `large` for hero metrics. */
export function stat(opts: StatOptions): StatHandle {
  const valueEl = el("div", {
    className: "dash-stat-value",
    text: opts.value ?? "—",
  });
  const root = el(
    "div",
    {
      className: ["dash-stat", opts.large ? "large" : "", opts.className ?? ""]
        .filter(Boolean)
        .join(" "),
    },
    el("div", { className: "dash-stat-label", text: opts.label }),
    valueEl,
  );
  return {
    root,
    set(value: string) {
      valueEl.textContent = value;
    },
    setWarn(on: boolean) {
      valueEl.classList.toggle("warn", on);
    },
  };
}

export type BtnVariant = "default" | "armed" | "danger";

export type BtnOptions = {
  label: string;
  onClick?: (ev: MouseEvent) => void;
  variant?: BtnVariant;
  disabled?: boolean;
  className?: string;
  type?: "button" | "submit" | "reset";
};

/** Sharp control matching TRAIN rail buttons. */
export function btn(opts: BtnOptions): HTMLButtonElement {
  const classes = ["dash-btn"];
  if (opts.variant && opts.variant !== "default") classes.push(opts.variant);
  if (opts.className) classes.push(opts.className);
  const props: ElProps = {
    className: classes.join(" "),
    text: opts.label,
    attrs: {
      type: opts.type ?? "button",
      ...(opts.disabled ? { disabled: true } : {}),
    },
  };
  if (opts.onClick) {
    props.on = { click: (ev) => opts.onClick!(ev as MouseEvent) };
  }
  return el("button", props);
}

export type CodeBlockOptions = {
  code: string;
  /** Optional language hint shown as meta; not syntax-highlighted. */
  lang?: string;
  className?: string;
};

/** Monospace pre block for commands, paths, JSON. */
export function codeBlock(opts: CodeBlockOptions): HTMLElement {
  const pre = el("pre", {
    className: ["dash-code", opts.className ?? ""].filter(Boolean).join(" "),
    text: opts.code,
  });
  if (opts.lang) pre.dataset.lang = opts.lang;
  return pre;
}

export type MountSparklineOptions = {
  className?: string;
  /** Initial CSS height (e.g. `7rem`). */
  height?: string;
  stroke?: string;
  maxPoints?: number;
};

export type SparklineHandle = {
  canvas: HTMLCanvasElement;
  sparkline: Sparkline;
  push: (timesteps: number, reward: number) => void;
  clear: () => void;
  resize: () => void;
  destroy: () => void;
};

/**
 * Append a sparkline canvas to `host` and wire resize.
 * Uses the TRAIN `Sparkline` (hairline grid + thick reward stroke).
 */
export function mountSparkline(
  host: HTMLElement,
  opts: MountSparklineOptions = {},
): SparklineHandle {
  const canvas = el("canvas", {
    className: ["dash-sparkline", opts.className ?? ""].filter(Boolean).join(" "),
  });
  if (opts.height) canvas.style.height = opts.height;
  host.appendChild(canvas);

  const sparkOpts: { stroke?: string; maxPoints?: number } = {};
  if (opts.stroke != null) sparkOpts.stroke = opts.stroke;
  if (opts.maxPoints != null) sparkOpts.maxPoints = opts.maxPoints;
  const sparkline = new Sparkline(canvas, sparkOpts);
  const onResize = (): void => sparkline.resize();
  window.addEventListener("resize", onResize);
  // Layout may settle after mount; one rAF catches flex/grid sizes.
  requestAnimationFrame(onResize);

  return {
    canvas,
    sparkline,
    push(timesteps, reward) {
      sparkline.push(timesteps, reward);
    },
    clear() {
      sparkline.clear();
    },
    resize: onResize,
    destroy() {
      window.removeEventListener("resize", onResize);
      canvas.remove();
    },
  };
}

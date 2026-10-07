/**
 * CRT / dither / scanline finish — presentation only, toggleable.
 * Mild always-on; `V` flips hard arcade mode.
 */

export class CrtOverlay {
  constructor(parent = document.getElementById("app")) {
    this.root = document.createElement("div");
    this.root.id = "crt-overlay";
    this.root.className = "crt-overlay crt-mild";
    this.root.setAttribute("aria-hidden", "true");
    this.root.innerHTML = [
      '<div class="crt-scanlines"></div>',
      '<div class="crt-vignette"></div>',
      '<div class="crt-dither"></div>',
    ].join("");
    parent.appendChild(this.root);
    this.mode = "mild"; // off | mild | hard
  }

  setMode(mode) {
    const next = mode === "hard" || mode === "off" || mode === "mild" ? mode : "mild";
    this.mode = next;
    this.root.classList.toggle("crt-off", next === "off");
    this.root.classList.toggle("crt-mild", next === "mild");
    this.root.classList.toggle("crt-hard", next === "hard");
  }

  /** Cycle off → mild → hard → off. */
  cycle() {
    const order = ["mild", "hard", "off"];
    const i = order.indexOf(this.mode);
    this.setMode(order[(i + 1) % order.length]);
    return this.mode;
  }

  get label() {
    return this.mode === "off" ? "CRT OFF"
      : this.mode === "hard" ? "CRT HARD"
      : "CRT";
  }
}

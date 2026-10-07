/**
 * Period rally HUD: stage card, pace tile, timer and a right-side tach stack.
 *
 * It remains DOM rather than canvas so the graphic overlays stay crisp and
 * independent of the 3D render. Every changing value comes from the replay.
 */

import type { Frame, LoadedReplay } from "./replay";

/** mm:ss.hh, matching the two-digit stage timer in the reference HUD. */
export function formatTime(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

const GEAR_LABEL = (g: number | undefined): string =>
  g === undefined ? "-" : g === 0 ? "N" : g < 0 ? "R" : String(g);

function callPresentation(note: string): { glyph: string; grade: string } {
  const lower = note.toLowerCase();
  const grade = lower.match(/\b([1-6])\b/)?.[1] ?? "";
  if (lower.includes("left")) return { glyph: "↰", grade };
  if (lower.includes("right")) return { glyph: "↱", grade };
  if (lower.includes("crest")) return { glyph: "⌃", grade };
  if (lower.includes("jump")) return { glyph: "↥", grade };
  if (lower.includes("caution")) return { glyph: "!", grade };
  return { glyph: "↑", grade };
}

export class Hud {
  private readonly root: HTMLElement;
  private readonly el: Record<string, HTMLElement> = {};
  private readonly tachSegments: HTMLElement[];

  constructor(parent: HTMLElement, replay: LoadedReplay) {
    const tach = Array.from(
      { length: 12 },
      (_, i) =>
        `<i data-level="${i}" class="${i >= 10 ? "hot" : i >= 8 ? "warm" : ""}"></i>`,
    ).join("");

    this.root = document.createElement("div");
    this.root.className = "hud";
    this.root.innerHTML = `
      <section class="hud-stage-card">
        <div class="hud-brand">
          <b>SUBARU</b>
          <span>WORLD RALLY TEAM</span>
        </div>
        <div class="hud-entry">
          <strong>5</strong>
          <span class="hud-555">555</span>
          <em data-k="stage"></em>
        </div>
      </section>

      <section class="hud-note" data-k="note">
        <div class="hud-note-tile">
          <span data-k="noteGlyph">↑</span>
          <b data-k="noteGrade"></b>
        </div>
        <small data-k="noteText"></small>
      </section>

      <section class="hud-time-panel">
        <small>STAGE TIME</small>
        <div class="hud-time" data-k="time">00:00.00</div>
        <div class="hud-time-meta">
          <span data-k="surf">GRAVEL</span>
          <span data-k="prog">0%</span>
        </div>
      </section>

      <section class="hud-instruments">
        <div class="hud-tach">
          <span>8</span>
          <div class="hud-tach-bars">${tach}</div>
          <small>RPM</small>
        </div>
        <div class="hud-gear" data-k="gear">N</div>
        <div class="hud-speed">
          <strong data-k="speed">0</strong>
          <small>KM/H</small>
        </div>
      </section>

      <div class="hud-progress"><i data-k="bar"></i></div>
    `;
    parent.appendChild(this.root);

    for (const n of this.root.querySelectorAll<HTMLElement>("[data-k]")) {
      this.el[n.dataset.k!] = n;
    }
    this.tachSegments = [
      ...this.root.querySelectorAll<HTMLElement>(".hud-tach-bars i"),
    ];

    const tier = replay.stage.generator?.tier;
    this.el.stage!.textContent =
      `${replay.stage.name}${tier === undefined ? "" : ` · T${tier}`}`.toUpperCase();
  }

  update(frame: Frame, time: number, speed: number): void {
    this.el.time!.textContent = formatTime(time);
    this.el.speed!.textContent = (speed * 3.6).toFixed(0);
    this.el.gear!.textContent = GEAR_LABEL(frame.gear);
    this.el.surf!.textContent = (frame.surf ?? "-").toUpperCase();

    const prog = Math.min(Math.max(frame.prog ?? 0, 0), 1);
    this.el.prog!.textContent = `${(prog * 100).toFixed(0)}%`;
    this.el.bar!.style.height = `${(prog * 100).toFixed(1)}%`;

    const rpm = Math.min(Math.max(frame.rpm ?? 0, 0), 8000);
    const lit = Math.round((rpm / 8000) * this.tachSegments.length);
    this.tachSegments.forEach((seg, i) => seg.classList.toggle("on", i < lit));

    const text = frame.note?.trim() ?? "";
    this.el.note!.classList.toggle("on", Boolean(text));
    this.el.noteText!.textContent = text.toUpperCase();
    if (text) {
      const call = callPresentation(text);
      this.el.noteGlyph!.textContent = call.glyph;
      this.el.noteGrade!.textContent = call.grade;
    }
  }

  /** Remove any transient finish/termination overlay after a seek or restart. */
  reset(): void {
    for (const banner of this.root.querySelectorAll(".hud-end")) banner.remove();
  }

  finish(replay: LoadedReplay): void {
    const { termination, time_s, clean } = replay.meta;
    const ok = termination === "finish";
    const banner = document.createElement("div");
    banner.className = `hud-end ${ok ? "ok" : "bad"}`;
    banner.innerHTML = ok
      ? `<b>FINISH</b><span>${formatTime(time_s)}${clean ? " · CLEAN" : ""}</span>`
      : `<b>${termination.replace("_", " ").toUpperCase()}</b>` +
        `<span>${formatTime(time_s)}</span>`;
    this.root.appendChild(banner);
    setTimeout(() => banner.remove(), 4000);
  }
}

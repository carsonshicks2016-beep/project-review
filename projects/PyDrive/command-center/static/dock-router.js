/* DOCK ROUTER — hand each run the instrument that fits it.
 *
 * The telemetry dock used to render one view for everything: the Pit Wall,
 * which is a Nordschleife ring instrument fed from /api/fable-*. A drift or
 * hybrid run has no sectors, no benchmark lap and no stage ladder, so that
 * board showed a wall of FAILs for series that never arrive.
 *
 * Routing now follows the run's lineage:
 *
 *   fable / ring runs        -> Pit Wall     (pitwall.js, unchanged)
 *   race / drift / hybrid    -> Lineage Wall (lineage.js)
 *
 * Loaded after app.js, pitwall.js and lineage.js. Nothing in those files
 * changes: this wraps PitWall.setActive and app.js's selectTask, both of
 * which are plain globals in a classic script.
 */
(function () {
  "use strict";

  const LW = window.LineageWall;
  const PW = window.PitWall;
  if (!LW || !PW) return;

  /* The ring-specific furniture. The session feed (#console) is deliberately
     NOT in here — a live log is worth having whatever the lineage. */
  const RING_SEL = "#dock .pw-chase, #dock .pw-maingrid, #dock .pw-lab, #dock .pw-beh,\n                   #dock #pw-rail";   // the Fable stage ladder: ring programme only
  function ringNodes() {
    return Array.prototype.slice.call(document.querySelectorAll(RING_SEL));
  }

  let current = null;   // "ring" | "lineage" | null

  function activeLabel() {
    const t = typeof window.pwActiveTask === "function" ? window.pwActiveTask() : null;
    return t ? t.label : "";
  }

  function wantedFor(label) {
    if (!label) return "ring";                 // nothing running: leave the default board
    if (LW.isRingRun(label)) return "ring";
    return LW.lineageOf(label) ? "lineage" : "ring";
  }

  function show(which) {
    const ring = which === "ring";
    ringNodes().forEach(n => { n.hidden = !ring; });
    const wall = document.querySelector("#lineage-wall");
    if (wall) wall.hidden = ring;
    if (current === which) {
      (ring ? PW : LW).refresh();
      return;
    }
    current = which;
    if (ring) { LW.setActive(false); PW.setActive(true); }
    else { PW.setActive(false); LW.setActive(true); }
  }

  /* --- wrap PitWall.setActive: app.js calls this to open/close the dock --- */
  const pwSetActive = PW.setActive;
  PW.setActive = function (on) {
    if (!on) {
      current = null;
      LW.setActive(false);
      return pwSetActive.call(PW, false);
    }
    show(wantedFor(activeLabel()));
  };

  /* --- re-route when the user switches dock tabs between runs --- */
  const sel = window.selectTask;
  if (typeof sel === "function") {
    window.selectTask = function (pid) {
      const r = sel.apply(this, arguments);
      const dock = document.querySelector("#dock");
      if (dock && !dock.classList.contains("hidden")) show(wantedFor(activeLabel()));
      return r;
    };
  }
})();

/* LINEAGE WALL — training viewer for the race / drift / hybrid PPO lineage.
 *
 * Why this exists
 * ---------------
 * The Pit Wall (pitwall.js) is a Nordschleife ring instrument. Every source it
 * reads is /api/fable-*, and its gates are hardcoded to the ring programme:
 * clean sectors >= 12/16, terminal rate <= 0.25, lap progress >= 0.90, pace vs
 * a Bellof benchmark. A drift or hybrid run emits none of those series — it is
 * a track-POOL generalist with no sectors, no benchmark lap and no stage
 * ladder. Pointed at a hybrid run the Pit Wall therefore shows a full board of
 * FAILs that are not failures at all: the numbers simply never arrive.
 *
 * This module is the instrument those runs actually deserve. It reads only
 * what the lineage really emits, straight off the live SSE metric stream that
 * app.js already maintains (window.pwActiveTask()):
 *
 *   per iteration   x, return, laps|drift, difficulty,
 *                   policy_loss, value_loss, entropy, kl, kl_stop
 *   per eval        eval_lap, eval_drift (style), eval_score, jumps, air
 *   run events      restart / max_restarts, converged
 *
 * The scoring it visualises is the real thing from supra/ppo.py:
 *
 *   drift    metric = drift * (0.3 + 0.7 * lap)     completion-GATED
 *   hybrid   metric = lap   * (0.6 + 0.4 * style)   completion-DOMINANT
 *   race     metric = laps                          uncapped
 *
 * Those asymmetries are the whole story of the lineage, so they are the hero
 * of the view rather than a footnote.
 */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);

  /* ---------------- lineage definitions ---------------- */

  const LINEAGE = {
    hybrid: {
      label: "HYBRID",
      blurb: "race backbone + corner-drift style",
      // metric = lap * (0.6 + 0.4 * style)
      score: (lap, style) => lap * (0.6 + 0.4 * style),
      formula: "score = lap × (0.60 + 0.40 · style)",
      xKey: "eval_lap", yKey: "eval_drift",
      xName: "lap", yName: "style",
      // style is a multiplier band, never a driver on its own
      floor: 0.6, span: 0.4,
      styleFor: (s, lap) => (lap <= 0 ? NaN : (s / lap - 0.6) / 0.4),
    },
    drift: {
      label: "DRIFT",
      blurb: "slide angle, gated on completing the lap",
      // metric = drift * (0.3 + 0.7 * lap)
      score: (lap, style) => style * (0.3 + 0.7 * lap),
      formula: "score = drift × (0.30 + 0.70 · lap)",
      xKey: "eval_lap", yKey: "eval_drift",
      xName: "lap", yName: "drift",
      floor: 0.3, span: 0.7,
      styleFor: (s, lap) => s / (0.3 + 0.7 * lap),
    },
    race: {
      label: "RACE",
      blurb: "cumulative laps over the eval budget — uncapped",
      score: (lap) => lap,
      formula: "score = laps",
      xKey: "eval_lap", yKey: null,
      xName: "laps", yName: null,
      floor: 1, span: 0,
    },
  };

  /* Which lineage is running? The task label is the tag launch() was given —
     "hybrid_train", "drift_live", "ppo_continue", optionally ":<track>". */
  function lineageOf(label) {
    const s = String(label || "").toLowerCase();
    if (s.includes("hybrid")) return "hybrid";
    if (s.includes("drift")) return "drift";
    if (/\bppo|race|selfplay|multiagent/.test(s)) return "race";
    return null;
  }

  /* Fable/ring runs keep the Pit Wall — this viewer must not claim them. */
  function isRingRun(label) {
    const s = String(label || "").toLowerCase();
    return s.includes("fable") || s.includes("ring");
  }

  window.LineageWall = window.LineageWall || {};
  window.LineageWall.lineageOf = lineageOf;
  window.LineageWall.isRingRun = isRingRun;

  /* ---------------- tiny svg helpers ---------------- */

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  /* Size the viewBox to the host's real pixels. A fixed viewBox stretched with
     preserveAspectRatio:none distorts every glyph in the chart. */
  function svgHost(host, hFallback) {
    host.innerHTML = "";
    const w = Math.max(220, Math.round(host.clientWidth || 420));
    const h = Math.max(90, Math.round(host.clientHeight || hFallback || 150));
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: "100%" }, host);
    return { svg: s, w, h };
  }
  function label(svg, x, y, s, fill, size, extra) {
    return el("text", Object.assign({
      x, y, fill, "font-size": size || 10,
      "font-family": "ui-monospace,Menlo,monospace",
    }, extra || {}), svg).appendChild(document.createTextNode(s)).parentNode;
  }
  const fmt = (v, d = 2) => (v == null || !isFinite(v) ? "—" : (+v).toFixed(d));

  /* ---------------- state ---------------- */

  /* Mirrors supra/config.py — HybridReward.style_ramp_* and
     PPOSpec.hybrid_promote_lap. Kept here so the verdict can tell "the gate is
     shut" apart from "the policy refuses to slide". */
  const STYLE_RAMP_LO = 0.35, STYLE_RAMP_HI = 0.70, HYBRID_PROMOTE_LAP = 0.60;

  const S = { rows: [], label: "", mode: null, live: false };

  function pull() {
    const t = typeof window.pwActiveTask === "function" ? window.pwActiveTask() : null;
    S.rows = (t && t.metrics) || [];
    S.label = (t && t.label) || "";
    S.live = !!(t && t.live);
    S.mode = lineageOf(S.label);
    return !!t;
  }

  /* every row that carries a deterministic eval, in order */
  const evals = () => S.rows.filter(r => r && r.eval_lap != null);
  const last = arr => (arr.length ? arr[arr.length - 1] : null);
  function lastWith(key) {
    for (let i = S.rows.length - 1; i >= 0; i--)
      if (S.rows[i] && S.rows[i][key] != null) return S.rows[i];
    return null;
  }

  /* ---------------- verdict — the headline ----------------
     The most useful thing the Brain Lab does is say what happened in words.
     This does the same for a live run, from the numbers actually present. */

  function verdict() {
    const L = LINEAGE[S.mode];
    if (!L) return { tone: "idle", text: "No race, drift or hybrid run selected." };

    const ev = evals();
    const it = lastWith("entropy");
    const conv = lastWith("converged");
    const rs = lastWith("restart");

    if (!ev.length) {
      return { tone: "wait", text: it
        ? `Training — ${it.x ?? "?"} iterations in, no deterministic eval banked yet. `
          + "The rolling numbers move first; the [eval] line is the one to trust."
        : "Waiting for the first iteration." };
    }

    const e = last(ev);
    const lap = e.eval_lap ?? 0;
    const sty = e.eval_drift ?? 0;
    const score = e.eval_score ?? L.score(lap, sty);
    const best = Math.max(...ev.map(r => r.eval_score ?? L.score(r.eval_lap ?? 0, r.eval_drift ?? 0)));

    if (conv && conv.converged)
      return { tone: "bad", text: `Converged — the trainer exhausted its reseeds at best ${fmt(conv.converged_best, 3)}. `
        + "It will not improve further on this configuration." };

    /* entropy collapse — the classic silent death for these runs */
    if (it && it.entropy != null && it.entropy < 0.4)
      return { tone: "bad", text: `Entropy collapsed to ${fmt(it.entropy)} — the policy has stopped exploring `
        + "and is locked into whatever it does now. Reseed, or raise the entropy coefficient." };

    /* KL clipping most updates */
    const recent = S.rows.slice(-40).filter(r => r && r.kl != null);
    const clipped = recent.filter(r => r.kl_stop).length;
    if (recent.length >= 10 && clipped / recent.length > 0.6)
      return { tone: "warn", text: `KL early-stopped ${clipped} of the last ${recent.length} updates — `
        + "the trust region is fighting the step size. Lower the learning rate or raise target_kl." };

    /* Regression: the latest eval has fallen away from the banked peak. Worth
       saying before anything else — a run going backwards reads exactly like a
       run that never started if you only look at the current numbers. */
    const drop = best > 0 ? (best - score) / best : 0;
    if (ev.length >= 2 && drop > 0.25) {
      const peak = ev.reduce((a, r) => {
        const sc = r.eval_score ?? L.score(r.eval_lap ?? 0, r.eval_drift ?? 0);
        return sc > (a.sc ?? -1) ? { sc, r } : a;
      }, {});
      return { tone: "bad", text: `Regressing — score peaked at ${fmt(best, 3)} `
        + `(iteration ${peak.r && peak.r.x != null ? peak.r.x : "?"}) and is now ${fmt(score, 3)}, `
        + `${Math.round(drop * 100)}% down. Lap ${fmt(lap)}, ${L.yName || "laps"} ${fmt(sty)}. `
        + "The banked best checkpoint is still the one to keep — this run is walking away from it." };
    }

    if (S.mode === "race")
      return lap < 0.5
        ? { tone: "bad", text: `Completing ${fmt(lap)} laps per eval — the policy is not finishing laps yet.` }
        : { tone: "ok", text: `${fmt(lap)} laps per eval, best score ${fmt(best, 3)}.` };

    /* --- the hybrid / drift story: which term is the bottleneck? --- */
    if (S.mode === "hybrid") {
      /* Style now ramps in with curriculum difficulty (HybridReward.style_ramp_*),
         so style==0 below the gate is the design working, not the policy failing.
         Saying otherwise would send you tuning a knob that is deliberately shut. */
      const dRow = lastWith("difficulty");
      const diff = dRow ? dRow.difficulty : null;
      if (sty < 0.05 && diff != null && diff < STYLE_RAMP_LO)
        return { tone: "wait", text: `Style is ${fmt(sty)} by design — the style bonus ramps in `
          + `between difficulty ${STYLE_RAMP_LO} and ${STYLE_RAMP_HI}, and the curriculum is at `
          + `${fmt(diff)}. Right now this is training the race backbone; lap is at ${fmt(lap)} `
          + `and needs to reach ${HYBRID_PROMOTE_LAP} to advance.` };
      if (lap < 0.25 && sty < 0.05)
        return { tone: "bad", text: `Lap ${fmt(lap)}, style ${fmt(sty)}. The policy neither completes laps nor drifts. `
          + `Score is ${fmt(score, 3)} because style sits at its 0.60 floor — you are training a slow racer, not a hybrid. `
          + "Lap progress has to come first; style multiplies it, it cannot rescue it." };
      if (sty < 0.05)
        return { tone: "warn", text: `Style is ${fmt(sty)} — effectively zero. The multiplier is stuck at its 0.60 floor, `
          + `so score tracks lap alone (${fmt(lap)} → ${fmt(score, 3)}). There is 40% of the objective going unclaimed.` };
      if (lap < 0.5)
        return { tone: "warn", text: `Style is landing (${fmt(sty)}) but lap ${fmt(lap)} is the bottleneck — `
          + "score is completion-dominant, so lap progress buys more than style does from here." };
      return { tone: "ok", text: `Lap ${fmt(lap)} × style ${fmt(sty)} → ${fmt(score, 3)} `
        + `(best ${fmt(best, 3)}). Both terms contributing.` };
    }

    /* drift */
    if (lap < 0.15)
      return { tone: "bad", text: `Lap ${fmt(lap)} — drift score is completion-gated, so even a perfect slide `
        + `is capped at 0.30× until it finishes laps. Current ${fmt(score, 3)}.` };
    if (sty < 0.1)
      return { tone: "warn", text: `Drift angle ${fmt(sty)} is near zero — it is driving, not sliding.` };
    return { tone: "ok", text: `Drift ${fmt(sty)} gated by lap ${fmt(lap)} → ${fmt(score, 3)} (best ${fmt(best, 3)}).` };
  }

  /* ---------------- panel: score decomposition ---------------- */

  function renderScore() {
    const host = $("#lw-score"); if (!host) return;
    const L = LINEAGE[S.mode];
    const ev = evals(); const e = last(ev);
    if (!L || !e) { host.innerHTML = `<div class="lw-empty">no eval banked yet</div>`; return; }

    const lap = e.eval_lap ?? 0, sty = e.eval_drift ?? 0;
    const score = e.eval_score ?? L.score(lap, sty);

    if (S.mode === "race") {
      host.innerHTML =
        `<div class="lw-formula">${L.formula}</div>
         <div class="lw-bigrow"><span class="lw-big num">${fmt(lap)}</span><span class="lw-bigcap">laps / eval</span></div>`;
      return;
    }

    /* multiplier band: floor -> floor+span, driven by the style term */
    const mult = L.floor + L.span * (S.mode === "hybrid" ? sty : lap);
    const driver = S.mode === "hybrid" ? lap : sty;          // the dominant term
    const gate = S.mode === "hybrid" ? sty : lap;            // the multiplier term
    const driverName = S.mode === "hybrid" ? "lap" : "drift";
    const gateName = S.mode === "hybrid" ? "style" : "lap";
    const pct = Math.max(0, Math.min(1, (mult - L.floor) / L.span)) * 100;

    host.innerHTML =
      `<div class="lw-formula">${L.formula}</div>
       <div class="lw-term">
         <div class="lw-term-head"><span>${driverName}</span><b class="num">${fmt(driver)}</b></div>
         <div class="lw-bar"><i style="width:${Math.max(0, Math.min(1, driver)) * 100}%"></i></div>
         <div class="lw-term-note">the dominant term — score is linear in this</div>
       </div>
       <div class="lw-term">
         <div class="lw-term-head"><span>${gateName} multiplier</span>
           <b class="num">×${fmt(mult)}</b></div>
         <div class="lw-bar lw-bar-mult"><i style="width:${pct}%"></i>
           <u style="left:0">${L.floor.toFixed(2)}</u><u style="right:0">${(L.floor + L.span).toFixed(2)}</u></div>
         <div class="lw-term-note">${gateName} ${fmt(gate)} — can only scale the result inside this band</div>
       </div>
       <div class="lw-result"><span>score</span><b class="num">${fmt(score, 3)}</b></div>`;
  }

  /* ---------------- panel: lap x style frontier ---------------- */

  function renderFrontier() {
    const host = $("#lw-frontier"); if (!host) return;
    const L = LINEAGE[S.mode];
    const ev = evals();
    if (!L || L.yKey == null) { host.innerHTML = `<div class="lw-empty">single-objective run — no frontier</div>`; return; }
    if (!ev.length) { host.innerHTML = `<div class="lw-empty">no eval banked yet</div>`; return; }

    const box = svgHost(host, 260), svg = box.svg, W = box.w, H = box.h;
    const ml = 44, mr = 16, mt = 16, mb = 32;
    /* scale to the data — a fixed 0..1 floor buries a run sitting at lap 0.16 */
    const xMax = Math.max(0.05, ...ev.map(r => r.eval_lap ?? 0)) * 1.2;
    const yMax = Math.max(0.05, ...ev.map(r => r.eval_drift ?? 0)) * 1.2;
    const X = v => ml + (v / xMax) * (W - ml - mr);
    const Y = v => H - mb - (v / yMax) * (H - mt - mb);

    /* iso-score contours — the shape of the objective itself */
    const scores = ev.map(r => r.eval_score ?? L.score(r.eval_lap ?? 0, r.eval_drift ?? 0));
    const top = Math.max(...scores, 0.05);
    [0.25, 0.5, 0.75, 1.0].forEach(f => {
      const s = top * f;
      const pts = [];
      for (let i = 0; i <= 60; i++) {
        const x = (i / 60) * xMax;
        const y = L.styleFor(s, x);
        if (isFinite(y) && y >= 0 && y <= yMax && x > 0) pts.push(`${X(x)},${Y(y)}`);
      }
      if (pts.length > 1) {
        el("polyline", { points: pts.join(" "), fill: "none", stroke: "var(--line2)",
                         "stroke-width": 1, "stroke-dasharray": f === 1 ? "" : "2 3",
                         opacity: f === 1 ? .9 : .5 }, svg);
        if (f === 1) label(svg, X(xMax) - 4, Y(L.styleFor(s, xMax)) - 5,
                           `best ${fmt(s, 2)}`, "var(--dim)", 9, { "text-anchor": "end" });
      }
    });

    /* axes */
    el("line", { x1: ml, y1: H - mb, x2: W - mr, y2: H - mb, stroke: "var(--line)" }, svg);
    el("line", { x1: ml, y1: mt, x2: ml, y2: H - mb, stroke: "var(--line)" }, svg);
    label(svg, W - mr, H - 8, L.xName, "var(--muted)", 10, { "text-anchor": "end" });
    label(svg, 4, mt + 8, L.yName, "var(--muted)", 10);
    label(svg, ml - 5, H - mb + 3, "0", "var(--muted)", 9, { "text-anchor": "end" });
    label(svg, ml - 5, mt + 8, fmt(yMax, 1), "var(--muted)", 9, { "text-anchor": "end" });
    label(svg, W - mr, H - mb + 14, fmt(xMax, 1), "var(--muted)", 9, { "text-anchor": "end" });

    /* trajectory through the evals, oldest -> newest */
    const path = ev.map(r => `${X(r.eval_lap ?? 0)},${Y(r.eval_drift ?? 0)}`);
    if (path.length > 1)
      el("polyline", { points: path.join(" "), fill: "none", stroke: "var(--info)",
                       "stroke-width": 1.2, opacity: .45 }, svg);

    const bestI = scores.indexOf(Math.max(...scores));
    ev.forEach((r, i) => {
      const isLast = i === ev.length - 1, isBest = i === bestI;
      el("circle", { cx: X(r.eval_lap ?? 0), cy: Y(r.eval_drift ?? 0),
                     r: isLast ? 4.5 : isBest ? 4 : 2.2,
                     fill: isLast ? "var(--accent)" : isBest ? "var(--good)" : "var(--info)",
                     opacity: isLast || isBest ? 1 : .38,
                     stroke: isBest ? "var(--good)" : "none",
                     "stroke-width": isBest ? 1.5 : 0,
                     "fill-opacity": isBest && !isLast ? 0 : 1 }, svg);
    });
    /* the origin trap: everything piled at (0,0) means nothing is happening */
    if (ev.length > 3 && Math.max(...ev.map(r => r.eval_drift ?? 0)) < 0.02)
      label(svg, ml + 8, mt + 20, `${L.yName} flat at zero`, "var(--danger)", 10);
  }

  /* ---------------- generic line chart ---------------- */

  function lineChart(host, series, opts) {
    opts = opts || {};
    if (!host) return;
    const rows = S.rows.filter(r => r && r.x != null);
    if (rows.length < 2) { host.innerHTML = `<div class="lw-empty">waiting for iterations</div>`; return; }
    const box = svgHost(host, opts.h || 150), svg = box.svg, W = box.w, H = box.h;
    const ml = 46, mr = 14, mt = 12, mb = 24;
    const xs = rows.map(r => r.x);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const all = [];
    series.forEach(s => rows.forEach(r => { if (r[s.key] != null) all.push(r[s.key]); }));
    if (!all.length) { host.innerHTML = `<div class="lw-empty">no data</div>`; return; }
    let lo = opts.lo != null ? opts.lo : Math.min(...all);
    let hi = opts.hi != null ? opts.hi : Math.max(...all);
    if (hi - lo < 1e-6) { hi = lo + 1; }
    const X = v => ml + ((v - x0) / Math.max(1, x1 - x0)) * (W - ml - mr);
    const Y = v => H - mb - ((v - lo) / (hi - lo)) * (H - mt - mb);

    el("line", { x1: ml, y1: H - mb, x2: W - mr, y2: H - mb, stroke: "var(--line)" }, svg);
    label(svg, ml - 5, H - mb + 3, fmt(lo, 2), "var(--muted)", 9, { "text-anchor": "end" });
    label(svg, ml - 5, mt + 8, fmt(hi, 2), "var(--muted)", 9, { "text-anchor": "end" });
    label(svg, ml, H - 6, String(x0), "var(--muted)", 9);
    label(svg, W - mr, H - 6, String(x1), "var(--muted)", 9, { "text-anchor": "end" });

    series.forEach(s => {
      const pts = rows.filter(r => r[s.key] != null).map(r => `${X(r.x)},${Y(r[s.key])}`);
      if (pts.length > 1)
        el("polyline", { points: pts.join(" "), fill: "none", stroke: s.color,
                         "stroke-width": s.w || 1.4, opacity: s.op || 1 }, svg);
    });

    /* reseed kicks + kl early-stops as event ticks */
    if (opts.events) {
      rows.forEach(r => {
        if (r.restart != null)
          el("line", { x1: X(r.x), y1: mt, x2: X(r.x), y2: H - mb,
                       stroke: "var(--warn)", "stroke-width": 1, "stroke-dasharray": "2 2", opacity: .7 }, svg);
        if (r.kl_stop)
          el("circle", { cx: X(r.x), cy: H - mb - 2, r: 1.6, fill: "var(--danger)", opacity: .75 }, svg);
      });
    }
    return svg;
  }

  /* ---------------- panels ---------------- */

  function renderCurriculum() {
    lineChart($("#lw-curriculum"), [{ key: "difficulty", color: "var(--accent)", w: 1.8 }],
              { lo: 0, hi: 1.02, h: 140, events: true });
    const d = lastWith("difficulty");
    const cap = $("#lw-curriculum-cap");
    if (cap) cap.textContent = d
      ? `difficulty ${fmt(d.difficulty)} · ${d.difficulty >= 0.999 ? "curriculum maxed" : "still widening"}`
      : "";
  }

  function renderHealth() {
    lineChart($("#lw-entropy"), [{ key: "entropy", color: "var(--info2)", w: 1.6 }], { h: 120, events: true });
    lineChart($("#lw-loss"), [
      { key: "value_loss", color: "var(--warn)", w: 1.3 },
      { key: "policy_loss", color: "var(--info)", w: 1.3 },
    ], { h: 120 });
    const it = lastWith("entropy");
    const cap = $("#lw-health-cap");
    if (cap && it) {
      const bits = [`ent ${fmt(it.entropy)}`];
      if (it.kl != null) bits.push(`kl ${fmt(it.kl, 4)}${it.kl_stop ? " • stopped" : ""}`);
      bits.push(`vf ${fmt(it.value_loss)}`, `pi ${fmt(it.policy_loss, 3)}`);
      cap.textContent = bits.join("  ·  ");
    }
  }

  function renderObjective() {
    const L = LINEAGE[S.mode];
    if (!L) return;
    const series = [{ key: "eval_score", color: "var(--good)", w: 1.9 },
                    { key: "eval_lap", color: "var(--info)", w: 1.3, op: .8 }];
    if (L.yKey) series.push({ key: "eval_drift", color: "var(--info2)", w: 1.3, op: .8 });
    lineChart($("#lw-objective"), series, { h: 150, lo: 0 });
  }

  function renderState() {
    const host = $("#lw-state"); if (!host) return;
    const rs = lastWith("restart"), cv = lastWith("converged");
    const air = lastWith("air"), it = lastWith("difficulty");
    const ev = evals();
    const L = LINEAGE[S.mode];
    const best = ev.length
      ? Math.max(...ev.map(r => r.eval_score ?? (L ? L.score(r.eval_lap ?? 0, r.eval_drift ?? 0) : 0))) : null;
    const cells = [
      ["iterations", it && it.x != null ? String(it.x) : "—"],
      ["evals banked", String(ev.length)],
      ["best score", fmt(best, 3)],
      ["reseeds", rs ? `${rs.restart}/${rs.max_restarts}` : "0"],
      ["converged", cv && cv.converged ? "yes" : "no"],
    ];
    if (air && air.air != null) cells.push(["airtime", `${air.jumps ?? 0} jumps · ${fmt(air.air, 1)}s`]);
    host.innerHTML = cells.map(([k, v]) =>
      `<div><span>${k}</span><b class="num">${v}</b></div>`).join("");
  }

  function renderHeader() {
    const L = LINEAGE[S.mode];
    const kicker = $("#lw-kicker"), title = $("#lw-title"), sub = $("#lw-sub");
    if (kicker) kicker.textContent = L ? `${L.label} LINEAGE` : "LINEAGE";
    if (title) title.textContent = L ? L.blurb : "—";
    if (sub) {
      const spec = String(S.label || "").split(":")[1];
      sub.textContent = [S.label ? S.label.split(":")[0] : "",
                         spec ? `specialist · ${spec}` : "generalist · track pool",
                         S.live ? "live" : "finished"].filter(Boolean).join("  ·  ");
    }
    const v = verdict(), box = $("#lw-verdict");
    if (box) { box.textContent = v.text; box.dataset.tone = v.tone; }
  }

  /* ---------------- lifecycle ---------------- */

  function refresh() {
    if (!pull()) return;
    renderHeader(); renderScore(); renderFrontier();
    renderObjective(); renderCurriculum(); renderHealth(); renderState();
  }

  let timer = null, active = false;
  function setActive(on) {
    if (on === active) { if (on) refresh(); return; }
    active = on;
    const wall = $("#lineage-wall");
    if (wall) wall.hidden = !on;
    if (on) { refresh(); timer = setInterval(refresh, 2000); }
    else if (timer) { clearInterval(timer); timer = null; }
  }

  Object.assign(window.LineageWall, { refresh, setActive });
})();

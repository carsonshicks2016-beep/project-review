/* ==========================================================================
   PIT WALL — Quantum Particle Swarm training screen (tab-fable).
   Board renderer + particle engine, all fed by real endpoints:
     /api/fable-five      manifest (hero, gates, levers, garage, heat, ladder)
     /api/fable-pit-log   parsed pit-log tail -> eval timeline + feed + calm ladder
     /api/fable-track     centreline / landmarks / sector names (cached server-side)
     /api/fable-diag(/x)  Brain Lab telemetry trace + behaviour
     /api/status + SSE    live optimiser channels from the running trainer
   Loads after app.js: reuses its globals ($, api, esc, launch, fableParams).
   Exposed as window.PitWall { refresh, setActive }.
   ========================================================================== */
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const q = s => document.querySelector(s);
  const C = { cyan: "#6FE3FF", violet: "#C39BFF", green: "#7CFFB0", red: "#FF7A6E",
              amber: "#FFC24D", ink2: "#7E97AA", ink3: "#46596B",
              grid: "rgba(70,110,150,.14)" };
  const MONO = "ui-monospace,Menlo,monospace";
  const DISP = "Avenir Next Condensed,Arial Narrow,sans-serif";
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const BELLOF = 371.13, EVO = 319.55;
  // pit-wall gate thresholds — mirror supra/fable5.py (PitWall healthy_* +
  // _frontier_scale_next mastery). Displayed only; the trainer is the truth.
  const HEALTHY = { cleanFrac: 12 / 16, terminal: 0.25, progress: 0.90 };
  const MASTERY_PACE = 0.88;
  const STAGES = ["foundation", "flow", "finish", "fast", "frontier"];

  const fmt = (v, d = 2) => (v == null || isNaN(v)) ? "--" : Number(v).toFixed(d);
  const fmtLap = s => s == null ? "--" :
    `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`;

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function makeSVG(host, w, h) {
    host.innerHTML = "";
    const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%" });
    host.appendChild(svg);
    return svg;
  }
  function text(svg, x, y, s, fill, size, extra = {}) {
    const t = svgEl("text", { x, y, fill, "font-size": size,
      "font-family": MONO, ...extra }, svg);
    t.textContent = s;
    return t;
  }
  let tip;
  function showTip(ev, s) {
    if (!tip) { tip = document.createElement("div"); tip.id = "pw-tip";
                document.body.appendChild(tip); }
    tip.style.display = "block"; tip.textContent = s;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - r.width - 8) + "px";
    tip.style.top = Math.min(ev.clientY + 14, innerHeight - r.height - 8) + "px";
  }
  function hideTip() { if (tip) tip.style.display = "none"; }

  /* ======================= THE SWARM ENGINE =======================
     One canvas glued to the tab's content (local coordinates — it scrolls
     with the board). Homes: panel perimeters (flowing streams), the circuit
     river, the rasterised gap numeral, void dust. Forces: spring home,
     cursor gravity + swirl, click shockwaves, heat-scaled agitation.  */
  const swarm = (() => {
    let cv, ctx, tab, P = [], waves = [], sprites = {}, dpr = 1;
    let timer = null, rafAlive = false, running = false, last = 0;
    let virgin = true;      /* first build after activation plays the full
                               scatter->form entrance; the 20s data polls
                               rebuild QUIETLY (particles spawn near home) */
    const mouse = { x: -9e3, y: -9e3 };

    function sprite(col) {
      if (sprites[col]) return sprites[col];
      const s = document.createElement("canvas"); s.width = s.height = 24;
      const g = s.getContext("2d"), r = 12;
      const grad = g.createRadialGradient(r, r, 0, r, r, r);
      grad.addColorStop(0, col); grad.addColorStop(.4, col);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      g.globalAlpha = .5; g.fillStyle = grad; g.fillRect(0, 0, 24, 24);
      g.globalAlpha = 1; g.fillStyle = col;
      g.beginPath(); g.arc(r, r, 2.4, 0, 7); g.fill();
      g.fillStyle = "rgba(255,255,255,.9)";
      g.beginPath(); g.arc(r, r, 1.1, 0, 7); g.fill();
      sprites[col] = s; return s;
    }
    function add(hx, hy, col, o = {}) {
      const R = virgin ? 600 : 16;
      P.push({ hx, hy, x: hx + (Math.random() - .5) * R,
        y: hy + (Math.random() - .5) * R, vx: 0, vy: 0, col,
        size: o.size || (1.2 + Math.random() * 1.2),
        jitter: o.jitter ?? .1, flow: o.flow || null, el: o.el || null,
        scatter: 0, spring: o.spring || .03, alpha: o.alpha || .85 });
    }
    function local(el) {
      const r = el.getBoundingClientRect(), tr = tab.getBoundingClientRect();
      return { x: r.left - tr.left, y: r.top - tr.top, w: r.width, h: r.height };
    }
    function buildBorders() {
      const ACC = { amber: C.amber, cyan: C.cyan, violet: C.violet,
                    green: C.green, red: C.red };
      tab.querySelectorAll(".pw-panel").forEach(p => {
        const r = local(p), col = ACC[p.dataset.accent] || C.cyan;
        const per = 2 * (r.w + r.h), n = Math.min(220, Math.floor(per / 11));
        const path = tt => {
          const t = ((tt % 1) + 1) % 1, d = t * per;
          if (d < r.w) return [r.x + d, r.y];
          if (d < r.w + r.h) return [r.x + r.w, r.y + (d - r.w)];
          if (d < 2 * r.w + r.h) return [r.x + r.w - (d - r.w - r.h), r.y + r.h];
          return [r.x, r.y + r.h - (d - 2 * r.w - r.h)];
        };
        for (let i = 0; i < n; i++)
          add(...path(i / n), col, { flow: { path, t: i / n,
            speed: .9 + Math.random() * .5 }, size: 1 + Math.random(),
            jitter: .07, el: p, alpha: .7 });
      });
    }
    function buildTrack(pts, heat) {
      if (!pts || !pts.length) return;
      const hmax = Math.max(...heat, 1e-9), n = pts.length;
      const path = tt => pts[Math.floor((((tt % 1) + 1) % 1) * (n - 1))];
      const count = Math.min(1500, n * 3);
      for (let i = 0; i < count; i++) {
        const t = i / count, s = Math.min(15, Math.floor(t * 16));
        const h = (heat[s] || 0) / hmax;
        const col = h > .72 ? C.red : h > .45 ? C.amber : C.green;
        const p0 = path(t);
        add(p0[0], p0[1], col, { flow: { path, t,
          speed: .14 + Math.random() * .08 },
          size: h > .72 ? 1.5 + Math.random() * 1.3 : 1 + Math.random(),
          jitter: .1 + h * 1.1, alpha: .4 + h * .6 });
      }
    }
    function buildHero() {
      const el = q("#pw-gap");
      if (!el || REDUCED) return;
      el.classList.remove("swarmed");
      const str = el.dataset.swarm || el.textContent.trim();
      if (!str || str === "--") return;
      const r = local(el);
      if (r.w < 30) return;
      const off = document.createElement("canvas");
      const ow = Math.ceil(r.w) + 24, oh = Math.max(40, Math.ceil(r.h));
      off.width = ow; off.height = oh;
      const g = off.getContext("2d");
      g.font = "700 54px " + MONO; g.textBaseline = "middle"; g.fillStyle = "#fff";
      g.fillText(str, 0, oh / 2 + 1);
      const numW = g.measureText(str).width;
      g.font = "500 22px " + MONO;
      g.fillText(" s", numW, oh / 2 + 10);
      const img = g.getImageData(0, 0, ow, oh).data;
      for (let y = 0; y < oh; y += 3)
        for (let x = 0; x < ow; x += 3)
          if (img[(y * ow + x) * 4 + 3] > 128)
            add(r.x + x, r.y + y, Math.random() < .86 ? "#EAF6FF" : C.cyan,
                { size: 1 + Math.random() * .9, jitter: .06, spring: .085,
                  el, alpha: .95 });
      el.classList.add("swarmed");
    }
    function gWidth(g, s, font) {
      const f = g.font; g.font = font;
      const w = g.measureText(s).width; g.font = f; return w;
    }
    function buildDust() {
      const h = tab.scrollHeight;
      for (let i = 0; i < 200; i++)
        add(Math.random() * tab.clientWidth, Math.random() * h, "#7EA8CC",
            { size: .7 + Math.random() * .7, jitter: .4, spring: .0015, alpha: .2 });
    }
    function shockwave(x, y, el) {
      if (REDUCED || !running) return;
      waves.push({ x, y, max: 520, born: performance.now() });
      for (const p of P) {
        const dx = p.x - x, dy = p.y - y, d = Math.hypot(dx, dy) || 1;
        const own = el && p.el === el;
        const f = 44 * Math.exp(-d / 250) * (own ? 3.4 : 1);
        p.vx += dx / d * f * (.6 + Math.random() * .8);
        p.vy += dy / d * f * (.6 + Math.random() * .8);
        if (own) p.scatter = 1;
      }
      if (el) {
        el.classList.add("scattered");
        setTimeout(() => {
          for (const p of P) if (p.el === el) p.scatter = 0;
          el.classList.remove("scattered");
          el.classList.add("reformed");
          setTimeout(() => el.classList.remove("reformed"), 620);
        }, 470);
      }
    }
    function frame(ts) {
      /* frame-rate independent: big gaps between ticks (throttled timers,
         hidden panes, real-browser hiccups) become several UNIT substeps,
         not one slow-motion step — the swarm converges in wall time. */
      const gap = (ts - last) / 16.7 || 1; last = ts;
      const nsub = Math.max(1, Math.min(10, Math.round(gap)));
      const W = cv.width / dpr, H = cv.height / dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.globalCompositeOperation = "lighter";
      for (const p of P) {
        for (let s = 0; s < nsub; s++) {
          if (p.flow) {
            p.flow.t += p.flow.speed * .0011;
            const h = p.flow.path(p.flow.t);
            p.hx = h[0]; p.hy = h[1];
          }
          if (!p.scatter) {
            p.vx += (p.hx - p.x) * p.spring;
            p.vy += (p.hy - p.y) * p.spring;
          }
          const dx = mouse.x - p.x, dy = mouse.y - p.y, d2 = dx * dx + dy * dy;
          if (d2 < 19600) {
            const d = Math.sqrt(d2) || 1, g = (1 - d / 140) * 1.3;
            p.vx += dx / d * g + (-dy / d) * g * .8;
            p.vy += dy / d * g + (dx / d) * g * .8;
          }
          p.vx += (Math.random() - .5) * p.jitter;
          p.vy += (Math.random() - .5) * p.jitter;
          p.vx *= .86; p.vy *= .86;
          p.x += p.vx; p.y += p.vy;
        }
        const sz = p.size * 5;
        ctx.globalAlpha = p.alpha;
        ctx.drawImage(sprite(p.col), p.x - sz / 2, p.y - sz / 2, sz, sz);
      }
      ctx.globalAlpha = 1;
      const now = performance.now();
      waves = waves.filter(w => now - w.born < 700);
      for (const w of waves) {
        const k = (now - w.born) / 700;
        ctx.beginPath();
        ctx.arc(w.x, w.y, k * w.max, 0, 7);
        ctx.strokeStyle = `rgba(140,220,255,${.45 * (1 - k)})`;
        ctx.lineWidth = 1.5; ctx.stroke();
      }
      ctx.globalCompositeOperation = "source-over";
    }
    function drive() {
      if (!running) return;
      const loop = ts => { if (!running) return; rafAlive = true; frame(ts);
                           requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
      setTimeout(() => {          /* hosts that throttle rAF (hidden panes) */
        if (running && !rafAlive)
          timer = setInterval(() => frame(performance.now()), 33);
      }, 600);
    }
    function rebuild(trackPts, heat) {
      if (!tab) return;
      P = []; sprites = {};
      const w = tab.clientWidth, h = Math.max(tab.scrollHeight, tab.clientHeight);
      dpr = Math.max(.75, Math.min(devicePixelRatio || 1,
                                   Math.sqrt(9.5e6 / Math.max(1, w * h)), 1.5));
      cv.width = Math.floor(w * dpr); cv.height = Math.floor(h * dpr);
      cv.style.width = w + "px"; cv.style.height = h + "px";
      buildBorders(); buildDust(); buildHero();
      if (trackPts) buildTrack(trackPts, heat || []);
      virgin = false;
    }
    function init(tabEl) {
      if (tab) return;
      tab = tabEl;
      cv = document.createElement("canvas");
      cv.id = "pw-swarm"; cv.setAttribute("aria-hidden", "true");
      ctx = cv.getContext("2d");
      tab.appendChild(cv);
      tab.addEventListener("mousemove", e => {
        const r = tab.getBoundingClientRect();
        mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
      }, { passive: true });
      tab.addEventListener("mouseleave", () => { mouse.x = mouse.y = -9e3; });
      tab.addEventListener("click", e => {
        const r = tab.getBoundingClientRect();
        shockwave(e.clientX - r.left, e.clientY - r.top,
                  e.target.closest(".pw-bt"));
      });
    }
    function setRunning(on) {
      if (on === running) return;
      running = on;
      rafAlive = false;
      if (timer) { clearInterval(timer); timer = null; }
      if (on && !REDUCED) { last = performance.now(); drive(); }
      else if (ctx && cv) {
        ctx.clearRect(0, 0, cv.width, cv.height);
        virgin = true;               /* re-entering the tab replays the entrance */
      }
    }
    return { init, rebuild, setRunning, shockwave,
             ready: () => !!tab, isRunning: () => running,
             debug: () => ({ n: P.length, running, rafAlive,
               hero: P.filter(p => p.el && p.el.id === "pw-gap").slice(0, 4)
                 .map(p => ({ x: p.x | 0, y: p.y | 0, hx: p.hx | 0,
                              hy: p.hy | 0 })) }) };
  })();

  /* ======================= DATA + PANELS ======================= */
  const S = {           // latest fetched state
    manifest: null, pitRows: [], track: null, diag: null, diagName: null,
    metrics: [], live: null, es: null,
  };

  function bestLapInfo() {
    const m = S.manifest || {}, ev = m.latest_eval || {};
    const hofLap = ((m.hall_of_fame || {}).lap || {});
    const cands = [];
    if (hofLap.lap_time) cands.push({ lap: +hofLap.lap_time, src: "hall of fame" });
    if (ev.lap_time) cands.push({ lap: +ev.lap_time, src: "latest eval" });
    const bm = ((m.pit || {}).trends || {}).best_metric;
    if (bm && bm >= 130) cands.push({ lap: 100000 / bm, src: "banked metric" });
    if (!cands.length) return null;
    return cands.sort((a, b) => a.lap - b.lap)[0];
  }

  /* ---------- race control ---------- */
  function renderHeader() {
    const m = S.manifest || {}, auto = m.auto || {}, hist = auto.history || [];
    const cur = m.current_stage || "--";
    const rail = q("#pw-rail");
    rail.innerHTML = "";
    for (let i = 0; i < STAGES.length; i++) {
      const st = STAGES[i];
      const hs = hist.filter(h => h.stage === st);
      const lastH = hs[hs.length - 1];
      const gated = hs.some(h => h.gated);
      const div = document.createElement("div");
      div.className = "pw-st" + (st === cur ? " live" : gated ? " done" : "");
      let sub = "";
      if (lastH) sub = `${lastH.scale != null ? Number(lastH.scale).toFixed(2) : ""}`
        + ` · ${hs.reduce((a, h) => a + (h.iters || 0), 0)} it`;
      else if (st === cur && m.envelope_scale != null)
        sub = `scale ${Number(m.envelope_scale).toFixed(2)}`;
      if (st === cur && lastH && lastH.next_scale != null
          && lastH.next_scale !== lastH.scale)
        sub += ` → ${Number(lastH.next_scale).toFixed(2)}`;
      div.innerHTML = `${st}<span class="pw-sc">${sub || "&nbsp;"}</span>`;
      rail.appendChild(div);
      if (i < STAGES.length - 1) {
        const a = document.createElement("span");
        a.className = "pw-arrow"; a.textContent = "▸";
        rail.appendChild(a);
      }
    }
    const run = (m.pit || {}).run_name || "--";
    if (q("#pw-run")) q("#pw-run").textContent =
      `${run} · ${(m.active_checkpoint || m.active_best_checkpoint || "--")}`;
    if (q("#pw-runmeta")) q("#pw-runmeta").textContent =
      `stage ${cur} · scale ${m.envelope_scale != null
        ? Number(m.envelope_scale).toFixed(2) : "--"}`
      + (m.fable_reward_version ? ` · reward ${m.fable_reward_version}` : "");
    const led = q("#pw-led");
    if (led) {
      led.className = "pw-led" + (S.live ? " live" : "");
      led.innerHTML = `<i></i>${S.live ? "TRAINING" : "IDLE"}`;
    }
    const spent = auto.spent || 0, total = auto.total || 0;
    if (q("#pw-budget-lbl")) q("#pw-budget-lbl").textContent = total
      ? `${spent.toLocaleString()} / ${total.toLocaleString()} it` : "no auto run";
    if (q("#pw-budget-fill")) q("#pw-budget-fill").style.width = total
      ? Math.min(100, spent / total * 100).toFixed(1) + "%" : "0";
  }

  /* ---------- the chase ---------- */
  function renderChase() {
    const m = S.manifest || {};
    const best = bestLapInfo();
    const theo = m.theoretical_lap;
    const gapEl = q("#pw-gap");
    const toEl = q("#pw-gap-to");
    if (best) {
      const gap = best.lap - BELLOF;
      const str = (gap >= 0 ? "+" : "−") + Math.abs(gap).toFixed(2);
      gapEl.dataset.swarm = str;
      gapEl.innerHTML = `${str}<small> s</small>`;
      toEl.textContent = gap <= 0 ? "◈ SUPERHUMAN — line crossed"
        : "◈ Superhuman line not yet crossed";
      toEl.classList.toggle("superhuman", gap <= 0);
      q("#pw-best-lap").textContent = fmtLap(best.lap);
      q("#pw-best-meta").innerHTML =
        `${best.src} · scale ${m.envelope_scale != null
          ? Number(m.envelope_scale).toFixed(2) : "--"}<br>`
        + `theoretical ${fmtLap(theo)} · record ${fmtLap(EVO)}`;
    } else {
      gapEl.dataset.swarm = "";
      gapEl.textContent = "--";
      toEl.textContent = "◈ no clean lap banked yet";
      q("#pw-best-lap").textContent = "--";
      q("#pw-best-meta").innerHTML =
        `theoretical ${fmtLap(theo)} · Bellof ${fmtLap(BELLOF)}`;
    }
    /* axis */
    const host = q("#pw-axis"), w = 760, h = 96;
    const svg = makeSVG(host, w, h);
    const lapT = best ? best.lap : null;
    const t0 = 300, t1 = Math.max(660, (lapT || 0) + 45);
    const X = t => 30 + (t - t0) / (t1 - t0) * (w - 60), y = 58;
    svgEl("line", { x1: X(t0), x2: X(t1), y1: y, y2: y,
      stroke: "rgba(110,160,210,.3)", "stroke-width": 1,
      "stroke-dasharray": "1 4" }, svg);
    const step = t1 - t0 > 500 ? 60 : 30;
    for (let t = t0; t <= t1; t += step) {
      svgEl("circle", { cx: X(t), cy: y, r: 1.2, fill: C.ink3 }, svg);
      text(svg, X(t), y + 16, `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`,
           C.ink3, 9, { "text-anchor": "middle" });
    }
    const marks = [
      { t: EVO, lab: "919 EVO RECORD", col: "#5F7386", up: true },
      { t: BELLOF, lab: "BELLOF ’83", col: C.violet, up: true, big: true },
    ];
    if (theo) marks.push({ t: theo, lab: "PHYSICS FLOOR", col: C.cyan, up: false });
    if (lapT) marks.push({ t: lapT, lab: "BANKED BEST", col: C.green,
                           up: false, big: true });
    for (const mk of marks) {
      svgEl("line", { x1: X(mk.t), x2: X(mk.t), y1: mk.up ? 22 : y,
        y2: mk.up ? y : 78, stroke: mk.col,
        "stroke-width": mk.big ? 1.6 : 1, opacity: .85 }, svg);
      svgEl("circle", { cx: X(mk.t), cy: y, r: mk.big ? 3.4 : 2.2,
        fill: mk.col }, svg);
      if (mk.big) svgEl("circle", { cx: X(mk.t), cy: y, r: 8, fill: "none",
        stroke: mk.col, opacity: .35 }, svg);
      text(svg, X(mk.t), mk.up ? 14 : 92, mk.lab, mk.col, 9.5,
           { "text-anchor": "middle", "font-weight": 600,
             "letter-spacing": "1", "font-family": DISP });
    }
    if (lapT && lapT > BELLOF) {
      svgEl("path", { d: `M ${X(BELLOF)} 34 H ${X(lapT)}`, stroke: C.amber,
        "stroke-width": 1, "stroke-dasharray": "1 5", fill: "none",
        opacity: .9 }, svg);
      text(svg, (X(BELLOF) + X(lapT)) / 2, 30,
           `−${(lapT - BELLOF).toFixed(2)}s to go`, C.amber, 10,
           { "text-anchor": "middle" });
    }
  }

  /* ---------- circuit map ---------- */
  let mapPts = null, mapCursor = null, mapSvg = null;
  function currentHeat() {
    const m = S.manifest || {};
    const sh = m.sector_heat || {};
    return sh[m.current_stage] || sh.fast
      || Object.values(sh).filter(a => Array.isArray(a)).pop()
      || new Array(16).fill(0);
  }
  function sectorRows() {
    /* prefer the diag's measured sectors (pace/clean/reason); else eval hints */
    const names = (S.track || {}).sector_names || [];
    const diagSecs = (S.diag || {}).sectors || [];
    const ev = (S.manifest || {}).latest_eval || {};
    const worst = Object.fromEntries((ev.worst_sectors || [])
      .map(w2 => [w2.sector, w2]));
    const fails = new Set(ev.fail_sectors || []);
    return new Array(16).fill(0).map((_, i) => {
      const d = diagSecs[i] || {};
      const w2 = worst[i];
      return {
        name: d.name || names[i] || `S${i + 1}`,
        pace: d.pace ?? (w2 ? w2.pace : null),
        clean: d.clean ?? (w2 ? !!w2.clean : !fails.has(i)),
        reason: d.reason || (w2 ? w2.reason : null),
      };
    });
  }
  let ghosts = { resumed: null, best: null, latest: null };
  let mapGhostsSvg = { resumed: null, best: null, latest: null };
  let ghostRaf = null;

  function runGhostLoop() {
    if (ghostRaf) cancelAnimationFrame(ghostRaf);
    let start = performance.now();
    function tick(now) {
      if (!S.track || !mapPts) return;
      const L = S.track.len_m;
      const t_sec = ((now - start) / 1000.0) % 200; // Loop over 200 virtual seconds
      
      const updateG = (data, el) => {
        if (!el) return;
        if (!data || data.length === 0) { el.style.display = "none"; return; }
        
        let i = Math.floor(t_sec);
        let frac = t_sec - i;
        if (i >= data.length - 1) i = data.length - 2;
        if (i < 0) return;
        
        let s_m = data[i] + (data[i+1] - data[i]) * frac;
        let p_frac = s_m / L;
        let pt_idx = p_frac * mapPts.length;
        
        let idx = Math.floor(pt_idx);
        let pfrac = pt_idx - idx;
        if (idx >= mapPts.length - 1) idx = mapPts.length - 2;
        if (idx < 0) return;
        
        let x = mapPts[idx][0] + (mapPts[idx+1][0] - mapPts[idx][0]) * pfrac;
        let y = mapPts[idx][1] + (mapPts[idx+1][1] - mapPts[idx][1]) * pfrac;
        
        el.setAttribute("cx", x);
        el.setAttribute("cy", y);
        el.style.display = "block";
      };

      updateG(ghosts.resumed, mapGhostsSvg.resumed);
      updateG(ghosts.best, mapGhostsSvg.best);
      updateG(ghosts.latest, mapGhostsSvg.latest);

      ghostRaf = requestAnimationFrame(tick);
    }
    ghostRaf = requestAnimationFrame(tick);
  }

  async function fetchGhosts() {
    const man = S.manifest || {};
    if (man.resumed_from) {
      api(`/api/fable-diag/${encodeURIComponent(man.resumed_from)}`)
        .then(d => { if (d?.trace?.s_m) ghosts.resumed = d.trace.s_m; }).catch(()=>{});
    }
    if (man.checkpoint) {
      api(`/api/fable-diag/${encodeURIComponent(man.checkpoint)}`)
        .then(d => { if (d?.trace?.s_m) ghosts.best = d.trace.s_m; }).catch(()=>{});
    }
    const latest = man.latest_eval || {};
    if (latest.latest_trace && latest.latest_trace.length > 0) {
      ghosts.latest = latest.latest_trace;
    }
  }

  function renderMap() {
    try {
    const host = q("#pw-mapwrap");
    console.log("renderMap running! S.track:", !!S.track, "S.track.pts:", S.track ? !!S.track.pts : false);

    if (!S.track || !S.track.pts) {
      host.innerHTML = `<div class="pw-empty">track geometry unavailable</div>`;
      return;
    }
    const pts = S.track.pts, W = 1040, H = 1040, pad = 30;
    mapSvg = makeSVG(host, W, H);
    mapPts = pts.map(p => [pad + p[0] * (W - 2 * pad) / 1000,
                           pad + (1000 - p[1]) * (H - 2 * pad) / 1000]);
    const heat = currentHeat(), hmax = Math.max(...heat, 1e-9);
    const secs = sectorRows();
    svgEl("polyline", { points: mapPts.map(p => p.join(",")).join(" "),
      fill: "none", stroke: "rgba(70,110,150,.10)", "stroke-width": 7,
      "stroke-linejoin": "round", "stroke-linecap": "round" }, mapSvg);
    for (let s = 0; s < 16; s++) {
      const a = Math.floor(s / 16 * pts.length);
      const b = Math.min(pts.length - 1, Math.ceil((s + 1) / 16 * pts.length));
      const pl = svgEl("polyline", {
        points: mapPts.slice(a, b + 1).map(p => p.join(",")).join(" "),
        fill: "none", stroke: "rgba(0,0,0,0.001)", "stroke-width": 26,
        cursor: "pointer" }, mapSvg);
      pl.addEventListener("mousemove", ev2 => {
        const sec = secs[s];
        showTip(ev2, `S${s + 1} ${sec.name}\n`
          + (sec.pace != null ? `pace ${fmt(sec.pace)} · ` : "")
          + `heat ${fmt(heat[s], 1)}\n`
          + (sec.clean ? "clean" : `✕ ${sec.reason || "dirty"}`));
        selectTower(s);
      });
      pl.addEventListener("mouseleave", () => { hideTip(); selectTower(-1); });
    }
    svgEl("rect", { x: mapPts[0][0] - 5, y: mapPts[0][1] - 5, width: 10,
      height: 10, fill: "#DCEAF5",
      transform: `rotate(45 ${mapPts[0][0]} ${mapPts[0][1]})` }, mapSvg);
    const ev = (S.manifest || {}).latest_eval || {};
    for (const w2 of ev.worst_sectors || []) {
      const i = Math.floor((w2.sector + .5) / 16 * pts.length), p = mapPts[i];
      if (!p) continue;
      svgEl("circle", { cx: p[0], cy: p[1], r: 14, fill: "none",
        stroke: C.red, "stroke-width": 1, "stroke-dasharray": "2 4",
        opacity: .8 }, mapSvg);
      text(mapSvg, p[0], p[1] - 20, `S${w2.sector + 1}`, C.red, 22,
           { "text-anchor": "middle", "font-weight": 700, "font-family": DISP });
      if (w2.reason)
        text(mapSvg, p[0], p[1] + 34, w2.reason.replace(/_/g, " "), C.ink3, 15,
             { "text-anchor": "middle" });
    }
    const lmPick = ["Flugplatz", "Fuchsröhre", "Karussell", "Brünnchen",
                    "Döttinger Höhe", "Hatzenbach"];
    for (const lm of (S.track.landmarks || []).filter(l =>
        lmPick.includes(l.name))) {
      const i = Math.min(pts.length - 1,
                         Math.floor(lm.arc / S.track.len_m * pts.length));
      const p = mapPts[i];
      const t = svgEl("text", { x: p[0] + 12, y: p[1] - 8,
        fill: "rgba(90,130,170,.5)", "font-size": 15, "font-style": "italic",
        "font-family": "Avenir Next,Helvetica,sans-serif" }, mapSvg);
      t.textContent = lm.name;
    }
    mapCursor = svgEl("circle", { cx: mapPts[0][0], cy: mapPts[0][1], r: 7,
      fill: C.cyan, stroke: "#010208", "stroke-width": 2 }, mapSvg);

    mapGhostsSvg.resumed = svgEl("circle", { r: 6, fill: C.violet, opacity: 0.8 }, mapSvg);
    mapGhostsSvg.best = svgEl("circle", { r: 6, fill: C.green, opacity: 0.8 }, mapSvg);
    mapGhostsSvg.latest = svgEl("circle", { r: 6, fill: "#fff", opacity: 0.8 }, mapSvg);
    
    fetchGhosts().then(() => runGhostLoop());
    } catch(e) {
      console.error("renderMap ERROR:", e);
      document.querySelector('#pw-mapwrap').innerHTML = `<div style="color:red">ERROR: ${e.message}</div>`;
    }
  }
  function mapLocalPts() {
    /* circuit points in TAB coordinates for the swarm river */
    if (!mapPts || !mapSvg) return null;
    const tab = q("#tab-fable");
    const r = mapSvg.getBoundingClientRect(), tr = tab.getBoundingClientRect();
    const k = r.width / 1040;
    return mapPts.map(p => [r.left - tr.left + p[0] * k,
                            r.top - tr.top + p[1] * k]);
  }

  /* ---------- sector tower ---------- */
  function selectTower(s) {
    document.querySelectorAll("#pw-tower tbody tr").forEach((tr, i) =>
      tr.classList.toggle("sel", i === s));
  }
  const REASON = { offtrack_timeout: "OFF TRACK", left_track: "LEFT TRACK",
                   spin: "SPIN", stall: "STALL", "traction-break": "TRACTION" };
  function renderTower() {
    const tb = q("#pw-tower tbody");
    tb.innerHTML = "";
    const heat = currentHeat(), hmax = Math.max(...heat, 1e-9);
    sectorRows().forEach((s, i) => {
      const tr = document.createElement("tr");
      const pace = s.pace, h = (heat[i] || 0) / hmax;
      const pcol = pace == null ? "transparent" : pace >= .9
        ? "var(--pw-violet)" : pace >= .75 ? "var(--pw-green)" : "var(--pw-amber)";
      const hcol = h > .72 ? "var(--pw-red)" : h > .45 ? "var(--pw-amber)"
        : "rgba(40,90,60,.8)";
      tr.innerHTML =
        `<td>${String(i + 1).padStart(2, "0")}</td>`
        + `<td class="pw-name" title="${esc(s.name)}">${esc(s.name)}</td>`
        + `<td>${pace == null ? "--" : fmt(pace)}</td>`
        + `<td><span class="pw-pacebar"><i style="width:${pace == null ? 0
            : Math.min(100, pace * 100)}%;background:${pcol};color:${pcol}"></i></span></td>`
        + `<td><span class="pw-heatcell" style="background:${hcol};opacity:${
            .3 + h * .7};box-shadow:0 0 ${4 + h * 8}px ${hcol}"></span></td>`
        + `<td>${s.clean ? '<span class="pw-schip clean">CLEAN</span>'
            : `<span class="pw-schip dirty">${REASON[s.reason] || "DIRTY"}</span>`}</td>`;
      tr.addEventListener("mouseenter", () => {
        if (!mapPts || !mapCursor) return;
        const p = mapPts[Math.floor((i + .5) / 16 * mapPts.length)];
        mapCursor.setAttribute("cx", p[0]); mapCursor.setAttribute("cy", p[1]);
      });
      tb.appendChild(tr);
    });
  }

  /* ---------- eval timeline (pit log) ---------- */
  function currentRunRows() {
    const rows = S.pitRows || [];
    if (!rows.length) return [];
    const run = rows[rows.length - 1].run;
    return rows.filter(r => r.run === run);
  }
  function renderEvalChart() {
    const host = q("#pw-evalchart");
    const rows = currentRunRows();
    if (rows.length < 2) {
      host.innerHTML = `<div class="pw-empty">no pit-log evals yet — the
        timeline builds from fable5_pit_log.txt as the trainer evals</div>`;
      return;
    }
    const w = 720, h = 240, L = 46, R = 12, T = 14, B = 30;
    const svg = makeSVG(host, w, h);
    const best = Math.max(...rows.map(r => r.best), 1);
    const ymax = Math.max(best * 1.1, 20);
    const x = i => L + i / (rows.length - 1) * (w - L - R);
    const Y = m => T + (1 - (Math.max(-10, m) + 10) / (ymax + 10)) * (h - T - B);
    const gstep = ymax > 120 ? 50 : ymax > 40 ? 20 : 10;
    for (let g = 0; g <= ymax; g += gstep) {
      svgEl("line", { x1: L, x2: w - R, y1: Y(g), y2: Y(g), stroke: C.grid }, svg);
      text(svg, L - 6, Y(g) + 3, String(g), C.ink3, 9, { "text-anchor": "end" });
    }
    svgEl("polyline", { points: rows.map((r, i) =>
      `${x(i)},${Y(r.best)}`).join(" "), fill: "none", stroke: C.green,
      "stroke-width": 1.2, "stroke-dasharray": "4 4", opacity: .7 }, svg);
    svgEl("line", { x1: L, x2: w - R, y1: Y(.25 * best), y2: Y(.25 * best),
      stroke: C.red, "stroke-width": 1, "stroke-dasharray": "2 5",
      opacity: .8 }, svg);
    text(svg, w - R, Y(.25 * best) - 4, "collapse line (0.25 × best)", C.red,
         8.5, { "text-anchor": "end" });
    svgEl("polyline", { points: rows.map((r, i) =>
      `${x(i)},${Y(r.metric)}`).join(" "), fill: "none", stroke: C.violet,
      "stroke-width": 5, opacity: .15, "stroke-linejoin": "round" }, svg);
    svgEl("polyline", { points: rows.map((r, i) =>
      `${x(i)},${Y(r.metric)}`).join(" "), fill: "none", stroke: C.violet,
      "stroke-width": 1.8, "stroke-linejoin": "round" }, svg);
    rows.forEach((r, i) => {
      const cx = x(i), cy = Y(r.metric);
      if (r.lap) svgEl("circle", { cx, cy, r: 5.5, fill: "none",
        stroke: C.green, opacity: .5 }, svg);
      const dot = svgEl("circle", { cx, cy, r: 2.6,
        fill: r.lap ? C.green : C.violet, cursor: "pointer" }, svg);
      dot.addEventListener("mousemove", ev2 => showTip(ev2,
        `${r.t}\nmetric ${fmt(r.metric, 3)} · best ${fmt(r.best, 3)}`
        + (r.lap ? `\nlap ${fmt(r.lap)}s` : "\nno lap")
        + `\n⚑ ${r.decision.toUpperCase()}`));
      dot.addEventListener("mouseleave", hideTip);
      if (r.decision === "rollback")
        svgEl("path", { d: `M ${cx} ${h - B + 4} l 4 7 h -8 z`, fill: C.red }, svg);
      else if (/new best/.test(r.reason || "") || r.metric >= r.best - 1e-9)
        svgEl("path", { d: `M ${cx} ${h - B + 11} l 4 -7 h -8 z`,
          fill: C.green }, svg);
      else if (r.decision === "reseed" || r.decision === "consolidate")
        svgEl("path", { d: `M ${cx} ${h - B + 4} l 4 7 h -8 z`,
          fill: C.violet }, svg);
    });
    text(svg, L, h - 6,
         `run ${rows[0].run} · ${rows.length} evals   ▲ new best   ▼ pit call`,
         C.ink3, 9);
  }

  /* ---------- optimiser channels (live SSE) ---------- */
  function renderPPO() {
    const host = q("#pw-ppo");
    const hist = S.metrics.filter(m => m.policy_loss != null).slice(-80);
    if (hist.length < 3) {
      host.innerHTML = `<div class="pw-empty">${S.live
        ? "waiting for train-log iterations…"
        : "no live trainer — start a run and the channels go live"}</div>`;
      return;
    }
    host.innerHTML = "";
    const chans = [
      { k: "kl", col: C.cyan, cap: "approx-KL",
        note: "* = early stop", band: true },
      { k: "entropy", col: C.violet, cap: "entropy",
        note: "narrows per rollback" },
      { k: "value_loss", col: C.red, cap: "value loss",
        note: "spikes = collapse windows" },
      { k: "return", col: C.green, cap: "rollout return",
        note: "crash-dominated ↓" },
    ];
    for (const ch of chans) {
      const data = hist.map(m => m[ch.k]).filter(v => v != null);
      const div = document.createElement("div");
      div.className = "pw-sm";
      div.innerHTML = `<div class="pw-smcap"><b>${ch.cap}</b>` +
        `<span>${ch.note}</span></div><div class="pw-smplot"></div>`;
      host.appendChild(div);
      if (data.length < 3) {
        div.querySelector(".pw-smplot").innerHTML =
          `<div class="pw-empty">n/a</div>`;
        continue;
      }
      const w = 320, h = 74;
      const svg = makeSVG(div.querySelector(".pw-smplot"), w, h);
      const mn = Math.min(...data), mx = Math.max(...data), sp = mx - mn || 1;
      const x = i => 4 + i / (data.length - 1) * (w - 52);
      const y = v => 6 + (1 - (v - mn) / sp) * (h - 16);
      if (ch.band) {
        const stops = hist.filter(m => m.kl_stop).length;
        div.querySelector(".pw-smcap span").textContent =
          `${stops} early stop${stops === 1 ? "" : "s"} in window`;
      }
      svgEl("polyline", { points: data.map((v, i) =>
        `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: ch.col,
        "stroke-width": 4, opacity: .14 }, svg);
      svgEl("polyline", { points: data.map((v, i) =>
        `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: ch.col,
        "stroke-width": 1.4 }, svg);
      const lastV = data[data.length - 1];
      svgEl("circle", { cx: x(data.length - 1), cy: y(lastV), r: 2.6,
        fill: ch.col }, svg);
      text(svg, w - 46, y(lastV) + 3,
           Math.abs(lastV) >= 100 ? lastV.toFixed(0) : lastV.toFixed(3),
           "#DCEAF5", 10);
    }
  }

  /* ---------- calm ladder + envelope scale ---------- */
  function renderLadders() {
    const host = q("#pw-ladders");
    const rows = currentRunRows().filter(r => r.lr != null);
    const m = S.manifest || {}, hist = ((m.auto || {}).history || [])
      .filter(h => h.scale != null);
    if (!rows.length && !hist.length) {
      host.innerHTML = `<div class="pw-empty">no rollbacks and no ladder
        segments recorded yet</div>`;
      return;
    }
    const w = 720, h = 170, L = 50, R = 150, T = 12, B = 26;
    const svg = makeSVG(host, w, h);
    if (rows.length) {
      const lrs = rows.map(r => r.lr);
      const lmin = Math.log10(Math.min(...lrs, 2e-5) * .8);
      const lmax = Math.log10(Math.max(...lrs, 2e-4) * 1.2);
      const x = i => L + (rows.length === 1 ? 0
        : i / (rows.length - 1) * (w - L - R));
      const yl = v => T + (1 - (Math.log10(v) - lmin) / (lmax - lmin))
        * (h - T - B);
      for (const g of [2e-5, 5e-5, 1e-4, 2e-4]) {
        svgEl("line", { x1: L, x2: w - R, y1: yl(g), y2: yl(g),
          stroke: C.grid }, svg);
        text(svg, L - 5, yl(g) + 3, g.toExponential(0).replace("e-", "e−"),
             C.ink3, 8.5, { "text-anchor": "end" });
      }
      let d = `M ${x(0)} ${yl(rows[0].lr)}`;
      rows.forEach((r, i) => { if (i) d += ` H ${x(i)} V ${yl(r.lr)}`; });
      svgEl("path", { d, fill: "none", stroke: C.amber, "stroke-width": 4,
        opacity: .15 }, svg);
      svgEl("path", { d, fill: "none", stroke: C.amber,
        "stroke-width": 1.6 }, svg);
      rows.forEach((r, i) => {
        const c = svgEl("circle", { cx: x(i), cy: yl(r.lr), r: 2.6,
          fill: C.amber, cursor: "pointer" }, svg);
        c.addEventListener("mousemove", ev2 => showTip(ev2,
          `${r.t} ${r.decision}\nlr ${r.lr.toExponential(2)}`
          + (r.log_std_cut ? `\nlog-std −${fmt(r.log_std_cut)}` : "")));
        c.addEventListener("mouseleave", hideTip);
      });
      svgEl("line", { x1: L, x2: w - R, y1: yl(2e-5), y2: yl(2e-5),
        stroke: C.red, "stroke-dasharray": "2 4", opacity: .8 }, svg);
      text(svg, L + 4, yl(2e-5) - 4,
           "lr floor — a fresh segment re-opens the stage lr", C.red, 8.5);
      text(svg, L, h - 8, `calm ladder: ${rows.length} pit calls this run →`,
           C.ink3, 9);
    } else {
      text(svg, L, h / 2, "no rollbacks this run — calm ladder untouched",
           C.ink3, 10);
    }
    /* scale ladder inset */
    if (hist.length) {
      const seg = hist.slice(-8).map(h2 => ({
        lab: `${h2.stage}${h2.segment ? " seg " + h2.segment : ""}`,
        sc: +h2.scale, next: h2.next_scale != null ? +h2.next_scale : null }));
      const last = seg[seg.length - 1];
      if (last.next != null && last.next !== last.sc)
        seg.push({ lab: "next (planned)", sc: last.next, plan: true });
      const sx0 = w - R + 16, sw = R - 30;
      const vals = seg.map(s2 => s2.sc);
      const lo = Math.min(...vals) - .02, hi = Math.max(...vals) + .02;
      const sy = v => T + (1 - (v - lo) / (hi - lo)) * (h - T - B - 20);
      const sx = i => sx0 + (seg.length === 1 ? sw / 2
        : i / (seg.length - 1) * sw);
      let sd = `M ${sx(0)} ${sy(seg[0].sc)}`;
      seg.forEach((p, i) => { if (i) sd += ` H ${sx(i)} V ${sy(p.sc)}`; });
      svgEl("path", { d: sd, fill: "none", stroke: C.cyan,
        "stroke-width": 1.6 }, svg);
      seg.forEach((p, i) => {
        const c = svgEl("circle", { cx: sx(i), cy: sy(p.sc), r: 2.6,
          fill: p.plan ? "#5F7386" : C.cyan, cursor: "pointer" }, svg);
        c.addEventListener("mousemove", ev2 => showTip(ev2,
          `${p.lab}\nscale ${p.sc.toFixed(2)}`
          + (p.plan ? "\n(planned by the ladder)" : "")));
        c.addEventListener("mouseleave", hideTip);
      });
      text(svg, sx0, h - 8, "envelope scale/segment", C.ink3, 9);
    }
  }

  /* ---------- gates + levers ---------- */
  function renderGates() {
    const host = q("#pw-gates");
    const m = S.manifest || {}, ev = m.latest_eval || {};
    const pit = m.pit || {}, tr = pit.trends || {};
    host.innerHTML = "";
    const secN = ev.sector_count || 16;
    const rows = [
      { k: "Clean sectors", v: ev.clean_sectors,
        need: Math.round(HEALTHY.cleanFrac * secN), max: secN, dir: "≥" },
      { k: "Terminal rate", v: ev.terminal_rate, need: HEALTHY.terminal,
        max: .6, dir: "≤", inv: true },
      { k: "Lap progress", v: ev.progress_frac, need: HEALTHY.progress,
        max: 1.4, dir: "≥" },
      { k: "Pace ratio (mastery)", v: ev.pace_ratio, need: MASTERY_PACE,
        max: 1, dir: "≥" },
    ];
    let healthy = true, known = false;
    for (const r of rows) {
      const div = document.createElement("div");
      div.className = "pw-gate";
      if (r.v == null) {
        div.innerHTML = `<div class="pw-gcap"><b>${r.k}</b>` +
          `<span class="num">-- ${r.dir} ${r.need}</span></div>` +
          `<div class="pw-bullet"><span class="pw-tick" style="left:${
            Math.min(100, r.need / r.max * 100)}%"></span></div>`;
        host.appendChild(div);
        continue;
      }
      known = true;
      const ok = r.inv ? r.v <= r.need : r.v >= r.need;
      if (r.k !== "Pace ratio (mastery)" && !ok) healthy = false;
      const col = ok ? "var(--pw-green)" : "var(--pw-red)";
      div.innerHTML =
        `<div class="pw-gcap"><b>${r.k}</b><span class="num">${fmt(r.v,
          r.k === "Clean sectors" ? 0 : 2)} ${r.dir} ${r.need}${ok
          ? " · ok" : " · FAIL"}</span></div>`
        + `<div class="pw-bullet"><span class="pw-fill" style="width:${
          Math.min(100, r.v / r.max * 100)}%;background:${col};color:${col}">`
        + `</span><span class="pw-tick" style="left:${
          Math.min(100, r.need / r.max * 100)}%"></span></div>`;
      host.appendChild(div);
    }
    const v = q("#pw-verdict");
    const lapArmed = (tr.best_metric || 0) >= 60 && bestLapInfo();
    if (!known) {
      v.className = "pw-verdict"; v.textContent = "▯ no eval yet";
    } else if (healthy) {
      v.className = "pw-verdict ok";
      v.textContent = "▮ HEALTHY — refinement accumulates, no rollback";
    } else {
      v.className = "pw-verdict bad";
      v.textContent = `▮ UNHEALTHY — ${lapArmed ? "rollback armed" : "pit watching"}`
        + (tr.evals_since_best != null
           ? ` (${tr.evals_since_best} evals since best)` : "");
    }
    /* levers */
    const lv = q("#pw-levers");
    const rb = pit.rollbacks_since_best ?? 0, rs = pit.reseeds_since_best ?? 0;
    lv.innerHTML = [
      { v: `${rb}<small>/12</small>`, k: "Rollbacks", cls: rb >= 8 ? "hot" : "" },
      { v: `${rs}<small>/6</small>`, k: "Reseeds", cls: rs >= 4 ? "warm" : "" },
      { v: String(pit.consolidations ?? 0), k: "Consolid.", cls: "" },
      { v: String(tr.evals_since_best ?? "--"), k: "Since best",
        cls: (tr.evals_since_best || 0) >= 6 ? "warm" : "" },
    ].map(l => `<div class="pw-lever ${l.cls}"><div class="pw-v num">${l.v}` +
      `</div><div class="pw-k">${l.k}</div></div>`).join("");
  }

  /* ---------- pit feed ---------- */
  function renderFeed() {
    const host = q("#pw-feed");
    let rows = currentRunRows().slice(-40).reverse();
    if (!rows.length) {
      const dec = ((S.manifest || {}).pit || {}).decisions || [];
      rows = dec.slice().reverse().map(d => ({ t: d.t, decision: d.decision,
        reason: d.reason, metric: d.metric }));
    }
    host.innerHTML = rows.length ? rows.map(r => {
      const kind = r.decision === "rollback" ? "rollback"
        : r.decision === "reseed" ? "reseed"
        : r.decision === "consolidate" ? "consolidate"
        : /new best/.test(r.reason || "") ? "best"
        : /weak eval|regress/.test(r.reason || "") ? "watch" : "continue";
      const msg = (r.reason || "").length > 160
        ? r.reason.slice(0, 157) + "…" : (r.reason || "");
      return `<div class="pw-call ${kind}"><span class="pw-flag"></span>`
        + `<span class="pw-t">${esc(r.t || "")}</span>`
        + `<span class="pw-msg"><b>${esc((r.decision || "")
            .toUpperCase())}</b> — ${esc(msg)}</span></div>`;
    }).join("") : `<div class="pw-empty">no pit decisions yet</div>`;
  }

  /* ---------- garage ---------- */
  function renderGarage() {
    const host = q("#pw-garage");
    const hof = (S.manifest || {}).hall_of_fame || {};
    const cats = [
      { k: "lap", cap: "◆ Lap record",
        stat: r => r.lap_time ? fmtLap(+r.lap_time) : "--" },
      { k: "clean", cap: "◆ Cleanest",
        stat: r => `${r.clean_sectors ?? "--"}/16 · term ${fmt(r.terminal_rate)}` },
      { k: "progress", cap: "◆ Farthest",
        stat: r => `progress ${fmt(r.progress_frac)}×` },
    ];
    host.innerHTML = "";
    let any = false;
    for (const c of cats) {
      const rec = hof[c.k];
      if (!rec || !rec.path) continue;
      any = true;
      const file = String(rec.path).split("/").pop();
      const div = document.createElement("div");
      div.className = "pw-brain";
      div.innerHTML = `<span class="pw-cat">${c.cap}</span>`
        + `<span class="pw-stat num">${c.stat(rec)}</span>`
        + `<span class="pw-file" title="${esc(rec.path)}">${esc(file)}</span>`
        + `<span class="pw-acts">`
        + `<button class="pw-bt" data-act="watch">Watch</button>`
        + `<button class="pw-bt" data-act="diag">Diagnose</button></span>`;
      div.querySelector('[data-act="watch"]').onclick = () =>
        launch("fable_watch", fableParams({ checkpoint: rec.path }),
               `watch ${c.k} brain`);
      div.querySelector('[data-act="diag"]').onclick = () =>
        launch("fable_diagnose", fableParams({ checkpoint: rec.path,
          episodes: "1", max_steps: "" }), `diagnose ${c.k} brain`);
      host.appendChild(div);
    }
    if (!any)
      host.innerHTML = `<div class="pw-empty">no hall-of-fame brains banked
        yet — they appear as the run evals</div>`;
  }

  /* ---------- brain lab ---------- */
  function ds(arr, n) {
    if (!arr || arr.length <= n) return arr || [];
    const step = arr.length / n;
    return new Array(n).fill(0).map((_, i) => arr[Math.floor(i * step)]);
  }
  function renderLab() {
    const host = q("#pw-labstrip"), meta = q("#pw-labmeta");
    const d = S.diag;
    if (!d || !d.trace || !d.trace.s_m || d.trace.s_m.length < 10) {
      host.innerHTML = `<div class="pw-empty">no Brain Lab report yet — run
        Diagnose and the banked-best telemetry lands here</div>`;
      meta.innerHTML = "";
      return;
    }
    const N = 700;
    const t = {}; for (const k of ["s_m", "v", "vref", "thr", "brk", "gear",
                                   "slip", "lat"]) t[k] = ds(d.trace[k], N);
    const n = t.s_m.length, lenM = t.s_m[n - 1] || 20832;
    const w = 1560, hRows = [86, 60, 34, 40], gap = 12, L = 54, R = 16;
    const H = 24 + hRows.reduce((a, b) => a + b + gap, 0) + 18;
    const svg = makeSVG(host, w, H);
    const x = i => L + t.s_m[i] / lenM * (w - L - R);
    let top = 10;
    const z = ds((S.track || {}).z || [], 200);
    (function speedRow() {
      const h = hRows[0];
      const vmax = Math.max(62, ...t.v, ...t.vref) * 1.02;
      const y = v => top + (1 - v / vmax) * h;
      if (z.length > 2) {
        const zmn = Math.min(...z), zsp = Math.max(...z) - zmn || 1;
        const zx = i => L + i / (z.length - 1) * (w - L - R);
        let zd = `M ${zx(0)} ${top + h}`;
        z.forEach((v, i) => zd += ` L ${zx(i)} ${top + h - (v - zmn) / zsp * h * .85}`);
        zd += ` L ${zx(z.length - 1)} ${top + h} Z`;
        svgEl("path", { d: zd, fill: "rgba(40,80,130,.12)" }, svg);
      }
      for (const g of [20, 40, 60]) {
        svgEl("line", { x1: L, x2: w - R, y1: y(g), y2: y(g),
          stroke: C.grid }, svg);
        text(svg, L - 5, y(g) + 3, String(g), C.ink3, 8.5,
             { "text-anchor": "end" });
      }
      svgEl("polyline", { points: t.vref.map((v, i) =>
        `${x(i)},${y(v)}`).join(" "), fill: "none",
        stroke: "rgba(126,151,170,.6)", "stroke-width": 1,
        "stroke-dasharray": "4 3" }, svg);
      svgEl("polyline", { points: t.v.map((v, i) =>
        `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: C.cyan,
        "stroke-width": 3.5, opacity: .15 }, svg);
      svgEl("polyline", { points: t.v.map((v, i) =>
        `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: C.cyan,
        "stroke-width": 1.4 }, svg);
      text(svg, L + 4, top + 10,
           "v (m/s) — cyan · v_ref envelope — dashed · elevation ghosted",
           C.ink2, 9.5);
      top += h + gap;
    })();
    (function pedalRow() {
      const h = hRows[1], y = v => top + (1 - v) * h;
      let dt2 = `M ${x(0)} ${y(0)}`;
      t.thr.forEach((v, i) => dt2 += ` L ${x(i)} ${y(v)}`);
      dt2 += ` L ${x(n - 1)} ${y(0)} Z`;
      svgEl("path", { d: dt2, fill: C.green, opacity: .32 }, svg);
      let db = `M ${x(0)} ${y(0)}`;
      t.brk.forEach((v, i) => db += ` L ${x(i)} ${y(v)}`);
      db += ` L ${x(n - 1)} ${y(0)} Z`;
      svgEl("path", { d: db, fill: C.red, opacity: .38 }, svg);
      text(svg, L + 4, top + 10, "throttle ▲ / brake ▼", C.ink2, 9.5);
      top += h + gap;
    })();
    (function gearRow() {
      const h = hRows[2], y = v => top + (1 - (v - 1) / 4.2) * h;
      let d2 = `M ${x(0)} ${y(t.gear[0])}`;
      t.gear.forEach((v, i) => { if (i) d2 += ` H ${x(i)} V ${y(v)}`; });
      svgEl("path", { d: d2, fill: "none", stroke: C.violet,
        "stroke-width": 1.3 }, svg);
      text(svg, L + 4, top + 9, "gear (RaceBox + agent offset)", C.ink2, 9.5);
      top += h + gap;
    })();
    (function latRow() {
      const h = hRows[3], mid = top + h / 2;
      const half = (S.track || {}).half_width || 5;
      svgEl("line", { x1: L, x2: w - R, y1: mid, y2: mid,
        stroke: C.grid }, svg);
      svgEl("polyline", { points: t.lat.map((v, i) =>
        `${x(i)},${mid - Math.max(-1.4, Math.min(1.4, v / half)) * (h / 2 - 2)}`)
        .join(" "), fill: "none", stroke: C.amber, "stroke-width": 1.1,
        opacity: .85 }, svg);
      text(svg, L + 4, top + 9,
           "lateral offset vs centreline (± track half-width)", C.ink2, 9.5);
      top += h + gap;
    })();
    const secs = d.sectors || [];
    for (let s = 0; s <= 16; s++) {
      const xm = L + s / 16 * (w - L - R);
      svgEl("line", { x1: xm, x2: xm, y1: 6, y2: top - 6,
        stroke: "rgba(60,100,140,.12)" }, svg);
      if (s < 16)
        text(svg, xm + 3, top + 2, `S${s + 1}`,
             secs[s] && !secs[s].clean ? C.red : C.ink3, 8.5);
    }
    const cross = svgEl("line", { x1: x(0), x2: x(0), y1: 6, y2: top - 6,
      stroke: "#DCEAF5", "stroke-width": 1, opacity: .85 }, svg);
    const hit = svgEl("rect", { x: L, y: 0, width: w - L - R, height: H,
      fill: "transparent", cursor: "crosshair" }, svg);
    hit.addEventListener("mousemove", ev2 => {
      const r = svg.getBoundingClientRect();
      const px = (ev2.clientX - r.left) / r.width * w;
      const frac = Math.max(0, Math.min(1, (px - L) / (w - L - R)));
      const i = Math.min(n - 1, Math.round(frac * (n - 1)));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i));
      if (mapPts && mapCursor) {
        const p = mapPts[Math.min(mapPts.length - 1,
          Math.round(frac * (mapPts.length - 1)))];
        mapCursor.setAttribute("cx", p[0]); mapCursor.setAttribute("cy", p[1]);
      }
      showTip(ev2, `${(t.s_m[i] / 1000).toFixed(2)} km\n`
        + `v ${fmt(t.v[i], 1)} m/s · ref ${fmt(t.vref[i], 1)}\n`
        + `thr ${fmt(t.thr[i])} · brk ${fmt(t.brk[i])} · G${t.gear[i]}\n`
        + `slip ${fmt(t.slip[i])} · lat ${fmt(t.lat[i], 1)} m`);
    });
    hit.addEventListener("mouseleave", hideTip);
    const b = d.behavior || {};
    meta.innerHTML =
      `<span>report <b>${esc(S.diagName || "?")}</b></span>`
      + `<span>sim <b>${fmt(b.sim_km, 1)} km</b></span>`
      + `<span>mean pace on track <b>${fmt(b.mean_pace_on_track, 3)}</b></span>`
      + `<span>over envelope <b>${fmt((b.time_over_envelope || 0) * 100, 1)}%</b></span>`
      + `<span>verdict <b>${esc(d.verdict || "--")}</b></span>`;
  }
  function renderBehavior() {
    const host = q("#pw-beh");
    const b = ((S.diag || {}).behavior) || null;
    if (!b) { host.innerHTML = ""; return; }
    const tiles = [
      { v: fmt((b.full_throttle_frac || 0) * 100, 1), u: "%", k: "Full throttle" },
      { v: fmt((b.braking_frac || 0) * 100, 1), u: "%", k: "Braking" },
      { v: fmt((b.coasting_frac || 0) * 100, 1), u: "%", k: "Coasting" },
      { v: fmt(b.max_speed, 1), u: "m/s", k: "Top speed" },
      { v: fmt(b.mean_speed, 1), u: "m/s", k: "Mean speed" },
      { v: fmt(b.steer_abs_mean, 3), u: "", k: "Steer |mean|" },
    ];
    host.innerHTML = tiles.map(t2 =>
      `<div class="pw-tile"><div class="pw-v num">${t2.v}<small> ${t2.u}` +
      `</small></div><div class="pw-k">${t2.k}</div></div>`).join("");
    const hist = b.pace_histogram || {};
    const keys = Object.keys(hist);
    if (keys.length) {
      const d2 = document.createElement("div");
      d2.className = "pw-tile";
      d2.innerHTML = `<div class="pw-k" style="margin:0 0 6px">Pace
        distribution (v / v_ref)</div><div class="pw-histplot"></div>`;
      host.appendChild(d2);
      const vals = keys.map(k => hist[k]), vmax = Math.max(...vals, 1e-9);
      const w = 300, h = 64;
      const svg = makeSVG(d2.querySelector(".pw-histplot"), w, h);
      keys.forEach((k, i) => {
        const bw = (w - 30) / keys.length - 3;
        const bh = Math.max(2, vals[i] / vmax * (h - 22));
        const bx = 4 + i * ((w - 30) / keys.length);
        const hot = /^(0\.9|1)/.test(k);
        const r = svgEl("rect", { x: bx, y: h - 14 - bh, width: bw,
          height: bh, rx: 2, fill: hot ? C.violet : "rgba(60,110,160,.5)",
          cursor: "pointer" }, svg);
        r.addEventListener("mousemove", ev2 =>
          showTip(ev2, `pace ${k}\n${(vals[i] * 100).toFixed(1)}% of samples`));
        r.addEventListener("mouseleave", hideTip);
        if (i % 2 === 0)
          text(svg, bx + bw / 2, h - 3, k, C.ink3, 8,
               { "text-anchor": "middle" });
      });
    }
  }

  /* ---------- live SSE hookup ---------- */
  function connectLive() {
    api("/api/status").then(st => {
      const tasks = st.tasks || [];
      /* a fable TRAINER (not watch/diagnose): newest running one, else the
         newest finished one still in PROCS (its history is intact) */
      const isTrainer = t2 => {
        const cmd = Array.isArray(t2.cmd) ? t2.cmd.join(" ") : String(t2.cmd || "");
        return /--fable(\s|$)/.test(cmd) && !/--watch-fable|--fable-diag/.test(cmd);
      };
      const trainers = tasks.filter(isTrainer);
      const fable = trainers.filter(t2 => t2.running)[0] || trainers[0];
      const pid = fable && fable.pid;
      if (!pid) { S.live = null; return; }
      if (S.es && S.esPid === pid) {
        S.live = fable.running ? pid : null;
        return;                                    /* already subscribed */
      }
      if (S.es) { S.es.close(); S.es = null; }
      S.esPid = pid;
      S.live = fable.running ? pid : null;
      S.metrics = [];
      const es = new EventSource(`/api/stream/${pid}`);
      S.es = es;
      let paint = 0;
      es.onmessage = e => {
        try {
          const d = JSON.parse(e.data);
          if (d.snapshot && Array.isArray(d.history)) S.metrics = d.history;
          else if (d.metric) S.metrics.push(d.metric);
          else if (d.history) S.metrics = d.history;
          if (d.done) { S.live = null; es.close(); S.es = null; }
          const now = Date.now();
          if (now - paint > 1500) { paint = now; renderPPO(); renderHeader(); }
        } catch (_) { /* keep streaming */ }
      };
      es.onerror = () => { /* server restart etc. — poll re-attaches */ };
    }).catch(() => {});
  }

  /* ---------- refresh orchestration ---------- */
  let refreshing = false;
  async function refresh() {
    console.log("refresh() called. active:", active, "refreshing:", refreshing);
    if (refreshing || !q("#pw-rail")) return;
    refreshing = true;
    try {
      const [five, pitlog] = await Promise.all([
        api("/api/fable-five").catch(() => ({})),
        api("/api/fable-pit-log").catch(() => ({ rows: [] })),
      ]);
      S.manifest = five.manifest || {};
      S.pitRows = pitlog.rows || [];
      if (!S.track)
        S.track = await api("/api/fable-track").catch(() => null);
      /* newest fable diag, once per name */
      const diags = await api("/api/fable-diag").catch(() => []);
      const newest = (diags || []).filter(d => !d.error)[0];
      if (newest && newest.name !== S.diagName) {
        S.diagName = newest.name;
        S.diag = await api(`/api/fable-diag/${encodeURIComponent(newest.name)}`)
          .catch(() => null);
      }
      renderHeader(); renderChase(); renderMap(); renderTower();
      renderEvalChart(); renderPPO(); renderLadders(); renderGates();
      renderFeed(); renderGarage(); renderLab(); renderBehavior();
      connectLive();
      /* (re)build the swarm on fresh geometry — after layout settles.
         setTimeout, not rAF: throttled/embedded hosts may never fire rAF */
      if (swarm.ready())
        setTimeout(() => swarm.rebuild(mapLocalPts(), currentHeat()), 60);
    } finally { refreshing = false; }
  }

  /* ---------- lifecycle ---------- */
  let pollTimer = null, active = false;
  function setActive(on) {
    if (on === active) {
      if (on) refresh();
      return;
    }
    active = on;
    const tab = q("#tab-fable");
    if (on) {
      if (!swarm.ready() && tab) swarm.init(tab);
      swarm.setRunning(!REDUCED);
      refresh();
      pollTimer = setInterval(() => { refresh(); }, 20000);
    } else {
      swarm.setRunning(false);
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      if (S.es) { S.es.close(); S.es = null; }
    }
  }
  /* drawer toggle + resize */
  document.addEventListener("click", e => {
    if (e.target && e.target.id === "pw-board-toggle") {
      const d = q("#pw-drawer");
      d.classList.toggle("open");
      e.target.textContent = d.classList.contains("open")
        ? "Pit board ▴" : "Pit board ▾";
      setTimeout(() => { if (swarm.ready())
        swarm.rebuild(mapLocalPts(), currentHeat()); }, 60);
    }
  });
  let rsz;
  addEventListener("resize", () => {
    if (!active) return;
    clearTimeout(rsz);
    rsz = setTimeout(() => swarm.ready()
      && swarm.rebuild(mapLocalPts(), currentHeat()), 200);
  });

  window.PitWall = { refresh, setActive, debug: () => swarm.debug() };
})();

/* Asclepius console.
 *
 * The whole dataset is embedded, so every interaction is local and instant —
 * no server, no network, no round trip. Statistics are recomputed in the
 * browser on each change.
 *
 * Honesty note, because it matters for this project: correlations here are
 * reported with a Bartlett-adjusted effective N and a Fisher-z confidence
 * interval, so autocorrelation is not silently inflating them. What the browser
 * does NOT do is FDR correction across a family of tests, or regression with
 * confounders held fixed. For those use whoop_rank_drivers and whoop_regress.
 */
(function () {
  "use strict";

  var D = window.__ASCLEPIUS__;
  var M = D.metrics, DATES = D.dates, N = DATES.length;
  var KEYS = Object.keys(M);
  var $ = function (id) { return document.getElementById(id); };
  var el = function (tag, cls) { var e = document.createElement(tag); if (cls) e.className = cls; return e; };

  var S = {
    i0: 0, i1: N - 1,
    primary: M.recovery_score ? "recovery_score" : KEYS[0],
    overlay: [],
    roll: 7,
    driver: M.strain ? "strain" : KEYS[0],
    outcome: M.recovery_score ? "recovery_score" : KEYS[0],
    lag: 1,
    day: N - 1,
    cal: M.recovery_score ? "recovery_score" : KEYS[0]
  };

  // ------------------------------------------------------------- utilities
  function label(k) { return D.labels[k] || k; }
  function unit(k) { return D.units[k] || ""; }
  function slice(k, a, b) { return M[k].slice(a === undefined ? S.i0 : a, (b === undefined ? S.i1 : b) + 1); }
  function fmt(v, d) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return Number(v).toFixed(d === undefined ? 1 : d);
  }
  function clean(a) { return a.filter(function (v) { return v !== null && !isNaN(v); }); }
  function extent(a) {
    var c = clean(a);
    if (!c.length) return [0, 1];
    var lo = Math.min.apply(null, c), hi = Math.max.apply(null, c);
    if (lo === hi) { lo -= 1; hi += 1; }
    return [lo, hi];
  }
  function mean(a) { var c = clean(a); return c.length ? c.reduce(function (s, v) { return s + v; }, 0) / c.length : NaN; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]; }); }

  function rolling(a, w) {
    var out = [], i, j, s, n;
    for (i = 0; i < a.length; i++) {
      s = 0; n = 0;
      for (j = Math.max(0, i - w + 1); j <= i; j++) {
        if (a[j] !== null && !isNaN(a[j])) { s += a[j]; n++; }
      }
      out.push(n >= Math.min(3, w) ? s / n : null);
    }
    return out;
  }

  // ------------------------------------------------------------ statistics
  /* Positive lag means the driver leads the outcome, matching analyze.py:
     driver at t is paired with outcome at t+lag. */
  function pairAt(dk, ok, lag) {
    var xs = [], ys = [], i, j;
    for (i = S.i0; i <= S.i1; i++) {
      j = i + lag;
      if (j < 0 || j > S.i1) continue;
      var x = M[dk][i], y = M[ok][j];
      if (x === null || y === null || isNaN(x) || isNaN(y)) continue;
      xs.push(x); ys.push(y);
    }
    return [xs, ys];
  }

  function pearson(x, y) {
    var n = x.length, i, mx = 0, my = 0;
    if (n < 3) return NaN;
    for (i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    var sxy = 0, sxx = 0, syy = 0, a, b;
    for (i = 0; i < n; i++) {
      a = x[i] - mx; b = y[i] - my;
      sxy += a * b; sxx += a * a; syy += b * b;
    }
    return (sxx && syy) ? sxy / Math.sqrt(sxx * syy) : NaN;
  }

  function autocorr(v, k) {
    var n = v.length, i, m = 0;
    for (i = 0; i < n; i++) m += v[i];
    m /= n;
    var num = 0, den = 0, a;
    for (i = 0; i < n; i++) { a = v[i] - m; den += a * a; }
    for (i = 0; i < n - k; i++) num += (v[i] - m) * (v[i + k] - m);
    return den ? num / den : 0;
  }

  /* Bartlett's effective sample size. Two autocorrelated series correlate by
     accident, so the honest N is well below the number of days. */
  function effectiveN(x, y) {
    var n = x.length, s = 0, k, kmax = Math.min(Math.floor(n / 4), 20);
    for (k = 1; k <= kmax; k++) s += autocorr(x, k) * autocorr(y, k);
    var ne = n / (1 + 2 * s);
    return Math.max(4, Math.min(n, ne));
  }

  function fisherCI(r, ne) {
    if (isNaN(r) || ne <= 4) return [NaN, NaN];
    var z = Math.atanh(Math.max(-0.999999, Math.min(0.999999, r)));
    var se = 1 / Math.sqrt(ne - 3);
    return [Math.tanh(z - 1.96 * se), Math.tanh(z + 1.96 * se)];
  }

  function linfit(x, y) {
    var n = x.length, i, mx = 0, my = 0;
    for (i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    var sxy = 0, sxx = 0;
    for (i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    var b = sxx ? sxy / sxx : 0;
    return { slope: b, intercept: my - b * mx };
  }

  // ---------------------------------------------------------- svg plotting
  function svg(vb, body, cls) {
    return '<svg viewBox="' + vb + '" class="' + (cls || "") + '" preserveAspectRatio="none">' + body + "</svg>";
  }
  function path(vals, w, h, lo, hi) {
    var n = vals.length, step = n > 1 ? w / (n - 1) : w, out = [], down = false, i, v, y;
    for (i = 0; i < n; i++) {
      v = vals[i];
      if (v === null || isNaN(v)) { down = false; continue; }
      y = h - ((v - lo) / ((hi - lo) || 1)) * h;
      y = Math.max(-2, Math.min(h + 2, y));
      out.push((down ? "L" : "M") + (i * step).toFixed(2) + " " + y.toFixed(2));
      down = true;
    }
    return out.join(" ");
  }
  function ramp(t) {
    // dim -> amber -> cream, the console's own heat scale
    t = Math.max(0, Math.min(1, t));
    var a = [46, 26, 10], b = [255, 158, 44], c = [255, 220, 162], p, q, m;
    if (t < 0.6) { m = t / 0.6; p = a; q = b; } else { m = (t - 0.6) / 0.4; p = b; q = c; }
    return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(p[i] + (q[i] - p[i]) * m); }).join(",") + ")";
  }

  // ------------------------------------------------------------ range bar
  function renderRange() {
    var w = 1000, h = 40, rec = M[S.primary] || M[KEYS[0]], e = extent(rec);
    var sel0 = (S.i0 / (N - 1)) * w, sel1 = (S.i1 / (N - 1)) * w;
    var marks = "", seen = {}, lastX = -999;
    DATES.forEach(function (d, i) {
      var key = d.slice(0, 7);
      if (seen[key]) return; seen[key] = 1;
      var x = (i / (N - 1)) * w;
      marks += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="' + h + '" stroke="var(--dim)" stroke-width=".5" opacity=".5"/>';
      if (x - lastX < 58) return;   // a partial first month would collide
      lastX = x;
      marks += '<text x="' + (x + 3) + '" y="' + (h + 11) + '" class="ax">' + d.slice(2, 7) + "</text>";
    });
    var body = marks +
      '<rect x="0" y="0" width="' + w + '" height="' + h + '" fill="rgba(0,0,0,.35)"/>' +
      '<path d="' + path(rolling(rec, 7), w, h, e[0], e[1]) + '" fill="none" stroke="var(--dim)" stroke-width="1.2"/>' +
      '<rect x="' + sel0 + '" y="0" width="' + Math.max(2, sel1 - sel0) + '" height="' + h + '" fill="rgba(255,158,44,.14)" stroke="var(--amber)" stroke-width="1"/>' +
      '<rect id="rangeHit" x="0" y="0" width="' + w + '" height="' + h + '" fill="transparent" style="cursor:crosshair"/>';
    $("rangeChart").innerHTML = svg("0 -4 1000 60", body);
    $("rangeLabel").textContent = DATES[S.i0] + " → " + DATES[S.i1] + "  ·  " + (S.i1 - S.i0 + 1) + " DAYS";
    wireRange();
  }

  function wireRange() {
    var hit = $("rangeHit");
    if (!hit) return;
    var dragging = false, start = 0;
    function idxAt(ev) {
      var box = hit.getBoundingClientRect();
      var f = (ev.clientX - box.left) / box.width;
      return Math.max(0, Math.min(N - 1, Math.round(f * (N - 1))));
    }
    hit.addEventListener("mousedown", function (ev) { dragging = true; start = idxAt(ev); ev.preventDefault(); });
    window.addEventListener("mousemove", function (ev) {
      if (!dragging) return;
      var cur = idxAt(ev);
      S.i0 = Math.min(start, cur); S.i1 = Math.max(start, cur);
      if (S.i1 - S.i0 < 6) S.i1 = Math.min(N - 1, S.i0 + 6);
      drawAll();
    });
    window.addEventListener("mouseup", function () { dragging = false; });
  }

  // --------------------------------------------------------------- vitals
  var DIALS = [
    ["recovery_score", 0, 100, "RECOVERY", "%"],
    ["hrv_rmssd_milli", 0, 200, "HRV", "ms"],
    ["resting_hr", 35, 75, "RESTING HR", "bpm"],
    ["sleep_performance", 0, 100, "SLEEP PERF", "%"]
  ];

  function dial(k, lo, hi, name, u) {
    var vals = slice(k), cur = null, i;
    for (i = vals.length - 1; i >= 0; i--) { if (vals[i] !== null && !isNaN(vals[i])) { cur = vals[i]; break; } }
    var avg = mean(vals);
    var half = Math.floor((S.i1 - S.i0) / 2);
    var prev = mean(slice(k, S.i0, S.i0 + half));
    var now = mean(slice(k, S.i0 + half, S.i1));
    var delta = (!isNaN(prev) && !isNaN(now)) ? now - prev : NaN;

    var r = 54, cx = 70, cy = 70, sweep = 260, st = 90 + (360 - sweep) / 2;
    function pt(deg, rad) { var a = deg * Math.PI / 180; return [cx + rad * Math.cos(a), cy + rad * Math.sin(a)]; }
    function arc(f, rad) {
      f = Math.max(0, Math.min(1, f));
      if (f <= 0) return "";
      var p0 = pt(st, rad), p1 = pt(st + sweep * f, rad);
      return "M" + p0[0].toFixed(2) + " " + p0[1].toFixed(2) + "A" + rad + " " + rad + " 0 " +
        (sweep * f > 180 ? 1 : 0) + " 1 " + p1[0].toFixed(2) + " " + p1[1].toFixed(2);
    }
    var ticks = "", j;
    for (j = 0; j < 27; j++) {
      var deg = st + sweep * (j / 26), maj = j % 5 === 0;
      var a = pt(deg, r + 6), b = pt(deg, r + (maj ? 13 : 9));
      ticks += '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) +
        '" y2="' + b[1].toFixed(1) + '" stroke="' + (maj ? "var(--amber)" : "var(--dim)") + '" stroke-width="' + (maj ? 2 : 1) + '"/>';
    }
    var frac = cur === null ? 0 : (cur - lo) / (hi - lo);
    var marker = "";
    if (!isNaN(avg)) {
      var mf = Math.max(0, Math.min(1, (avg - lo) / (hi - lo)));
      var m0 = pt(st + sweep * mf, r - 9), m1 = pt(st + sweep * mf, r + 3);
      marker = '<line x1="' + m0[0].toFixed(1) + '" y1="' + m0[1].toFixed(1) + '" x2="' + m1[0].toFixed(1) +
        '" y2="' + m1[1].toFixed(1) + '" stroke="var(--hi)" stroke-width="2.5"/>';
    }
    var dcls = isNaN(delta) ? "" : (delta > 0 ? "up" : "down");
    var dtxt = isNaN(delta) ? "—" : (delta > 0 ? "▲" : "▼") + " " + Math.abs(delta).toFixed(1) + " vs 1st half";
    // Resting HR is the one dial where lower is better.
    if (k === "resting_hr" && !isNaN(delta)) dcls = delta < 0 ? "up" : "down";

    return '<div class="gauge" data-metric="' + k + '" role="button" tabindex="0" title="Plot ' + esc(label(k)) + '">' +
      svg("0 0 140 140", ticks +
        '<path d="' + arc(1, r) + '" fill="none" stroke="var(--dim)" stroke-width="7" stroke-linecap="round" opacity=".35"/>' +
        '<path d="' + arc(frac, r) + '" fill="none" stroke="var(--amber)" stroke-width="7" stroke-linecap="round"/>' +
        marker +
        '<text x="70" y="68" class="g-val">' + (cur === null ? "—" : (Math.abs(cur) >= 10 ? cur.toFixed(0) : cur.toFixed(1))) + "</text>" +
        '<text x="70" y="86" class="g-unit">' + u + "</text>", "lit") +
      '<div class="g-label">' + name + '</div><div class="g-delta ' + dcls + '">' + dtxt + "</div></div>";
  }

  function renderVitals() {
    $("vitals").innerHTML = DIALS.filter(function (d) { return M[d[0]]; })
      .map(function (d) { return dial(d[0], d[1], d[2], d[3], d[4]); }).join("");
    Array.prototype.forEach.call($("vitals").querySelectorAll(".gauge"), function (g) {
      function go() { S.primary = g.dataset.metric; S.overlay = []; drawAll(); }
      g.addEventListener("click", go);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
  }

  // ------------------------------------------------------------- explorer
  var SERIES_COLOURS = ["var(--amber)", "var(--hi)", "var(--hot)"];

  function renderExplorer() {
    var w = 1000, h = 210;
    var keys = [S.primary].concat(S.overlay);
    var grid = [0, .25, .5, .75, 1].map(function (f) {
      return '<line x1="0" y1="' + (h * f) + '" x2="' + w + '" y2="' + (h * f) + '" stroke="var(--dim)" stroke-width=".5" opacity=".3"/>';
    }).join("");

    var months = "", seen = {}, lastX = -999;
    for (var i = S.i0; i <= S.i1; i++) {
      var key = DATES[i].slice(0, 7);
      if (seen[key]) continue; seen[key] = 1;
      var x = ((i - S.i0) / Math.max(1, S.i1 - S.i0)) * w;
      months += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="' + h + '" stroke="var(--dim)" stroke-width=".5" opacity=".45"/>';
      if (x - lastX < 58) continue;
      lastX = x;
      months += '<text x="' + (x + 4) + '" y="' + (h + 13) + '" class="ax">' + DATES[i].slice(2, 7) + "</text>";
    }

    var paths = "", legend = "";
    keys.forEach(function (k, idx) {
      var raw = slice(k), e = extent(raw), roll = rolling(raw, S.roll);
      var col = SERIES_COLOURS[idx];
      if (idx === 0) {
        paths += '<path d="' + path(raw, w, h, e[0], e[1]) + '" fill="none" stroke="' + col + '" stroke-width="1" opacity=".22"/>';
      }
      paths += '<path d="' + path(roll, w, h, e[0], e[1]) + '" fill="none" stroke="' + col +
        '" stroke-width="' + (idx === 0 ? 2.6 : 1.8) + '"' + (idx === 0 ? ' class="lit"' : "") + "/>";
      legend += '<span><i style="background:' + col + '"></i>' + esc(label(k)) +
        " <b style='color:var(--dim)'>" + fmt(e[0], 1) + "–" + fmt(e[1], 1) + " " + esc(unit(k)) + "</b></span>";
    });

    $("expLegend").innerHTML = legend + '<span class="readout" id="expRead">HOVER FOR DAILY VALUES</span>';
    $("expChart").innerHTML = svg("-6 -10 1020 245",
      grid + months + paths +
      '<line id="expCross" x1="0" y1="0" x2="0" y2="' + h + '" stroke="var(--hi)" stroke-width="1" opacity="0"/>' +
      '<rect id="expHit" x="0" y="0" width="' + w + '" height="' + h + '" fill="transparent" style="cursor:crosshair"/>');

    var hit = $("expHit"), cross = $("expCross"), read = $("expRead");
    function at(ev) {
      var box = hit.getBoundingClientRect();
      var f = (ev.clientX - box.left) / box.width;
      return Math.max(S.i0, Math.min(S.i1, S.i0 + Math.round(f * (S.i1 - S.i0))));
    }
    hit.addEventListener("mousemove", function (ev) {
      var i = at(ev), x = ((i - S.i0) / Math.max(1, S.i1 - S.i0)) * w;
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("opacity", ".8");
      read.textContent = DATES[i] + "  ·  " + keys.map(function (k) {
        return label(k).toUpperCase() + " " + fmt(M[k][i], 1);
      }).join("  ·  ");
    });
    hit.addEventListener("mouseleave", function () {
      cross.setAttribute("opacity", "0"); read.textContent = "HOVER FOR DAILY VALUES";
    });
    hit.addEventListener("click", function (ev) { S.day = at(ev); renderDay(); renderCal(); });
  }

  function renderMetricList() {
    var box = $("mlist");
    box.innerHTML = "";
    KEYS.forEach(function (k) {
      var b = el("button");
      b.textContent = label(k);
      b.title = k + "  ·  " + unit(k);
      var slot = k === S.primary ? 0 : (S.overlay.indexOf(k) >= 0 ? S.overlay.indexOf(k) + 1 : -1);
      b.setAttribute("aria-pressed", slot >= 0 ? "true" : "false");
      if (slot > 0) b.dataset.slot = String(slot);
      b.addEventListener("click", function (ev) {
        if (ev.shiftKey && k !== S.primary) {
          var at = S.overlay.indexOf(k);
          if (at >= 0) S.overlay.splice(at, 1);
          else if (S.overlay.length < 2) S.overlay.push(k);
        } else { S.primary = k; S.overlay = S.overlay.filter(function (o) { return o !== k; }); }
        drawAll();
      });
      box.appendChild(b);
    });
  }

  // ------------------------------------------------------ correlation lab
  function renderCorr() {
    var pr = pairAt(S.driver, S.outcome, S.lag);
    var xs = pr[0], ys = pr[1];
    var r = pearson(xs, ys), ne = xs.length >= 4 ? effectiveN(xs, ys) : NaN;
    var ci = fisherCI(r, ne);

    $("corrStats").innerHTML = xs.length < 4
      ? '<span class="empty">Not enough overlapping days in this range.</span>'
      : ["r <b>" + (r >= 0 ? "+" : "") + fmt(r, 3) + "</b>",
      "95% CI <b>" + fmt(ci[0], 2) + " … " + fmt(ci[1], 2) + "</b>",
      "n <b>" + xs.length + "</b>",
      "eff. n <b>" + fmt(ne, 0) + "</b>"].join("&nbsp;&nbsp;·&nbsp;&nbsp;");

    // lag profile: r at every lag from -7 to +7
    var w = 420, h = 130, lags = [], k;
    for (k = -7; k <= 7; k++) {
      var p = pairAt(S.driver, S.outcome, k);
      lags.push({ lag: k, r: p[0].length >= 4 ? pearson(p[0], p[1]) : NaN });
    }
    var maxAbs = Math.max(0.25, Math.max.apply(null, lags.map(function (d) { return isNaN(d.r) ? 0 : Math.abs(d.r); })));
    var bw = w / lags.length;
    var bars = lags.map(function (d, i) {
      if (isNaN(d.r)) return "";
      var hh = (Math.abs(d.r) / maxAbs) * (h / 2 - 6);
      var y = d.r >= 0 ? h / 2 - hh : h / 2;
      var on = d.lag === S.lag;
      return '<rect x="' + (i * bw + 2).toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + (bw - 4).toFixed(1) +
        '" height="' + Math.max(1, hh).toFixed(1) + '" fill="' + (d.r < 0 ? "var(--hot)" : "var(--amber)") +
        '" opacity="' + (on ? 1 : .45) + '"' + (on ? ' class="lit"' : "") + "/>" +
        '<text x="' + (i * bw + bw / 2).toFixed(1) + '" y="' + (h + 12) + '" class="ax c">' + d.lag + "</text>";
    }).join("");
    $("lagChart").innerHTML = svg("-48 -6 500 155",
      '<line x1="0" y1="' + h / 2 + '" x2="' + w + '" y2="' + h / 2 + '" stroke="var(--dim)" stroke-width=".75"/>' +
      '<text x="-6" y="8" class="ax l">+' + maxAbs.toFixed(2) + "</text>" +
      '<text x="-6" y="' + (h + 2) + '" class="ax l">−' + maxAbs.toFixed(2) + "</text>" +
      bars);

    // scatter with fit
    var sw = 420, sh = 210;
    if (xs.length >= 4) {
      var ex = extent(xs), ey = extent(ys), fit = linfit(xs, ys);
      var pts = xs.map(function (x, i) {
        var px = ((x - ex[0]) / (ex[1] - ex[0])) * sw;
        var py = sh - ((ys[i] - ey[0]) / (ey[1] - ey[0])) * sh;
        return '<circle cx="' + px.toFixed(1) + '" cy="' + py.toFixed(1) + '" r="2.6" fill="var(--amber)" opacity=".55"/>';
      }).join("");
      var y0 = fit.intercept + fit.slope * ex[0], y1 = fit.intercept + fit.slope * ex[1];
      var ly0 = sh - ((y0 - ey[0]) / (ey[1] - ey[0])) * sh, ly1 = sh - ((y1 - ey[0]) / (ey[1] - ey[0])) * sh;
      $("scatter").innerHTML = svg("-34 -10 470 250",
        '<rect x="0" y="0" width="' + sw + '" height="' + sh + '" fill="rgba(0,0,0,.25)" stroke="var(--hair)"/>' +
        pts +
        '<line x1="0" y1="' + ly0.toFixed(1) + '" x2="' + sw + '" y2="' + ly1.toFixed(1) +
        '" stroke="var(--hi)" stroke-width="1.8" class="lit"/>' +
        '<text x="-6" y="8" class="ax l">' + fmt(ey[1], 0) + "</text>" +
        '<text x="-6" y="' + sh + '" class="ax l">' + fmt(ey[0], 0) + "</text>" +
        '<text x="0" y="' + (sh + 14) + '" class="ax">' + fmt(ex[0], 0) + "</text>" +
        '<text x="' + sw + '" y="' + (sh + 14) + '" class="ax l">' + fmt(ex[1], 0) + "</text>");
      $("scatterAx").innerHTML = "X " + esc(label(S.driver)) + " (" + esc(unit(S.driver)) + ")  ·  Y " +
        esc(label(S.outcome)) + " (" + esc(unit(S.outcome)) + ")  ·  LAG " + S.lag;
    } else {
      $("scatter").innerHTML = '<div class="empty">Not enough paired days.</div>';
      $("scatterAx").textContent = "";
    }
  }

  // ------------------------------------------------------------- calendar
  function renderCal() {
    var vals = M[S.cal], e = extent(vals), box = $("cal");
    box.innerHTML = "";

    var axis = el("div", "wk dow");
    ["", "M", "", "W", "", "F", ""].forEach(function (t) {
      var c = el("div", "dl"); c.textContent = t; axis.appendChild(c);
    });
    box.appendChild(axis);

    var wk = el("div", "wk"), dow = new Date(DATES[0] + "T00:00:00").getDay(), i;
    for (i = 0; i < dow; i++) wk.appendChild(el("div", "d"));
    DATES.forEach(function (d, idx) {
      var cell = el("div", "d");
      var v = vals[idx];
      cell.style.background = (v === null || isNaN(v)) ? "transparent" : ramp((v - e[0]) / ((e[1] - e[0]) || 1));
      if (idx === S.day) cell.classList.add("sel");
      if (idx < S.i0 || idx > S.i1) cell.style.opacity = ".28";
      cell.title = d + "  ·  " + label(S.cal) + " " + fmt(v, 1) + " " + unit(S.cal);
      cell.addEventListener("click", function () { S.day = idx; renderDay(); renderCal(); });
      wk.appendChild(cell);
      if (new Date(d + "T00:00:00").getDay() === 6) { box.appendChild(wk); wk = el("div", "wk"); }
      if (idx === DATES.length - 1) box.appendChild(wk);
    });
    $("calScale").innerHTML = 'LOW ' + [0, .25, .5, .75, 1].map(function (t) {
      return '<i style="background:' + ramp(t) + '"></i>';
    }).join("") + " HIGH  ·  " + esc(label(S.cal)) + " " + fmt(e[0], 0) + "–" + fmt(e[1], 0);
  }

  // ----------------------------------------------------------- day detail
  var DAY_ROWS = ["recovery_score", "hrv_rmssd_milli", "resting_hr", "spo2_percentage",
    "skin_temp_celsius", "strain", "sleep_hours", "sleep_performance", "rem_hours",
    "sws_hours", "sleep_debt_hours", "respiratory_rate", "bedtime_hr", "waketime_hr"];

  function clock(v) {
    if (v === null || isNaN(v)) return "—";
    var t = ((v % 24) + 24) % 24, hh = Math.floor(t), mm = Math.round((t - hh) * 60);
    if (mm === 60) { mm = 0; hh = (hh + 1) % 24; }
    return (hh < 10 ? "0" : "") + hh + ":" + (mm < 10 ? "0" : "") + mm;
  }

  function renderDay() {
    var i = S.day, out = "";
    DAY_ROWS.filter(function (k) { return M[k]; }).forEach(function (k) {
      var v = M[k][i];
      var shown = (k === "bedtime_hr" || k === "waketime_hr") ? clock(v) : fmt(v, 1) + " " + unit(k);
      out += "<dt>" + esc(label(k)) + "</dt><dd>" + shown + "</dd>";
    });
    $("dayTitle").textContent = DATES[i];
    $("dayBody").innerHTML = '<dl class="kv">' + out + "</dl>";
    var todays = (D.workouts || []).filter(function (w) { return w.date === DATES[i]; });
    $("dayWo").innerHTML = todays.length
      ? '<div class="wo">' + todays.map(function (w) {
        return '<div class="wo-row"><span>' + esc(w.sport || "activity") + "</span><span>" +
          (w.strain === null ? "—" : "strain " + w.strain.toFixed(1)) + "  ·  " + Math.round(w.minutes) + " min</span></div>";
      }).join("") + "</div>"
      : '<div class="wo"><div class="empty">No workouts logged</div></div>';
  }

  // --------------------------------------------------------------- sports
  function renderSports() {
    var from = DATES[S.i0], to = DATES[S.i1], agg = {};
    (D.workouts || []).forEach(function (w) {
      if (w.date < from || w.date > to) return;
      var a = agg[w.sport || "activity"] || (agg[w.sport || "activity"] = { n: 0, strain: 0, min: 0 });
      a.n++; a.strain += w.strain || 0; a.min += w.minutes || 0;
    });
    var rows = Object.keys(agg).map(function (k) { return [k, agg[k]]; })
      .sort(function (a, b) { return b[1].min - a[1].min; });
    var maxMin = rows.length ? rows[0][1].min : 1;
    $("sports").innerHTML = rows.length ? rows.map(function (r) {
      var cells = "", lit = Math.round((r[1].min / maxMin) * 8), j;
      for (j = 0; j < 8; j++) {
        cells += '<span class="cell" style="background:' + (j < lit ? "var(--amber)" : "transparent") +
          ';border-color:' + (j < lit ? "var(--amber)" : "var(--dim)") + ';opacity:' + (j < lit ? 1 : .45) + '"></span>';
      }
      return "<tr><td>" + esc(r[0]) + '</td><td class="num">' + r[1].n + '</td><td><span class="cells">' + cells +
        '</span></td><td class="num">' + Math.round(r[1].min) + '</td><td class="num">' + r[1].strain.toFixed(1) + "</td></tr>";
    }).join("") : '<tr><td colspan="5" class="empty">No workouts in this range</td></tr>';
  }

  // ------------------------------------------------------------- controls
  function fillSelect(sel, chosen) {
    sel.innerHTML = KEYS.map(function (k) {
      return '<option value="' + k + '"' + (k === chosen ? " selected" : "") + ">" + esc(label(k)) + "</option>";
    }).join("");
  }

  function drawAll() {
    renderRange(); renderVitals(); renderExplorer(); renderMetricList();
    renderCorr(); renderCal(); renderDay(); renderSports();
    Array.prototype.forEach.call(document.querySelectorAll("[data-preset]"), function (b) {
      var d = Number(b.dataset.preset);
      b.setAttribute("aria-pressed", String((d === 0 ? S.i0 === 0 : S.i1 - S.i0 + 1 === d) && S.i1 === N - 1));
    });
  }

  function setRangeDays(d) {
    S.i1 = N - 1;
    S.i0 = d === 0 ? 0 : Math.max(0, N - d);
    if (S.day < S.i0) S.day = S.i1;
    drawAll();
  }

  function init() {
    fillSelect($("driverSel"), S.driver);
    fillSelect($("outcomeSel"), S.outcome);
    fillSelect($("calSel"), S.cal);

    $("driverSel").addEventListener("change", function () { S.driver = this.value; renderCorr(); });
    $("outcomeSel").addEventListener("change", function () { S.outcome = this.value; renderCorr(); });
    $("calSel").addEventListener("change", function () { S.cal = this.value; renderCal(); });
    $("lag").addEventListener("input", function () {
      S.lag = Number(this.value); $("lagVal").textContent = (S.lag > 0 ? "+" : "") + S.lag; renderCorr();
    });
    $("roll").addEventListener("input", function () {
      S.roll = Number(this.value); $("rollVal").textContent = S.roll + "d"; renderExplorer();
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-preset]"), function (b) {
      b.addEventListener("click", function () { setRangeDays(Number(b.dataset.preset)); });
    });
    $("swap").addEventListener("click", function () {
      var t = S.driver; S.driver = S.outcome; S.outcome = t;
      $("driverSel").value = S.driver; $("outcomeSel").value = S.outcome; renderCorr();
    });

    document.addEventListener("keydown", function (e) {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowLeft") { S.day = Math.max(0, S.day - 1); renderDay(); renderCal(); }
      else if (e.key === "ArrowRight") { S.day = Math.min(N - 1, S.day + 1); renderDay(); renderCal(); }
      else if (e.key === "r" || e.key === "R") setRangeDays(0);
      else return;
      e.preventDefault();
    });

    drawAll();

    var boot = $("boot");
    if (boot) setTimeout(function () { boot.classList.add("done"); }, 700);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

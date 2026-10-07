/* charts.js — tiny offline canvas line-chart module for the Command Center.
   Replaces the CDN Chart.js: zero network, draws multi-series line charts on a
   <canvas> from {labels, datasets:[{label,data,color,width,points}]}.
   Kept deliberately small — the dock's other instruments are SVG in pitwall.js. */
(function () {
  "use strict";
  const MONO = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  const dpr = () => Math.max(1, Math.min(window.devicePixelRatio || 1, 2));

  function fmtTick(v) {
    const a = Math.abs(v);
    if (a >= 1000) return (v / 1000).toFixed(1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(1);
    return v.toFixed(2);
  }

  function MiniChart(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
  }
  MiniChart.prototype.destroy = function () {
    if (this.ctx) this.ctx.clearRect(0, 0, this.c.width, this.c.height);
  };
  MiniChart.prototype.render = function (labels, datasets) {
    const canvas = this.c, ctx = this.ctx;
    if (!ctx) return;
    const box = canvas.parentElement || canvas;
    const w = Math.max(120, canvas.clientWidth || box.clientWidth || 320);
    const h = Math.max(80, canvas.clientHeight || box.clientHeight || 180);
    const k = dpr();
    canvas.width = Math.floor(w * k);
    canvas.height = Math.floor(h * k);
    ctx.setTransform(k, 0, 0, k, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pl = 40, pr = 10, pt = 12, pb = 20;
    let mn = Infinity, mx = -Infinity;
    datasets.forEach(d => d.data.forEach(v => {
      if (v != null && isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
    }));
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    if (mn === mx) { mn -= 1; mx += 1; }
    const n = labels.length;
    const X = i => pl + (n <= 1 ? 0 : i / (n - 1)) * (w - pl - pr);
    const Y = v => pt + (1 - (v - mn) / (mx - mn)) * (h - pt - pb);

    // grid + y ticks
    ctx.font = MONO;
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const val = mn + (mx - mn) * g / 4, y = Y(val);
      ctx.strokeStyle = "rgba(140,160,180,.09)";
      ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(w - pr, y); ctx.stroke();
      ctx.fillStyle = "#6b7885"; ctx.textAlign = "right";
      ctx.fillText(fmtTick(val), pl - 5, y);
    }
    // x ticks
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = "#6b7885";
    const xt = Math.min(6, n);
    for (let t = 0; t < xt; t++) {
      const i = Math.round((n - 1) * t / (xt - 1 || 1));
      ctx.fillText(String(labels[i] ?? ""), X(i), h - pb + 4);
    }
    // series
    datasets.forEach(d => {
      ctx.strokeStyle = d.color || "#8aa"; ctx.lineWidth = d.width || 1.3;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let started = false;
      d.data.forEach((v, i) => {
        if (v == null || !isFinite(v)) { started = false; return; }
        const x = X(i), y = Y(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      if (d.points) {
        ctx.fillStyle = d.color || "#8aa";
        d.data.forEach((v, i) => {
          if (v != null && isFinite(v)) { ctx.beginPath(); ctx.arc(X(i), Y(v), 2.2, 0, 7); ctx.fill(); }
        });
      }
      // endpoint dot
      for (let i = d.data.length - 1; i >= 0; i--) {
        const v = d.data[i];
        if (v != null && isFinite(v)) {
          ctx.fillStyle = d.color || "#8aa";
          ctx.beginPath(); ctx.arc(X(i), Y(v), 2.6, 0, 7); ctx.fill();
          break;
        }
      }
    });
  };

  window.MiniChart = MiniChart;
})();

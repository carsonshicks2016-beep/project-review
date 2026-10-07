// Telemetry plot for the training run.
//
// Three series share one canvas because they answer one question together: "is it
// learning?". Fitness (best + population average) is on the left axis; gates cleared is
// on the right axis, because it is the number a human actually reads as progress and it
// lives on a completely different scale.
const COLORS = {
  best: '#00e5ff',
  avg: '#ff007f',
  gates: '#ffb020',
  axis: 'rgba(255,255,255,0.28)',
  grid: 'rgba(255,255,255,0.07)',
  label: 'rgba(220,225,240,0.65)'
};

const PAD = { top: 14, right: 46, bottom: 22, left: 52 };

function formatFitness(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'k';
  return v.toFixed(0);
}

export class FitnessGraph {
  constructor(canvasId, totalGates = 16) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.data = []; // { gen, best, avg, gates }
    this.totalGates = totalGates;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    // Back the canvas with real device pixels; without this the lines are soft on any
    // HiDPI display, since the CSS size and the buffer size disagree.
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
    this.draw();
  }

  setTotalGates(n) {
    this.totalGates = Math.max(1, n);
    this.draw();
  }

  addDataPoint(gen, best, avg, gates = 0) {
    this.data.push({ gen, best, avg, gates });
    this.draw();
  }

  reset() {
    this.data = [];
    this.draw();
  }

  draw() {
    const { ctx, data } = this;
    const w = this.w;
    const h = this.h;
    ctx.clearRect(0, 0, w, h);

    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;
    if (plotW <= 10 || plotH <= 10) return;

    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'middle';

    if (data.length === 0) {
      ctx.fillStyle = COLORS.label;
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for the first generation…', w / 2, h / 2);
      return;
    }

    // ── Scales ──
    const maxGen = Math.max(10, data[data.length - 1].gen);
    let hi = 0;
    let lo = 0;
    for (const d of data) {
      hi = Math.max(hi, d.best, d.avg);
      lo = Math.min(lo, d.best, d.avg);
    }
    if (hi === lo) hi = lo + 1;
    hi += (hi - lo) * 0.12; // headroom so the newest point is not on the frame

    const xMap = (gen) => PAD.left + (gen / maxGen) * plotW;
    const yMap = (v) => PAD.top + plotH - ((v - lo) / (hi - lo)) * plotH;
    const yGates = (g) => PAD.top + plotH - (g / this.totalGates) * plotH;

    // ── Grid + left axis labels ──
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.label;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    const TICKS = 4;
    for (let i = 0; i <= TICKS; i++) {
      const v = lo + ((hi - lo) * i) / TICKS;
      const y = Math.round(yMap(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
      ctx.fillText(formatFitness(v), PAD.left - 6, y);
    }

    // ── Right axis: gates cleared ──
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(255,176,32,0.7)';
    for (let g = 0; g <= this.totalGates; g += Math.max(1, Math.round(this.totalGates / 4))) {
      ctx.fillText(String(g), PAD.left + plotW + 6, yGates(g));
    }

    // ── X axis labels ──
    ctx.fillStyle = COLORS.label;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 4; i++) {
      const g = Math.round((maxGen * i) / 4);
      ctx.fillText(String(g), xMap(g), PAD.top + plotH + 6);
    }
    ctx.textBaseline = 'middle';

    // ── Frame ──
    ctx.strokeStyle = COLORS.axis;
    ctx.beginPath();
    ctx.moveTo(PAD.left + 0.5, PAD.top);
    ctx.lineTo(PAD.left + 0.5, PAD.top + plotH + 0.5);
    ctx.lineTo(PAD.left + plotW, PAD.top + plotH + 0.5);
    ctx.stroke();

    const plot = (key, color, map, width, dashed) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed || []);
      data.forEach((d, i) => {
        const x = xMap(d.gen);
        const y = map(d[key]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // ── Gates cleared: a step line, since it only moves in whole gates ──
    ctx.beginPath();
    ctx.strokeStyle = COLORS.gates;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    data.forEach((d, i) => {
      const x = xMap(d.gen);
      const y = yGates(d.gates);
      if (i === 0) ctx.moveTo(x, y);
      else {
        ctx.lineTo(x, yGates(data[i - 1].gates));
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Best fitness, with a fill so it separates from the average at a glance ──
    const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + plotH);
    grad.addColorStop(0, 'rgba(0,229,255,0.22)');
    grad.addColorStop(1, 'rgba(0,229,255,0)');
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = xMap(d.gen);
      const y = yMap(d.best);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(xMap(data[data.length - 1].gen), PAD.top + plotH);
    ctx.lineTo(xMap(data[0].gen), PAD.top + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    plot('avg', COLORS.avg, yMap, 1.5);
    plot('best', COLORS.best, yMap, 2);

    // ── Latest values, pinned at the right edge of each line ──
    const last = data[data.length - 1];
    const lastX = xMap(last.gen);
    const dot = (y, color) => {
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(lastX, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    };
    dot(yMap(last.best), COLORS.best);
    dot(yMap(last.avg), COLORS.avg);
    dot(yGates(last.gates), COLORS.gates);

    // ── Legend ──
    ctx.textAlign = 'left';
    const entries = [
      ['Best', COLORS.best, formatFitness(last.best)],
      ['Avg', COLORS.avg, formatFitness(last.avg)],
      ['Gates', COLORS.gates, `${last.gates}/${this.totalGates}`]
    ];
    let lx = PAD.left + 6;
    for (const [label, color, value] of entries) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, PAD.top + 3, 8, 2);
      lx += 12;
      ctx.fillStyle = COLORS.label;
      const text = `${label} ${value}`;
      ctx.fillText(text, lx, PAD.top + 4);
      lx += ctx.measureText(text).width + 14;
    }
  }
}

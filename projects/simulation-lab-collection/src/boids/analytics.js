/**
 * ApexFlock - Ecological Telemetry & Lotka-Volterra Phase-Space Analytics
 * High-precision mathematical ecological dashboard:
 * - Lotka-Volterra phase-plane attractor with theoretical vector flow field
 * - Demographic time-series curves with soft organic area fills
 * - Ecological stability metrics and equilibrium nullclines
 */

export class AnalyticsDashboard {
  constructor(phaseCanvas, historyCanvas) {
    this.phaseCanvas = phaseCanvas;
    this.phaseCtx = phaseCanvas ? phaseCanvas.getContext('2d') : null;
    this.historyCanvas = historyCanvas;
    this.historyCtx = historyCanvas ? historyCanvas.getContext('2d') : null;
    this.pulse = 0;
  }

  render(ecosystem) {
    this.pulse += 0.05;
    if (this.phaseCtx) this._renderPhasePlane(ecosystem);
    if (this.historyCtx) this._renderHistoryChart(ecosystem);
  }

  _renderPhasePlane(ecosystem) {
    const ctx = this.phaseCtx;
    const w = this.phaseCanvas.width;
    const h = this.phaseCanvas.height;

    // Dark gallery background
    ctx.fillStyle = 'rgba(8, 12, 18, 0.96)';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 36;
    const padBottom = 26;
    const padTop = 18;
    const padRight = 14;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const maxPrey = 280;
    const maxPred = 30;

    // 1. LOTKA-VOLTERRA THEORETICAL VECTOR FLOW FIELD
    // dx/dt = alpha*x - beta*x*y, dy/dt = delta*x*y - gamma*y
    const alpha = 0.8, beta = 0.08, gamma = 0.5, delta = 0.005;
    const nullclineX = gamma / delta; // ~ 100 prey
    const nullclineY = alpha / beta;  // ~ 10 pred

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;

    for (let gx = 40; gx < maxPrey; gx += 45) {
      for (let gy = 4; gy < maxPred; gy += 5) {
        const px = padLeft + (gx / maxPrey) * plotW;
        const py = padTop + plotH - (gy / maxPred) * plotH;

        const vx = alpha * gx - beta * gx * gy;
        const vy = delta * gx * gy - gamma * gy;
        const len = Math.hypot(vx, vy);
        if (len > 1e-4) {
          const arrowLen = 5.0;
          const dx = (vx / len) * arrowLen;
          const dy = -(vy / len) * arrowLen; // invert Y for screen space

          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px + dx, py + dy);
          ctx.stroke();
        }
      }
    }

    // 2. EQUILIBRIUM NULLCLINES (Subtle dashed guides)
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(217, 180, 110, 0.2)';
    // Predator nullcline (vertical line)
    const nPx = padLeft + (nullclineX / maxPrey) * plotW;
    ctx.beginPath();
    ctx.moveTo(nPx, padTop);
    ctx.lineTo(nPx, padTop + plotH);
    ctx.stroke();

    // Prey nullcline (horizontal line)
    const nPy = padTop + plotH - (nullclineY / maxPred) * plotH;
    ctx.beginPath();
    ctx.moveTo(padLeft, nPy);
    ctx.lineTo(padLeft + plotW, nPy);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // 3. AXES & TICKS
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    ctx.fillStyle = 'rgba(160, 175, 195, 0.7)';
    ctx.font = '8px "Space Grotesk", sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText("PRED", padLeft - 4, padTop + 8);
    ctx.textAlign = 'center';
    ctx.fillText("PREY DENSITY", padLeft + plotW / 2, h - 6);

    const history = ecosystem.history;
    if (history.length < 2) return;

    // 4. HISTORICAL LIMIT CYCLE TRAJECTORY
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const pt = history[i];
      const px = padLeft + Math.min(1, pt.preyCount / maxPrey) * plotW;
      const py = padTop + plotH - Math.min(1, pt.predCount / maxPred) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = 'rgba(217, 180, 110, 0.35)';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Trajectory head glow (recent points)
    const tailLen = Math.min(20, history.length);
    for (let i = history.length - tailLen; i < history.length - 1; i++) {
      const pt1 = history[i];
      const pt2 = history[i + 1];
      const x1 = padLeft + Math.min(1, pt1.preyCount / maxPrey) * plotW;
      const y1 = padTop + plotH - Math.min(1, pt1.predCount / maxPred) * plotH;
      const x2 = padLeft + Math.min(1, pt2.preyCount / maxPrey) * plotW;
      const y2 = padTop + plotH - Math.min(1, pt2.predCount / maxPred) * plotH;

      const alphaProgress = (i - (history.length - tailLen)) / tailLen;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(210, 88, 104, ${alphaProgress * 0.85})`;
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }

    // Current ecological coordinate point
    const cur = history[history.length - 1];
    const curX = padLeft + Math.min(1, cur.preyCount / maxPrey) * plotW;
    const curY = padTop + plotH - Math.min(1, cur.predCount / maxPred) * plotH;

    // Pulsing ripple ring
    const rippleR = 3.5 + Math.sin(this.pulse) * 1.5;
    ctx.beginPath();
    ctx.arc(curX, curY, rippleR, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(210, 88, 104, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(curX, curY, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#d25868';
    ctx.fill();
  }

  _renderHistoryChart(ecosystem) {
    const ctx = this.historyCtx;
    const w = this.historyCanvas.width;
    const h = this.historyCanvas.height;

    ctx.fillStyle = 'rgba(8, 12, 18, 0.96)';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 28;
    const padBottom = 20;
    const padTop = 14;
    const padRight = 10;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const history = ecosystem.history;
    if (history.length < 2) return;

    // Horizontal grid line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop + plotH * 0.5);
    ctx.lineTo(padLeft + plotW, padTop + plotH * 0.5);
    ctx.stroke();

    const maxVal = 260;
    const stepX = plotW / Math.max(1, history.length - 1);

    // 1. Prey Series (Soft Celadon)
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = padLeft + i * stepX;
      const y = padTop + plotH - Math.min(1, history[i].preyCount / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#7ec7b0';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // 2. Food Series (Warm Pollen Amber)
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = padLeft + i * stepX;
      const y = padTop + plotH - Math.min(1, history[i].foodCount / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#e4bc73';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // 3. Predator Series (Carmine Rose, scaled x8)
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = padLeft + i * stepX;
      const y = padTop + plotH - Math.min(1, (history[i].predCount * 8.0) / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#d25868';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Legend
    ctx.font = '8.5px "Space Grotesk", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#7ec7b0';
    ctx.fillText(`Prey ${ecosystem.boids.length}`, padLeft + 4, padTop + 9);
    ctx.fillStyle = '#d25868';
    ctx.fillText(`Predators ${ecosystem.predators.length}`, padLeft + 72, padTop + 9);
    ctx.fillStyle = '#e4bc73';
    ctx.fillText(`Food ${ecosystem.foods.length}`, padLeft + 155, padTop + 9);
  }
}

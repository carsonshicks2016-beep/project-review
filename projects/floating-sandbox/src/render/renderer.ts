import { materialByIndex } from '../sim/materials';
import {
  BEAM_SKIN,
  FLAG_BROKEN,
  FLAG_FLOODED,
  FLAG_PINNED,
  type Ship,
  type TankBounds,
  type Vec2,
  type WaterState,
} from '../sim/types';
import type { Camera } from './camera';

const SKY_TOP = '#c5d6e8';
const SKY_HORIZON = '#e8eef4';
const SEABED = '#6b5a45';
const SEABED_DEEP = '#4a3d30';
const WATER_FILL = 'rgba(45, 110, 160, 0.55)';
const WATER_DEEP = 'rgba(28, 72, 110, 0.72)';
const WATER_SURFACE = 'rgba(210, 235, 248, 0.95)';
const BRUSH_STROKE = 'rgba(40, 48, 56, 0.55)';

/** Max water points drawn per frame (subsample beyond this). */
const WATER_DRAW_BUDGET = 14_000;
/** Columns used for surface silhouette. */
const SURFACE_BINS = 160;

export class SimRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private surfaceBins = new Float32Array(SURFACE_BINS);
  private surfaceHit = new Uint8Array(SURFACE_BINS);

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.canvas = canvas;
    this.ctx = ctx;
    this.resize();
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = this.canvas.clientWidth || this.canvas.parentElement?.clientWidth || 800;
    const cssH = this.canvas.clientHeight || this.canvas.parentElement?.clientHeight || 600;
    this.canvas.width = Math.max(1, Math.floor(cssW * dpr));
    this.canvas.height = Math.max(1, Math.floor(cssH * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  render(args: {
    camera: Camera;
    bounds: TankBounds;
    water: WaterState;
    ship: Ship | null;
    stressOverlay: boolean;
    toolRadius?: number;
    cursorWorld?: Vec2 | null;
  }): void {
    const { camera, bounds, water, ship, stressOverlay } = args;
    const ctx = this.ctx;
    const cssW = this.canvas.clientWidth || this.canvas.width;
    const cssH = this.canvas.clientHeight || this.canvas.height;
    const dpr = this.canvas.width / Math.max(1, cssW);

    // CSS-pixel coordinate system (device pixels via dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.drawSky(cssW, cssH);
    this.drawSeabed(camera, bounds, cssW, cssH);

    // World transform for sim content
    ctx.save();
    ctx.translate(cssW * 0.5, cssH * 0.5);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-camera.x, -camera.y);

    this.drawTankFrame(bounds, camera.zoom);
    this.drawWater(water, bounds, camera.zoom);
    if (ship) {
      this.drawShip(ship, stressOverlay, camera.zoom);
    }

    if (args.cursorWorld && args.toolRadius != null && args.toolRadius > 0) {
      this.drawBrush(args.cursorWorld, args.toolRadius, camera.zoom);
    }

    ctx.restore();
  }

  private drawSky(w: number, h: number): void {
    const g = this.ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, SKY_TOP);
    g.addColorStop(0.55, SKY_HORIZON);
    g.addColorStop(1, '#d4cfc4');
    this.ctx.fillStyle = g;
    this.ctx.fillRect(0, 0, w, h);
  }

  private drawSeabed(
    camera: Camera,
    bounds: TankBounds,
    cssW: number,
    cssH: number,
  ): void {
    const ctx = this.ctx;
    const topLeft = camera.screenToWorld(0, 0, this.canvas);
    const bottomRight = camera.screenToWorld(cssW, cssH, this.canvas);

    // Soft underwater wash below mid-tank waterline
    const waterLineY = bounds.y0 + (bounds.y1 - bounds.y0) * 0.35;
    const sy0 = (waterLineY - camera.y) * camera.zoom + cssH * 0.5;
    if (sy0 < cssH) {
      const ug = ctx.createLinearGradient(0, Math.max(0, sy0), 0, cssH);
      ug.addColorStop(0, 'rgba(120, 160, 185, 0.18)');
      ug.addColorStop(1, 'rgba(70, 100, 120, 0.28)');
      ctx.fillStyle = ug;
      ctx.fillRect(0, Math.max(0, sy0), cssW, cssH - Math.max(0, sy0));
    }

    // Seabed strip at tank floor
    const sx0 = (bounds.x0 - camera.x) * camera.zoom + cssW * 0.5;
    const sx1 = (bounds.x1 - camera.x) * camera.zoom + cssW * 0.5;
    const sy = (bounds.y1 - camera.y) * camera.zoom + cssH * 0.5;
    const depth = Math.max(24, (bottomRight.y - topLeft.y) * camera.zoom * 0.08);

    if (sy < cssH + depth) {
      const bed = ctx.createLinearGradient(0, sy, 0, sy + depth);
      bed.addColorStop(0, SEABED);
      bed.addColorStop(1, SEABED_DEEP);
      ctx.fillStyle = bed;
      ctx.fillRect(sx0, sy, sx1 - sx0, depth + cssH);

      ctx.strokeStyle = 'rgba(90, 72, 52, 0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const step = Math.max(18, (sx1 - sx0) / 24);
      for (let x = sx0; x < sx1; x += step) {
        const amp = 3 + ((x * 0.07) % 4);
        ctx.moveTo(x, sy + 6);
        ctx.quadraticCurveTo(x + step * 0.5, sy + 6 + amp, x + step, sy + 8);
      }
      ctx.stroke();
    }
  }

  private drawTankFrame(bounds: TankBounds, zoom: number): void {
    const ctx = this.ctx;
    ctx.strokeStyle = 'rgba(50, 56, 62, 0.45)';
    ctx.lineWidth = 2 / Math.max(zoom, 1);
    ctx.strokeRect(bounds.x0, bounds.y0, bounds.x1 - bounds.x0, bounds.y1 - bounds.y0);
  }

  private drawWater(water: WaterState, bounds: TankBounds, zoom: number): void {
    const { count, x, y, spacing } = water;
    if (count <= 0) return;

    const ctx = this.ctx;
    const step = Math.max(1, Math.ceil(count / WATER_DRAW_BUDGET));
    const r = Math.max(0.55, spacing * 0.72);
    const screenR = r * zoom;
    const y0 = bounds.y0;
    const ySpan = Math.max(1e-6, bounds.y1 - bounds.y0);
    const binW = Math.max(1e-6, (bounds.x1 - bounds.x0) / SURFACE_BINS);

    // Surface = uppermost water (min y; y increases downward)
    this.surfaceBins.fill(Infinity);
    this.surfaceHit.fill(0);

    if (screenR < 2.2) {
      const alpha = Math.min(0.85, 0.35 + 0.5 / Math.sqrt(step));
      for (let i = 0; i < count; i += step) {
        const px = x[i]!;
        const py = y[i]!;
        const depth = (py - y0) / ySpan;
        ctx.fillStyle =
          depth > 0.55
            ? `rgba(28, 72, 110, ${alpha})`
            : `rgba(45, 110, 160, ${alpha * 0.9})`;
        ctx.fillRect(px - r * 0.55, py - r * 0.55, r * 1.1, r * 1.1);
        this.noteSurface(px, py, bounds.x0, binW);
      }
    } else {
      for (let i = 0; i < count; i += step) {
        const px = x[i]!;
        const py = y[i]!;
        const depth = (py - y0) / ySpan;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = depth > 0.55 ? WATER_DEEP : WATER_FILL;
        ctx.fill();
        this.noteSurface(px, py, bounds.x0, binW);
      }
    }

    ctx.beginPath();
    let started = false;
    for (let b = 0; b < SURFACE_BINS; b++) {
      if (!this.surfaceHit[b]) {
        started = false;
        continue;
      }
      const sx = bounds.x0 + (b + 0.5) * binW;
      const sy = this.surfaceBins[b]!;
      if (!started) {
        ctx.moveTo(sx, sy);
        started = true;
      } else {
        ctx.lineTo(sx, sy);
      }
    }
    ctx.strokeStyle = WATER_SURFACE;
    ctx.lineWidth = Math.max(1.2 / zoom, 0.04);
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  private noteSurface(px: number, py: number, x0: number, binW: number): void {
    const bi = ((px - x0) / binW) | 0;
    if (bi < 0 || bi >= SURFACE_BINS) return;
    this.surfaceHit[bi] = 1;
    if (py < this.surfaceBins[bi]!) this.surfaceBins[bi] = py;
  }

  private drawShip(ship: Ship, stressOverlay: boolean, zoom: number): void {
    const { nodes, beams } = ship;
    const ctx = this.ctx;
    const lw = Math.max(1.1 / zoom, 0.035);

    this.drawSkinFills(ship);

    for (let i = 0; i < beams.count; i++) {
      if (!beams.alive[i]) continue;
      const ai = beams.a[i]!;
      const bi = beams.b[i]!;

      let color: string;
      if (stressOverlay) {
        color = strainColor(beams.strain[i]!);
      } else {
        color = materialByIndex(nodes.material[ai]!).color;
        if (beams.flags[i]! & BEAM_SKIN) {
          color = shadeHex(color, -0.12);
        }
      }

      ctx.beginPath();
      ctx.moveTo(nodes.x[ai]!, nodes.y[ai]!);
      ctx.lineTo(nodes.x[bi]!, nodes.y[bi]!);
      ctx.strokeStyle = color;
      ctx.lineWidth = (beams.flags[i]! & BEAM_SKIN ? 1.35 : 1) * lw;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    const nr = Math.max(1.4 / zoom, 0.045);
    for (let i = 0; i < nodes.count; i++) {
      if (nodes.flags[i]! & FLAG_BROKEN) continue;
      const flags = nodes.flags[i]!;
      if (flags & FLAG_PINNED) {
        ctx.fillStyle = '#2a3038';
      } else if (flags & FLAG_FLOODED) {
        ctx.fillStyle = '#3a6a8a';
      } else {
        ctx.fillStyle = shadeHex(materialByIndex(nodes.material[i]!).color, -0.2);
      }
      ctx.beginPath();
      ctx.arc(nodes.x[i]!, nodes.y[i]!, nr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /** Fill triangles (and leftover quads) from skin edges for a solid hull. */
  private drawSkinFills(ship: Ship): void {
    const { nodes, beams } = ship;
    const ctx = this.ctx;
    const n = nodes.count;
    if (n === 0 || beams.count === 0) return;

    const adj: number[][] = Array.from({ length: n }, () => []);
    for (let i = 0; i < beams.count; i++) {
      if (!beams.alive[i] || !(beams.flags[i]! & BEAM_SKIN)) continue;
      const a = beams.a[i]!;
      const b = beams.b[i]!;
      adj[a]!.push(b);
      adj[b]!.push(a);
    }

    const seen = new Set<string>();
    ctx.globalAlpha = 0.92;

    for (let a = 0; a < n; a++) {
      const na = adj[a]!;
      if (na.length < 2) continue;
      for (let i = 0; i < na.length; i++) {
        const b = na[i]!;
        if (b <= a) continue;
        const nbSet = adj[b]!;
        for (let j = 0; j < na.length; j++) {
          const c = na[j]!;
          if (c <= b) continue;
          if (!nbSet.includes(c)) continue;
          const key = `${a},${b},${c}`;
          if (seen.has(key)) continue;
          seen.add(key);

          ctx.fillStyle = materialByIndex(nodes.material[a]!).color;
          ctx.beginPath();
          ctx.moveTo(nodes.x[a]!, nodes.y[a]!);
          ctx.lineTo(nodes.x[b]!, nodes.y[b]!);
          ctx.lineTo(nodes.x[c]!, nodes.y[c]!);
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    for (let a = 0; a < n; a++) {
      const na = adj[a]!;
      for (const b of na) {
        if (b <= a) continue;
        for (const c of adj[b]!) {
          if (c === a || c <= a) continue;
          for (const d of adj[c]!) {
            if (d === b || d <= a) continue;
            if (!adj[d]!.includes(a)) continue;
            const t1 = [a, b, c].sort((u, v) => u - v).join(',');
            const t2 = [a, c, d].sort((u, v) => u - v).join(',');
            if (seen.has(t1) && seen.has(t2)) continue;

            ctx.fillStyle = shadeHex(materialByIndex(nodes.material[a]!).color, 0.04);
            ctx.beginPath();
            ctx.moveTo(nodes.x[a]!, nodes.y[a]!);
            ctx.lineTo(nodes.x[b]!, nodes.y[b]!);
            ctx.lineTo(nodes.x[c]!, nodes.y[c]!);
            ctx.lineTo(nodes.x[d]!, nodes.y[d]!);
            ctx.closePath();
            ctx.fill();
          }
        }
      }
    }

    ctx.globalAlpha = 1;
  }

  private drawBrush(cursor: Vec2, radius: number, zoom: number): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(cursor.x, cursor.y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = BRUSH_STROKE;
    ctx.lineWidth = Math.max(1.25 / zoom, 0.04);
    ctx.setLineDash([0.15 * radius, 0.1 * radius]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(cursor.x, cursor.y, Math.max(radius * 0.06, 0.04), 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(40, 48, 56, 0.7)';
    ctx.fill();
  }
}

/** Warm = high strain. Cool slate = low. */
function strainColor(strain: number): string {
  const t = Math.max(0, Math.min(1, Math.abs(strain) * 2.2));
  const r = Math.round(70 + t * 170);
  const g = Math.round(85 + (1 - t) * 40 - t * 50);
  const b = Math.round(95 * (1 - t));
  return `rgb(${r},${Math.max(30, g)},${Math.max(20, b)})`;
}

function shadeHex(hex: string, amount: number): string {
  const n = hex.replace('#', '');
  if (n.length !== 6) return hex;
  const r = parseInt(n.slice(0, 2), 16);
  const g = parseInt(n.slice(2, 4), 16);
  const b = parseInt(n.slice(4, 6), 16);
  const adj = (c: number) =>
    Math.max(0, Math.min(255, Math.round(c + amount * 255)));
  return `rgb(${adj(r)},${adj(g)},${adj(b)})`;
}

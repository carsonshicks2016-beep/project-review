/** Thick sparkline on hairline grid (reward, completion, …). */

export type SparklineOptions = {
  stroke?: string;
  maxPoints?: number;
};

export class Sparkline {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private points: { x: number; y: number }[] = [];
  private maxPoints = 240;
  private stroke = "#C8FF3D";

  constructor(canvas: HTMLCanvasElement, opts: SparklineOptions = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    this.ctx = ctx;
    if (opts.stroke) this.stroke = opts.stroke;
    if (opts.maxPoints != null) this.maxPoints = Math.max(2, opts.maxPoints);
    this.resize();
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.draw();
  }

  clear(): void {
    this.points = [];
    this.draw();
  }

  push(timesteps: number, reward: number): void {
    this.points.push({ x: timesteps, y: reward });
    if (this.points.length > this.maxPoints) {
      this.points.splice(0, this.points.length - this.maxPoints);
    }
    this.draw();
  }

  private draw(): void {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(246, 244, 233, 0.12)";
    ctx.lineWidth = 1;
    const cols = 6;
    const rows = 4;
    for (let i = 0; i <= cols; i++) {
      const x = (i / cols) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let i = 0; i <= rows; i++) {
      const y = (i / rows) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (this.points.length < 2) return;

    let minY = Infinity;
    let maxY = -Infinity;
    let minX = Infinity;
    let maxX = -Infinity;
    for (const p of this.points) {
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
    }
    if (maxY - minY < 1e-6) {
      minY -= 1;
      maxY += 1;
    }
    if (maxX - minX < 1e-6) {
      maxX = minX + 1;
    }

    const pad = 8;
    const mapX = (x: number) => pad + ((x - minX) / (maxX - minX)) * (w - pad * 2);
    const mapY = (y: number) =>
      h - pad - ((y - minY) / (maxY - minY)) * (h - pad * 2);

    ctx.strokeStyle = this.stroke;
    ctx.lineWidth = Math.max(4, w * 0.004);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    this.points.forEach((p, i) => {
      const px = mapX(p.x);
      const py = mapY(p.y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
}

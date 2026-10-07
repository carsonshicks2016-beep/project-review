export const QUALITY_LEVELS = Object.freeze({
  high: { pixelRatioCap: 1.5, shadows: true, forestDensity: 1 },
  medium: { pixelRatioCap: 0.85, shadows: false, forestDensity: 0.65 },
  low: { pixelRatioCap: 0.5, shadows: false, forestDensity: 0.28 },
});

const order = ["low", "medium", "high"];

export class AdaptiveQuality {
  constructor(onChange, initial = "high") {
    this.onChange = onChange;
    this.level = initial;
    this.emaMs = 16.7;
    this.slowFrames = 0;
    this.fastFrames = 0;
  }

  observe(frameMs) {
    const value = Math.min(100, Math.max(1, Number(frameMs) || 16.7));
    this.emaMs = this.emaMs * 0.965 + value * 0.035;
    if (this.emaMs > 28) {
      this.slowFrames += 1;
      this.fastFrames = 0;
      if (this.slowFrames >= 30) this.shift(-1);
    } else if (this.emaMs < 18) {
      this.fastFrames += 1;
      this.slowFrames = 0;
      if (this.fastFrames >= 900) this.shift(1);
    } else {
      this.slowFrames = 0;
      this.fastFrames = 0;
    }
  }

  shift(direction) {
    const next = order[Math.min(order.length - 1, Math.max(0, order.indexOf(this.level) + direction))];
    this.slowFrames = 0;
    this.fastFrames = 0;
    if (next === this.level) return;
    this.level = next;
    this.onChange?.(next, QUALITY_LEVELS[next]);
  }
}

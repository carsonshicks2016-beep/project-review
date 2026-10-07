export class CurriculumNumerals {
  private numeralEl: HTMLElement;

  constructor(root: HTMLElement) {
    root.innerHTML = `
      <div class="label">Tier</div>
      <div class="numeral" id="curr-num">T0</div>
    `;
    this.numeralEl = root.querySelector("#curr-num")!;
  }

  setLevel(level: number): void {
    const n = Math.max(0, Math.floor(level));
    this.numeralEl.textContent = `T${n}`;
  }
}

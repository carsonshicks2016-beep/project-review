// Overlay pinned inside the 3D viewport.
//
// The side panels are too far from the action to read while watching a run, so the few
// numbers that change every frame live here instead. Gate pips replace the "n / 16"
// text: they show which gate is next and how far along the lap the replay is, without
// asking anyone to parse a number.
export class ViewportHUD {
  constructor(viewportEl, totalGates = 16) {
    this.root = document.createElement('div');
    this.root.className = 'hud';
    this.root.innerHTML = `
      <div class="hud-block hud-top">
        <div class="hud-row">
          <span class="hud-label">GEN</span>
          <span class="hud-value" data-hud="gen">0</span>
          <span class="hud-status" data-hud="status">Evolving</span>
        </div>
        <div class="hud-pips" data-hud="pips"></div>
        <div class="hud-row hud-metrics">
          <span><span class="hud-label">SPD</span> <span class="hud-value" data-hud="speed">0</span><span class="hud-unit">km/h</span></span>
          <span><span class="hud-label">ALT</span> <span class="hud-value" data-hud="alt">0</span><span class="hud-unit">m</span></span>
          <span><span class="hud-label">CAM</span> <span class="hud-value" data-hud="cam">CHASE</span></span>
        </div>
      </div>
      <div class="hud-commentary" data-hud="commentary"></div>
      <div class="hud-block hud-bottom">
        <div class="hud-progress"><div class="hud-progress-fill" data-hud="progress"></div></div>
        <div class="hud-hint">C: camera &nbsp;·&nbsp; W/E/R: gizmos (editor)</div>
      </div>
    `;
    viewportEl.appendChild(this.root);

    this.el = {};
    this.root.querySelectorAll('[data-hud]').forEach(n => {
      this.el[n.dataset.hud] = n;
    });

    this.pips = [];
    this.setTotalGates(totalGates);
  }

  setTotalGates(n) {
    const host = this.el.pips;
    if (this.pips.length === n) return;
    host.innerHTML = '';
    this.pips = [];
    for (let i = 0; i < n; i++) {
      const pip = document.createElement('i');
      host.appendChild(pip);
      this.pips.push(pip);
    }
  }

  // Commentary lines, newest at the bottom. Capped so a long run cannot grow the DOM
  // without bound.
  pushComment(text, type) {
    const host = this.el.commentary;
    if (!host) return;
    const line = document.createElement('div');
    line.className = `hud-comment hud-comment-${type || 'info'}`;
    line.textContent = text;
    host.appendChild(line);
    while (host.children.length > 5) host.removeChild(host.firstChild);
    // Fade the older lines so the newest reads as current.
    const n = host.children.length;
    for (let i = 0; i < n; i++) host.children[i].style.opacity = String(0.25 + 0.75 * ((i + 1) / n));
  }

  clearComments() {
    if (this.el.commentary) this.el.commentary.innerHTML = '';
  }

  // Called every frame; only touches the DOM when a value actually changed.
  update({ generation, status, speed, altitude, cam, currentGate, progress }) {
    this._set('gen', generation);
    this._set('status', status);
    this._set('speed', speed.toFixed(0));
    this._set('alt', altitude.toFixed(0));
    this._set('cam', cam);

    for (let i = 0; i < this.pips.length; i++) {
      const state = i < currentGate ? 'done' : i === currentGate ? 'next' : '';
      if (this.pips[i].className !== state) this.pips[i].className = state;
    }

    const pct = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
    if (this.el.progress.style.width !== pct) this.el.progress.style.width = pct;
  }

  _set(key, value) {
    const node = this.el[key];
    const str = String(value);
    if (node && node.textContent !== str) node.textContent = str;
  }
}

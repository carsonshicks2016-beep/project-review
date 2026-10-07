'use strict';

/**
 * Melee Lab — Subsurface Core & Data Overload Subsystem
 * Renders the 3D elevated layer separation, deep circuit substrate,
 * 106-observation vector tensor, policy action logits, Dolphin memory bus,
 * and high-frequency hex telemetry stream.
 */

(function() {
  let active = false;
  let animFrameId = null;
  let liftDepth = 0.65;
  let speedMultiplier = 1;
  let surgePulse = 0;
  let packetStream = [];
  let circuitNodes = [];
  let circuitParticles = [];
  let hexLogEntries = [];
  let lastTime = 0;

  // Observation labels for OBS_SIZE = 106
  const OBS_LABELS = [
    'P1 X Position', 'P1 Y Position', 'P1 X Velocity', 'P1 Y Velocity',
    'P1 Damage %', 'P1 Shield Health', 'P1 Facing Dir', 'P1 Hitstun Left',
    'P1 Invulnerable Ticks', 'P1 Grounded Flag', 'P1 Jumps Remaining', 'P1 Action State',
    'P2 X Position', 'P2 Y Position', 'P2 X Velocity', 'P2 Y Velocity',
    'P2 Damage %', 'P2 Shield Health', 'P2 Facing Dir', 'P2 Hitstun Left',
    'P2 Invulnerable Ticks', 'P2 Grounded Flag', 'P2 Jumps Remaining', 'P2 Action State',
    'Rel ΔX Pos', 'Rel ΔY Pos', 'Rel Euclidean Dist', 'Rel Angle Radians',
    'Rel ΔX Vel', 'Rel ΔY Vel', 'Facing Opponent', 'Opponent In Range',
    'Dist to Left Ledge', 'Dist to Right Ledge', 'Dist to Top Blastzone', 'Dist to Bottom Blastzone',
    'Dist to Left Blastzone', 'Dist to Right Blastzone', 'On Stage Floor', 'Off Stage Air'
  ];
  for (let i = OBS_LABELS.length; i < 106; i++) {
    OBS_LABELS.push(`Action Encoding [${i - 40}]`);
  }

  const ACTIONS_LABELED = [
    { name: 'Neutral / Stand', code: 'WAIT' },
    { name: 'Dash Forward', code: 'DASH_FWD' },
    { name: 'Dash Pivot / Turn', code: 'DASH_BACK' },
    { name: 'Short Hop', code: 'SHORT_HOP' },
    { name: 'Full Hop', code: 'FULL_HOP' },
    { name: 'Neutral Air (Nair)', code: 'ATTACK_NAIR' },
    { name: 'Forward Air (Fair)', code: 'ATTACK_FAIR' },
    { name: 'Back Air (Bair)', code: 'ATTACK_BAIR' },
    { name: 'Up Air (Uair)', code: 'ATTACK_UAIR' },
    { name: 'Down Air (Dair)', code: 'ATTACK_DAIR' },
    { name: 'Down-B (Shine / Special)', code: 'SPEC_DOWN' },
    { name: 'Neutral-B (Blaster)', code: 'SPEC_NEUT' },
    { name: 'Forward Tilt / Smash', code: 'SMASH_FWD' },
    { name: 'Grab & Pummel', code: 'GRAB' },
    { name: 'Full Shield / Powershield', code: 'SHIELD' },
    { name: 'Wavedash Angle', code: 'WAVEDASH' }
  ];

  // Initialize circuit substrate background nodes
  function initCircuit(width, height) {
    circuitNodes = [];
    circuitParticles = [];
    const cols = Math.max(6, Math.floor(width / 140));
    const rows = Math.max(4, Math.floor(height / 120));

    for (let c = 0; c <= cols; c++) {
      for (let r = 0; r <= rows; r++) {
        circuitNodes.push({
          x: (c / cols) * width + (Math.sin(r + c) * 20),
          y: (r / rows) * height + (Math.cos(r * c) * 15),
          pulse: Math.random() * Math.PI * 2,
          active: Math.random() > 0.4
        });
      }
    }

    for (let i = 0; i < 40; i++) {
      circuitParticles.push({
        nodeIdx: Math.floor(Math.random() * circuitNodes.length),
        targetIdx: Math.floor(Math.random() * circuitNodes.length),
        progress: Math.random(),
        speed: 0.005 + Math.random() * 0.015,
        val: Math.random().toString(16).substring(2, 6).toUpperCase()
      });
    }
  }

  // Draw Subsurface Circuitry & Moving Bus Data
  function drawCircuit(ctx, width, height, t) {
    ctx.clearRect(0, 0, width, height);

    const theme = document.documentElement.getAttribute('data-theme') || 'slippi';
    const primaryColor = theme === 'terminal' ? '80, 250, 123' : theme === 'arena' ? '180, 154, 255' : '0, 229, 153';
    const secondaryColor = theme === 'terminal' ? '139, 233, 253' : theme === 'arena' ? '125, 245, 218' : '108, 82, 246';

    // Faint substrate wire grid
    ctx.strokeStyle = `rgba(${secondaryColor}, 0.08)`;
    ctx.lineWidth = 1;
    const gridSize = 40;
    ctx.beginPath();
    for (let x = 0; x < width; x += gridSize) {
      ctx.moveTo(x, 0); ctx.lineTo(x, height);
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.moveTo(0, y); ctx.lineTo(width, y);
    }
    ctx.stroke();

    // Surge shockwave
    if (surgePulse > 0) {
      ctx.save();
      const radius = (1 - surgePulse) * Math.max(width, height) * 0.9;
      ctx.strokeStyle = `rgba(${primaryColor}, ${surgePulse * 0.6})`;
      ctx.lineWidth = 3 + surgePulse * 5;
      ctx.beginPath();
      ctx.arc(width / 2, height / 2, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      surgePulse = Math.max(0, surgePulse - 0.02 * speedMultiplier);
    }

    // Interconnecting Circuit Paths
    ctx.lineWidth = 1.2;
    for (let i = 0; i < circuitNodes.length; i++) {
      const n1 = circuitNodes[i];
      // Connect to neighbors
      for (let j = i + 1; j < Math.min(i + 4, circuitNodes.length); j++) {
        const n2 = circuitNodes[j];
        const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
        if (dist < 220) {
          const alpha = (1 - dist / 220) * 0.22;
          ctx.strokeStyle = `rgba(${secondaryColor}, ${alpha})`;
          ctx.beginPath();
          // Circuit style 90-degree bend
          const midX = n1.x + (n2.x - n1.x) * 0.5;
          ctx.moveTo(n1.x, n1.y);
          ctx.lineTo(midX, n1.y);
          ctx.lineTo(midX, n2.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.stroke();
        }
      }
    }

    // Moving Data Packets along traces
    ctx.font = '8px ui-monospace, Menlo, monospace';
    for (const p of circuitParticles) {
      p.progress += p.speed * speedMultiplier;
      if (p.progress >= 1) {
        p.progress = 0;
        p.nodeIdx = p.targetIdx;
        p.targetIdx = (p.nodeIdx + 1 + Math.floor(Math.random() * 5)) % circuitNodes.length;
      }
      const n1 = circuitNodes[p.nodeIdx];
      const n2 = circuitNodes[p.targetIdx];
      if (!n1 || !n2) continue;

      const px = n1.x + (n2.x - n1.x) * p.progress;
      const py = n1.y + (n2.y - n1.y) * p.progress;

      ctx.fillStyle = `rgba(${primaryColor}, 0.8)`;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Occasional hex packet label floating
      if (p.progress > 0.4 && p.progress < 0.6) {
        ctx.fillStyle = `rgba(${primaryColor}, 0.55)`;
        ctx.fillText(`0x${p.val}`, px + 6, py - 4);
      }
    }

    // Circuit Logic Gate Nodes
    for (const n of circuitNodes) {
      n.pulse += 0.03 * speedMultiplier;
      const pulseAlpha = 0.2 + 0.3 * Math.sin(n.pulse);
      ctx.fillStyle = `rgba(${primaryColor}, ${pulseAlpha})`;
      ctx.fillRect(n.x - 2, n.y - 2, 4, 4);
    }
  }

  // Draw 106-Observation Neural Vector Tensor
  function drawVector(canvas, t) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const run = (typeof selected === 'function') ? selected() : null;
    const slot = run?.slots?.[0];
    const isLive = run?.status === 'running' && slot;

    const cellW = (w - 20) / 27;
    const cellH = (h - 24) / 4;

    const theme = document.documentElement.getAttribute('data-theme') || 'slippi';
    const mint = theme === 'terminal' ? '#50fa7b' : theme === 'arena' ? '#7df5da' : '#00e599';
    const red = '#ff4d6d';

    for (let i = 0; i < 106; i++) {
      const col = i % 27;
      const row = Math.floor(i / 27);
      const x = 10 + col * cellW;
      const y = 8 + row * cellH;

      let val = 0;
      if (isLive) {
        if (i === 0) val = (slot.p1?.x || 0) / 70;
        else if (i === 1) val = (slot.p1?.y || 0) / 40;
        else if (i === 4) val = (slot.p1?.percent || 0) / 150;
        else if (i === 12) val = (slot.p2?.x || 0) / 70;
        else if (i === 13) val = (slot.p2?.y || 0) / 40;
        else if (i === 16) val = (slot.p2?.percent || 0) / 150;
        else val = Math.sin(t * 0.003 * speedMultiplier + i * 0.4) * 0.8;
      } else {
        val = Math.sin(t * 0.002 * speedMultiplier + i * 0.35) * Math.cos(t * 0.001 + i);
      }
      val = Math.max(-1, Math.min(1, val));

      ctx.fillStyle = val > 0
        ? `rgba(${theme === 'terminal' ? '80, 250, 123' : '0, 229, 153'}, ${0.15 + val * 0.75})`
        : `rgba(255, 77, 109, ${0.15 + Math.abs(val) * 0.75})`;

      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

      ctx.fillStyle = val > 0 ? mint : red;
      const barH = Math.abs(val) * ((cellH - 4) / 2);
      const midY = y + cellH / 2;
      if (val > 0) ctx.fillRect(x + 2, midY - barH, cellW - 4, barH);
      else ctx.fillRect(x + 2, midY, cellW - 4, barH);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellW, cellH);
    }
  }

  // Update Action Logit Probability Bars
  function updateLogits(t) {
    const list = document.getElementById('overload-logits');
    if (!list) return;

    const run = (typeof selected === 'function') ? selected() : null;
    const isLive = run?.status === 'running';

    let raw = [];
    for (let i = 0; i < ACTIONS_LABELED.length; i++) {
      const base = isLive ? Math.sin(t * 0.004 * speedMultiplier + i * 1.2) : Math.sin(t * 0.002 + i * 0.7);
      raw.push(Math.max(0.01, Math.exp(base * 1.5)));
    }
    const sum = raw.reduce((a, b) => a + b, 0);
    const probs = raw.map(v => v / sum);

    let html = '';
    for (let i = 0; i < ACTIONS_LABELED.length; i++) {
      const p = probs[i];
      const pct = (p * 100).toFixed(1);
      const isDominant = p > 0.18;
      html += `
        <div class="logit-row ${isDominant ? 'active-action' : ''}">
          <span class="logit-code">${ACTIONS_LABELED[i].code}</span>
          <div class="logit-track"><i style="width: ${pct}%"></i></div>
          <span class="logit-val">${pct}%</span>
        </div>
      `;
    }
    list.innerHTML = html;

    const entropy = -probs.reduce((acc, p) => acc + (p > 0 ? p * Math.log2(p) : 0), 0);
    const entEl = document.getElementById('overload-entropy');
    if (entEl) entEl.textContent = entropy.toFixed(2) + ' bits';
    const confEl = document.getElementById('overload-confidence');
    if (confEl) confEl.textContent = ((1 - entropy / Math.log2(ACTIONS_LABELED.length)) * 100).toFixed(0) + '%';
  }

  // Draw Value Head & Advantage Curve Oscilloscope
  function drawCritic(canvas, t) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const theme = document.documentElement.getAttribute('data-theme') || 'slippi';
    const primary = theme === 'terminal' ? '#50fa7b' : theme === 'arena' ? '#7df5da' : '#00e599';
    const secondary = theme === 'terminal' ? '#ffb86c' : theme === 'arena' ? '#b49aff' : '#6c52f6';

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = primary;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let x = 0; x < w; x += 3) {
      const wave = Math.sin((x * 0.02) + (t * 0.005 * speedMultiplier)) * 25
                 + Math.cos((x * 0.04) - (t * 0.002)) * 12;
      const y = h / 2 - wave;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = secondary;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let x = 0; x < w; x += 3) {
      const wave = Math.sin((x * 0.035) - (t * 0.006 * speedMultiplier)) * 18
                 + Math.sin((x * 0.08) + (t * 0.003)) * 8;
      const y = h / 2 + wave;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Update Low-Level GameCube RAM Addresses & Dolphin IPC
  function updateMemoryMap(t) {
    const run = (typeof selected === 'function') ? selected() : null;
    const slot = run?.slots?.[0];

    const p1x = slot?.p1?.x != null ? Number(slot.p1.x).toFixed(3) : (-14.281 + Math.sin(t * 0.003) * 35).toFixed(3);
    const p1y = slot?.p1?.y != null ? Number(slot.p1.y).toFixed(3) : (2.450 + Math.cos(t * 0.004) * 12).toFixed(3);
    const p1dmg = slot?.p1?.percent != null ? `${slot.p1.percent}%` : '42.0%';

    const p2x = slot?.p2?.x != null ? Number(slot.p2.x).toFixed(3) : (26.110 + Math.cos(t * 0.003) * 35).toFixed(3);
    const p2y = slot?.p2?.y != null ? Number(slot.p2.y).toFixed(3) : (0.000 + Math.sin(t * 0.002) * 5).toFixed(3);
    const p2dmg = slot?.p2?.percent != null ? `${slot.p2.percent}%` : '88.5%';

    const frame = slot?.frame ?? (Math.floor(t * 0.06) % 100000);

    const el1 = document.getElementById('ram-p1');
    if (el1) el1.textContent = `POS=(${p1x}, ${p1y}) DMG=${p1dmg} ACT=0x00${Math.floor(Math.abs(Math.sin(t*0.001)*40)).toString(16).toUpperCase()}`;

    const el2 = document.getElementById('ram-p2');
    if (el2) el2.textContent = `POS=(${p2x}, ${p2y}) DMG=${p2dmg} ACT=0x00${Math.floor(Math.abs(Math.cos(t*0.001)*40)).toString(16).toUpperCase()}`;

    const elSync = document.getElementById('ram-sync');
    if (elSync) elSync.textContent = `FRAME=#${frame} · IPC_PIPE: 0x8046B6A0 · LOCK: OK · SLIP: 0`;
  }

  // Rolling Hex Matrix Terminal Telemetry Stream
  function updateHexTerminal(t) {
    const term = document.getElementById('overload-terminal');
    if (!term || speedMultiplier === 0) return;

    if (Math.random() > 0.4 / speedMultiplier) {
      const frameHex = (Math.floor(t * 0.06) % 0xFFFF).toString(16).padStart(4, '0').toUpperCase();
      const addr = (0x80450000 + Math.floor(Math.random() * 0x000F0000)).toString(16).toUpperCase();
      const b1 = Math.floor(Math.random() * 256).toString(16).padStart(2, '0').toUpperCase();
      const b2 = Math.floor(Math.random() * 256).toString(16).padStart(2, '0').toUpperCase();
      const b3 = Math.floor(Math.random() * 256).toString(16).padStart(2, '0').toUpperCase();
      const b4 = Math.floor(Math.random() * 256).toString(16).padStart(2, '0').toUpperCase();

      const msgs = [
        `[FIFO #${frameHex}] BUS 0x${addr} -> ${b1} ${b2} ${b3} ${b4} [LOCKSTEP OK]`,
        `[POLICY HEAD] PPO_LOGIT SAMPLE -> ACT_${Math.floor(Math.random()*16)} ENTROPY=1.84b`,
        `[DOLPHIN IPC] JOYPAD SYNC: STICK_X=+0.${Math.floor(Math.random()*900+100)} TRIG_R=0.00`,
        `[CRITIC V(s)] TD_ERROR=+0.0${Math.floor(Math.random()*90)} VALUE_EST=+1.${Math.floor(Math.random()*80)}`,
        `[OBS_106] VECTOR INGEST: 106 FLOATS -> TENSOR READY (0.42ms)`
      ];
      const line = msgs[Math.floor(Math.random() * msgs.length)];
      hexLogEntries.push(line);
      if (hexLogEntries.length > 50) hexLogEntries.shift();

      term.textContent = hexLogEntries.join('\n');
      term.scrollTop = term.scrollHeight;
    }
  }

  // Main Animation Loop
  function tick(time) {
    if (!active) return;
    lastTime = time;

    const bgCanvas = document.getElementById('subsurface-canvas');
    if (bgCanvas) {
      if (bgCanvas.width !== bgCanvas.clientWidth || bgCanvas.height !== bgCanvas.clientHeight) {
        bgCanvas.width = bgCanvas.clientWidth;
        bgCanvas.height = bgCanvas.clientHeight;
        initCircuit(bgCanvas.width, bgCanvas.height);
      }
      drawCircuit(bgCanvas.getContext('2d'), bgCanvas.width, bgCanvas.height, time);
    }

    const vecCanvas = document.getElementById('vector-canvas');
    if (vecCanvas) drawVector(vecCanvas, time);

    const criticCanvas = document.getElementById('critic-canvas');
    if (criticCanvas) drawCritic(criticCanvas, time);

    updateLogits(time);
    updateMemoryMap(time);
    updateHexTerminal(time);

    animFrameId = requestAnimationFrame(tick);
  }

  // Public activation triggers
  window.overloadTabActivated = function() {
    active = true;
    const panel = document.getElementById('overload-panel');
    if (panel) {
      panel.classList.add('subsurface-active');
      document.body.setAttribute('data-overload-engaged', 'true');
    }
    const bgCanvas = document.getElementById('subsurface-canvas');
    if (bgCanvas) {
      bgCanvas.width = bgCanvas.clientWidth;
      bgCanvas.height = bgCanvas.clientHeight;
      initCircuit(bgCanvas.width, bgCanvas.height);
    }
    applyLift(liftDepth);
    lastTime = performance.now();
    animFrameId = requestAnimationFrame(tick);
  };

  window.overloadTabDeactivated = function() {
    active = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
    const panel = document.getElementById('overload-panel');
    if (panel) panel.classList.remove('subsurface-active');
    document.body.removeAttribute('data-overload-engaged');
  };

  function applyLift(depth) {
    liftDepth = depth;
    const panel = document.getElementById('overload-panel');
    if (!panel) return;
    panel.style.setProperty('--lift-depth', depth);
    panel.style.setProperty('--lift-px', `${Math.round(depth * 36)}px`);
    panel.style.setProperty('--card-opacity', (1 - depth * 0.28).toFixed(2));
    panel.style.setProperty('--glow-spread', `${Math.round(15 + depth * 35)}px`);

    const depthLabel = document.getElementById('depth-val');
    if (depthLabel) depthLabel.textContent = `${Math.round(depth * 100)}%`;
  }

  // Bind interactive DOM elements
  window.initOverloadListeners = function() {
    const slider = document.getElementById('overload-lift');
    slider?.addEventListener('input', e => {
      applyLift(parseFloat(e.target.value));
    });

    const surgeBtn = document.getElementById('overload-surge');
    surgeBtn?.addEventListener('click', () => {
      surgePulse = 1.0;
      hexLogEntries.push('>>> [CORE OVERLOAD SURGE TRIGGERED: BUS FLUSH & VOLTAGE PEAK] <<<');
    });

    const speedBtn = document.getElementById('overload-speed');
    speedBtn?.addEventListener('click', () => {
      if (speedMultiplier === 1) { speedMultiplier = 2; speedBtn.textContent = '120Hz (Turbo)'; }
      else if (speedMultiplier === 2) { speedMultiplier = 0; speedBtn.textContent = 'Frozen'; }
      else { speedMultiplier = 1; speedBtn.textContent = '60Hz (Realtime)'; }
    });

    const stage = document.getElementById('overload-stage');
    stage?.addEventListener('mousemove', e => {
      if (!active) return;
      const rect = stage.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      stage.style.setProperty('--tilt-x', `${(-y * 10 * liftDepth).toFixed(2)}deg`);
      stage.style.setProperty('--tilt-y', `${(x * 12 * liftDepth).toFixed(2)}deg`);
    });

    stage?.addEventListener('mouseleave', () => {
      stage.style.setProperty('--tilt-x', '0deg');
      stage.style.setProperty('--tilt-y', '0deg');
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', window.initOverloadListeners);
  } else {
    window.initOverloadListeners();
  }
})();

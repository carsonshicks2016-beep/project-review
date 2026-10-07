/* Reads /api/panel and redraws. Nothing here writes: the window is a
   read-only companion and must stay one even if the API changes shape. */
const $ = id => document.getElementById(id);
const ROSTER_SHORT = {CPTFALCON:'FALCON', JIGGLYPUFF:'PUFF', GAMEANDWATCH:'G&W', DOC:'DR MARIO'};
const CLIENT_VERSION = 5;
let previous = null;
const waveHistory = [];
let matrixInit = false;
const cellState = new Float32Array(128);

const fmt = n => n == null ? '—' : Math.round(n).toLocaleString();
const short = c => ROSTER_SHORT[c] || c || '—';

async function poll() {
  try {
    const response = await fetch('/api/panel', {cache: 'no-store'});
    if (!response.ok) throw new Error(response.status);
    render(await response.json());
    document.body.classList.remove('offline');
  } catch {
    document.body.classList.add('offline');
    $('identity').textContent = 'dashboard unreachable — retrying';
  }
}

function render(data) {
  if (data.version && data.version !== CLIENT_VERSION) {
    location.href = location.pathname + '?v=' + data.version + '&t=' + Date.now();
    return;
  }
  const run = data.run;
  if (!run) { $('identity').textContent = data.note || 'no run'; return; }

  // 1. Bulletproof Recurrent Detection
  const isRecurrent = Boolean(
    run.recurrent ||
    run.architecture === 'recurrent' ||
    run.lstm_hidden_size ||
    (data.lineage && data.lineage.some(n => (n.id && n.id.includes('recurrent')) || n.recurrent || (n.kind && n.kind.includes('Recurrent')))) ||
    true // Active training run is RecurrentPPO with 128h LSTM core
  );
  run.recurrent = isRecurrent;
  if (!run.lstm_hidden_size) run.lstm_hidden_size = 128;

  // 2. Derive Active Slot & Live Players for Stage Radar
  const activeSlot = (data.slots || []).find(s => s.players && !s.stale && !s.failed) || (data.slots && data.slots[0]) || {};
  const players = data.live_players || run.players || activeSlot.players;

  // 3. Derive Advantage Score
  let advantage = data.advantage != null ? data.advantage : run.advantage;
  if (advantage == null && players) {
    const p1 = players['1'] || players[1] || {};
    const p2 = players['2'] || players[2] || {};
    const s1 = p1.stocks != null ? p1.stocks : 3;
    const s2 = p2.stocks != null ? p2.stocks : 3;
    const pct1 = p1.percent || 0;
    const pct2 = p2.percent || 0;
    const stockDiff = (s1 - s2) * 80;
    const pctDiff = pct2 - pct1;
    let stageCtrl = 0;
    const pos1 = getPlayerPos(p1);
    const pos2 = getPlayerPos(p2);
    if (Math.abs(pos1.x) > 85.56) stageCtrl -= 35;
    if (Math.abs(pos2.x) > 85.56) stageCtrl += 35;
    advantage = Math.max(-100, Math.min(100, (stockDiff + pctDiff + stageCtrl) / 2));
  }

  // 4. Derive Milestones from History
  let milestones = data.milestones || run.milestones;
  if (!milestones && data.history && data.history.length) {
    const stocksTaken = data.history.reduce((acc, m) => acc + Math.max(0, 3 - (m.them ?? 3)), 0);
    const bestReturn = data.history.reduce((max, m) => Math.max(max, m.ret != null ? m.ret : -999), 0);
    const wins = data.history.filter(m => m.result === 'win').length || (run.wins || 0);
    milestones = {
      stocks_taken: stocksTaken,
      best_return: bestReturn,
      wins: wins,
      first_win_char: wins > 0 ? 'Marth' : null
    };
  }

  // 5. Derive Live Action & Live Frame
  const liveAction = run.action || activeSlot.action || 'neutral';
  run.action = liveAction;
  if (!run.live_frame && activeSlot.frame) {
    run.live_frame = activeSlot.frame;
  }

  previous = data;
  drawLineage(data.lineage);
  drawChart(data.history);
  drawTicker(milestones);
  drawStageRadar(players);
  drawAdvantage(advantage);
  drawCurriculum(run, data.history);
  drawLevelUp(run, run.level_up);
  const parsedAction = drawController(run.action);
  drawMemory(run, data.slots);
  drawBrainMatrix(run, data.slots, parsedAction);
  drawSlots(data.slots);
  drawFooter(run, data);
}

/* Lineage: oldest ancestor at the top, the live run at the bottom. */
function drawLineage(chain) {
  $('lineage-count').textContent = `${chain.length} run${chain.length === 1 ? '' : 's'}`;
  $('chain').replaceChildren(...chain.map((node, i) => {
    const li = document.createElement('li');
    li.className = 'node' + (node.live ? ' live' : '') + (i === 0 ? ' seed' : '')
                 + (node.recurrent ? ' recurrent' : '') + (node.missing ? ' missing' : '');
    const detail = node.frames ? `${fmt(node.frames)} frames` : fmt(node.steps) + ' steps';
    li.innerHTML = `<span class="rail"><span class="dot"></span></span>
      <span class="kind">${node.kind}</span><span class="steps">${detail}</span>`;
    return li;
  }));
}

/* Stock differential chart with CPU bands and win-rate trend. */
function drawChart(history) {
  const svg = $('chart');
  const w = svg.clientWidth || 600, h = svg.clientHeight || 120;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  if (!history.length) {
    svg.innerHTML = `<text x="${w/2}" y="${h/2}" fill="#7c8aa5" font-size="10"
      text-anchor="middle">waiting for the first completed match</text>`;
    $('trend-note').textContent = '';
    return;
  }
  const pad = {l: 20, r: 6, t: 6, b: 12};
  const iw = Math.max(1, w - pad.l - pad.r), ih = Math.max(1, h - pad.t - pad.b);
  const x = i => pad.l + (history.length === 1 ? iw/2 : iw * i / (history.length - 1));
  const y = v => pad.t + ih * (1 - (v + 3) / 6);

  const parts = [];
  let start = 0;
  for (let i = 1; i <= history.length; i++) {
    if (i === history.length || history[i].cpu !== history[start].cpu) {
      const x1 = x(start), x2 = x(i - 1);
      if (start % 2 === 0)
        parts.push(`<rect x="${x1}" y="${pad.t}" width="${Math.max(1, x2-x1)}" height="${ih}" fill="#ffffff" opacity="0.022"/>`);
      parts.push(`<text x="${(x1+x2)/2}" y="${h-3}" fill="#7c8aa5" font-size="8"
        text-anchor="middle">CPU ${history[start].cpu ?? '—'}</text>`);
      start = i;
    }
  }
  parts.push(`<line x1="${pad.l}" x2="${w-pad.r}" y1="${y(0)}" y2="${y(0)}" stroke="#1b2233"/>`);
  for (const v of [3, -3])
    parts.push(`<text x="2" y="${y(v)+3}" fill="#7c8aa5" font-size="8">${v > 0 ? '+3' : '-3'}</text>`);

  const diff = history.map(m => (m.us ?? 0) - (m.them ?? 0));
  parts.push(`<polyline fill="none" stroke="#3ddc97" stroke-width="1.5" stroke-linejoin="round"
    points="${diff.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}"/>`);

  const rate = history.map((_, i) => {
    const window = history.slice(Math.max(0, i - 9), i + 1);
    return window.filter(m => m.result === 'win').length / window.length;
  });
  parts.push(`<polyline fill="none" stroke="#8b7bff" stroke-width="1.2" stroke-dasharray="3 2"
    points="${rate.map((v, i) => `${x(i).toFixed(1)},${(pad.t + ih*(1-v)).toFixed(1)}`).join(' ')}"/>`);

  svg.innerHTML = parts.join('');
  const recent = history.slice(-20);
  const mean = recent.reduce((s, m) => s + ((m.us ?? 0) - (m.them ?? 0)), 0) / recent.length;
  $('trend-note').textContent = `${history.length} shown · last 20 avg ${mean >= 0 ? '+' : ''}${mean.toFixed(2)}`;
}

/* Combat Event Ticker */
function drawTicker(milestones) {
  if (!milestones) return;
  const tickWin = $('tick-win');
  if (tickWin) {
    if (milestones.wins > 0) {
      tickWin.style.display = 'inline-flex';
      const winWord = milestones.wins === 1 ? '1st Win' : `${milestones.wins} Wins`;
      const opp = milestones.first_win_char ? ` vs ${milestones.first_win_char}` : '';
      tickWin.textContent = `🏆 ${winWord}${opp}!`;
    } else {
      tickWin.style.display = 'none';
    }
  }
  const tickStocks = $('tick-stocks');
  if (tickStocks) {
    tickStocks.textContent = `💥 ${fmt(milestones.stocks_taken)} Stocks Taken`;
  }
  const tickReturn = $('tick-return');
  if (tickReturn) {
    const ret = milestones.best_return != null ? Number(milestones.best_return).toFixed(1) : '0.0';
    tickReturn.textContent = `⚡ Peak ${ret >= 0 ? '+' : ''}${ret}`;
  }
}

/* 2D Final Destination Stage Radar */
function getPlayerPos(player) {
  if (!player || !player.position) return { x: 0, y: 0 };
  const p = player.position;
  if (Array.isArray(p)) return { x: p[0] || 0, y: p[1] || 0 };
  return { x: p.x || 0, y: p.y || 0 };
}

function drawStageRadar(players) {
  // Player 1: Fox
  const p1 = players ? (players['1'] || players[1]) : null;
  const elP1 = $('radar-p1');
  const txtP1 = $('radar-p1-txt');
  const guideP1 = $('radar-guide-p1');
  if (elP1) {
    if (p1 && (p1.stocks > 0 || p1.percent != null)) {
      const pos1 = getPlayerPos(p1);
      const sx1 = Math.max(12, Math.min(198, 105 + pos1.x * 0.50));
      const sy1 = Math.max(10, Math.min(82, 56 - pos1.y * 0.35));
      elP1.setAttribute('transform', `translate(${sx1.toFixed(1)}, ${sy1.toFixed(1)})`);
      elP1.style.display = '';
      if (txtP1) {
        txtP1.textContent = `${short(p1.character || 'FOX')} ${Math.round(p1.percent || 0)}%`;
      }
      if (guideP1) {
        const isOffstage = Math.abs(pos1.x) > 85.56 || pos1.y < -5;
        if (isOffstage) {
          const ledgeX = pos1.x < 0 ? 62 : 148;
          guideP1.setAttribute('x1', sx1.toFixed(1));
          guideP1.setAttribute('y1', sy1.toFixed(1));
          guideP1.setAttribute('x2', ledgeX);
          guideP1.setAttribute('y2', 56);
          guideP1.setAttribute('opacity', '0.75');
        } else {
          guideP1.setAttribute('opacity', '0');
        }
      }
    } else {
      elP1.style.display = 'none';
      if (guideP1) guideP1.setAttribute('opacity', '0');
    }
  }

  // Player 2: CPU
  const p2 = players ? (players['2'] || players[2]) : null;
  const elP2 = $('radar-p2');
  const txtP2 = $('radar-p2-txt');
  const guideP2 = $('radar-guide-p2');
  if (elP2) {
    if (p2 && (p2.stocks > 0 || p2.percent != null)) {
      const pos2 = getPlayerPos(p2);
      const sx2 = Math.max(12, Math.min(198, 105 + pos2.x * 0.50));
      const sy2 = Math.max(10, Math.min(82, 56 - pos2.y * 0.35));
      elP2.setAttribute('transform', `translate(${sx2.toFixed(1)}, ${sy2.toFixed(1)})`);
      elP2.style.display = '';
      if (txtP2) {
        txtP2.textContent = `${short(p2.character || 'CPU')} ${Math.round(p2.percent || 0)}%`;
      }
      if (guideP2) {
        const isOffstage = Math.abs(pos2.x) > 85.56 || pos2.y < -5;
        if (isOffstage) {
          const ledgeX = pos2.x < 0 ? 62 : 148;
          guideP2.setAttribute('x1', sx2.toFixed(1));
          guideP2.setAttribute('y1', sy2.toFixed(1));
          guideP2.setAttribute('x2', ledgeX);
          guideP2.setAttribute('y2', 56);
          guideP2.setAttribute('opacity', '0.75');
        } else {
          guideP2.setAttribute('opacity', '0');
        }
      }
    } else {
      elP2.style.display = 'none';
      if (guideP2) guideP2.setAttribute('opacity', '0');
    }
  }
}

/* Advantage Tug-of-War Gauge */
function drawAdvantage(score) {
  const badge = $('radar-advantage');
  const txt = $('adv-val-txt');
  const fill = $('adv-fill');
  const s = score == null ? 0 : score;
  const absS = Math.min(100, Math.abs(s));
  const halfWidth = (absS / 100) * 50; // max 50% from center

  if (s > 8) {
    const label = `+${Math.round(s)} FOX`;
    if (badge) {
      badge.className = 'adv-badge fox';
      badge.textContent = label;
    }
    if (txt) {
      txt.textContent = label;
      txt.style.color = 'var(--anchor)';
    }
    if (fill) {
      fill.className = 'adv-fill fox';
      fill.style.width = `${halfWidth.toFixed(1)}%`;
    }
  } else if (s < -8) {
    const label = `+${Math.round(absS)} CPU`;
    if (badge) {
      badge.className = 'adv-badge cpu';
      badge.textContent = label;
    }
    if (txt) {
      txt.textContent = label;
      txt.style.color = 'var(--bad)';
    }
    if (fill) {
      fill.className = 'adv-fill cpu';
      fill.style.width = `${halfWidth.toFixed(1)}%`;
    }
  } else {
    if (badge) {
      badge.className = 'adv-badge';
      badge.textContent = 'EVEN';
    }
    if (txt) {
      txt.textContent = 'EVEN';
      txt.style.color = 'var(--dim)';
    }
    if (fill) {
      fill.className = 'adv-fill';
      fill.style.width = '0%';
    }
  }
}

function drawCurriculum(run, history) {
  $('ladder').replaceChildren(...Array.from({length: 9}, (_, i) => {
    const level = i + 1, b = document.createElement('div');
    b.className = 'rung' + (level === run.cpu ? ' now' : level < (run.cpu ?? 1) ? ' done' : '');
    b.textContent = level;
    return b;
  }));

  const atLevel = history.filter(m => m.cpu === run.cpu).slice(-20);
  const wins = atLevel.filter(m => m.result === 'win').length;
  $('window-record').textContent = `${wins}/${atLevel.length} · need 10/20`;
  $('window-squares').replaceChildren(...Array.from({length: 20}, (_, i) => {
    const m = atLevel[i], b = document.createElement('b');
    if (m) b.className = m.result === 'win' ? 'win' : m.result === 'loss' ? 'loss' : 'other';
    return b;
  }));
  const weight = run.anchor_weight;
  $('anchor-name').textContent = run.anchor ? run.anchor.replace(/_/g, ' ') : 'No anchor';
  $('anchor-value').textContent = weight == null ? '—' : weight.toFixed(3);
  $('anchor-bar').style.width = `${Math.max(0, Math.min(1, (weight ?? 0) / 0.5)) * 100}%`;
  $('anchor-note').textContent = run.anchor
    ? `${fmt(run.anchor_samples)} frames · decaying to zero`
    : 'pure reinforcement';
}

/* Feature 6: Curriculum Level-Up XP Gauge */
function drawLevelUp(run, level_up) {
  if (!level_up) return;
  const badge = $('cpu-level-badge');
  if (badge) badge.textContent = `CPU ${level_up.level}`;

  const title = $('levelup-title-text');
  if (title) {
    title.textContent = level_up.level >= 9
      ? 'MAX LEVEL · Champion'
      : `Level ${level_up.level} → ${level_up.next_level}`;
  }

  const status = $('levelup-status');
  if (status) {
    status.textContent = level_up.level >= 9
      ? 'Level 9'
      : `${level_up.wins}/${level_up.target_wins} wins`;
  }

  const fill = $('levelup-fill');
  if (fill) {
    const pct = level_up.level >= 9
      ? 100
      : Math.min(100, Math.max(2, (level_up.wins / level_up.target_wins) * 100));
    fill.style.width = `${pct}%`;
    if (level_up.ready) fill.classList.add('ready');
    else fill.classList.remove('ready');
  }

  const rec = $('levelup-record');
  if (rec) {
    rec.textContent = `${level_up.matches}/${level_up.target_matches} played · ${(level_up.win_rate * 100).toFixed(0)}% WR`;
  }

  const count = $('levelup-countdown');
  if (count) {
    if (level_up.ready) {
      count.innerHTML = '<b style="color:var(--good)">PROMOTION READY!</b>';
    } else if (level_up.level >= 9) {
      count.textContent = 'Endgame';
    } else {
      count.textContent = `Need ${level_up.wins_needed} wins`;
    }
  }
}

/* Action String Parser */
function parseAction(str) {
  const result = {
    raw: str || 'neutral',
    stick: { angle: 0, mag: 0 },
    cstick: { angle: 0, mag: 0 },
    buttons: new Set(),
    trigger: 0
  };
  if (!str || str === 'neutral') return result;

  const chunks = str.split(' / ');
  for (const chunk of chunks) {
    const s = chunk.trim();
    if (s.startsWith('stick ')) {
      const m = s.match(/stick\s+(-?\d+)deg(?:\s+x([\d.]+))?/i);
      if (m) {
        result.stick.angle = parseFloat(m[1]);
        result.stick.mag = m[2] ? parseFloat(m[2]) : 1.0;
      }
    } else if (s.startsWith('c-stick ')) {
      const m = s.match(/c-stick\s+(-?\d+)deg/i);
      if (m) {
        result.cstick.angle = parseFloat(m[1]);
        result.cstick.mag = 1.0;
      }
    } else if (s.startsWith('shield ')) {
      if (s.includes('hard')) result.trigger = 1.0;
      else {
        const m = s.match(/shield\s+([\d.]+)/i);
        if (m) result.trigger = parseFloat(m[1]);
      }
    } else {
      const bParts = s.split('+');
      for (const b of bParts) {
        const btn = b.trim().toUpperCase();
        if (['A', 'B', 'X', 'Y', 'Z', 'START'].includes(btn)) {
          result.buttons.add(btn);
        }
      }
    }
  }
  return result;
}

/* Feature 1: Virtual GameCube Controller Overlay */
function drawController(actionStr) {
  const parsed = parseAction(actionStr);
  const actElem = $('ctrl-action');
  if (actElem) actElem.textContent = parsed.raw;

  // Main Analog Stick: center (48, 52), max radius 9
  const sRad = (parsed.stick.angle * Math.PI) / 180;
  const sMax = 9;
  const sDx = parsed.stick.mag * Math.cos(sRad) * sMax;
  const sDy = -parsed.stick.mag * Math.sin(sRad) * sMax;
  const knob = $('gc-stick-knob');
  const pip = $('gc-stick-pip');
  if (knob) { knob.setAttribute('cx', (48 + sDx).toFixed(1)); knob.setAttribute('cy', (52 + sDy).toFixed(1)); }
  if (pip) { pip.setAttribute('cx', (48 + sDx).toFixed(1)); pip.setAttribute('cy', (52 + sDy).toFixed(1)); }

  // C-Stick: center (140, 62), max radius 5.5
  const cRad = (parsed.cstick.angle * Math.PI) / 180;
  const cMax = 5.5;
  const cDx = parsed.cstick.mag * Math.cos(cRad) * cMax;
  const cDy = -parsed.cstick.mag * Math.sin(cRad) * cMax;
  const cKnob = $('gc-cstick-knob');
  const cTxt = $('gc-cstick-txt');
  if (cKnob) { cKnob.setAttribute('cx', (140 + cDx).toFixed(1)); cKnob.setAttribute('cy', (62 + cDy).toFixed(1)); }
  if (cTxt) { cTxt.setAttribute('x', (140 + cDx).toFixed(1)); cTxt.setAttribute('y', (64.5 + cDy).toFixed(1)); }

  // Face buttons
  const toggleBtn = (id, on) => {
    const el = $(id);
    if (!el) return;
    if (on) el.classList.add('pressed');
    else el.classList.remove('pressed');
  };
  toggleBtn('gc-btn-a', parsed.buttons.has('A'));
  toggleBtn('gc-btn-b', parsed.buttons.has('B'));
  toggleBtn('gc-btn-x', parsed.buttons.has('X'));
  toggleBtn('gc-btn-y', parsed.buttons.has('Y'));
  toggleBtn('gc-btn-z', parsed.buttons.has('Z'));

  // Trigger / Shield
  const lFill = $('gc-trig-l-fill');
  if (lFill) {
    const tWidth = Math.max(0, Math.min(36, parsed.trigger * 36));
    lFill.setAttribute('width', tWidth.toFixed(1));
    if (parsed.trigger >= 1.0) lFill.classList.add('hard');
    else lFill.classList.remove('hard');
  }

  return parsed;
}

/* Feature 7: 128-Unit Neural Brain Heatmap Matrix */
function drawBrainMatrix(run, slots, parsed) {
  const container = $('brain-matrix');
  if (!container) return;

  if (!matrixInit) {
    container.replaceChildren(...Array.from({length: 128}, (_, i) => {
      const cell = document.createElement('div');
      cell.className = 'm-cell';
      cell.id = `m-c-${i}`;
      return cell;
    }));
    matrixInit = true;
  }

  const frame = run.live_frame || 0;
  const isRecurrent = !!run.recurrent;
  const tag = $('matrix-entropy');
  if (tag) {
    tag.textContent = isRecurrent ? 'LSTM 128h' : 'FEEDFORWARD';
    tag.style.color = isRecurrent ? '#b3a6ff' : 'var(--dim)';
  }

  const stickActive = parsed.stick.mag > 0.1;
  const btnActive = parsed.buttons.size > 0;
  const trigActive = parsed.trigger > 0;
  const t = Date.now() / 400;

  for (let i = 0; i < 128; i++) {
    let target = 0.05;
    if (isRecurrent) {
      if (i < 32) {
        // Layer 0: Reflex (0-31) - high responsiveness to sticks and inputs
        const freq = 1.8 + (i % 8) * 0.3;
        const wave = Math.sin(t * freq + i) * 0.5 + 0.5;
        if (stickActive || btnActive || trigActive) {
          target = 0.4 + wave * 0.55;
        } else {
          target = 0.08 + wave * 0.2;
        }
      } else if (i < 80) {
        // Layer 1: Tactical Read (32-79) - tracking spacing and sequences
        const wave = Math.sin(t * 0.9 + i * 0.15) * 0.5 + 0.5;
        const depth = Math.min(1.0, frame / 300);
        target = 0.1 + wave * (0.3 + depth * 0.5);
      } else {
        // Layer 2: Macro & Habits (80-127) - deep persistent memory
        const depth = Math.min(1.0, frame / 600);
        const slow = Math.sin(t * 0.3 + i * 0.08) * 0.5 + 0.5;
        target = 0.12 + slow * (0.2 + depth * 0.65);
      }
    } else {
      target = 0.05 + Math.random() * 0.08;
    }

    cellState[i] += (target - cellState[i]) * 0.35;
    const v = cellState[i];
    const el = document.getElementById(`m-c-${i}`);
    if (el) {
      if (i < 32) {
        el.style.background = `rgba(79, 195, 247, ${v.toFixed(2)})`;
        el.style.boxShadow = v > 0.7 ? '0 0 3px #4fc3f7' : 'none';
      } else if (i < 80) {
        el.style.background = `rgba(139, 123, 255, ${v.toFixed(2)})`;
        el.style.boxShadow = v > 0.7 ? '0 0 3px #8b7bff' : 'none';
      } else {
        el.style.background = `rgba(61, 220, 151, ${v.toFixed(2)})`;
        el.style.boxShadow = v > 0.7 ? '0 0 3px #3ddc97' : 'none';
      }
    }
  }
}

function drawMemory(run, slots) {
  const active = (slots || []).filter(s => !s.stale && !s.failed);
  const frame = run.live_frame || (active.length ? Math.max(...active.map(s => s.frame || 0)) : 0);
  const sec = frame / 60;

  let regime, label, color;
  if (run.recurrent) {
    if (frame < 60) { regime = 'reflex'; label = 'REFLEX'; color = '#4fc3f7'; }
    else if (frame < 180) { regime = 'read'; label = 'TACTICAL READ'; color = '#8b7bff'; }
    else if (frame < 600) { regime = 'tech'; label = 'HABIT TRACKING'; color = '#ba68c8'; }
    else { regime = 'deep'; label = 'DEEP CONTEXT'; color = '#3ddc97'; }

    const pct = Math.min(100, Math.max(3, (frame / 600) * 100));
    $('mem-val').innerHTML = `${sec.toFixed(1)}s <small style="font-size:8px;opacity:.75;font-weight:400">(${fmt(frame)}f)</small>`;
    const fill = $('mem-fill');
    fill.style.width = `${pct}%`;
    if (frame >= 600) fill.classList.add('deep'); else fill.classList.remove('deep');
    const reg = $('mem-regime');
    reg.className = 'mem-pill ' + regime;
    reg.textContent = label;
    $('mem-detail').textContent = `${fmt(frame)}f context`;
  } else {
    regime = 'feedforward';
    label = 'FEEDFORWARD';
    color = '#7c8aa5';
    $('mem-val').innerHTML = `2 frames <small style="font-size:8px;opacity:.75;font-weight:400">(~33ms)</small>`;
    const fill = $('mem-fill');
    fill.style.width = '6%';
    fill.classList.remove('deep');
    const reg = $('mem-regime');
    reg.className = 'mem-pill feedforward';
    reg.textContent = label;
    $('mem-detail').textContent = `Match: ${sec.toFixed(1)}s (${fmt(frame)}f)`;
  }

  // Real-time oscilloscope memory wave
  const base = run.recurrent ? Math.min(0.82, 0.18 + (frame / 700) * 0.58) : 0.18;
  const t = Date.now() / 700;
  const wobble = run.recurrent ? Math.sin(t * 3.1) * 0.12 + Math.cos(t * 6.3) * 0.06 : Math.sin(t * 1.5) * 0.03;
  const waveVal = Math.max(0.08, Math.min(0.92, base + wobble));
  waveHistory.push(waveVal);
  if (waveHistory.length > 24) waveHistory.shift();

  const svg = $('mem-wave');
  if (svg) {
    const N = waveHistory.length;
    const pts = waveHistory.map((v, i) => {
      const x = (i / Math.max(1, N - 1)) * 200;
      const y = 22 - (v * 16 + 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const polyPts = [`0,22`, ...pts, `200,22`].join(' ');
    svg.innerHTML = `<defs><linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity="0.32"/><stop offset="100%" stop-color="${color}" stop-opacity="0.0"/></linearGradient></defs><polygon fill="url(#memGrad)" points="${polyPts}"/><polyline fill="none" stroke="${color}" stroke-width="1.2" stroke-linejoin="round" points="${pts.join(' ')}"/>`;
  }
}

function drawSlots(slots) {
  const bad = slots.filter(s => s.stale || s.failed).length;
  $('slot-summary').textContent = `${slots.length - bad}/${slots.length} live`;
  $('slot-grid').replaceChildren(...slots.map(s => {
    const el = document.createElement('div');
    const age = s.age_seconds;
    el.className = 'slot ' + (s.failed ? 'dead' : s.stale ? 'stale' : age > 8 ? 'slow' : 'ok');
    const opponent = short(s.players?.['2']?.character);
    const sFrame = s.frame || 0;
    const sSec = (sFrame / 60).toFixed(1);
    const bits = [];
    bits.push(`<span class="mem-tag">🧠 ${sSec}s</span>`);
    if (s.recoveries) bits.push(`${s.recoveries} rec`);
    bits.push(age == null ? 'no data' : `${age.toFixed(0)}s`);
    el.innerHTML = `<span class="id">${String(s.index).padStart(2, '0')}</span>
      <span class="opp">${opponent}</span><span class="meta">${bits.join(' · ')}</span>
      <div class="slot-mem-bar" style="width:${Math.min(100, (sFrame / 600) * 100).toFixed(1)}%"></div>`;
    return el;
  }));
}

function drawFooter(run, data) {
  $('identity').innerHTML = `<b>${run.character}</b> · ${run.id} · ${run.phase || run.status}`;
  const arch = $('arch');
  if (arch) {
    if (run.recurrent) {
      arch.className = 'recurrent';
      arch.innerHTML = `🧠 <b>LSTM</b> ${run.lstm_hidden_size || 128}h`;
    } else {
      arch.className = 'feedforward';
      arch.innerHTML = `⚡ <b>MLP</b> 2-frame`;
    }
  }
  const session = Math.max(0, run.steps - run.initial);
  const rate = run.elapsed ? session / run.elapsed : 0;
  $('throughput').innerHTML = `<b>${fmt(rate)}</b> decisions/s`;
  const budget = run.budget ? `<b>${fmt(session)}</b> / ${fmt(run.budget)}` : `<b>${fmt(session)}</b>`;
  const left = run.budget && rate > 0.5 ? ` · ${((run.budget - session)/rate/3600).toFixed(1)}h left` : '';
  $('progress').innerHTML = budget + left;
  $('clock').innerHTML = run.live
    ? `<b>${run.wins ?? 0}</b>/${run.matches ?? 0} matches`
    : `<b>${run.status}</b>`;
}

poll();
setInterval(poll, 300);
addEventListener('resize', () => previous && drawChart(previous.history));


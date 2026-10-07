'use strict';
const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? '–' : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (n) => (n == null ? '–' : `${Math.round(n * 100)}%`);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
const toast = (msg) => { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 3500); };

const S = { run: null, status: null, games: [], metrics: [], runs: [] };

// ------------------------------------------------------------------ charts
function lineChart(el, series, opts = {}) {
  const w = el.clientWidth || 600, h = el.clientHeight || 220, pad = { l: 34, r: opts.right ? 34 : 8, t: 8, b: 20 };
  const all = series.flatMap((s) => s.points);
  if (!all.length) { el.innerHTML = '<div class="empty">Waiting for data…</div>'; return; }
  const xs = all.map((p) => p[0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const range = (axis) => {
    const pts = series.filter((s) => (s.axis || 'left') === axis).flatMap((s) => s.points.map((p) => p[1]));
    let lo = opts[axis + 'Min'] ?? Math.min(...pts), hi = opts[axis + 'Max'] ?? Math.max(...pts);
    if (!(hi > lo)) { hi = lo + 1; }
    return [lo, hi];
  };
  const L = range('left'), R = opts.right ? range('right') : L;
  const auto = ([lo, hi]) => { const r = hi - lo; const d = r < 0.05 ? 4 : r < 0.5 ? 3 : r < 5 ? 2 : r < 50 ? 1 : 0; return (v) => fmt(v, d); };
  const lf = opts.leftFmt || auto(L), rf = opts.rightFmt || auto(R);
  const X = (x) => pad.l + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (w - pad.l - pad.r);
  const Y = (y, ax) => { const [lo, hi] = ax === 'right' ? R : L; return pad.t + (1 - (y - lo) / (hi - lo)) * (h - pad.t - pad.b); };
  let svg = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">`;
  for (let i = 0; i <= 4; i++) {
    const yv = L[0] + (L[1] - L[0]) * i / 4, y = Y(yv, 'left');
    svg += `<line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${y}" y2="${y}" stroke-dasharray="${i ? '2 4' : ''}"/>`;
    svg += `<text x="${pad.l - 4}" y="${y + 3}" text-anchor="end">${lf(yv)}</text>`;
    if (opts.right) { const rv = R[0] + (R[1] - R[0]) * i / 4; svg += `<text x="${w - pad.r + 4}" y="${y + 3}">${rf(rv)}</text>`; }
  }
  svg += `<text x="${pad.l}" y="${h - 4}">${esc(opts.xFmt ? opts.xFmt(x0) : fmt(x0))}</text><text x="${w - pad.r}" y="${h - 4}" text-anchor="end">${esc(opts.xFmt ? opts.xFmt(x1) : fmt(x1))}</text>`;
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1], s.axis || 'left').toFixed(1)}`).join('');
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.8}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} stroke-linejoin="round"/>`;
  }
  svg += '</svg>';
  el.innerHTML = svg + `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div>`;
}

function rolling(rows, n, fn) {
  const out = [];
  for (let i = 0; i < rows.length; i++) {
    const win = rows.slice(Math.max(0, i - n + 1), i + 1);
    if (win.length >= Math.min(n, 10)) out.push([i + 1, fn(win)]);
  }
  return out;
}

// ------------------------------------------------------------------ render
function renderKpis(st) {
  const m = st.metrics || {};
  const k = [
    ['Game time played', `${fmt(st.game_hours_played, 1)} h${st.world_frames_estimated ? '*' : ''}`,
      `${fmt(st.experience_hours, 1)} h of experience (self-play counts both players)${st.world_frames_estimated ? ' · *estimated from the game log' : ''}`],
    ['Training speed', st.realtime_multiple == null ? '–' : `${fmt(st.realtime_multiple, 1)}× real time`,
      st.realtime_multiple == null ? 'not recorded for this older run' : `${fmt(st.world_frames_per_second)} game frames/s · ${fmt(st.frames_per_second)} experience/s`],
    ['Emulators', `${fmt(st.emulator_fps)} fps`, `${(st.workers || []).filter((w) => w.alive).length} of ${(st.workers || []).length} running`],
    ['CPU frontier', `Level ${st.ladder?.frontier ?? '–'}`, `${fmt(st.games)} games played`],
    ['Policy version', fmt(st.version), st.critic_warmup_left > 0 ? `value warm-up · ${fmt(st.critic_warmup_left)} updates left`
      : `update ${fmt(m.seconds, 2)}s · lag ${fmt(m.mean_lag, 1)} · ${esc(st.device)}`],
    ['Run time', dur(st.elapsed), st.state === 'running' ? `updated ${fmt(st.age, 0)}s ago` : st.state],
  ];
  const sw = st.stall;
  if (sw) {
    k.splice(4, 0, ['Time offstage', sw.offstage_share == null ? '–' : pct(sw.offstage_share),
      `${sw.games < sw.window ? `${sw.games}/${sw.window} games so far` : `last ${sw.window} games`} · games ${fmt(sw.game_minutes, 1)} min · warns at ${pct(sw.warn)}`,
      sw.level === 'ok' ? '' : `lvl-${sw.level}`]);
  }
  $('#kpis').innerHTML = k.map(([l, v, s, c]) => `<div class="kpi ${c || ''}"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
}

function renderAlert(st) {
  const el = $('#alert'), sw = st.stall || {};
  const ok = sw.last_ok_checkpoint;
  let html = '', lvl = '';
  if (sw.level === 'warn' || sw.level === 'alarm') {
    lvl = sw.level;
    html = `<div class="msg"><b>${sw.level === 'alarm' ? 'Stalling alarm' : 'Possible stalling'}</b>Puff spent ${pct(sw.offstage_share)} of the last ${sw.games} games offstage
      (healthy runs stay under ${pct(sw.warn)}); games average ${fmt(sw.game_minutes, 1)} min. She may be hiding to put off losing stocks.
      ${ok ? `<span class="muted">Last checkpoint before this: ${esc(ok.split('/').pop())}</span>` : ''}</div>
      ${ok ? `<button class="small" data-eval="${esc(ok)}">Evaluate that checkpoint</button>` : ''}`;
  } else if (st.critic_warmup_left > 0 && ['running', 'starting'].includes(st.state)) {
    lvl = 'info';
    html = `<div class="msg"><b>Value warm-up</b>The planning horizon changed, so for ${fmt(st.critic_warmup_left)} more updates only the value estimate
      trains; Puff plays exactly as the starting weights did until it has adjusted.</div>`;
  }
  el.hidden = !html;
  el.className = `alert ${lvl ? `lvl-${lvl}` : ''}`;
  if (el._html !== html) { el.innerHTML = html; el._html = html; }
}
const dur = (s) => { s = Math.round(s || 0); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`; };

function renderWorkers(st) {
  const ws = st.workers || [];
  $('#workers').innerHTML = ws.map((w) => {
    const phase = !w.alive ? (w.phase === 'restarting' ? 'restarting' : 'dead') : (w.phase || 'starting');
    return `<div class="worker ${phase}"><div class="top"><span>#${w.worker}${w.visible ? ' 👁' : ''}</span><span class="fps">${fmt(w.fps)} fps</span></div>
      <div class="setup" title="${esc(w.setup)}">${esc(w.setup || phase)}</div>
      <div class="muted">${esc(phase)} · ${fmt(w.games)} games · ${fmt(w.errors)} err${w.restarts ? ` · ${w.restarts} restarts` : ''}${w.phillip_ms ? ` · Phillip ${fmt(w.phillip_ms, 1)} ms/f` : ''}</div></div>`;
  }).join('') || '<div class="empty">No emulators</div>';
  $('#fleet-note').textContent = `${(st.config?.act_every ?? 3)} frames per decision · ${fmt(st.dropped_stale)} stale fragments dropped`;
}

function renderLadder(st) {
  const lv = st.ladder?.levels || {}, f = st.ladder?.frontier;
  let html = '';
  for (let l = 1; l <= 9; l++) {
    const r = lv[l] || {};
    html += `<div class="rung ${l === f ? 'front' : ''}"><b>${l}</b><span class="wr">${r.games ? pct(r.win_rate) : '·'}</span><div class="muted">${r.games || 0}g</div></div>`;
  }
  $('#ladder').innerHTML = html;
  const c = st.config || {};
  $('#ladder-note').textContent = `promote at ${pct(c.promote_at)} over ${c.ladder_window} games · demote at ${pct(c.demote_at)} · ×N = extra focus on losing matchups`;
  const bo = st.summary?.by_opponent || {}, sp = st.summary?.selfplay || {};
  const mu = st.ladder?.matchups || {};
  $('#by-opponent').innerHTML = Object.entries(bo).sort((a, b) => b[1].games - a[1].games)
    .map(([k, v]) => `<span class="chip" title="training focus ×${mu[k]?.focus ?? '–'} (more games vs matchups it is losing)">${esc(k)} <b>${pct(v.win_rate)}</b> <span class="muted">${v.games}g${mu[k] ? ` · ×${mu[k].focus}` : ''}</span></span>`).join('') +
    Object.entries(sp).map(([k, v]) => `<span class="chip">${esc(k)} <b>${pct(v.win_rate)}</b> <span class="muted">${v.games}g</span></span>`).join('') +
    Object.entries(st.summary?.by_phillip || {}).map(([k, v]) => `<span class="chip" title="vs Slippi-AI Phillip">🤖 ${esc(k)} <b>${pct(v.win_rate)}</b> <span class="muted">${v.games}g</span></span>`).join('');
}

const GROUPS = [
  { name: 'Movement & drift', re: /^(noop|left|right|walk|up|down)/, color: '#6aa9ff' },
  { name: 'Jumps', re: /^(short_hop|jump)/, color: '#a88bff' },
  { name: 'Aerials', re: /^(c_|sh_)/, color: '#f49ac1' },
  { name: 'Ground attacks', re: /^a(_|$)/, color: '#f5b642' },
  { name: 'Rest', re: /^rest/, color: '#4fd08a' },
  { name: 'Pound', re: /^pound/, color: '#7fe0c3' },
  { name: 'Shield, roll, grab', re: /^(shield|roll|spot|grab)/, color: '#9aa1b5' },
  { name: 'Tech direction', re: /^tech/, color: '#ffc4dd' },
];
const groupOf = (n) => GROUPS.find((g) => g.re.test(n)) || { name: 'Other', color: '#888' };
const moveName = (n) => n.replaceAll('_', ' ').replace(/^c /, 'C-stick ').replace(/^sh /, 'short-hop ');
function renderBrain(st) {
  const share = st.summary?.action_share || {};
  const rows = Object.entries(share);
  if (!rows.length) { $('#brain').innerHTML = '<div class="empty">No games yet.</div>'; return; }
  const max = Math.max(...rows.map((r) => r[1]));
  const s = st.summary;
  $('#brain').innerHTML = `<div class="chips" style="margin-bottom:8px">
      <span class="chip">Rests/game <b>${fmt(s.rests_per_game, 2)}</b></span>
      <span class="chip">Rest hit rate <b>${pct(s.rest_hit_rate)}</b></span>
      <span class="chip">Self-destruct share <b>${pct(s.self_destruct_share)}</b></span>
      <span class="chip">Damage ratio <b>${fmt(s.damage_ratio, 2)}</b></span>
      <span class="chip" title="knockdowns teched">Tech rate <b>${pct(s.tech_rate)}</b></span>
      <span class="chip" title="average frames of aerial landing lag; auto L-cancel halves it">Aerial landing lag <b>${fmt(s.aerial_landing_lag, 1)}f</b></span>
      <span class="chip" title="should stay 0: tech clicks are only legal over the stage">Offstage air-dodges <b>${fmt(s.offstage_airdodges)}</b></span></div>` +
    rows.map(([n, v]) => {
      const col = groupOf(n).color;
      return `<div class="bar"><span>${esc(n.replaceAll('_', ' '))}</span><div class="track"><div class="fill" style="width:${(v / max) * 100}%;background:${col}"></div></div><span class="n">${(v * 100).toFixed(1)}%</span></div>`;
    }).join('');
}

function renderGames() {
  const rows = S.games.slice(-60).reverse();
  $('#games tbody').innerHTML = rows.map((g) => `<tr>
    <td>${esc(g.opponent_kind === 'cpu' ? g.opponent : g.opponent_kind === 'phillip' ? `🤖 Phillip ${g.opponent}`
      : g.opponent_kind === 'mirror' ? 'Puff (mirror)' : `Puff (${g.snapshot})`)}</td>
    <td>${g.opponent_kind === 'cpu' ? g.cpu_level : '–'}</td>
    <td class="${g.result}">${g.result}${g.timeout ? ' (time)' : ''}</td>
    <td>${g.stocks_left}–${g.opp_stocks_left}</td>
    <td>${fmt(g.damage_dealt)} / ${fmt(g.damage_received)}</td>
    <td>${g.self_destructs}</td><td>${g.rest_hits}/${g.rests}</td></tr>`).join('');
  const done = S.games.filter(completed);   // capped / interrupted games have no result
  const cpu = done.filter((g) => g.opponent_kind === 'cpu');
  lineChart($('#chart-win'), [
    { name: 'win rate', color: '#4fd08a', points: rolling(cpu, 60, (w) => w.filter((g) => g.result === 'win').length / w.length) },
    { name: 'CPU level (right)', color: '#f49ac1', axis: 'right', points: rolling(cpu, 60, (w) => w.reduce((a, g) => a + g.cpu_level, 0) / w.length) },
  ], { leftMin: 0, leftMax: 1, leftFmt: pct, right: true, rightMin: 1, rightMax: 9, xFmt: (x) => `game ${fmt(x)}` });
  lineChart($('#chart-play'), [
    { name: 'damage ratio (dealt/taken)', color: '#f5b642', points: rolling(done, 60, (w) => w.reduce((a, g) => a + g.damage_dealt, 0) / Math.max(1, w.reduce((a, g) => a + g.damage_received, 0))) },
    { name: 'self-destruct share (right)', color: '#f0616d', axis: 'right', points: rolling(done, 60, (w) => { const l = w.reduce((a, g) => a + g.stocks_lost, 0); return l ? w.reduce((a, g) => a + g.self_destructs, 0) / l : 0; }) },
    { name: 'rest hits/game (right)', color: '#4fd08a', axis: 'right', points: rolling(done, 60, (w) => w.reduce((a, g) => a + g.rest_hits, 0) / w.length) },
  ], { leftMin: 0, right: true, rightMin: 0, rightMax: 1, rightFmt: (v) => fmt(v, 2), xFmt: (x) => `game ${fmt(x)}` });
  const cfg = S.status?.config || {}, warn = cfg.stall_offstage_warn ?? 0.5, alarm = cfg.stall_offstage_alarm ?? 0.6;
  const n = cfg.stall_window ?? 100;
  const off = rolling(done, n, (w) => w.reduce((a, g) => a + (g.offstage_frames || 0), 0) / Math.max(1, w.reduce((a, g) => a + g.frames, 0)));
  const line = (y) => (off.length ? [[off[0][0], y], [off[off.length - 1][0], y]] : []);
  lineChart($('#chart-stall'), [
    { name: 'time offstage', color: '#a88bff', points: off },
    { name: 'warning line', color: '#f5b642', dash: '4 4', width: 1, points: line(warn) },
    { name: 'alarm line', color: '#f0616d', dash: '4 4', width: 1, points: line(alarm) },
    { name: 'game length, min (right)', color: '#6aa9ff', axis: 'right', points: rolling(done, n, (w) => w.reduce((a, g) => a + g.frames, 0) / w.length / 3600) },
  ], { leftMin: 0, leftMax: 1, leftFmt: pct, right: true, rightMin: 0, rightFmt: (v) => fmt(v, 1), xFmt: (x) => `game ${fmt(x)}` });
}

function renderMetrics() {
  const m = S.metrics;
  lineChart($('#chart-learn'), [
    { name: 'entropy', color: '#a88bff', points: m.map((r) => [r.frames_trained / 1e6, r.entropy]) },
    { name: 'value loss (right)', color: '#6aa9ff', axis: 'right', points: m.map((r) => [r.frames_trained / 1e6, r.value_loss]) },
  ], { leftMin: 0, right: true, rightMin: 0, xFmt: (x) => `${fmt(x, 1)}M frames` });
  const speed = [];
  for (let i = 1; i < m.length; i++) {
    const dt = m[i].time - m[i - 1].time;
    if (dt > 0) speed.push([m[i].frames_trained / 1e6, (m[i].frames_trained - m[i - 1].frames_trained) / dt]);
  }
  const smooth = speed.map((p, i) => [p[0], speed.slice(Math.max(0, i - 9), i + 1).reduce((a, q) => a + q[1], 0) / Math.min(i + 1, 10)]);
  lineChart($('#chart-speed'), [{ name: 'frames/s (10-update mean)', color: '#f49ac1', points: smooth }], { leftMin: 0, leftFmt: (v) => fmt(v), xFmt: (x) => `${fmt(x, 1)}M frames` });
  const last = m[m.length - 1];
  $('#learner-note').textContent = last ? `KL ${fmt(last.approx_kl, 4)} · clip ${pct(last.clip_frac)} · entropy coef ${fmt(last.entropy_coef, 4)}` : '';
}

function renderEvents(st) {
  $('#events').innerHTML = (st.errors || []).slice().reverse().map((e) =>
    `<li><span class="t">${new Date((e.time || 0) * 1000).toLocaleTimeString()}</span>${e.worker >= 0 ? `#${e.worker} ` : ''}${esc(e.error)}</li>`).join('') || '<li class="muted">Quiet.</li>';
}

async function renderCheckpoints() {
  if (!S.run) return;
  const cks = await api(`/api/runs/${S.run}/checkpoints`).catch(() => []);
  S.checkpoints = cks;
  $('#checkpoints tbody').innerHTML = cks.slice(0, 40).map((c) => `<tr><td>${c.champion ? '👑 ' : ''}${esc(c.name.replace('.pt', ''))}</td><td>${fmt(c.frames_m, 2)}M</td>
    <td><div class="actions"><button class="small" data-eval="${esc(c.path)}">Evaluate</button><button class="small" data-watch="${esc(c.path)}">Watch</button><button class="small" data-crown="${esc(c.path)}">Crown</button></div></td></tr>`).join('')
    || '<tr><td colspan="3" class="muted">First checkpoint arrives after the checkpoint interval.</td></tr>';
}

async function renderEvals() {
  const evs = await api('/api/evals').catch(() => []);
  $('#evals').innerHTML = evs.slice(0, 8).map((e) => {
    const o = e.overall || {};
    const per = Object.entries(e.by_opponent || {}).map(([k, v]) => `<span class="chip">${esc(k)} <b>${pct(v.win_rate)}</b> <span class="muted">${v.wins}/${v.games}</span></span>`).join('');
    const short = Object.entries(e.requested || {}).filter(([k, v]) => (e.counted || {})[k] < v).map(([k, v]) => `${k} ${(e.counted || {})[k]}/${v}`);
    const state = e.state === 'complete' ? 'complete' : `${e.state}${short.length ? ' · missing ' + short.join(', ') : ''}`;
    return `<div class="eval"><div class="head"><span>${esc((e.checkpoint || '').split('/').slice(-3).join('/'))}</span><span class="${e.state === 'complete' || e.state === 'finished' ? 'muted' : 'warn'}">${esc(state)} · ${e.kind === 'phillip' ? 'vs Phillip' : `CPU ${e.level}`}</span></div>
      <div><span class="big">${pct(o.win_rate)}</span> <span class="muted">${o.wins ?? 0}/${o.games ?? 0} · 95% CI ${pct(o.win_rate_95ci?.[0])}–${pct(o.win_rate_95ci?.[1])} · SD share ${pct(o.self_destruct_share)} · stocks ${fmt(o.stocks_taken_per_game, 2)}/${fmt(o.stocks_lost_per_game, 2)}</span></div>
      <div class="chips" style="margin-top:6px">${per}</div></div>`;
  }).join('') || '<div class="muted">No evaluations yet.</div>';
  const champs = await api('/api/champions').catch(() => []);
  $('#champions').innerHTML = champs.length ? champs.map((c) => `<div>👑 ${esc(c.name)} <span class="${c.without_evaluation ? 'warn' : 'muted'}">${c.without_evaluation ? 'no evaluation' : c.evaluation ? `${pct(c.evaluation.overall?.win_rate)} vs CPU ${c.evaluation.level}` : ''}</span></div>`).join('') : 'None yet.';
}

// ------------------------------------------------------------------ report tab
const NAMES = { CPTFALCON: 'Falcon', JIGGLYPUFF: 'Puff', DOC: 'Dr. Mario', GAMEANDWATCH: 'Game & Watch' };
const charName = (c) => NAMES[c] || (c ? c[0] + c.slice(1).toLowerCase() : '–');
const GREEN = 'rgba(79,208,138,A)', RED = 'rgba(240,97,109,A)', AMBER = 'rgba(245,182,66,A)';
const tint = (v, lo, hi, color) => {
  if (v == null) return '';
  const a = Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * 0.4;
  return a > 0.01 ? ` style="background:${color.replace('A', a.toFixed(2))}"` : '';
};
const diverge = (v, mid, span) => (v == null ? '' : v >= mid ? tint(v, mid, mid + span, GREEN) : tint(mid - v, 0, span, RED));
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const ago = (t) => { if (!t) return '–'; const s = Date.now() / 1000 - t; return s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${fmt(s / 3600, 1)} h ago` : `${fmt(s / 86400, 1)} days ago`; };
let reportHours = 12;

async function renderReport() {
  if (!S.run) return;
  const r = await api(`/api/runs/${S.run}/report?hours=${reportHours}`).catch(() => null);
  if (!r) return;
  $('#report-headline').innerHTML = r.headline.map((l) => `<li>${esc(l)}</li>`).join('');
  const w = r.window;
  $('#report-note').textContent = w ? `${new Date(w.start * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' })} to ${clock(w.end)} · colors: green good, red bad` : '';
  const warn = r.warn ?? 0.5;
  $('#report-hours tbody').innerHTML = (r.hours || []).map((h) => `<tr>
    <td>${esc(h.label)}</td><td class="num">${fmt(h.games)}</td>
    <td class="heat"${diverge(h.cpu9_rate, 0.5, 0.4)}>${h.cpu9_games ? `${pct(h.cpu9_rate)} <span class="muted">${h.cpu9_wins}/${h.cpu9_games}</span>` : '–'}</td>
    <td class="heat"${diverge(h.cpu_rate, 0.5, 0.4)}>${h.cpu_games ? `${pct(h.cpu_rate)} <span class="muted">lvl ${fmt(h.cpu_level, 1)}</span>` : '–'}</td>
    <td class="heat"${h.phillip_wins ? tint(1, 0, 1, GREEN) : ''}>${h.phillip_games ? `${h.phillip_wins}/${h.phillip_games}` : '–'}</td>
    <td class="heat"${tint(h.phillip_taken, 1, 3, GREEN)}>${fmt(h.phillip_taken, 2)}</td>
    <td class="heat"${tint(h.tech_rate, 0.2, 0.8, GREEN)}>${pct(h.tech_rate)}</td>
    <td class="heat"${h.offstage >= warn ? tint(h.offstage, warn - 0.1, warn + 0.15, RED) : h.offstage >= warn - 0.08 ? tint(0.5, 0, 1, AMBER) : ''}>${pct(h.offstage)}</td>
    <td class="num">${fmt(h.minutes, 1)}</td>
    <td class="heat"${diverge(h.damage_ratio, 1.0, 0.5)}>${fmt(h.damage_ratio, 2)}</td>
    <td class="num"${tint(h.sd_per_game, 0.05, 0.5, RED)}>${fmt(h.sd_per_game, 2)}</td>
    <td class="muted">${h.versions ? `v${h.versions[0]}–${h.versions[1]}` : ''}</td></tr>`).join('')
    || '<tr><td colspan="12" class="muted">No games in this window.</td></tr>';
  const flaps = (r.events || []).filter((e) => /^(Top level beaten|CPU \d no longer beaten)/.test(e.error));
  const rest = (r.events || []).filter((e) => !flaps.includes(e));
  const items = rest.map((e) => [e.time, esc(e.error)]);
  if (flaps.length > 2) {
    items.push([flaps[flaps.length - 1].time, `Ladder: the top level flipped between beaten and not beaten ${flaps.length} times (last: ${esc(flaps[flaps.length - 1].error)})`]);
  } else flaps.forEach((e) => items.push([e.time, esc(e.error)]));
  for (const c of r.scorecards || []) {
    if (!rest.some((e) => e.error.startsWith('Scorecard') && e.error.includes(c.checkpoint_name)) && c.state === 'complete') {
      items.push([c.finished || c.started, `Scorecard ${esc(c.checkpoint_name)}: ${c.wins}/${c.games} vs CPU 9`]);
    }
  }
  items.sort((a, b) => b[0] - a[0]);
  $('#report-events').innerHTML = items.map(([t, h]) => `<li><span class="t">${clock(t)}</span>${h}</li>`).join('')
    || `<li class="muted">${r.event_log ? 'Nothing notable.' : 'This run predates the event log.'}</li>`;
  const wins = (r.phillip_wins || []).slice().reverse();
  $('#report-wins-note').textContent = wins.length ? `${wins.length} in this window` : '';
  $('#report-wins').innerHTML = wins.map((g) => `<span class="chip win">${clock(g.time)} · ${esc(charName(g.opponent))} <span class="muted">${g.stocks_left} stock${g.stocks_left === 1 ? '' : 's'} left · v${g.version}</span></span>`).join('')
    || '<span class="muted">None in this window.</span>';
}

// ------------------------------------------------------------------ phillip tab
async function renderPhillip() {
  if (!S.run) return;
  const p = await api(`/api/runs/${S.run}/phillip`).catch(() => null);
  if (!p || !p.overall) {
    $('#ph-kpis').innerHTML = '<div class="kpi"><div class="v">No Phillip games in this run</div><div class="s">Set “Phillip opponents” when starting a run.</div></div>';
    $('#chart-phillip').innerHTML = ''; $('#ph-chars tbody').innerHTML = ''; $('#ph-wins').innerHTML = '';
    return;
  }
  const o = p.overall, f = p.first100, l = p.last100;
  const chars = Object.entries(p.characters).filter(([, v]) => v.games >= 10);
  const best = chars.sort((a, b) => b[1].taken - a[1].taken)[0];
  const k = [
    ['Record', `${o.wins}–${o.games - o.wins}`, `${pct(o.win_rate)} of ${fmt(o.games)} games`],
    ['Last 100 games', `${l.wins} win${l.wins === 1 ? '' : 's'}`, o.games > 100 ? `first 100: ${f.wins}` : 'fewer than 100 so far'],
    ['Stocks taken per game', fmt(l.taken, 2), o.games > 100 ? `last 100 · first 100: ${fmt(f.taken, 2)} · 4 = a win` : '4 = a win'],
    ['Stocks lost per game', fmt(l.lost, 2), o.games > 100 ? `last 100 · first 100: ${fmt(f.lost, 2)}` : ''],
    ['Damage ratio', fmt(l.damage_ratio, 2), o.games > 100 ? `last 100 · first 100: ${fmt(f.damage_ratio, 2)}` : 'dealt / taken'],
    ['Best matchup', best ? charName(best[0]) : '–', best ? `${fmt(best[1].taken, 2)} stocks taken per game` : 'needs 10 games per character'],
  ];
  $('#ph-kpis').innerHTML = k.map(([a, v, s]) => `<div class="kpi"><div class="l">${a}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
  const gs = p.games;
  lineChart($('#chart-phillip'), [
    { name: 'stocks taken / game', color: '#4fd08a', points: rolling(gs, 50, (w) => w.reduce((a, g) => a + g.stocks_taken, 0) / w.length) },
    { name: 'stocks lost / game', color: '#f0616d', points: rolling(gs, 50, (w) => w.reduce((a, g) => a + g.stocks_lost, 0) / w.length) },
    { name: 'win rate (right)', color: '#f5b642', axis: 'right', points: rolling(gs, 50, (w) => w.filter((g) => g.result === 'win').length / w.length) },
  ], { leftMin: 0, leftMax: 4, leftFmt: (v) => fmt(v, 1), right: true, rightMin: 0, rightFmt: pct, xFmt: (x) => `Phillip game ${fmt(x)}` });
  $('#ph-chars tbody').innerHTML = Object.entries(p.characters).map(([c, v]) => `<tr><td>${esc(charName(c))}</td><td class="num">${v.games}</td>
    <td class="num"${v.wins ? tint(1, 0, 1, GREEN) : ''}>${v.wins}</td><td class="num"${tint(v.taken, 1, 3, GREEN)}>${fmt(v.taken, 2)}</td>
    <td class="num">${fmt(v.lost, 2)}</td><td class="num"${diverge(v.damage_ratio, 1.0, 0.6)}>${fmt(v.damage_ratio, 2)}</td><td class="muted">${ago(v.last_win)}</td></tr>`).join('');
  $('#ph-wins').innerHTML = p.wins.slice().reverse().map((g) => `<span class="chip win">${new Date(g.time * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' })} · ${esc(charName(g.opponent))} <span class="muted">${g.stocks_left} left · v${g.version}</span></span>`).join('')
    || '<span class="muted">No wins yet. Stocks taken per game is the number to watch until then.</span>';
}

// ------------------------------------------------------------------ progress tab
const RUN_COLORS = ['#f49ac1', '#6aa9ff', '#4fd08a', '#f5b642', '#a88bff', '#7fe0c3'];
function scoreChart(el, cards, colorOf) {
  const w = el.clientWidth || 600, h = el.clientHeight || 220, pad = { l: 40, r: 20, t: 18, b: 34 };
  if (!cards.length) { el.innerHTML = '<div class="empty">No scorecards yet. The first automatic one runs 3 hours into a run.</div>'; return; }
  const n = cards.length;
  const X = (i) => pad.l + (n === 1 ? (w - pad.l - pad.r) / 2 : (i * (w - pad.l - pad.r)) / (n - 1));
  const Y = (v) => pad.t + (1 - v) * (h - pad.t - pad.b);
  let svg = `<svg viewBox="0 0 ${w} ${h}">`;
  for (let i = 0; i <= 4; i++) {
    const v = i / 4, y = Y(v);
    svg += `<line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${y}" y2="${y}" stroke-dasharray="${i ? '2 4' : ''}"/><text x="${pad.l - 6}" y="${y + 3}" text-anchor="end">${pct(v)}</text>`;
  }
  const done = cards.map((c, i) => [c, i]).filter(([c]) => c.state === 'complete');
  if (done.length > 1) svg += `<path d="${done.map(([c, i], j) => `${j ? 'L' : 'M'}${X(i)},${Y(c.win_rate)}`).join('')}" fill="none" stroke="#8b90a0" stroke-dasharray="3 4" stroke-width="1"/>`;
  const every = Math.ceil(n / 12);
  cards.forEach((c, i) => {
    const x = X(i), col = colorOf(c.run);
    if (c.state === 'complete' && c.ci) {
      svg += `<line x1="${x}" x2="${x}" y1="${Y(c.ci[1])}" y2="${Y(c.ci[0])}" stroke="${col}" stroke-width="2" opacity=".55"/>`;
      svg += `<line x1="${x - 5}" x2="${x + 5}" y1="${Y(c.ci[1])}" y2="${Y(c.ci[1])}" stroke="${col}" opacity=".55"/><line x1="${x - 5}" x2="${x + 5}" y1="${Y(c.ci[0])}" y2="${Y(c.ci[0])}" stroke="${col}" opacity=".55"/>`;
      svg += `<circle cx="${x}" cy="${Y(c.win_rate)}" r="5" fill="${col}"/>`;
      svg += `<text x="${x + 8}" y="${Y(c.win_rate) + 4}" style="fill:var(--text)">${c.wins}/${c.games}</text>`;
    } else {
      const v = c.games ? c.wins / c.games : 0.5;
      svg += `<circle cx="${x}" cy="${Y(v)}" r="5" fill="none" stroke="${col}" stroke-width="2"/><text x="${x + 8}" y="${Y(v) + 4}">${esc(c.state)} ${c.games || 0}/60</text>`;
    }
    if (i % every === 0 || i === n - 1) {
      const d = new Date((c.started || 0) * 1000);
      svg += `<text x="${x}" y="${h - 18}" text-anchor="middle">${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</text><text x="${x}" y="${h - 6}" text-anchor="middle">v${c.version ?? '?'}</text>`;
    }
  });
  const runs = [...new Set(cards.map((c) => c.run))];
  el.innerHTML = svg + '</svg>' + `<div class="legend">${runs.map((r) => `<span><i style="background:${colorOf(r)}"></i>${esc(r || 'other')}</span>`).join('')}<span>bars: likely range (95%)</span></div>`;
}

function stackChart(el, buckets, layers) {
  const w = el.clientWidth || 600, h = el.clientHeight || 300, pad = { l: 38, r: 8, t: 8, b: 20 };
  if (!buckets.length) { el.innerHTML = '<div class="empty">Waiting for games…</div>'; return; }
  const n = buckets.length;
  const X = (i) => pad.l + (n === 1 ? 0 : (i * (w - pad.l - pad.r)) / (n - 1));
  const Y = (v) => pad.t + (1 - v) * (h - pad.t - pad.b);
  let svg = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">`;
  const base = new Array(n).fill(0);
  for (const L of layers) {
    const top = L.values.map((v, i) => base[i] + v);
    const d = top.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('') +
      base.map((v, i) => [i, v]).reverse().map(([i, v]) => `L${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('') + 'Z';
    svg += `<path d="${d}" fill="${L.color}" fill-opacity=".8" stroke="${L.color}" stroke-width=".5"><title>${esc(L.name)}</title></path>`;
    top.forEach((v, i) => { base[i] = v; });
  }
  for (let i = 0; i <= 4; i++) { const y = Y(i / 4); svg += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${y}" y2="${y}" stroke="rgba(15,17,23,.5)" stroke-dasharray="2 4"/><text x="${pad.l - 4}" y="${y + 3}" text-anchor="end">${pct(i / 4)}</text>`; }
  svg += `<text x="${pad.l}" y="${h - 4}">game ${fmt(buckets[0].first_game)}</text><text x="${w - pad.r}" y="${h - 4}" text-anchor="end">game ${fmt(buckets[n - 1].last_game)}</text>`;
  const now = layers.map((L) => L.values[n - 1]);
  el.innerHTML = svg + '</svg>' + `<div class="legend">${layers.map((L, j) => `<span><i style="background:${L.color}"></i>${esc(L.name)} ${pct(now[j])}</span>`).join('')}</div>`;
}

async function renderProgress() {
  const [cards, mv] = await Promise.all([api('/api/scorecards').catch(() => []), S.run ? api(`/api/runs/${S.run}/moves`).catch(() => null) : null]);
  const runs = [...new Set(cards.map((c) => c.run))];
  const colorOf = (r) => RUN_COLORS[runs.indexOf(r) % RUN_COLORS.length];
  scoreChart($('#chart-scorecards'), cards, colorOf);
  const sc = S.status?.scorecard;
  const next = sc?.running ? 'a scorecard is running now' : sc?.next ? `next automatic one in ${dur(Math.max(0, sc.next - Date.now() / 1000))}` : 'automatic scorecards start with the next run';
  $('#sc-note').textContent = `60 games vs level-9 Fox, Falco, Marth, Falcon, Peach, Puff · ${next}`;
  $('#sc-table tbody').innerHTML = cards.slice().reverse().map((c) => `<tr>
    <td>${new Date((c.started || 0) * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}${c.label === 'scorecard' ? ' <span class="muted">auto</span>' : ''}</td>
    <td><span class="chip run-dot"><i style="background:${colorOf(c.run)}"></i>${esc(c.run || '–')}</span></td>
    <td>${esc(c.checkpoint_name.replace('.pt', ''))}</td>
    <td class="heat"${c.state === 'complete' ? diverge(c.win_rate, 0.5, 0.45) : ''}>${c.state === 'complete' ? `${c.wins}/${c.games} (${pct(c.win_rate)})` : `${esc(c.state)} · ${c.games || 0}/60`}</td>
    <td>${c.ci ? `${pct(c.ci[0])}–${pct(c.ci[1])}` : '–'}</td>
    <td>${fmt(c.stocks_taken, 2)} / ${fmt(c.stocks_lost, 2)}</td>
    <td><div class="chips">${Object.entries(c.by_opponent || {}).map(([k, v]) => `<span class="chip">${esc(charName(k))} <b>${v.wins}/${v.games}</b></span>`).join('')}</div></td></tr>`).join('')
    || '<tr><td colspan="7" class="muted">None yet.</td></tr>';
  if (!mv) return;
  const names = mv.names, bs = mv.buckets;
  const layers = GROUPS.map((g) => ({ name: g.name, color: g.color, values: bs.map((b) => names.reduce((a, n, j) => a + (g.re.test(n) ? b.share[j] : 0), 0)) }));
  stackChart($('#chart-moves'), bs, layers);
  $('#moves-note').textContent = bs.length ? `share of decisions · each block is ${mv.size} games · hover a band for its name` : '';
  if (bs.length >= 2) {
    const a = bs[0].share, b = bs[bs.length - 1].share;
    const d = names.map((n, j) => [n, b[j] - a[j], b[j]]).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 10);
    $('#moves-movers').innerHTML = d.map(([n, dv, now]) => `<span class="chip"><i style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:${groupOf(n).color}"></i>${esc(moveName(n))}
      <b class="${dv > 0 ? 'up' : 'down'}">${dv > 0 ? '+' : ''}${(dv * 100).toFixed(1)} pts</b> <span class="muted">now ${(now * 100).toFixed(1)}%</span></span>`).join('');
  } else $('#moves-movers').innerHTML = '<span class="muted">Needs two blocks of games.</span>';
}

// ------------------------------------------------------------------ showcase & viewer
let SC = { cpu_characters: ['FOX'], phillip_characters: ['FOX'] };
const MEDALS = ['🥇', '🥈', '🥉'];
async function renderShowcase() {
  const r = await api('/api/showcase').catch(() => null);
  if (!r) return;
  SC = r;
  renderViewer(r.viewer);
  $('#showcase-list').innerHTML = r.rows.map((x, i) => {
    const c = x.cpu, p = x.phillip;
    const cpuLine = c ? `CPU 9 scorecard ${c.wins}/${c.games}` : 'no CPU scorecard yet';
    const phLine = p ? `Phillip ${p.wins}/${p.games} · ${fmt(p.taken, 1)} stocks/game` : 'not tested vs Phillip';
    return `<div class="sc-row ${i === 0 ? 'top' : ''}">
      <div class="rank">${MEDALS[i] || `#${i + 1}`}</div>
      <div class="score"><b>${fmt(x.score, 0)}</b><span>of 100</span></div>
      <div class="who"><div class="name">${esc(x.run || '')} · v${x.version ?? '?'}</div>
        <div class="muted">${esc(x.name)} · ${x.frames ? `${fmt(x.frames / 1e6, 1)}M frames · ` : ''}tested ${x.tested ? new Date(x.tested * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '–'}</div>
        <div class="split" title="${fmt(x.cpu_points, 1)} points from CPUs, ${fmt(x.phillip_points, 1)} from Phillip"><i style="width:${x.cpu_points}%;background:var(--green)"></i><i style="width:${x.phillip_points}%;background:var(--pink)"></i></div>
        <div class="muted" style="margin-bottom:6px">${cpuLine} · ${phLine}</div>
        <div class="chips">${x.badges.map((b) => `<span class="chip">${esc(b)}</span>`).join('')}</div></div>
      <div class="btns"><button class="primary small" data-watch="${esc(x.path)}">▶ Watch</button>
        ${p ? '' : `<button class="small" data-phillip-test="${esc(x.path)}" title="24 games vs Phillip (Fox, Falco, Marth, Falcon, Peach, Puff), about 30 minutes">Score vs Phillip</button>`}
        ${c ? '' : `<button class="small" data-scorecard="${esc(x.path)}" title="the 60-game CPU-9 scorecard">Run scorecard</button>`}</div></div>`;
  }).join('') || '<div class="empty">No tested checkpoints yet.</div>';
}

const LAYOUT_MAP = {
  switch: [['A', 'B (right button)'], ['B', 'A (bottom button)'], ['X', 'Y (top button)'], ['Y', 'X (left button)']],
  xbox: [['A', 'A (bottom button)'], ['B', 'B (right button)'], ['X', 'X (left button)'], ['Y', 'Y (top button)']],
};
const COMMON_MAP = [['Z', 'RB or LB (bumpers)'], ['L / R', 'LT / RT (analog triggers)'], ['Control stick', 'Left stick'],
  ['C-stick', 'Right stick'], ['D-pad', 'D-pad'], ['Start', 'Menu button (☰)']];
function renderControls() {
  const layout = $('#play-form').layout.value;
  $('#controls-map tbody').innerHTML = [...LAYOUT_MAP[layout], ...COMMON_MAP].map(([g, x]) => `<tr><td><b>${esc(g)}</b></td><td>${esc(x)}</td></tr>`).join('');
  $('#controls-note').textContent = layout === 'switch' ? 'buttons sit where they do on a Switch Pro controller' : 'buttons match the letters on the pad';
}
let playLoaded = false;
async function renderPlay(refresh = false) {
  const o = await api(`/api/play/options${refresh ? '?refresh=true' : ''}`).catch(() => null);
  if (!o) return;
  const f = $('#play-form');
  const pad = o.controllers[0];
  $('#pad-status').className = `pad-status ${pad ? 'ok' : 'missing'}`;
  $('#pad-status').innerHTML = `<span>${pad ? `🎮 <b>${esc(pad.split('/').slice(2).join('/'))}</b> connected` : '⚠️ No controller found: turn it on or reconnect it'}</span><button type="button" class="small" id="pad-check">Check again</button>`;
  if (!playLoaded) {
    f.checkpoint.innerHTML = o.models.map((m) => `<option value="${esc(m.path)}">${esc(m.label)}</option>`).join('');
    f.character.innerHTML = o.characters.map((c) => `<option value="${esc(c)}">${esc(charName(c))}</option>`).join('');
    f.character.value = 'FOX';
    playLoaded = true;
  }
  renderControls();
  renderSession(o.viewer);
}
function renderSession(v) {
  if (!v || v.kind !== 'versus') { $('#session').innerHTML = '<span class="muted">No match running.</span>'; $('#session-note').textContent = ''; return; }
  const you = v.results.filter((r) => r === 'loss').length, bot = v.results.filter((r) => r === 'win').length;
  $('#session-note').textContent = `you ${you} – ${bot} Puff`;
  $('#session').innerHTML = v.results.map((r, i) => `<span class="chip ${r === 'loss' ? 'win' : 'lost'}">Game ${i + 1}: ${r === 'loss' ? 'you won' : r === 'win' ? 'Puff won' : 'draw'}</span>`).join('')
    || '<span class="muted">First game in progress…</span>';
}
$('#play-form').layout.addEventListener('change', renderControls);
$('#play-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData($('#play-form')));
  try {
    await post('/api/versus', { checkpoint: f.checkpoint, character: f.character, layout: f.layout, delay: +f.delay });
    toast('Starting… a Dolphin window opens in a few seconds. Hands off the controller until the match begins.');
    setTimeout(pollViewer, 1500);
  } catch (err) { toast(err.message); }
});
document.addEventListener('click', (e) => { if (e.target.id === 'pad-check') { e.target.textContent = 'Checking…'; renderPlay(true); } });

function renderViewer(v) {
  const el = $('#viewer-bar');
  if (tab === 'play') renderSession(v);
  if (!v) { el.hidden = true; el._html = ''; return; }
  if (v.kind === 'versus') {
    const you = v.results.filter((r) => r === 'loss').length, bot = v.results.filter((r) => r === 'win').length;
    const html = `<div class="msg"><b>🎮 Me vs AI</b>You (${esc(charName(v.opponent))}) vs Puff ${esc(v.name.replace('.pt', ''))}${v.delay ? ` · bot reacts ${v.delay} frames late` : ''} ·
      ${v.state === 'playing' || v.results.length ? `<b style="color:var(--text)">you ${you} – ${bot} Puff</b>` : 'setting up the match… hands off the controller'}</div>
      <button class="danger" id="viewer-stop" ${v.stopping ? 'disabled' : ''}>${v.stopping ? 'Closing…' : '■ End match'}</button>`;
    el.hidden = false;
    if (el._html !== html) { el.innerHTML = html; el._html = html; }
    return;
  }
  const who = v.kind === 'phillip' ? `🤖 Phillip's ${charName(v.opponent)}` : `level-${v.level} ${charName(v.opponent)} CPU`;
  const games = v.games >= 99 ? 'until stopped' : `${v.games} game${v.games === 1 ? '' : 's'}`;
  const score = v.played ? ` · ${v.wins}–${v.played - v.wins} so far${v.last ? ` (last: ${v.last.result} ${v.last.stocks_left}–${v.last.opp_stocks_left})` : ''}` : ' · loading the game…';
  const html = `<div class="msg"><b>📺 Viewer open</b>${esc(v.name.replace('.pt', ''))} vs ${esc(who)} · ${games}${score}</div>
    <button class="danger" id="viewer-stop" ${v.stopping ? 'disabled' : ''}>${v.stopping ? 'Stopping…' : '■ Stop viewer'}</button>`;
  el.hidden = false;
  if (el._html !== html) { el.innerHTML = html; el._html = html; }
}
async function pollViewer() { const r = await api('/api/watch').catch(() => null); if (r) renderViewer(r.viewer); }

let watchTarget = null;
function fillCharacters() {
  const f = $('#watch-form'), kind = f.kind.value;
  const list = kind === 'phillip' ? SC.phillip_characters : SC.cpu_characters;
  const cur = f.opponent.value;
  f.opponent.innerHTML = list.map((c) => `<option value="${esc(c)}">${esc(charName(c))}</option>`).join('');
  if (list.includes(cur)) f.opponent.value = cur;
  $('#watch-level').hidden = kind === 'phillip';
}
async function openWatch(path) {
  watchTarget = path;
  if (!SC.rows) { const r = await api('/api/showcase').catch(() => null); if (r) SC = r; }
  $('#watch-name').textContent = path.split('/').pop().replace('.pt', '');
  fillCharacters();
  $('#watch-dialog').showModal();
}
$('#watch-form select[name=kind]').addEventListener('change', fillCharacters);
$('#watch-dialog').addEventListener('close', async () => {
  if ($('#watch-dialog').returnValue !== 'ok' || !watchTarget) return;
  const f = Object.fromEntries(new FormData($('#watch-form')));
  try {
    await post('/api/watch', { checkpoint: watchTarget, kind: f.kind, opponent: f.opponent, level: +f.level, games: +f.games, speed: +f.speed });
    toast(f.kind === 'phillip' ? 'Opening the viewer… Phillip takes ~30 s to load.' : 'Opening the viewer…');
    setTimeout(pollViewer, 1500);
  } catch (err) { toast(err.message); }
});
document.addEventListener('click', async (e) => {
  const t = e.target;
  if (t.id === 'viewer-stop') { t.disabled = true; t.textContent = 'Stopping…'; await post('/api/watch/stop').catch(() => {}); toast('Closing the viewer.'); setTimeout(pollViewer, 1000); }
  const run = async (body, msg) => { try { await post('/api/evaluate', body); toast(msg); } catch (err) { toast(err.message); } };
  if (t.dataset.phillipTest) {
    t.disabled = true;
    run({ checkpoint: t.dataset.phillipTest, kind: 'phillip', games: 4, workers: 2, opponents: ['FOX', 'FALCO', 'MARTH', 'CPTFALCON', 'PEACH', 'JIGGLYPUFF'] },
      'Phillip test started: 24 games, about 30 minutes. It appears here when complete.');
  }
  if (t.dataset.scorecard) {
    t.disabled = true;
    run({ checkpoint: t.dataset.scorecard, kind: 'cpu', level: 9, games: 10, workers: 2 }, 'Scorecard started: 60 games, about 20–30 minutes.');
  }
});

// ------------------------------------------------------------------ tabs
const TABS = ['live', 'showcase', 'play', 'report', 'phillip', 'progress'];
let tab = 'live';
function refreshTab() {
  if (tab === 'live') refreshSlow();
  else if (tab === 'showcase') renderShowcase();
  else if (tab === 'play') renderPlay();
  else if (tab === 'report') renderReport();
  else if (tab === 'phillip') renderPhillip();
  else renderProgress();
}
function showTab() {
  const want = location.hash.slice(1);
  tab = TABS.includes(want) ? want : 'live';
  for (const t of TABS) $(`#tab-${t}`).hidden = t !== tab;
  document.querySelectorAll('#tabs a').forEach((a) => a.classList.toggle('on', a.dataset.tab === tab));
  refreshTab();
}
window.addEventListener('hashchange', showTab);
$('#report-window').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-hours]');
  if (!b) return;
  reportHours = +b.dataset.hours;
  document.querySelectorAll('#report-window button').forEach((x) => x.classList.toggle('on', x === b));
  renderReport();
});

// ------------------------------------------------------------------ polling
async function loadRuns() {
  S.runs = await api('/api/runs').catch(() => []);
  const sel = $('#run-select');
  const cur = S.run;
  sel.innerHTML = S.runs.map((r) => `<option value="${esc(r.run)}">${esc(r.run)} · ${esc(r.state)}</option>`).join('') || '<option value="">No runs yet</option>';
  if (!S.run || !S.runs.find((r) => r.run === S.run)) S.run = (S.runs.find((r) => r.state === 'running') || S.runs[0] || {}).run || null;
  sel.value = S.run || '';
  if (cur !== S.run) { S.games = []; S.metrics = []; refreshTab(); }
}

async function refreshStatus() {
  if (!S.run) { $('#kpis').innerHTML = '<div class="kpi"><div class="v">No runs yet</div><div class="s">Press “New run” to start training.</div></div>'; return; }
  try {
    const st = await api(`/api/runs/${S.run}/status`);
    S.status = st;
    const b = $('#state'); b.textContent = st.state; b.className = `badge ${st.state}`;
    $('#stop-btn').hidden = !['running', 'starting'].includes(st.state);
    $('#resume-btn').hidden = !['stopped', 'crashed'].includes(st.state);
    renderKpis(st); renderAlert(st); renderWorkers(st); renderLadder(st); renderBrain(st); renderEvents(st);
  } catch (e) { /* run just created; status not written yet */ }
}

async function refreshSlow() {
  if (!S.run) return;
  [S.games, S.metrics] = await Promise.all([
    api(`/api/runs/${S.run}/games?tail=1500`).catch(() => S.games),
    api(`/api/runs/${S.run}/metrics?tail=800`).catch(() => S.metrics),
  ]);
  renderGames(); renderMetrics();
}
const completed = (g) => ['win', 'loss', 'draw'].includes(g.result);

// ------------------------------------------------------------------ actions
$('#run-select').addEventListener('change', (e) => { S.run = e.target.value; S.games = []; S.metrics = []; refreshStatus(); refreshTab(); renderCheckpoints(); });
$('#stop-btn').addEventListener('click', async () => {
  if (!confirm('Stop this run? It saves a checkpoint and shuts every emulator down.')) return;
  await post(`/api/runs/${S.run}/stop`); toast('Stopping — saving a checkpoint.');
});
$('#resume-btn').addEventListener('click', async () => {
  try { await post('/api/train', { resume: S.run }); toast('Resuming…'); } catch (e) { toast(e.message); }
});
$('#new-btn').addEventListener('click', async () => {
  const sel = $('#new-form select[name=init]');
  const champs = await api('/api/champions').catch(() => []);
  const cks = S.checkpoints || [];
  sel.innerHTML = '<option value="">Fresh network</option>' +
    champs.map((c) => `<option value="${esc(c.path)}">👑 ${esc(c.name)}</option>`).join('') +
    cks.map((c) => `<option value="${esc(c.path)}">${esc(S.run)} / ${esc(c.name)}</option>`).join('');
  $('#new-dialog').showModal();
});
$('#new-dialog').addEventListener('close', async () => {
  if ($('#new-dialog').returnValue !== 'ok') return;
  const f = Object.fromEntries(new FormData($('#new-form')));
  try {
    const r = await post('/api/train', { name: f.name, workers: +f.workers, visible: +f.visible, visible_speed: +f.visible_speed, phillip_workers: +f.phillip_workers, selfplay: +f.selfplay, level: +f.level, init: f.init || null });
    S.run = r.run; toast(`Started ${r.run}`); setTimeout(loadRuns, 1500);
  } catch (e) { toast(e.message); }
});
let evalTarget = null;
document.addEventListener('click', async (e) => {
  const t = e.target;
  if (t.dataset.eval) { evalTarget = t.dataset.eval; $('#eval-name').textContent = evalTarget.split('/').pop(); $('#eval-dialog').showModal(); }
  if (t.dataset.watch) openWatch(t.dataset.watch);
  if (t.dataset.crown) {
    const note = prompt('Crown as champion. Optional note:', '');
    if (note === null) return;
    try {
      await post('/api/champions', { path: t.dataset.crown, note });
      toast('Crowned 👑 (backed by a complete evaluation)');
    } catch (err) {
      if (!/No complete evaluation/.test(err.message)) { toast(err.message); return; }
      if (!confirm(`${err.message}\n\nCrown it anyway? It will be marked as crowned without evidence.`)) return;
      try { await post('/api/champions', { path: t.dataset.crown, note, force: true }); toast('Crowned 👑 (no evaluation)'); } catch (e2) { toast(e2.message); return; }
    }
    renderCheckpoints(); renderEvals();
  }
});
$('#eval-dialog').addEventListener('close', async () => {
  if ($('#eval-dialog').returnValue !== 'ok' || !evalTarget) return;
  const f = Object.fromEntries(new FormData($('#eval-form')));
  try {
    await post('/api/evaluate', { checkpoint: evalTarget, kind: f.kind, level: +f.level, games: +f.games, workers: +f.workers, opponents: f.opponents.split(',').map((s) => s.trim().toUpperCase()).filter(Boolean) });
    toast('Evaluation started. Results appear under Evaluations.');
  } catch (err) { toast(err.message); }
});

loadRuns().then(() => { refreshStatus(); showTab(); renderCheckpoints(); renderEvals(); });
setInterval(refreshStatus, 1500);
setInterval(pollViewer, 3000);
pollViewer();
setInterval(() => { if (tab === 'live') refreshSlow(); }, 6000);
setInterval(() => { if (tab !== 'live') refreshTab(); }, 30000);
setInterval(() => { loadRuns(); renderCheckpoints(); renderEvals(); }, 10000);
window.addEventListener('resize', () => { if (tab === 'live') { renderGames(); renderMetrics(); } else refreshTab(); });

'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => Number(value || 0).toLocaleString();
const number = value => value == null ? '—' : Number(value).toFixed(1);
const percent = value => value == null ? '—' : (value * 100).toFixed(1) + '%';
const when = value => value ? new Date(value * 1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '—';
const policyLabel = kind => ({ppo:'PPO training',evaluation:'Evaluation','scripted demonstration':'Scripted demo','self-play':'Frozen opponent','human challenge':'Human challenge'}[kind] || kind);
const activeStates = ['starting','running','paused'];
let state = null, selectedRun = null, currentTab = 'monitor', offset = 0, pageTotal = 0;
let pending = false, connected = false, pollRunning = false, analyticsKey = '', dataEpoch = 0;
let promotePath = null, policySignature = '', noticeTimer, samples = new Map(), latestSeries = [], plotPoints = [];
const queryCache = new Map();

function message(text, error = false) {
  clearTimeout(noticeTimer);
  $('notice').hidden = false;
  $('notice').textContent = text;
  $('notice').className = error ? 'error' : '';
  noticeTimer = setTimeout(() => { $('notice').hidden = true; }, error ? 15000 : 7000);
}
async function api(path, data) {
  const response = await fetch(path, {method:data === undefined ? 'GET' : 'POST', headers:{'Content-Type':'application/json'}, body:data === undefined ? undefined : JSON.stringify(data)});
  const body = await response.json();
  if (!response.ok) {
    let detail = body.detail;
    if (Array.isArray(detail)) detail = detail.map(e => e.msg).join(' ');
    else if (typeof detail === 'object') detail = detail.message;
    throw new Error(detail || 'The request could not be completed.');
  }
  return body;
}
async function act(fn) {
  if (pending) return;
  pending = true; renderControls();
  try { await fn(); await refresh(); }
  catch (error) { message(error.message, true); }
  finally { pending = false; renderControls(); }
}
function selected() { return state?.runs.find(r => r.id === selectedRun); }
function active() { return state?.runs.find(r => activeStates.includes(r.status)); }
function option(value, text) { const o = document.createElement('option'); o.value = value; o.textContent = text; return o; }
function setOptions(id, entries, value) {
  const select = $(id), signature = JSON.stringify(entries);
  if (select.dataset.signature === signature) {
    if (value !== undefined && !Array.isArray(value) && select.value !== value) select.value = value;
    return;
  }
  select.replaceChildren(...entries.map(([v,t]) => option(v,t)));
  if (Array.isArray(value)) for (const o of select.options) o.selected = value.includes(o.value);
  else select.value = value ?? '';
  select.dataset.signature = signature;
}
// How a trial was played is part of the result. A report from before the shared
// controller rules cannot say whether the agent was helped, so it says that instead.
function executionHTML(stats) {
  if (!stats || !stats.execution_mode) return '';
  if (!stats.execution_verified) return '<small class="interval">execution rules unverified</small>';
  const altered = stats.measured_override_rate;
  const rules = stats.execution_mode === 'raw' ? 'raw controller, no assistance'
    : stats.scripted_overrides ? 'assisted rules' : 'no assistance applied';
  return `<small class="interval">${esc(rules)}${altered ? ` · ${percent(altered)} of inputs altered` : ''}</small>`;
}
function rateHTML(stats, context = '') {
  if (!stats || !stats.episodes || stats.win_rate == null) return '<div class="rate"><strong>Not evaluated</strong><small>No deterministic evaluation evidence yet.</small></div>';
  const ci = stats.win_rate_95_interval;
  return `<div class="rate"><strong>${percent(stats.win_rate)}</strong><small class="interval">95% CI ${ci ? `${percent(ci[0])}–${percent(ci[1])}` : 'unavailable'}</small><small>${fmt(stats.wins)} wins / ${fmt(stats.episodes)} episodes${context ? ' · ' + esc(context) : ''}</small>${executionHTML(stats)}</div>`;
}
function outcomesHTML(stats) {
  const counts = stats?.outcomes || {};
  return Object.entries(counts).filter(([,n]) => n > 0).map(([k,n]) => `<span class="outcome-${esc(k)}">${fmt(n)} ${esc(k.replaceAll('_',' '))}</span>`).join(' · ') || 'No recorded matches';
}
function conditions(config) { return config ? `${config.character} vs ${config.opponent} · ${config.stocks} stocks · ${String(config.stage).replaceAll('_',' ')}` : 'Configuration unavailable'; }
function elapsed(seconds) { if (seconds == null) return '—'; const n = Math.floor(seconds); return `${String(Math.floor(n/3600)).padStart(2,'0')}:${String(Math.floor(n/60)%60).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`; }
function showTab(name) {
  const prev = currentTab;
  currentTab = name;
  for (const panel of ['monitor','arena','readiness','data','policies','overload','diagnostic','rankings']) {
    const p = $(panel+'-panel'); if (p) p.hidden = panel !== name;
    const t = $(panel+'-tab'); if (t) { t.setAttribute('aria-selected', String(panel === name)); t.tabIndex = panel === name ? 0 : -1; }
  }
  if (name === 'data') requestAnimationFrame(() => drawChart(latestSeries));
  if (name === 'overload' && typeof window.overloadTabActivated === 'function') window.overloadTabActivated();
  else if (prev === 'overload' && typeof window.overloadTabDeactivated === 'function') window.overloadTabDeactivated();
  if (name === 'diagnostic' && typeof window.diagnosticTabActivated === 'function') window.diagnosticTabActivated();
  if (name === 'rankings' && typeof window.rankingsTabActivated === 'function') window.rankingsTabActivated();
}

function updateSelectors() {
  const runEntries = state.runs.map(r => [r.id, `${r.id} · ${r.status} · ${r.envs}×`]);
  setOptions('run-select', runEntries.length ? runEntries : [['','No runs yet']], selectedRun);
  const compare = Array.from($('compare-runs').selectedOptions, o => o.value);
  setOptions('compare-runs', runEntries.filter(([id]) => id !== selectedRun), compare);
  const chosenChar = $('agent-character')?.value || state.config?.character || 'FOX';
  const matching = (state.checkpoints || []).filter(c => (c.config?.character || 'FOX') === chosenChar);
  const others = (state.checkpoints || []).filter(c => (c.config?.character || 'FOX') !== chosenChar);
  let chosen = $('checkpoint').value;
  const chosenCp = (state.checkpoints || []).find(c => c.path === chosen);

  if (chosen && (chosenCp?.config?.character || 'FOX') !== chosenChar) {
    chosen = matching.length ? matching[0].path : '';
    $('checkpoint').value = chosen;
  } else if (!chosen && !$('checkpoint').dataset.userExplicitEmpty && matching.length) {
    chosen = matching[0].path;
    $('checkpoint').value = chosen;
  }

  const charLabel = chosenChar === 'JIGGLYPUFF' ? 'New Jigglypuff' : chosenChar === 'FOX' ? 'New Fox' : `New ${chosenChar}`;
  const options = [['', charLabel]];
  for (const c of matching) {
    const isCtrl = c.config?.action_set === 'controller' || (typeof c.actions === 'object' && c.actions?.space === 'controller-v1') || c.actions === 'controller';
    const badge = isCtrl ? '🎮' : '🕹️';
    options.push([c.path, `${c.champion ? '★ ' : ''}${badge} ${c.name} · ${fmt(c.steps)}`]);
  }
  for (const c of others) {
    const isCtrl = c.config?.action_set === 'controller' || (typeof c.actions === 'object' && c.actions?.space === 'controller-v1') || c.actions === 'controller';
    const badge = isCtrl ? '🎮' : '🕹️';
    options.push([c.path, `[${c.config?.character || 'FOX'}] ${badge} ${c.name} · ${fmt(c.steps)}`]);
  }
  setOptions('checkpoint', options, chosen);
}
function renderControls() {
  if (!state) return;
  const running = active(), availability = state.availability || {can_start:false,message:'Checking availability…'};
  const evaluation = $('mode').value === 'evaluate';
  const envs = Number($('envs').value);
  if (evaluation) $('envs').value = '1';
  const parallel = !evaluation && envs > 1;
  const checkpoint = (state.checkpoints || []).find(c => c.path === $('checkpoint').value);
  const charSelect = $('agent-character');
  if (charSelect) {
    if (checkpoint && checkpoint.config?.character) {
      charSelect.value = checkpoint.config.character;
    } else if (!charSelect.dataset.userChanged && state.config.character) {
      charSelect.value = state.config.character;
    }
  }
  if (checkpoint) {
    if (checkpoint.architecture === 'recurrent' || checkpoint.recurrent) {
      if ($('recurrent')) $('recurrent').checked = true;
    }
    const isCtrlCp = checkpoint.config?.action_set === 'controller' || checkpoint.actions === 'controller' || (typeof checkpoint.actions === 'object' && checkpoint.actions?.space === 'controller-v1');
    if (isCtrlCp) {
      if ($('fast-inputs')) $('fast-inputs').checked = true;
    }
  }
  $('envs').disabled = evaluation;
  $('budget-label').hidden = evaluation; $('episodes-label').hidden = !evaluation;
  $('budget').disabled = evaluation; $('episodes').disabled = !evaluation;
  $('bootstrap-label').hidden = evaluation;
  $('bootstrap').disabled = evaluation || parallel || !!checkpoint;
  $('curriculum-label').hidden = evaluation;
  $('curriculum').disabled = evaluation;
  if (evaluation) $('curriculum').checked = false;
  $('launch').textContent = evaluation ? 'Start evaluation ↗' : checkpoint ? 'Resume training ↗' : 'Start training ↗';
  let invalid = '', hint = 'One emulator can learn from demonstrations and advance CPU difficulty.';
  const isCtrlCp = checkpoint ? (checkpoint.config?.action_set === 'controller' || checkpoint.actions === 'controller' || (typeof checkpoint.actions === 'object' && checkpoint.actions?.space === 'controller-v1')) : false;
  if (evaluation) hint = `One emulator. ${isCtrlCp ? 'Actions sampled from the policy' : 'Deterministic network actions'}. Every outcome is retained, including timeouts.`;
  else if (parallel) hint = `${envs} emulators. Demonstration warm-up is skipped; the curriculum works across all of them.`;
  if (evaluation && !checkpoint) invalid = 'Select a saved checkpoint or champion to evaluate.';
  if (!evaluation && (!Number.isInteger(envs) || envs < 1 || envs > 12)) invalid = 'Choose a whole number of emulators from 1 to 12.';
  if (!evaluation && parallel && !checkpoint && !$('anchor')?.value && !$('recurrent')?.checked) {
    invalid = 'Parallel emulators for a fresh agent require either 1 emulator to start, checking Recurrent Memory, or choosing tournament footage.';
  }
  if (checkpoint) {
    if (charSelect && checkpoint.config?.character && checkpoint.config.character !== charSelect.value) {
      invalid = `This policy was trained for ${checkpoint.config.character}, but ${charSelect.value} is currently selected.`;
    }
    if ($('anchor')?.value && !isCtrlCp) {
      invalid = 'Tournament footage anchoring requires a full 7-axis controller policy. Choose a controller policy (🎮) or start a fresh agent.';
    }
    const keys = ['stage','stocks'];
    if (!state.config.randomize_opponent&&!$('opponent-policy').value&&!$('benchmark').checked) keys.push('opponent');
    const mismatch = keys.find(k => checkpoint.config[k] !== state.config[k]);
    if (mismatch && !invalid) invalid = `This policy's ${mismatch.replaceAll('_',' ')} differs from the current setup.`;
  }
  $('constraints').textContent = invalid || hint;
  $('constraints').className = 'constraint' + (invalid ? ' invalid' : '');
  const activeChar = (checkpoint ? checkpoint.config.character : charSelect?.value) || state.config.character || 'FOX';
  const defaultContext = activeChar === 'JIGGLYPUFF'
    ? 'Start Jigglypuff with dedicated Rest Specialist rewards (+4.0 lethal hit, -1.5 whiff penalty, Up-throw/CC confirms).'
    : 'Start from random weights, with optional demonstrations.';
  const ctrlNote = checkpoint ? (isCtrlCp ? '🎮 Full 7-axis controller.' : '🕹️ Discrete actions.') : '';
  $('policy-context').textContent = checkpoint ? `${fmt(checkpoint.steps)} PPO steps · trained at CPU ${checkpoint.training_cpu_level}. ${ctrlNote} ${checkpoint.champion ? 'Kept champion copy.' : 'The input will be snapshotted at launch.'}` : defaultContext;
  const oppSummary = state.config.randomize_opponent ? 'Random (Competitive Roster)' : state.config.opponent;
  const charSummary = activeChar === 'JIGGLYPUFF' ? 'Jigglypuff (Rest Specialist)' : activeChar;
  $('launch-conditions').textContent = `${checkpoint ? 'Resume' : 'New run'}: ${charSummary} vs ${oppSummary} · CPU ${state.config.cpu_level} · ${state.config.speed === 0 ? 'unlimited speed' : state.config.speed+'× speed'}. Change in Setup.`;
  $('launch').disabled = pending || !connected || !state.ready || !availability.can_start || !!invalid;
  $('active-context').textContent = running ? `Controls → ${running.id}` : 'No active run';
  $('pause').textContent = running?.status === 'paused' || running?.control?.pause ? 'Resume' : 'Pause';
  const stopping = availability.reason === 'stopping';
  for (const id of ['pause','save','stop']) $(id).disabled = pending || !connected || !running || stopping;
  $('save').disabled ||= running?.mode !== 'train';
  if (typeof renderMissionControls==='function') renderMissionControls();
  $('availability').textContent = availability.message;
  $('settings').disabled = pending || !connected || !availability.can_start;
  document.querySelectorAll('[data-action="train"],[data-action="evaluate"],[data-action="play"]').forEach(button => { button.disabled = pending || !connected || !state.ready || !availability.can_start; });
  document.querySelectorAll('[data-action="promote"]').forEach(button => { button.disabled = pending || !connected; });
}
function renderMonitor() {
  const run = selected();
  $('run-status').textContent = run ? `${run.status.toUpperCase()} / ${run.mode === 'play' ? 'HUMAN CHALLENGE' : run.mode === 'evaluate' ? 'EVALUATION' : 'TRAINING'}` : 'NO RUN';
  $('steps').textContent = run ? fmt(run.steps) : '—';
  $('elapsed').textContent = elapsed(run?.elapsed);
  $('phase').textContent = run?.phase || 'No run selected';
  const slots = run?.slots || [];
  $('slot-count').textContent = run ? String(run.envs) : '—';
  const failed = slots.filter(s => s.failed).length, stale = slots.filter(s => s.stale).length;
  $('slot-summary').textContent = failed ? `${failed} failed · inspect below` : stale ? `${stale} with delayed updates` : run?.status === 'running' ? 'All reported slots healthy' : 'Last recorded state';
  $('floor-count').textContent = run ? `${slots.length} / ${run.envs} SLOTS` : '';
  const samplesForRun = samples.get(run?.id) || [];
  const diff = samplesForRun.length > 1 ? (samplesForRun.at(-1).steps-samplesForRun[0].steps)/(samplesForRun.at(-1).time-samplesForRun[0].time) : null;
  $('throughput').textContent = run?.status === 'running' && diff != null ? Math.max(0,diff).toFixed(1) : '—';
  $('run-error').hidden = !run?.error;
  if (run?.error) {
    const cooling = run.error.includes('owns the emulator');
    $('run-error').innerHTML = cooling ? 'The previous emulators were still closing when this run tried to start.<small>Wait until Run controls says ready, then start again. No training results were produced by this attempt.</small>' : `${esc(run.error)}<small>The worker log and any saved checkpoints remain in this run's folder.</small>`;
  }
  const openDetails = new Set(Array.from($('slots').querySelectorAll('details[open]'), d => d.dataset.slot));
  $('slots').innerHTML = slots.length ? slots.map(s => {
    let label = s.failed ? 'FAILED' : s.waiting ? 'WAITING TO BOOT' : run.status === 'paused' ? 'PAUSED' : !activeStates.includes(run.status) ? 'LAST STATE' : s.stale ? 'UPDATE DELAYED' : (s.connection || 'CONNECTED').toUpperCase();
    const players = s.players || {};
    const playerHTML = port => { const p = players[port]; return `<div><small>P${port} / ${esc(p?.character || (port === 1 ? run.config?.character : run.config?.opponent) || '—')}</small><strong>${p ? number(p.percent) + '%' : '—'}</strong><span class="stocks">${p ? '●'.repeat(Math.max(0,Math.min(5,p.stocks))) || '0 stocks' : ''}</span><small title="${esc(p?.action)}">${esc(p?.action || 'No frame yet')}</small></div>`; };
    return `<article class="slot ${s.failed ? 'failed' : s.stale ? 'stale' : ''}"><div class="slot-head"><strong>ENV ${String(s.index).padStart(2,'0')} <span class="muted">/ CPU ${s.cpu_level ?? run.cpu_level ?? '—'}</span></strong><span class="slot-state">${esc(label)}</span></div><div class="slot-players">${playerHTML(1)}<span class="vs">vs</span>${playerHTML(2)}</div><canvas data-slot="${s.index}" width="600" height="170" aria-label="Emulator ${s.index} player positions"></canvas>${s.failed ? `<div class="failure">${esc(s.failed)}</div><details data-slot="${s.index}" ${openDetails.has(String(s.index)) ? 'open' : ''}><summary>Failure details</summary><pre>${esc(s.traceback || 'No traceback available.')}</pre></details>` : ''}<div class="slot-footer"><span>${esc(s.action || 'Waiting for input')}</span><span>F ${fmt(s.frame)} · ${s.age_seconds == null ? 'no update' : Math.round(s.age_seconds)+'s ago'}</span></div></article>`;
  }).join('') : '<div class="empty"><strong>No emulator frames yet.</strong>Start a run to see every emulator here.</div>';
  for (const canvas of $('slots').querySelectorAll('canvas')) drawArena(canvas, slots.find(s => s.index === Number(canvas.dataset.slot))?.players);
}
function drawArena(canvas, players) {
  const c = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  c.clearRect(0,0,w,h); c.strokeStyle='#273237'; c.lineWidth=1;
  for (let x=30;x<w;x+=45) { c.beginPath(); c.moveTo(x,8); c.lineTo(x,h-15); c.stroke(); }
  for (let y=25;y<h;y+=35) { c.beginPath(); c.moveTo(16,y); c.lineTo(w-16,y); c.stroke(); }
  const scale=2, left=w/2-85*scale, right=w/2+85*scale, ground=110;
  c.fillStyle='#354342'; c.beginPath(); c.moveTo(left,ground); c.lineTo(right,ground); c.lineTo(right-70,ground+19); c.lineTo(left+70,ground+19); c.closePath(); c.fill();
  c.fillStyle='#758b6c'; c.fillRect(left,ground, right-left,2);
  if (!players) { c.fillStyle='#7e939b'; c.font='11px monospace'; c.textAlign='center'; c.fillText('Awaiting game state',w/2,70); return; }
  for (const [port,p] of Object.entries(players)) {
    const x=Math.min(w-14,Math.max(14,w/2+p.x*scale)), y=Math.min(h-12,Math.max(20,ground-8-p.y*.8));
    c.fillStyle=port==='1'?'#c2f781':'#efac7c'; c.globalAlpha=.1; c.beginPath(); c.arc(x,y,17,0,Math.PI*2); c.fill(); c.globalAlpha=1;
    c.beginPath(); c.arc(x,y,5,0,Math.PI*2); c.fill(); c.font='9px monospace'; c.textAlign='center'; c.fillText('P'+port,x,y-14);
  }
}

async function query(run, kind, cpu, outcome, pageOffset) {
  const key = JSON.stringify([run.id,run.match_revision,kind,cpu,outcome,pageOffset]);
  if (queryCache.has(key)) return queryCache.get(key);
  const params = new URLSearchParams({kind,offset:pageOffset,limit:25});
  if (cpu) params.set('cpu',cpu); if (outcome) params.set('outcome',outcome);
  const value = await api(`/api/runs/${encodeURIComponent(run.id)}/matches?${params}`);
  queryCache.set(key,value); if (queryCache.size>60) queryCache.delete(queryCache.keys().next().value);
  return value;
}
async function refreshAnalytics(force = false) {
  const run=selected(); if (!run) return;
  const comparisons=Array.from($('compare-runs').selectedOptions,o=>o.value).slice(0,3);
  const kind=$('kind').value,cpu=$('data-cpu').value,outcome=$('outcome').value;
  const targets=[run,...comparisons.map(id=>state.runs.find(r=>r.id===id)).filter(Boolean)];
  const key=JSON.stringify([targets.map(r=>[r.id,r.match_revision]),kind,cpu,outcome,offset]);
  if (!force && key===analyticsKey) return;
  analyticsKey=key; const epoch=++dataEpoch;
  try {
    const [pulse,...reports]=await Promise.all([query(run,run.opponent_checkpoint?'self-play':run.mode==='play'?'human challenge':run.mode==='evaluate'?'evaluation':'ppo','','',0),...targets.map((r,i)=>query(r,kind,cpu,i===0?outcome:'',i===0?offset:0))]);
    if (epoch!==dataEpoch) return;
    $('pulse').innerHTML=pulse.groups.length ? pulse.groups.map(g=>`<div class="pulse-group"><span>CPU ${g.cpu_level}</span>${rateHTML(g)}<div class="outcomes">${outcomesHTML(g)}</div></div>`).join('') : '<p class="muted">No recorded PPO matches yet. Scripted demonstrations are kept in Match data.</p>';
    renderData(targets,reports);
  } catch (error) { if (epoch===dataEpoch) { analyticsKey=''; message('Could not load match data: '+error.message,true); } }
}
function renderData(targets,reports) {
  const report=reports[0]; pageTotal=report.total;
  $('record-count').textContent=`${fmt(report.recorded)} records on disk${report.invalid_records ? ` · ${report.invalid_records} unreadable` : ''}`;
  $('data-summary').innerHTML=report.groups.length ? report.groups.map(g=>`<div class="stat-card"><h3>${esc(policyLabel(g.kind))} / CPU ${g.cpu_level}</h3>${rateHTML(g)}<div class="outcomes">${outcomesHTML(g)}</div><small class="muted">Mean reward ${number(g.mean_return)}</small></div>`).join('') : '<div class="empty">No matches for this policy type and CPU level.</div>';
  $('compare-body').innerHTML=reports.flatMap((r,i)=>r.groups.map(g=>`<tr><td class="name">${esc(targets[i].id)}<small>${esc(policyLabel(g.kind))}</small></td><td>${g.cpu_level}</td><td>${rateHTML(g)}</td><td>${fmt(g.episodes)}</td><td class="outcomes">${outcomesHTML(g)}</td><td>${number(g.mean_return)}</td></tr>`)).join('') || '<tr><td colspan="6" class="muted">No matching completed records.</td></tr>';
  $('match-body').innerHTML=report.rows.map(r=>`<tr><td class="name">${r.number}<small>${when(r.time)}</small></td><td>${esc(policyLabel(r.kind))}</td><td>${r.cpu_level}</td><td class="outcome-${esc(r.result)}">${esc(r.result.replaceAll('_',' '))}</td><td>${r.players?.['1']?.stocks ?? '—'} / ${r.players?.['2']?.stocks ?? '—'}</td><td>${number(r.return)}</td></tr>`).join('') || '<tr><td colspan="6" class="muted">No matches for this filter.</td></tr>';
  $('page-info').textContent=report.total ? `${offset+1}–${Math.min(offset+25,report.total)} of ${fmt(report.total)} · newest first` : 'No matches';
  $('previous').disabled=offset===0; $('next').disabled=offset+25>=report.total;
  latestSeries=reports.flatMap((r,i)=>r.series.map(s=>({...s,run:targets[i].id})));
  if (currentTab==='data') drawChart(latestSeries);
}
function drawChart(series) {
  const canvas=$('reward-chart'), rect=canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio=window.devicePixelRatio||1, width=rect.width,height=270;
  canvas.width=width*ratio; canvas.height=height*ratio;
  const c=canvas.getContext('2d'); c.scale(ratio,ratio); c.clearRect(0,0,width,height);
  const colors=['#c2f781','#84bfd1','#efac7c','#d8a3dc','#d8ce83','#d2ecdd'];
  const all=series.flatMap(s=>s.points).filter(p=>p.reward!=null); plotPoints=[];
  $('chart-empty').hidden=all.length>0;
  let ymin=Math.min(-1,...all.map(p=>p.reward)), ymax=Math.max(1,...all.map(p=>p.reward));
  const pad=(ymax-ymin)*.12; ymin-=pad;ymax+=pad;
  const xmax=Math.max(1,...all.map(p=>p.match)), left=47,right=15,top=22,bottom=35;
  const px=m=>left+(width-left-right)*m/xmax,py=y=>top+(height-top-bottom)*(1-(y-ymin)/(ymax-ymin));
  c.font='9px monospace';c.textAlign='right';
  for(let i=0;i<=4;i++){let y=ymin+(ymax-ymin)*i/4;c.strokeStyle='#303b40';c.beginPath();c.moveTo(left,py(y));c.lineTo(width-right,py(y));c.stroke();c.fillStyle='#8e9fa7';c.fillText(y.toFixed(1),left-8,py(y)+3);}
  c.textAlign='center';for(let i=0;i<=4;i++){let n=Math.round(xmax*i/4);c.fillStyle='#8e9fa7';c.fillText(String(n),px(n),height-19);}
  c.fillText('Recorded match within policy / CPU group',width/2,height-3);
  if(ymin<0&&ymax>0){c.setLineDash([3,4]);c.strokeStyle='#75888b';c.beginPath();c.moveTo(left,py(0));c.lineTo(width-right,py(0));c.stroke();c.setLineDash([]);}
  series.forEach((s,i)=>{const points=s.points.filter(p=>p.reward!=null);c.strokeStyle=colors[i%colors.length];c.fillStyle=colors[i%colors.length];c.lineWidth=1.8;c.beginPath();points.forEach((p,j)=>{const x=px(p.match),y=py(p.reward);j?c.lineTo(x,y):c.moveTo(x,y);plotPoints.push({x,y,p,s,color:colors[i%colors.length]});});c.stroke();if(points.length===1){c.beginPath();c.arc(px(points[0].match),py(points[0].reward),3,0,Math.PI*2);c.fill();}});
  $('chart-legend').innerHTML=series.map((s,i)=>`<span><i style="background:${colors[i%colors.length]}"></i>${esc(s.run)} / ${esc(policyLabel(s.kind))} / CPU ${s.cpu_level} · ${s.bin_size} match${s.bin_size===1?'':'es'}/bin</span>`).join('');
}
$('reward-chart').addEventListener('mousemove',event=>{
  const rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
  const nearest=plotPoints.reduce((best,p)=>!best||Math.hypot(p.x-x,p.y-y)<Math.hypot(best.x-x,best.y-y)?p:best,null);
  $('chart-tooltip').hidden=!nearest||Math.hypot(nearest.x-x,nearest.y-y)>45;
  if(nearest)$('chart-tooltip').textContent=`${nearest.s.run}\n${policyLabel(nearest.s.kind)} · CPU ${nearest.s.cpu_level}\nMatch ${nearest.p.match} · ${nearest.p.count} in bin\nMean reward ${number(nearest.p.reward)}`;
});
$('reward-chart').addEventListener('mouseleave',()=>{$('chart-tooltip').hidden=true;});
window.addEventListener('resize',()=>{if(currentTab==='data')drawChart(latestSeries);});

function actionButtons(policy, promote = true) {
  return `<div class="row-actions"><button data-action="play" data-path="${esc(policy.path)}">Challenge</button><button data-action="evaluate" data-path="${esc(policy.path)}">Evaluate</button><button data-action="train" data-path="${esc(policy.path)}">Train</button><button data-action="diagnose" data-path="${esc(policy.path)}" title="Stress-test this policy in Diagnostic Lab">Diagnose 🔬</button>${promote ? `<button data-action="promote" data-path="${esc(policy.path)}">Keep ★</button>` : ''}</div>`;
}
function renderPolicies(force = false) {
  const signature=JSON.stringify([state.checkpoints,$('policy-search').value]);
  if (!force&&signature===policySignature) return;
  policySignature=signature;
  const champions=state.champions||[];
  $('champion-count').textContent=`${champions.length} kept`;
  $('champions').innerHTML=champions.length ? champions.map(c=>{const e=c.evaluations[0];return `<article class="champion"><p class="eyebrow">★ KEPT POLICY / ${esc(c.champion.id)}</p><h3>${esc(c.name)}</h3><p class="note">${esc(c.champion.note||'No note added.')}</p><p class="help">${fmt(c.steps)} steps · trained at CPU ${c.training_cpu_level}</p>${rateHTML(e,e?`CPU ${e.config.cpu_level}`:'')}<p class="help">${e?`Latest trial · ${when(e.evaluated_at)} · ${esc(conditions(e.config))}`:'Preserved checkpoint. Performance has not been evaluated.'}</p><div class="button-row">${actionButtons(c,false)}</div></article>`;}).join('') : '<div class="empty"><strong>No champions promoted yet.</strong>Evaluate a checkpoint, then keep the policies worth returning to.<br>Unevaluated copies are clearly marked.</div>';
  $('champion-compare').innerHTML=champions.flatMap(c=>c.evaluations.length ? c.evaluations.map(e=>`<tr><td>${esc(c.name)}<small>${fmt(c.steps)} steps</small></td><td>CPU ${e.config.cpu_level}<small>${esc(conditions(e.config))}</small></td><td>${rateHTML(e)}</td><td>${fmt(e.episodes)}</td><td class="outcomes">${outcomesHTML(e)}</td><td>${when(e.evaluated_at)}</td></tr>`) : [`<tr><td>${esc(c.name)}</td><td>—</td><td>Not evaluated</td><td>0</td><td>—</td><td>—</td></tr>`]).join('') || '<tr><td colspan="6" class="muted">Promoted policies and their evaluation evidence will appear here.</td></tr>';
  const search=$('policy-search').value.toLowerCase();
  const candidates=state.checkpoints.filter(c=>!c.champion&&c.name.toLowerCase().includes(search));
  $('policy-body').innerHTML=candidates.map(c=>{const e=c.evaluations[0];return `<tr><td class="name">${esc(c.name)}${c.imported?' <span class="tag" style="font-size:8px;padding:2px 5px;vertical-align:middle;">IMPORTED</span>':''}<small title="${esc(c.sha256)}">ID ${esc(c.sha256.slice(0,12))}</small></td><td>${fmt(c.steps)}<small>${fmt(c.bootstrap_samples)} demo samples</small></td><td>CPU ${c.training_cpu_level}<small>${esc(conditions(c.config))}</small></td><td>${rateHTML(e,e?`CPU ${e.config.cpu_level}`:'')}${c.evaluations.length>1?`<details><summary>${c.evaluations.length} evaluation trials</summary>${c.evaluations.map(v=>`<p>${when(v.evaluated_at)}</p>${rateHTML(v,'CPU '+v.config.cpu_level)}`).join('')}</details>`:''}</td><td>${actionButtons(c)}</td></tr>`;}).join('') || '<tr><td colspan="5" class="muted">No matching checkpoints.</td></tr>';
}
function requestPromotion(path) {
  const cp=state.checkpoints.find(c=>c.path===path); if(!cp)return;
  promotePath=path;$('champion-name').value='';$('champion-note').value='';
  $('promote-policy').textContent=`${cp.name} · ${fmt(cp.steps)} PPO steps`;
  $('promote-evidence').innerHTML=cp.evaluations.length?rateHTML(cp.evaluations[0],'CPU '+cp.evaluations[0].config.cpu_level):'This policy has no deterministic evaluation yet. Its champion card will say “Not evaluated.”';
  $('promote-dialog').showModal();$('champion-name').focus();
}
async function refresh() {
  if(pollRunning)return;pollRunning=true;
  try {
    const previous=state;
    state=await api('/api/state'+(selectedRun?'?run_id='+encodeURIComponent(selectedRun):''));connected=true;
    if(!selectedRun||!state.runs.some(r=>r.id===selectedRun))selectedRun=active()?.id||state.runs[0]?.id||null;
    const run=selected();
    if(run){let history=samples.get(run.id)||[];const now=performance.now()/1000;history=history.filter(s=>now-s.time<20);history.push({time:now,steps:Number(run.steps||0)});samples.set(run.id,history);}
    $('connection').textContent=active()?active().status==='paused'?'Paused':'Local / '+active().mode:'Local / connected';$('connection').className='badge';
    $('updated').textContent='Updated '+new Date().toLocaleTimeString();
    if(!previous){for(const [id,key]of[['iso','iso'],['dolphin','dolphin'],['cpu','cpu_level'],['speed','speed']])$(id).value=state.config[key];}
    updateSelectors();renderMonitor();renderPolicies();renderControls();if(typeof renderMission==='function')renderMission();
    refreshAnalytics();
  }catch(error){connected=false;$('connection').textContent='Disconnected';$('connection').className='badge bad';$('updated').textContent='Connection lost · live values are stale';renderControls();}
  finally{pollRunning=false;}
}

document.querySelector('nav').addEventListener('click',e=>{const button=e.target.closest('[data-tab]');if(button)showTab(button.dataset.tab);});
document.querySelector('nav').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;const names=['monitor','arena','readiness','data','policies','overload','diagnostic'],i=names.indexOf(currentTab),next=names[(i+(e.key==='ArrowRight'?1:names.length-1))%names.length];showTab(next);$(next+'-tab').focus();e.preventDefault();});
$('run-select').onchange=()=>{selectedRun=$('run-select').value;offset=0;analyticsKey='';refresh();};
$('follow-active').onclick=()=>{selectedRun=active()?.id||state?.runs[0]?.id;offset=0;analyticsKey='';refresh();};
for(const id of ['mode','envs','recurrent','anchor'])$(id)?.addEventListener('change',renderControls);
$('envs').addEventListener('input',renderControls);

$('agent-character')?.addEventListener('change',()=>{
  $('agent-character').dataset.userChanged='true';
  delete $('checkpoint').dataset.userExplicitEmpty;
  const chosenChar=$('agent-character').value;
  const matching=(state?.checkpoints||[]).filter(c=>(c.config?.character||'FOX')===chosenChar);
  $('checkpoint').value=matching.length?matching[0].path:'';
  const targetCp=matching[0];
  if(targetCp){
    if(targetCp.architecture==='recurrent'||targetCp.recurrent){if($('recurrent'))$('recurrent').checked=true;}
    if(targetCp.config?.action_set==='controller'){if($('fast-inputs'))$('fast-inputs').checked=true;}
    else{if($('fast-inputs'))$('fast-inputs').checked=false;}
  }
  updateSelectors();
  renderControls();
});

$('checkpoint')?.addEventListener('change',()=>{
  const chosenPath=$('checkpoint').value;
  if(!chosenPath){
    $('checkpoint').dataset.userExplicitEmpty='true';
  }else{
    delete $('checkpoint').dataset.userExplicitEmpty;
    const cp=(state?.checkpoints||[]).find(c=>c.path===chosenPath);
    if(cp?.config?.character&&$('agent-character')){
      $('agent-character').value=cp.config.character;
      $('agent-character').dataset.userChanged='true';
    }
    if(cp&&(cp.architecture==='recurrent'||cp.recurrent)){if($('recurrent'))$('recurrent').checked=true;}
    if(cp?.config?.action_set==='controller'){if($('fast-inputs'))$('fast-inputs').checked=true;}
    else{if($('fast-inputs'))$('fast-inputs').checked=false;}
  }
  renderControls();
});

$('launch-form').onsubmit=e=>{
  e.preventDefault();
  if($('launch').disabled)return;
  act(async()=>{
    const evaluation=$('mode').value==='evaluate';
    const checkpoint=$('checkpoint').value||null;
    const envs=evaluation?1:Number($('envs').value);
    const isRec=Boolean($('recurrent')?.checked);
    const hasAnchor=Boolean($('anchor')?.value);
    const chosenCp=checkpoint?(state.checkpoints||[]).find(c=>c.path===checkpoint):null;
    const character=chosenCp?.config?.character||((!evaluation&&$('agent-character')?.value)?$('agent-character').value:null);
    const result=await api('/api/start',{
      mode:evaluation?'evaluate':'train',
      execution_mode:$('execution-mode')?.value||'assisted',
      league_checkpoints:!evaluation&&$('league-enabled')?.checked?Array.from($('league-history').selectedOptions,o=>o.value):null,
      checkpoint,
      envs,
      character:evaluation?null:character,
      steps:+$('budget').value,
      episodes:+$('episodes').value,
      bootstrap:evaluation||checkpoint||envs>1?0:+$('bootstrap').value,
      curriculum:evaluation?false:$('curriculum').checked,
      level:$('training-level').value?Number($('training-level').value):null,
      opponent_checkpoint:evaluation||$('league-enabled')?.checked?null:$('opponent-policy').value||null,
      benchmark:evaluation&&$('benchmark').checked,
      fast_inputs:!evaluation&&($('fast-inputs').checked||hasAnchor||isRec||chosenCp?.config?.action_set==='controller'),
      anchor:evaluation?null:$('anchor').value||null,
      finetune:!evaluation&&hasAnchor,
      recurrent:!evaluation&&(isRec||chosenCp?.architecture==='recurrent')
    });
    selectedRun=result.id;offset=0;analyticsKey='';
    showTab('monitor');
    message(evaluation?'Evaluation started. The exact input policy has been preserved.':(character==='JIGGLYPUFF'?(checkpoint?'Jigglypuff Rest training resumed.':'Jigglypuff Rest training started.'):'Training started.'));
  });
};
for(const action of ['pause','save','stop'])$(action).onclick=()=>act(async()=>{const command=action==='pause'&&$('pause').textContent==='Resume'?'resume':action;await api('/api/control/'+command,{});message(command==='stop'?'Stopping: saving the policy and closing every emulator. Start will unlock when teardown finishes.':command==='save'?'Checkpoint save requested.':command==='pause'?'Pause requested.':'Resuming training.');});
$('open-setup').onclick=()=>{if(!state)return;for(const [id,key]of[['iso','iso'],['dolphin','dolphin'],['cpu','cpu_level'],['speed','speed']])$(id).value=state.config[key];if($('setup-character'))$('setup-character').value=state.config.character||'FOX';$('randomize-opponent').checked=Boolean(state.config.randomize_opponent);$('setup-issue').textContent=state.issue||(!state.availability.can_start?'Finish the current run before changing setup.':'Local paths verified.');$('setup-dialog').showModal();};
document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$('setup-form').onsubmit=e=>{
  e.preventDefault();
  act(async()=>{
    const chosenChar=$('setup-character')?.value||'FOX';
    await api('/api/config',{iso:$('iso').value,dolphin:$('dolphin').value,cpu_level:+$('cpu').value,speed:+$('speed').value,randomize_opponent:$('randomize-opponent').checked,character:chosenChar});
    if($('agent-character')){
      $('agent-character').value=chosenChar;
      delete $('checkpoint').dataset.userExplicitEmpty;
    }
    $('setup-dialog').close();
    message('Setup saved for the next run.');
  });
};
for(const id of ['kind','data-cpu','outcome'])$(id).onchange=()=>{offset=0;refreshAnalytics(true);};
$('compare-runs').onchange=()=>{const chosen=Array.from($('compare-runs').selectedOptions);if(chosen.length>3){chosen.slice(3).forEach(o=>o.selected=false);message('Compare up to three additional runs.');}refreshAnalytics(true);};
$('previous').onclick=()=>{offset=Math.max(0,offset-25);refreshAnalytics(true);};$('next').onclick=()=>{if(offset+25<pageTotal){offset+=25;refreshAnalytics(true);}};
$('policy-search').oninput=()=>{renderPolicies(true);renderControls();};
$('policies-panel').addEventListener('click',e=>{
  const button=e.target.closest('[data-action]');
  if(!button||button.disabled)return;
  const path=button.dataset.path;
  if(button.dataset.action==='play'){$('challenge-policy').value=path;showTab('arena');renderMission();return;}
  if(button.dataset.action==='promote'){requestPromotion(path);return;}
  if(button.dataset.action==='diagnose'){
    showTab('diagnostic');
    if(typeof window.selectDiagnosticTarget==='function')window.selectDiagnosticTarget(path);
    return;
  }
  $('checkpoint').value=path;
  delete $('checkpoint').dataset.userExplicitEmpty;
  const cp=(state?.checkpoints||[]).find(c=>c.path===path);
  if(cp?.config?.character&&$('agent-character')){
    $('agent-character').value=cp.config.character;
    $('agent-character').dataset.userChanged='true';
  }
  if(cp&&(cp.architecture==='recurrent'||cp.recurrent)){if($('recurrent'))$('recurrent').checked=true;}
  if(cp?.config?.action_set==='controller'){if($('fast-inputs'))$('fast-inputs').checked=true;}
  $('mode').value=button.dataset.action==='evaluate'?'evaluate':'train';
  if(button.dataset.action==='evaluate')$('envs').value='1';
  updateSelectors();
  renderControls();
  $('launch-form').scrollIntoView({behavior:'smooth',block:'center'});
  $('launch').focus();
  message(`Selected policy. Review ${button.dataset.action==='evaluate'?'the episode count and CPU level':'the training options'}, then start.`);
});
$('promote-form').onsubmit=e=>{e.preventDefault();act(async()=>{$('promote-submit').disabled=true;try{const result=await api('/api/champions',{checkpoint:promotePath,name:$('champion-name').value,note:$('champion-note').value});$('promote-dialog').close();policySignature='';message(`Champion “${result.name}” preserved with its own checkpoint copy.`);}finally{$('promote-submit').disabled=false;}});};

let pendingImportFile=null;
let pendingImportPath=null;

function openImportDialog(fileOrPath, metadata=null) {
  const errEl=$('import-error');
  if(errEl){errEl.hidden=true;errEl.textContent='';}
  let fileName='';
  if(typeof fileOrPath==='string'){
    pendingImportPath=fileOrPath;
    pendingImportFile=null;
    fileName=fileOrPath.split('/').pop();
  } else {
    pendingImportFile=fileOrPath;
    pendingImportPath=null;
    fileName=fileOrPath.name;
  }

  const ext=fileName.split('.').pop().toLowerCase();
  const isGci=(ext==='gci'||ext==='raw');

  $('import-file-name').textContent=fileName;
  if(isGci){
    $('import-dialog-title').textContent='Import GameCube Save File';
    $('import-policy-fields').hidden=true;
    $('import-gci-fields').hidden=false;
    $('import-submit').textContent='Install into Dolphin Card A ↗';
  } else {
    $('import-dialog-title').textContent='Import Policy Checkpoint';
    $('import-policy-fields').hidden=false;
    $('import-gci-fields').hidden=true;
    $('import-submit').textContent='Import into Policy Vault ↗';

    const lower=fileName.toLowerCase();
    let char='JIGGLYPUFF';
    if(lower.includes('fox'))char='FOX';
    else if(lower.includes('marth'))char='MARTH';
    else if(lower.includes('falco'))char='FALCO';
    else if(lower.includes('falcon')||lower.includes('cpt'))char='CPTFALCON';
    else if(lower.includes('peach'))char='PEACH';
    else if(lower.includes('sheik'))char='SHEIK';
    else if(lower.includes('puff')||lower.includes('jigg'))char='JIGGLYPUFF';
    if(metadata?.character&&metadata.character!=='UNKNOWN')char=metadata.character;

    $('import-character').value=char;

    let stem=fileName.replace(/\.zip$/i,'').replace(/[-_]/g,' ');
    stem=stem.charAt(0).toUpperCase()+stem.slice(1);
    $('import-display-name').value=metadata?.name||stem;
  }

  $('import-save-dialog').showModal();
}
window.openImportDialog=openImportDialog;

$('import-save-btn')?.addEventListener('click',()=>$('policy-save-input')?.click());
$('sidebar-import-save')?.addEventListener('click',()=>$('policy-save-input')?.click());
$('policy-save-input')?.addEventListener('change',e=>{
  if(e.target.files&&e.target.files[0]){
    openImportDialog(e.target.files[0]);
    e.target.value='';
  }
});

$('import-save-form')?.addEventListener('submit',e=>{
  e.preventDefault();
  act(async()=>{
    $('import-submit').disabled=true;
    const errEl=$('import-error');
    if(errEl)errEl.hidden=true;
    try{
      const char=$('import-character').value;
      const dispName=$('import-display-name').value;
      let url='/api/checkpoints/import?';
      let options={method:'POST'};

      if(pendingImportFile){
        url+=`filename=${encodeURIComponent(pendingImportFile.name)}&character=${encodeURIComponent(char)}&name=${encodeURIComponent(dispName)}`;
        options.body=pendingImportFile;
      } else if(pendingImportPath){
        const fname=pendingImportPath.split('/').pop();
        url+=`filename=${encodeURIComponent(fname)}&source_path=${encodeURIComponent(pendingImportPath)}&character=${encodeURIComponent(char)}&name=${encodeURIComponent(dispName)}`;
      } else {
        throw new Error('No save file selected.');
      }

      const resp=await fetch(url,options);
      if(!resp.ok){
        const err=await resp.json();
        throw new Error(err.detail||'Import failed');
      }
      const data=await resp.json();
      $('import-save-dialog').close();
      policySignature='';
      message(data.message||'Save file imported successfully!');

      await refresh();

      if(data.type==='checkpoint'&&data.path){
        $('checkpoint').value=data.path;
        delete $('checkpoint').dataset.userExplicitEmpty;
        if(data.character&&$('agent-character')){
          $('agent-character').value=data.character;
          $('agent-character').dataset.userChanged='true';
        }
        if(data.architecture==='recurrent'&&$('recurrent')){
          $('recurrent').checked=true;
        }
        updateSelectors();
        renderControls();
        showTab('policies');
      }
    }catch(err){
      if(errEl){
        errEl.textContent=err.message;
        errEl.hidden=false;
      }else{
        alert(err.message);
      }
    }finally{
      $('import-submit').disabled=false;
    }
  });
});

function applyTheme(name){const valid=['slippi','terminal','arena'];const theme=valid.includes(name)?name:'slippi';document.documentElement.setAttribute('data-theme',theme);const picker=$('theme-select');if(picker&&picker.value!==theme)picker.value=theme;try{localStorage.setItem('melee-theme',theme);}catch(_){}const metaTheme=document.querySelector('meta[name="theme-color"]');if(metaTheme){metaTheme.content=theme==='terminal'?'#0f1015':theme==='arena'?'#0b0d16':'#0a0c16';}}
try{applyTheme(localStorage.getItem('melee-theme')||'slippi');}catch(_){applyTheme('slippi');}
$('theme-select')?.addEventListener('change',e=>applyTheme(e.target.value));
showTab('monitor');refresh();setInterval(refresh,1000);

'use strict';
let featuredIndex=0, inputSocket=null, inputArmed=false, pressedKeys=new Set(), lastInputSent=0, diagnosticRun=null;
const neutralFrame=()=>({buttons:[],main:[.5,.5],c:[.5,.5],triggers:[0,0]});
function renderMissionControls(){
  const evaluation=$('mode').value==='evaluate', frozen=!!$('opponent-policy').value;
  $('fast-inputs-label').hidden=evaluation;
  $('fast-inputs-help').hidden=evaluation||!$('fast-inputs').checked;
  $('opponent-policy-label').hidden=evaluation;
  const isRecurrent = !!$('recurrent')?.checked;
  const already=state.checkpoints.find(c=>c.path===$('checkpoint').value)?.config?.action_set==='controller';
  if((isRecurrent || !!$('anchor')?.value) && !$('checkpoint').value) $('fast-inputs').checked=true;
  const cloned=$('checkpoint').value?already:($('fast-inputs').checked||isRecurrent||!!$('anchor')?.value);
  // Offering the controller transfer for a policy that already has it can only do harm.
  $('fast-inputs-label').hidden=evaluation||already||isRecurrent;
  if(already)$('fast-inputs').checked=false;
  if($('recurrent-label')){
    const isRecurrentCheckpoint=state.checkpoints.find(c=>c.path===$('checkpoint').value)?.architecture==='recurrent';
    $('recurrent-label').hidden=evaluation;
    $('recurrent-help').hidden=evaluation||!$('recurrent').checked;
    if(isRecurrentCheckpoint){$('recurrent').checked=true;$('recurrent').disabled=true;}
    else{$('recurrent').disabled=false;}
  }
  $('anchor-label').hidden=evaluation;
  $('anchor-help').hidden=evaluation||!$('anchor').value;
  $('benchmark-label').hidden=!evaluation;
  $('benchmark-help').hidden=!evaluation||!$('benchmark').checked;
  $('level-label').hidden=(!evaluation&&frozen)||(evaluation&&$('benchmark').checked);
  if(frozen&&!evaluation){$('curriculum').disabled=true;$('constraints').textContent='Train against a frozen checkpoint. The opponent is snapshotted; its weights stay fixed throughout the run.';}
  $('start-challenge').disabled=pending||!connected||!state?.ready||!state?.availability.can_start||!$('challenge-policy').value;
  $('arm-input').disabled=!connected||active()?.mode!=='play';
  if(inputArmed&&(!connected||active()?.mode!=='play'))releaseInput();
}
function renderMission(){
  if(!state)return;
  const candidates=state.checkpoints.filter(c=>!c.path.endsWith('opponent-policy.zip'));
  const entries=candidates.map(c=>[c.path,`${c.champion?'★ ':''}${c.name} · ${fmt(c.steps)} steps`]);
  setOptions('challenge-policy',entries.length?entries:[['','No saved policies yet']],$('challenge-policy').value||entries[0]?.[0]);
  setOptions('opponent-policy',[['','Built-in CPU / curriculum'],...entries],$('opponent-policy').value);
  const sets=(state.datasets||[]).map(d=>[d.path,`${d.name} · ${d.samples?d.samples.toLocaleString():'?'} frames${d.returns?'':' · no value targets'}${d.features&&d.features!==1836?' · INCOMPATIBLE':''}`]);
  setOptions('anchor',[['','None · pure PPO'],...sets],$('anchor').value);
  const policy=candidates.find(c=>c.path===$('challenge-policy').value);
  const e=policy?.evaluations[0];
  $('challenge-evidence').textContent=policy?`${fmt(policy.steps)} training decisions · ${e?`${percent(e.win_rate)} CPU evaluation win rate (${e.episodes} games).`:'No linked evaluation yet.'} Human skill is unproven.`:'Save a policy from a training run first.';
  const run=selected(),slots=run?.slots||[];
  if(!slots.some(s=>s.index===featuredIndex))featuredIndex=slots[0]?.index||0;
  setOptions('featured-slot',slots.length?slots.map(s=>[String(s.index),`ENV ${String(s.index).padStart(2,'0')}`]):[['0','NO EMULATOR']],String(featuredIndex));
  const slot=slots.find(s=>s.index===featuredIndex),players=slot?.players||{};
  $('featured-label').textContent=!connected?'CONNECTION LOST':run?`${activeStates.includes(run.status)?run.status.toUpperCase():'RECORDED STATE'} / FRAME ${fmt(slot?.frame)}`:'AWAITING SIGNAL';
  $('featured-score').innerHTML=[1,2].map(port=>{const p=players[port];return `<div class="fighter fighter-${port}"><span class="fighter-port">P${port}</span><div><small>${port===1?'NEURAL AGENT':run?.mode==='play'?'YOU':slot?.opponent_type==='policy'?'FROZEN POLICY':'OPPONENT'}</small><strong>${esc(p?.character||'—')}</strong></div><b>${p?number(p.percent)+'%':'—'}</b><span class="stock-pips">${p?'●'.repeat(Math.max(0,Math.min(5,p.stocks))):'○ ○ ○'}</span></div>`;}).join('<span class="score-versus">VS</span>');
  const start=run?.initial_steps??null, budget=run?.requested_steps||0;
  const done=start!=null?Math.max(0,(run.steps||0)-start):null;
  const isTraining=run?.mode==='train', count=isTraining?done:run?.matches, target=isTraining?budget:run?.target_episodes;
  $('session-progress').textContent=run?(count!=null&&target?`${fmt(count)} / ${fmt(target)} ${isTraining?'decisions':'games'}`:run.phase||'Initializing'):'Waiting for a run';
  $('progress-fill').style.width=count!=null&&target?Math.min(100,count/target*100)+'%':'0%';
  $('progress-detail').textContent=run?.mode==='train'&&start==null?'This older run does not report a session starting counter.':run?.mode==='play'?'Real-time challenge · automatic rematches · results from the bot’s perspective':'Session budget excludes decisions from earlier training runs.';
  const r=state.readiness;
  if(r){
    $('readiness-title').textContent=r.label;$('readiness-explanation').textContent=r.explanation;
    const stats=[['EVALUATION GAMES',fmt(r.evidence.evaluation_games),'Exact checkpoint identity'],['ROSTER BENCHMARKS',fmt(r.evidence.benchmark_trials),'Six CPU-9 matchups'],['HUMAN SESSIONS',fmt(r.evidence.human_sessions),'Completed local sessions'],['FREE STORAGE',r.storage.free_gb+' GB','Replays and checkpoints']];
    $('evidence-metrics').innerHTML=stats.map(([label,value,note])=>`<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
    $('readiness-checks').innerHTML=r.checks.map((c,i)=>`<article class="readiness-check"><div><span class="check-number">${String(i+1).padStart(2,'0')}</span><span class="capability ${esc(c.status)}">${esc(c.status)}</span></div><h3>${esc(c.name)}</h3><p>${esc(c.detail)}</p></article>`).join('');
    const a=r.architecture;
    $('architecture').innerHTML=`<div><small>POLICY</small><b>${esc(a.policy)}</b></div><span>→</span><div><small>OBSERVATION</small><b>${fmt(a.observation_size)} features</b></div><span>→</span><div><small>CONTROLLER</small><b>${a.actions} macro actions</b></div><span>→</span><div><small>ENVIRONMENT</small><b>Real Melee / ${a.stocks} stocks</b></div>`;
  }
  const challenge=run?.mode==='play'?run:active()?.mode==='play'?active():null;
  const score=challengeScores.get(challenge?.id);
  $('challenge-score').innerHTML=challenge?`<div><small>BOT WINS</small><strong>${challenge.wins||0}</strong></div><span>:</span><div><small>YOUR WINS</small><strong>${score?.outcomes.loss??'—'}</strong></div><p>${fmt(challenge.matches)} games recorded · ${esc(challenge.status)}<br><span class="muted">${esc(challenge.id)}</span></p>`:'<p>No challenge selected. Start a session or choose a previous play run above.</p>';
  if(challenge)loadChallengeScore(challenge);
  renderMissionControls();
  if($('diagnostics').open&&diagnosticRun!==selectedRun)loadDiagnostics();
}
let challengeScoreKey='';
const challengeScores=new Map();
async function loadChallengeScore(run){
  const key=JSON.stringify([run.id,run.match_revision]);if(key===challengeScoreKey)return;challengeScoreKey=key;
  try{const report=await query(run,'human challenge','','',0);if(challengeScoreKey!==key)return;const g=report.groups[0];if(!g)return;
    $('challenge-score').innerHTML=`<div><small>BOT WINS</small><strong>${g.wins}</strong></div><span>:</span><div><small>YOUR WINS</small><strong>${g.outcomes.loss}</strong></div><p>${fmt(g.episodes)} games · ${g.outcomes.draw} draws · ${g.outcomes.timeout} timeouts<br><span class="muted">${esc(run.id)}</span></p>`;
    // Keep the scoreboard stable during the next telemetry poll.
    challengeScores.set(run.id,g);
  }catch(e){challengeScoreKey='';}
}
async function loadDiagnostics(){
  const id=selectedRun;if(!id)return;diagnosticRun=id;
  try{const result=await api(`/api/runs/${encodeURIComponent(id)}/diagnostics`);if(id!==selectedRun)return;
    $('worker-log').textContent=result.log;
    $('artifact-list').innerHTML=result.artifacts.length?result.artifacts.map(a=>`<a download href="/api/runs/${encodeURIComponent(id)}/artifact/${a.name.split('/').map(encodeURIComponent).join('/')}"><span>${esc(a.name)}</span><small>${(a.bytes/1024).toFixed(1)} KB ↓</small></a>`).join(''):'<p class="help">No artifacts recorded yet.</p>';
  }catch(e){$('worker-log').textContent=e.message;diagnosticRun=null;}
}
function releaseInput(){
  if(inputSocket?.readyState===WebSocket.OPEN)inputSocket.send(JSON.stringify(neutralFrame()));
  inputSocket?.close();inputSocket=null;inputArmed=false;pressedKeys.clear();
  $('input-status').textContent='DISARMED';$('arm-input').textContent='Connect controls';
}
$('arm-input').onclick=()=>{
  if(inputArmed){releaseInput();return;}
  const run=active();if(run?.mode!=='play')return;
  inputSocket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/play/${encodeURIComponent(run.id)}/input`);
  const socket=inputSocket;
  socket.onopen=()=>{if(inputSocket!==socket)return;inputArmed=true;$('input-status').textContent='CONNECTED / P2';$('arm-input').textContent='Release controls';$('arm-input').blur();};
  socket.onclose=event=>{if(inputSocket!==socket)return;releaseInput();if(event.code===1008)message('Input is unavailable: another tab may own the controller, or the challenge has ended.',true);};
  socket.onerror=()=>{if(inputSocket===socket)message('Controller connection failed.',true);};
};
const keys=new Set(['KeyW','KeyA','KeyS','KeyD','KeyJ','KeyK','KeyL','Space','ShiftLeft','ShiftRight','ArrowUp','ArrowDown','ArrowLeft','ArrowRight']);
window.addEventListener('keydown',e=>{if(e.code==='Escape'){releaseInput();return;}if(!inputArmed||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;if(keys.has(e.code)){e.preventDefault();pressedKeys.add(e.code);}});
window.addEventListener('keyup',e=>{pressedKeys.delete(e.code);if(inputArmed&&keys.has(e.code))e.preventDefault();});
window.addEventListener('blur',releaseInput);document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseInput();});window.addEventListener('pagehide',releaseInput);
/* A lone Joy-Con held sideways is not a "standard" gamepad: the browser reports it with
   its own button numbering and with the stick still in the Joy-Con's upright frame, so
   pushing towards the SL/SR rail reads as sideways. Rather than guess at a hardware
   convention, the stick is rotated by a quarter turn you can spin until it matches, and
   the raw axes and pressed button indices are shown live so a wrong mapping is visible
   instead of mysterious. The choice is remembered.

   One stick means no C-stick. In Melee that costs nothing that matters: smashes come
   from stick plus A, which is how most of the inputs in the training footage were made
   anyway. */
const PAD = {
  profile: localStorage.getItem('padProfile') || 'standard',
  rotation: Number(localStorage.getItem('padRotation') || 0),   // quarter turns
};
// Face buttons first, then the shoulder rails. Indices are a starting point; the
// readout below tells you what your Joy-Con actually reports.
const SINGLE_PAD_BUTTONS = [[0,'A'],[1,'B'],[2,'Y'],[3,'Z'],[4,'L'],[5,'R']];
const STANDARD_PAD_BUTTONS = [[0,'A'],[1,'X'],[2,'B'],[3,'Y'],[4,'L'],[5,'Z']];

function syncPadUI(){
  $('pad-profile').value = PAD.profile;
  const single = PAD.profile === 'single';
  $('pad-rotate').hidden = !single;
  $('pad-readout').hidden = !single;
  $('pad-rotation').textContent = `${PAD.rotation * 90}°`;
}
$('pad-profile').onchange = () => {
  PAD.profile = $('pad-profile').value;
  localStorage.setItem('padProfile', PAD.profile); syncPadUI();
};
$('pad-rotate').onclick = () => {
  PAD.rotation = (PAD.rotation + 1) % 4;
  localStorage.setItem('padRotation', String(PAD.rotation)); syncPadUI();
};
syncPadUI();

function rotate(x, y, quarters){
  for (let i = 0; i < quarters; i++) [x, y] = [y, -x];
  return [x, y];
}

function pollController(now){
  const pads = Array.from(navigator.getGamepads?.() || []).filter(Boolean);
  // Prefer a standard mapping when one exists, but never refuse a pad for lacking it --
  // that is exactly what a single Joy-Con lacks.
  const pad = pads.find(p => p.mapping === 'standard') || pads[0] || null;
  const single = PAD.profile === 'single';
  $('gamepad-name').textContent = pad
    ? `${pad.id}${pad.mapping === 'standard' ? ' · standard mapping' : ' · custom mapping'}`
    : 'Keyboard ready · connect a gamepad or Joy-Con for analog input.';
  if (single && pad) {
    const held = pad.buttons.map((b, i) => b.pressed ? i : -1).filter(i => i >= 0);
    $('pad-readout').textContent =
      `axes ${pad.axes.map(a => (a || 0).toFixed(2)).join('  ')}\nheld ${held.length ? held.join(', ') : '—'}`;
  }
  if(inputArmed&&inputSocket?.readyState===WebSocket.OPEN&&now-lastInputSent>=1000/60){
    const f=neutralFrame(),has=k=>pressedKeys.has(k);
    f.main=[.5+.5*(Number(has('KeyD'))-Number(has('KeyA'))),.5+.5*(Number(has('KeyW'))-Number(has('KeyS')))];
    f.c=[.5+.5*(Number(has('ArrowRight'))-Number(has('ArrowLeft'))),.5+.5*(Number(has('ArrowUp'))-Number(has('ArrowDown')))];
    for(const [key,button]of [['KeyJ','A'],['KeyK','B'],['KeyL','Z'],['Space','Y'],['ShiftLeft','L'],['ShiftRight','R']])if(has(key))f.buttons.push(button);
    if(pad){
      const axis=i=>Math.abs(pad.axes[i]||0)<.12?0:Math.max(-1,Math.min(1,pad.axes[i]||0));
      if(axis(0)||axis(1)){
        const [x,y]=single?rotate(axis(0),axis(1),PAD.rotation):[axis(0),axis(1)];
        f.main=[(x+1)/2,(1-y)/2];
      }
      // A lone Joy-Con has no second stick, so the C-stick stays on the arrow keys.
      if(!single&&(axis(2)||axis(3)))f.c=[(axis(2)+1)/2,(1-axis(3))/2];
      for(const [i,b]of (single?SINGLE_PAD_BUTTONS:STANDARD_PAD_BUTTONS))if(pad.buttons[i]?.pressed)f.buttons.push(b);
      f.buttons=Array.from(new Set(f.buttons));
      if(!single){
        f.triggers=[pad.buttons[6]?.value||0,pad.buttons[7]?.value||0];
        if(f.triggers[0]>.9&&!f.buttons.includes('L'))f.buttons.push('L');
        if(f.triggers[1]>.9&&!f.buttons.includes('R'))f.buttons.push('R');
      } else if(f.buttons.includes('L')) f.triggers=[1,0];   // SL is a digital shield
    }
    inputSocket.send(JSON.stringify(f));lastInputSent=now;
    $('stick-display').style.transform=`translate(${(f.main[0]-.5)*18}px,${(.5-f.main[1])*18}px)`;
    for(const b of ['a','b','y'])$('button-'+b).classList.toggle('lit',f.buttons.includes(b.toUpperCase()));
  }
  requestAnimationFrame(pollController);
}
requestAnimationFrame(pollController);
$('challenge-form').onsubmit=e=>{e.preventDefault();if($('start-challenge').disabled)return;act(async()=>{releaseInput();const result=await api('/api/start',{mode:'play',checkpoint:$('challenge-policy').value,human_character:$('human-character').value,episodes:Number($('challenge-games').value),bootstrap:0,curriculum:false,envs:1});selectedRun=result.id;challengeScoreKey='';message('Challenge starting. When Dolphin is ready, connect controls here and keep the dashboard focused.');});};
$('challenge-policy').onchange=renderMission;
for(const id of ['opponent-policy','benchmark','training-level','fast-inputs','anchor'])$(id).onchange=renderControls;
$('featured-slot').onchange=()=>{featuredIndex=Number($('featured-slot').value);renderMission();};
$('fullscreen-arena').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.featured').requestFullscreen();}catch(e){message('Full-screen is unavailable in this browser.');}};
$('diagnostics').ontoggle=()=>{if($('diagnostics').open)loadDiagnostics();};$('refresh-diagnostics').onclick=loadDiagnostics;

let arenaFrame=0;
function drawFeatured(time){
  if(time-arenaFrame<33||currentTab!=='monitor'){requestAnimationFrame(drawFeatured);return;}arenaFrame=time;
  const canvas=$('featured-arena'),rect=canvas.getBoundingClientRect();if(!rect.width){requestAnimationFrame(drawFeatured);return;}
  const ratio=Math.min(window.devicePixelRatio||1,2),w=rect.width,h=rect.height;
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const c=canvas.getContext('2d');c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,w,h);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches,t=reduced?0:time/1000;
  const glow=c.createRadialGradient(w*.53,h*.46,10,w*.53,h*.46,w*.6);glow.addColorStop(0,'#241b48');glow.addColorStop(.55,'#111323');glow.addColorStop(1,'#0c0f1c');c.fillStyle=glow;c.fillRect(0,0,w,h);
  c.strokeStyle='#383054';c.lineWidth=.5;
  const horizon=h*.52;
  for(let i=-12;i<=12;i++){c.beginPath();c.moveTo(w/2+i*20,horizon);c.lineTo(w/2+i*95,h);c.stroke();}
  for(let i=0;i<8;i++){let y=horizon+((i/8)**2)*(h-horizon);c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke();}
  for(let i=0;i<36;i++){let x=(i*131.7)%w,y=((i*47.3+t*(i%3+1))%(h*.7));c.fillStyle=i%3?'#605c86':'#9b94cb';c.globalAlpha=.2+(i%4)*.12;c.fillRect(x,y,1,1);}c.globalAlpha=1;
  const ground=h*.65,scale=Math.min(w/255,h/110),left=w/2-85*scale,right=w/2+85*scale;
  c.strokeStyle='#685695';c.fillStyle='#171e36';c.beginPath();c.moveTo(left,ground);c.lineTo(right,ground);c.lineTo(right-35*scale,ground+25);c.lineTo(w/2,ground+44);c.lineTo(left+35*scale,ground+25);c.closePath();c.fill();c.stroke();
  c.fillStyle='#252343';c.beginPath();c.moveTo(left,ground);c.lineTo(left+20,ground-13);c.lineTo(right-20,ground-13);c.lineTo(right,ground);c.closePath();c.fill();
  c.shadowColor='#b494ff';c.shadowBlur=14;c.strokeStyle='#c6a7ff';c.lineWidth=2;c.beginPath();c.moveTo(left,ground);c.lineTo(right,ground);c.stroke();c.shadowBlur=0;
  c.strokeStyle='#8b60c1';c.lineWidth=1;for(let i=0;i<5;i++){c.beginPath();c.moveTo(left+(right-left)*i/4,ground+1);c.lineTo(w/2,ground+44);c.stroke();}
  const run=selected(),slot=run?.slots?.find(s=>s.index===featuredIndex),players=slot?.players;
  if(players){for(const [port,p]of Object.entries(players)){
    const x=Math.max(18,Math.min(w-18,w/2+p.x*scale)),y=Math.max(35,Math.min(h-20,ground-14-p.y*scale));
    const color=port==='1'?'#7df5da':'#ff8fac';
    c.strokeStyle=color;c.globalAlpha=.25;c.setLineDash([3,5]);c.beginPath();c.moveTo(x,y);c.lineTo(x,ground);c.stroke();c.setLineDash([]);c.globalAlpha=1;
    c.shadowBlur=22;c.shadowColor=color;c.fillStyle=color;
    c.beginPath();c.moveTo(x,y-11);c.lineTo(x+8,y);c.lineTo(x,y+11);c.lineTo(x-8,y);c.closePath();c.fill();c.shadowBlur=0;
    c.font='10px ui-monospace,monospace';c.textAlign='center';c.fillText(`P${port} · ${p.character}`,x,y-25);
    c.fillStyle='#8d94ae';c.font='9px ui-monospace,monospace';c.fillText(`${p.x.toFixed(1)}, ${p.y.toFixed(1)}`,x,y+28);
  }}else{c.fillStyle='#8d94ae';c.font='11px ui-monospace,monospace';c.textAlign='center';c.fillText('AWAITING EMULATOR TELEMETRY',w/2,h*.34);}
  c.font='9px ui-monospace,monospace';c.fillStyle='#797797';c.textAlign='center';c.fillText('FINAL DESTINATION / SCHEMATIC',w/2,h-20);
  requestAnimationFrame(drawFeatured);
}
requestAnimationFrame(drawFeatured);

let moveCatalog=null, moveCatalogKey='';
async function loadMoves(){
  const expanded=$('expanded-moves').checked;
  try{moveCatalog=await api('/api/moves?expanded='+expanded);renderMoves();}catch(e){$('move-summary').textContent=e.message;}
}
function renderMoves(){
  if(!moveCatalog)return;
  const d=moveCatalog,q=$('move-search').value.toLowerCase();
  $('move-summary').innerHTML=`<div><small>INPUT VOCABULARY</small><strong>${d.count}</strong></div><div><small>DECISION FLOOR</small><strong>${d.decision_floor_ms} ms</strong></div><div><small>ENGINE INPUT LIMIT</small><strong>${d.max_decisions_per_game_second}/s</strong></div><p>${esc(d.note)}</p>`;
  $('move-body').innerHTML=d.moves.filter(m=>m.name.toLowerCase().includes(q)).map(m=>`<tr><td>${esc(m.name)}</td><td>${m.frames}</td><td>${m.hold?'Held / composable':m.inputs.length>1?'Macro sequence':'Pulse'}</td><td>${m.inputs[0].stick.join(', ')}</td><td>${esc(m.inputs.map(i=>i.buttons.map(b=>b.replace('BUTTON_','')).join('+')||'—').join(' → '))}${m.inputs[0].triggers?.some(v=>v>0)?' / analog shield':''}</td></tr>`).join('');
}
$('expanded-moves').onchange=loadMoves;$('move-search').oninput=renderMoves;loadMoves();

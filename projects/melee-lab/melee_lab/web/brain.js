'use strict';
const $ = id => document.getElementById(id);
const fmt = n => n == null ? '—' : Number(n).toLocaleString();
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const short = n => n == null ? '—' : n >= 1e6 ? (n/1e6).toFixed(2)+'m' : n >= 1e3 ? (n/1e3).toFixed(1)+'k' : fmt(n);
let data = null, mode = 'activity', selected = 0, lineageKey = '', samples = [], lastReceipt = 0;
let lastDraw = 0, lastRun = '', previousFrames = [], pulses = [], width = 0, height = 0;
const canvas = $('flow'), ctx = canvas.getContext('2d');

function switchMode(next) {
  mode = next;
  $('activity-tab').setAttribute('aria-selected', String(mode === 'activity'));
  $('lineage-tab').setAttribute('aria-selected', String(mode === 'lineage'));
  canvas.hidden = mode !== 'activity'; $('lineage-view').hidden = mode !== 'lineage';
  $('slot-label').hidden = mode !== 'activity';
  $('diagram-title').textContent = mode === 'activity' ? 'ONE SHARED POLICY · PARALLEL EXPERIENCE' : 'RECORDED CHECKPOINTS → CURRENT RUN';
  $('caption').textContent = mode === 'activity' ? 'Live game telemetry · flow is schematic, not neuron activations' : 'Counters show the frozen input used at each resume · scroll to explore';
  if (mode === 'lineage') $('lineage-view').scrollLeft = $('lineage-view').scrollWidth;
  resize();
}
$('activity-tab').onclick = () => switchMode('activity');
$('lineage-tab').onclick = () => switchMode('lineage');
$('slot').onchange = () => { selected = Number($('slot').value); };
document.querySelector('.tabs').addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    switchMode(mode === 'activity' ? 'lineage' : 'activity');
    $(mode+'-tab').focus(); e.preventDefault();
  }
});

function renderLineage(d) {
  const key = JSON.stringify(d.lineage.map(n => [n.id, n.kind, n.current ? null : n.steps, n.sha256]));
  if (key === lineageKey) return;
  lineageKey = key;
  const chain = d.lineage, compact = chain.slice(-3);
  $('depth').textContent = chain.length+' runs';
  $('ancestors').innerHTML = compact.map(n => `<div class="ancestor ${n.current ? 'current' : ''}"><i></i><div><strong>${esc(n.current ? 'Training now' : n.kind)}</strong><small>${n.current ? 'Live policy · '+d.slots.length+' emulators' : fmt(n.steps)+' decisions at resume'}</small><small class="run-id" title="${esc(n.id)}">${esc(n.id)}</small></div></div>`).join('');
  $('source-note').textContent = chain.length > 3 ? `${chain.length-3} earlier runs · open Full lineage` : 'Recorded resume chain · inputs are frozen';
  $('lineage-view').innerHTML = '<div class="lineage-chain">'+chain.map((n,i) => `<article class="lineage-card ${n.current ? 'current' : ''}"><small>${n.current ? 'CURRENT RUN' : 'ANCESTOR '+(i+1)}</small><div class="node-kind">${esc(n.kind)}</div><b>${n.current ? 'Learning live' : short(n.steps)+' decisions'}</b><small>${esc(n.id)}</small>${n.samples ? `<small>${fmt(n.samples)} demonstration samples</small>` : ''}${n.sha256 ? `<small title="${esc(n.sha256)}">Snapshot ${esc(n.sha256.slice(0,12))}</small>` : ''}${n.missing ? '<small>Source folder unavailable</small>' : ''}</article>`).join('')+'</div>';
  if (mode === 'lineage') $('lineage-view').scrollLeft = $('lineage-view').scrollWidth;
}

window.renderBrain = function(d) {
  if (!d || !Array.isArray(d.slots)) return;
  const now = performance.now();
  if (lastRun !== d.id) { samples = []; previousFrames = []; pulses = []; lastRun = d.id; lineageKey = ''; }
  if (!data || d.timestamp !== data.timestamp) {
    lastReceipt = now;
    d.slots.forEach((s,i) => {
      if (s.frame !== previousFrames[i] && s.frame != null && !s.stale && !s.failed && d.status === 'running') pulses[i] = now;
      previousFrames[i] = s.frame;
    });
    samples.push({time:d.timestamp, steps:d.steps});
    samples = samples.filter(s => d.timestamp-s.time <= 20);
  }
  data = d;
  if ($('slot').options.length !== d.slots.length) {
    $('slot').replaceChildren(...d.slots.map(s => new Option('Emulator '+(s.index+1), s.index)));
    selected = Math.min(selected, d.slots.length-1); $('slot').value = selected;
  }
  $('identity').textContent = d.character+' / '+d.id;
  $('steps').textContent = fmt(d.session_steps);
  $('budget').textContent = '/ '+short(d.requested_steps)+' decisions';
  const fraction = d.requested_steps ? Math.min(1,d.session_steps/d.requested_steps) : 0;
  $('progress-bar').style.width = fraction*100+'%';
  $('completion').textContent = (fraction*100).toFixed(2)+'% · '+short(d.steps)+' total';
  $('cpu').textContent = 'CPU '+(d.cpu_level ?? '—');
  const first = samples[0], seconds = d.timestamp-first.time;
  $('rate').textContent = d.status === 'running' && seconds >= 3 ? fmt(Math.max(0,Math.round((d.steps-first.steps)/seconds)))+' /s' : '— /s';
  const wins = d.recent.filter(r => r.result === 'win').length;
  $('record').textContent = d.recent.length ? wins+'/'+d.recent.length+' wins' : 'No matches yet';
  $('results').innerHTML = d.recent.map(r => `<span class="result ${r.result === 'win' ? 'win' : r.result === 'loss' ? 'loss' : 'other'}" title="${esc(r.result)} · CPU ${esc(r.cpu_level)} · return ${Number(r.return || 0).toFixed(1)}"></span>`).join('') || '<span class="result"></span>';
  $('save').textContent = d.saved_age == null ? 'No save yet' : d.saved_age < 60 ? 'Saved '+Math.floor(d.saved_age)+'s ago' : 'Saved '+Math.floor(d.saved_age/60)+'m ago';
  $('anchor').textContent = d.anchor ? 'PRO DEMONSTRATIONS · '+d.anchor.replaceAll('_',' ') + (d.anchor_weight == null ? '' : ' · weight '+d.anchor_weight.toFixed(2)) : 'PPO · '+(d.action_set === 'controller' ? 'Full controller policy' : d.action_set+' action policy');
  $('placement').textContent = d.placement+' · close anytime';
  renderLineage(d); health();
};

function health() {
  if (!data) return;
  const late = performance.now()-lastReceipt > 5000 || Date.now()/1000-data.timestamp > 8;
  const stale = data.slots.filter(s => s.stale || s.failed).length;
  $('state').textContent = late ? 'DISCONNECTED' : data.status.toUpperCase();
  $('state').style.color = late || data.status === 'failed' || stale ? '#d9b182' : '#79f2ce';
  $('health').textContent = late ? 'Telemetry connection lost · showing last recorded data' : `${data.slots.length-stale}/${data.slots.length} slots reporting · ${data.phase}` + (stale ? ' · delayed or unavailable slots shown dimmed' : ' · updates every second');
}
setInterval(health, 1000);

function resize() {
  const box = canvas.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 1.5);
  width = box.width; height = box.height;
  canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0);
}
new ResizeObserver(resize).observe(canvas);

function text(label, x, y, color='#8ea6b6', size=10, maxWidth=null, align='left') {
  ctx.fillStyle=color; ctx.font=`${size}px -apple-system, sans-serif`;ctx.textAlign=align;
  label=String(label);
  if(maxWidth){while(label.length>1 && ctx.measureText(label).width>maxWidth) label=label.slice(0,-2)+'…';}
  ctx.fillText(label,x,y);
}
function dot(x,y,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();}
function curve(x1,y1,x2,y2,color){ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x1,y1);const mid=(x1+x2)/2;ctx.bezierCurveTo(mid,y1,mid,y2,x2,y2);ctx.stroke();}
function draw(now) {
  requestAnimationFrame(draw);
  if (now-lastDraw < 33 || document.hidden || mode !== 'activity' || !width || !height) return;
  lastDraw = now;
  ctx.clearRect(0,0,width,height);
  if (!data) {text('Waiting for training telemetry…',width/2,height/2,'#8ea6b6',12,null,'center');return;}
  const small=width<480, left=small?8:14, hubX=width*(small?.35:.38), hubY=height/2;
  const outputX=width*(small?.56:.62), radius=small?26:32;
  const slot=data.slots[selected] || data.slots[0], p=slot?.players?.['1'], q=slot?.players?.['2'];
  const inputs=[['POSITION',p?`${p.x.toFixed(0)}, ${p.y.toFixed(0)}`:'—'],['DAMAGE',p?`${p.percent.toFixed(0)}% / ${q?.percent?.toFixed(0) ?? '—'}%`:'—'],['STOCKS',p?`${p.stocks} / ${q?.stocks ?? '—'}`:'—'],['DISTANCE',p&&q?`${Math.abs(p.x-q.x).toFixed(0)} units`:'—']];
  const streamLive=data.status==='running' && now-lastReceipt<5000 && Date.now()/1000-data.timestamp<8;
  const live=streamLive && !slot?.stale && !slot?.failed;
  const inputEnd=left+(small?65:90);
  inputs.forEach(([label,value],i)=>{
    const y=18+(height-36)*(i+.5)/4;
    text(label,left,y-5,'#667f91',8);text(value,left,y+9,'#c0d5e3',small?10:12);
    curve(inputEnd,y,hubX-radius,hubY,'#29404b');dot(inputEnd,y,2,live?'#7ea9b8':'#3c4d5a');
  });
  ctx.strokeStyle=live?'#315b60':'#273d48';ctx.lineWidth=1;ctx.beginPath();ctx.arc(hubX,hubY,radius+7,0,Math.PI*2);ctx.stroke();
  const glow=ctx.createRadialGradient(hubX,hubY,4,hubX,hubY,radius+25);glow.addColorStop(0,'#60dfc027');glow.addColorStop(1,'#60dfc000');ctx.fillStyle=glow;ctx.fillRect(hubX-radius-25,hubY-radius-25,(radius+25)*2,(radius+25)*2);
  dot(hubX,hubY,radius,'#142c33');
  // Deliberately an abstract policy symbol, never claimed as network topology.
  const nodes=[[-13,-9],[0,-17],[13,-7],[-9,9],[10,11],[0,0]];
  [[0,1],[1,2],[0,3],[2,4],[3,4],[0,5],[1,5],[2,5],[4,5],[3,5]].forEach(([a,b])=>{ctx.strokeStyle='#488d85';ctx.beginPath();ctx.moveTo(hubX+nodes[a][0],hubY+nodes[a][1]);ctx.lineTo(hubX+nodes[b][0],hubY+nodes[b][1]);ctx.stroke();});
  nodes.forEach(([x,y])=>dot(hubX+x,hubY+y,2.5,live?'#91f4d9':'#5f8a89'));
  text('SHARED PPO',hubX,hubY+radius+24,'#a0d7cd',small?8:9,null,'center');
  if(height>160)text(data.action_set==='controller'?'controller policy':'action policy',hubX,hubY+radius+38,'#597786',8,null,'center');
  data.slots.forEach((s,i)=>{
    const y=10+(height-20)*(i+.5)/data.slots.length;
    const active=streamLive&&!s.stale&&!s.failed, color=s.failed?'#c1798f':s.stale?'#a39360':active?'#79e5c7':'#47676b';
    curve(hubX+radius,hubY,outputX,y,i===selected?'#4a8b82':'#263f49');
    const phase=(now-(pulses[i]??-9999))/1000;
    if(active&&phase>=0&&phase<=1){const t=phase, inv=1-t, x1=hubX+radius, mid=(x1+outputX)/2;dot(inv**3*x1+3*inv*inv*t*mid+3*inv*t*t*mid+t**3*outputX,inv**3*hubY+3*inv*inv*t*hubY+3*inv*t*t*y+t**3*y,2,color);}
    dot(outputX,y,i===selected?4:3,color);
    text(String(i+1).padStart(2,'0'),outputX+10,y+3,color,9);
    text(s.failed?'Slot failed':s.stale?'Waiting for telemetry':s.action||s.connection||'Starting…',outputX+32,y+3,i===selected?'#d5ebe7':'#8da8b5',small?9:10,width-outputX-35);
  });
}
requestAnimationFrame(draw);

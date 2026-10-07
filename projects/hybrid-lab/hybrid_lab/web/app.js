import {Viewer} from './viewer.js';

const $=id=>document.getElementById(id);
const names={club:'Club circuit',national:'National circuit',coast:'Coastal run',sprint:'Sprint circuit',tech:'Technical circuit',oval2:'Speedway',akina:'Akina pass',pass:'Mountain pass'};
const cars={supra:'Toyota Supra',rx7:'Mazda RX-7',skyline:'Nissan Skyline'};
let viewer=null,state={},runs=[],history=[],activeRun=null,options=null,page='watch',failures=0,geometryGeneration=null,manual=false,metricsRun=null;
let toastTimer;
const fmt=(v,d=0)=>v==null?'—':Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function toast(message,error=false){$('toast').textContent=message;$('toast').className=error?'error':'';$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,error?8000:4000);}
async function api(path,body){
  const response=await fetch('/api/'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(15000)}:{cache:'no-store',signal:AbortSignal.timeout(10000)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;
}
async function command(body){return api('command',body);}
function bind(id,fn){$(id).addEventListener('click',async e=>{const button=e.currentTarget;button.disabled=true;try{await fn();}catch(err){toast(err.message,true);}finally{button.disabled=false;}});}
function showPage(next){page=next;document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-'+next));document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.page===next));location.hash=next;viewer?.resize();if(next==='runs')refreshRuns().catch(e=>toast(e.message,true));if(next==='train')drawChart();}

document.querySelectorAll('.nav').forEach(b=>b.onclick=()=>showPage(b.dataset.page));
for(const id of ['open-training','footer-train'])$(id).onclick=()=>showPage('train');
document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>viewer?.setCamera(b.dataset.camera));
$('show-rays').onchange=e=>{if(viewer)viewer.rays.visible=e.target.checked;};
$('show-trails').onchange=e=>{if(viewer)viewer.trails.visible=e.target.checked;};
bind('fullscreen',async()=>{if(document.fullscreenElement)await document.exitFullscreen();else await $('viewer').requestFullscreen();});
bind('play-pause',()=>command({op:'viewer',action:'pause',paused:!state.watching?.paused}));
bind('reset-view',()=>command({op:'viewer',action:'reset'}));
$('playback-rate').onchange=async e=>{try{await command({op:'viewer',action:'speed',rate:Number(e.target.value)});}catch(err){toast(err.message,true);}};
let loadedRun=null;   // the saved policy currently in the viewer, if any
$('driver-source').onchange=()=>{$('manual-help').hidden=$('driver-source').value!=='manual';};

async function watchRun(run,checkpoint='latest.pt',live=false){
  await command({op:'watch',run,checkpoint,live});manual=false;showPage('watch');
  // Point the controls at what is actually loaded. Without this they keep their
  // previous values, so the panel reads "Reference driver / Club circuit" while
  // the viewer is running a saved Akina policy — and the next "Load into viewer"
  // silently throws that policy away.
  loadedRun={run,checkpoint,live};
  const cfg=runs.find(r=>r.id===run)?.config;
  if(cfg){$('watch-car').value=cfg.car;$('watch-track').value=cfg.track;}
  $('driver-source').value=live?'live':'saved';
  $('manual-help').hidden=true;
  toast(live?'Following saved updates from the active run.':'Saved policy loaded.');
}
bind('apply-view',async()=>{
  const choice=$('driver-source').value;
  if(choice==='live'){
    if(!state.training?.run)throw new Error('Start a training run first.');
    return watchRun(state.training.run,'latest.pt',true);
  }
  if(choice==='saved'){
    if(!loadedRun)throw new Error('Open a saved policy from the Runs page first.');
    return watchRun(loadedRun.run,loadedRun.checkpoint,false);
  }
  loadedRun=null;
  await command({op:'watch',config:{car:$('watch-car').value,track:$('watch-track').value}});
  manual=choice==='manual';
  if(manual){await command({op:'viewer',action:'manual',controls:[0,0,0]});await command({op:'viewer',action:'pause',paused:false});}
  toast(manual?'Manual driving ready. Use WASD or the arrow keys.':'Reference driver loaded.');
});

$('style-weight').oninput=e=>$('style-output').value=Number(e.target.value).toFixed(3);
const TEXT_FIELDS=['name','car','track','mode'];
const BOOL_FIELDS=['anneal','hills'];
$('training-form').onsubmit=async e=>{
  e.preventDefault();const button=$('start-training');button.disabled=true;
  try{
    const config=Object.fromEntries(new FormData(e.target));
    // 'Continue from' is a request wrapper, not a Config field — the server
    // rejects unknown settings, so pull it out before sending.
    const resumeChoice=config.resume_from||'';delete config.resume_from;
    for(const key of BOOL_FIELDS)if(key in config)config[key]=config[key]==='true';
    for(const key of Object.keys(config))
      if(!TEXT_FIELDS.includes(key)&&!BOOL_FIELDS.includes(key))config[key]=Number(config[key]);
    let request={op:'start',config};
    if(resumeChoice){
      const [run,checkpoint]=resumeChoice.split('|');
      const parent=runs.find(r=>r.id===run);
      // Car, circuit and objective are fixed by the checkpoint's action contract.
      if(parent){config.car=parent.config.car;config.track=parent.config.track;
                 config.mode=parent.config.mode;config.hills=parent.config.hills;}
      // The saved config may carry a huge 'updates' budget; the anneal curve is
      // update/updates, so an unbounded budget means the rate never decays.
      // Take the horizon from this form instead.
      const done=parent?.status?.update||0;
      config.updates=Math.max(Number(config.updates)||0,done+1);
      request={op:'start',resume_run:run,checkpoint,additional_updates:1,config};
    }
    const result=await command(request);activeRun=result.run;history=[];metricsRun=result.run;
    await refreshRuns();toast(resumeChoice?'Continuing from the saved policy.':'Training started.');
  }catch(err){toast(err.message,true);}finally{button.disabled=false;}
};
$('resume-from').onchange=()=>{
  const on=!!$('resume-from').value;
  $('resume-hint').hidden=!on;
  for(const n of ['car','track','mode'])
    $('training-form').querySelector(`[name="${n}"]`).disabled=on;
};
bind('pause-training',()=>command({op:state.training?.status==='paused'?'resume':'pause'}));
bind('save-training',async()=>{await command({op:'save'});toast('Save requested. The checkpoint is written at the next safe point.');});
bind('stop-training',async()=>{await command({op:'stop'});toast('Stopping and saving the current policy.');});
bind('watch-training',()=>watchRun(state.training.run,'latest.pt',true));
bind('refresh-runs',()=>refreshRuns());

const keys=new Set();
function isTyping(){return ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName);}
window.addEventListener('keydown',e=>{if(manual&&!isTyping()&&page==='watch'&&['w','a','s','d','ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key)){keys.add(e.key);e.preventDefault();}});
window.addEventListener('keyup',e=>keys.delete(e.key));
window.addEventListener('blur',()=>keys.clear());
document.addEventListener('visibilitychange',()=>{if(document.hidden)keys.clear();});
let manualBusy=false;
setInterval(async()=>{
  if(!manual||manualBusy)return;manualBusy=true;
  const enabled=page==='watch'&&!document.hidden&&!isTyping();
  const has=(a,b)=>enabled&&(keys.has(a)||keys.has(b));
  const steer=(has('a','ArrowLeft')?1:0)-(has('d','ArrowRight')?1:0);
  const longitudinal=has('s','ArrowDown')?-1:has('w','ArrowUp')?1:0;
  try{await command({op:'viewer',action:'manual',controls:[steer,longitudinal,enabled&&keys.has(' ')?1:0]});}catch{}finally{manualBusy=false;}
},100);

function updateUI(){
  const t=state.training||{},w=state.watching||{},f=w.frame;
  const active=['starting','training','paused','evaluating'].includes(t.status);
  $('global-status').textContent=t.status==='idle'?'Ready to train':t.status==='failed'?'Training needs attention':t.status?.[0]?.toUpperCase()+t.status?.slice(1);
  $('global-steps').textContent=fmt(t.steps||0)+' steps';$('status-dot').classList.toggle('running',active);
  for(const id of ['pause-training','save-training','stop-training'])$(id).disabled=!active;
  $('start-training').disabled=active;
  const run=runs.find(r=>r.id===t.run);
  $('watch-training').disabled=!run?.checkpoints.includes('latest.pt');
  $('pause-training').textContent=t.status==='paused'?'Resume':'Pause';
  $('run-title').textContent=run?.config.name||'Waiting for an experiment';$('run-status').textContent=t.status?.toUpperCase()||'IDLE';
  $('training-progress').style.width=run?Math.min(100,(t.update||0)/run.config.updates*100)+'%':'0%';
  $('update-count').textContent=fmt(t.update||0)+' / '+fmt(run?.config.updates||0)+' updates';$('step-count').textContent=fmt(t.steps||0)+' environment steps';
  $('training-caption').textContent=run?`${run.config.name} · ${t.status} · update ${fmt(t.update||0)} of ${fmt(run.config.updates)}`:'No run active. Create an experiment to start learning.';
  if(t.error)$('save-status').textContent=t.error;
  else if(t.saved_at)$('save-status').textContent='Checkpoint saved '+new Date(t.saved_at*1000).toLocaleTimeString()+'.';
  else if(t.status==='evaluating')$('save-status').textContent='Evaluating the current policy on three repeatable starts…';
  else $('save-status').textContent='Checkpoints save automatically. The live viewer follows each saved update.';
  const m=history.at(-1)||t.metrics||{};
  $('metric-return').textContent=fmt(m.reward,2);$('metric-fps').textContent=fmt(m.fps);$('metric-progress').textContent=fmt(m.progress,3);
  for(const [id,key] of [['policy-loss','policy_loss'],['value-loss','value_loss'],['kl','kl']])$(id).textContent=fmt(m[key],4);
  $('style-ramp').textContent=m.style_scale==null?'—':fmt(m.style_scale*100)+'%';
  if(f){
    $('speed').textContent=fmt(f.speed);$('gear').textContent=f.gear;$('rpm').textContent=fmt(f.rpm)+' RPM';$('rpm-bar').style.width=Math.min(100,f.rpm/8500*100)+'%';
    $('track-title').textContent=names[w.track]||w.track;
    $('driver-label').textContent=w.source==='Reference driver'?'Reference driver · not a trained policy':w.source;
    $('play-pause').textContent=w.paused?'▶':'Ⅱ';$('play-pause').setAttribute('aria-label',w.paused?'Play simulation':'Pause simulation');
    const minutes=Math.floor(f.elapsed/60);$('lap-time').textContent=String(minutes).padStart(2,'0')+':'+(f.elapsed%60).toFixed(2).padStart(5,'0');
    $('lap-progress').innerHTML=fmt(Math.max(0,f.progress)*100,1)+'<span>%</span>';
    $('slip-angle').innerHTML=fmt(f.slip,1)+'<span>°</span>';$('drift-share').innerHTML=fmt(f.drift*100)+'<span>%</span>';
    for(const key of ['throttle','brake','handbrake']){$(key+'-meter').style.width=f[key]*100+'%';$(key+'-value').textContent=fmt(f[key]*100)+'%';}
    $('steer-marker').style.left=(50-f.steer*50)+'%';$('steer-value').textContent=fmt(f.steer,2);
    $('surface-state').textContent=f.airborne?'AIRBORNE':f.offtrack?'OFF TRACK':'ON TRACK';$('surface-state').style.color=f.offtrack?'var(--orange)':'var(--lime)';
    for(const key of ['progress','pace','style','penalty'])$('reward-'+key).textContent=fmt(f.parts?.[key],3);
    $('episode-reward').textContent=fmt(f.reward,2);
    if(w.paused)$('scene-note').textContent='Simulation paused';else if(f.reason)$('scene-note').textContent=f.reason+' · resetting';else $('scene-note').textContent=manual?'WASD / arrows to drive · Space to handbrake':'Drag to orbit · scroll to zoom';
    viewer?.setFrame(w);
  }
  if(!state.viewer_alive && state.viewer_alive!==undefined){$('scene-loading').hidden=false;$('scene-loading').textContent='The viewer stopped. Restart Hybrid Lab to reconnect.';}
  const eventText=JSON.stringify(state.events);
  if($('events').dataset.last!==eventText){$('events').dataset.last=eventText;$('events').innerHTML=(state.events||[]).slice(-6).reverse().map(e=>`<div class="event"><time>${new Date(e.time*1000).toLocaleTimeString()}</time><span>${escape(e.message)}</span></div>`).join('');}
}

async function refreshRuns(){
  runs=await api('runs');$('run-total').textContent=runs.length+' RUN'+(runs.length===1?'':'S');
  const saved=runs.filter(r=>r.checkpoints.length);
  $('runs-list').innerHTML=runs.length?runs.map(r=>{
    const ev=r.evaluation,c=r.config,s=r.status;
    let evaluation='';
    if(ev?.status==='evaluating')evaluation='<div class="evaluation">Evaluation in progress…</div>';
    else if(ev?.error)evaluation=`<div class="evaluation">Evaluation failed: ${escape(ev.error)}</div>`;
    else if(ev?.score!=null)evaluation=`<div class="evaluation">Evaluation · return <b>${fmt(ev.score,2)}</b> · progress <b>${fmt(ev.progress*100,1)}%</b> · drift <b>${fmt(ev.drift*100,1)}%</b> · off track <b>${fmt(ev.offtrack*100,1)}%</b><br>Three fixed starts · ${ev.seconds}s per start · ${escape(names[ev.track]||ev.track)}</div>`;
    const link=file=>`/api/download?run=${encodeURIComponent(r.id)}&file=${file}`;
    return `<article class="run-card"><div class="run-card-top"><div><h3>${escape(c.name)}</h3><p>${escape(cars[c.car])} / ${escape(names[c.track])} / ${escape(c.mode)} · ${fmt(s.update||0)} updates · ${fmt(s.steps||0)} steps</p><p>${escape(r.id)}</p></div><span class="tiny-badge">${escape(s.status||'saved').toUpperCase()}</span></div>${s.error?`<div class="evaluation">${escape(s.error)}</div>`:''}${evaluation}<div class="run-card-bottom"><div class="run-card-actions"><select aria-label="Checkpoint for ${escape(c.name)}" data-checkpoint="${r.id}" ${r.checkpoints.length?'':'disabled'}>${r.checkpoints.map(n=>`<option value="${n}">${n==='best.pt'?'Best evaluated':'Latest policy'}</option>`).join('')||'<option>Not saved yet</option>'}</select><button class="button" data-action="watch" data-run="${r.id}" ${r.checkpoints.length?'':'disabled'}>Watch ↗</button><button class="button" data-action="evaluate" data-run="${r.id}" ${r.checkpoints.length?'':'disabled'}>Evaluate</button><button class="button" data-action="resume" data-run="${r.id}" ${r.checkpoints.length?'':'disabled'}>Resume +100</button></div><div class="run-links"><a href="${link('config.json')}">Settings ↓</a>${(s.update||0)>0?`<a href="${link('metrics.jsonl')}">Metrics ↓</a>`:''}${r.checkpoints.map(n=>`<a href="${link(n)}">${n==='best.pt'?'Best':'Latest'} ↓</a>`).join('')}</div></div></article>`;
  }).join(''):'<div class="empty-state">No experiments yet.<br><span>Your runs and checkpoints will appear here.</span></div>';
  const picker=$('resume-from');
  if(picker){
    const keep=picker.value;
    picker.innerHTML='<option value="">Start fresh — new policy</option>'+
      saved.flatMap(r=>r.checkpoints.map(n=>{
        const label=n==='best.pt'?'best':'latest';
        const upd=r.status?.update||0;
        return `<option value="${r.id}|${n}">${escape(r.config.name)} · ${escape(names[r.config.track]||r.config.track)} · ${label} (update ${fmt(upd)})</option>`;
      })).join('');
    if([...picker.options].some(o=>o.value===keep))picker.value=keep;
    picker.onchange();
  }
  updateUI();
}
$('runs-list').addEventListener('click',async e=>{
  const b=e.target.closest('[data-action]');if(!b)return;
  const run=b.dataset.run,checkpoint=document.querySelector(`[data-checkpoint="${run}"]`).value;
  b.disabled=true;
  try{
    if(b.dataset.action==='watch')await watchRun(run,checkpoint);
    if(b.dataset.action==='evaluate'){await command({op:'evaluate',run,checkpoint});toast('Evaluation started. Results will appear here.');}
    if(b.dataset.action==='resume'){
      const result=await command({op:'start',resume_run:run,checkpoint,additional_updates:100,config:{updates:1}});
      activeRun=result.run;history=[];metricsRun=result.run;showPage('train');toast('Resumed into a separate run.');
    }
    await refreshRuns();
  }catch(err){toast(err.message,true);}finally{b.disabled=false;}
});

function drawChart(){
  const canvas=$('learning-chart'),rect=canvas.getBoundingClientRect();if(!rect.width)return;
  const dpr=Math.min(devicePixelRatio,2);canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;
  const data=history.filter(r=>r.reward!=null);$('chart-empty').hidden=data.length>0;
  const lo=data.length?Math.min(...data.map(r=>r.reward)):0,hi=data.length?Math.max(...data.map(r=>r.reward)):1;
  const pad=36,range=Math.max(1,hi-lo),x=i=>pad+i/Math.max(1,data.length-1)*(w-pad-12),y=v=>h-22-(v-lo)/range*(h-44);
  ctx.font='10px monospace';ctx.textAlign='right';
  for(let i=0;i<5;i++){const yy=20+i*(h-42)/4;ctx.strokeStyle='#394434';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(pad,yy);ctx.lineTo(w,yy);ctx.stroke();ctx.fillStyle='#809274';ctx.fillText((hi-i*range/4).toFixed(0),pad-8,yy+3);}
  if(data.length){
    ctx.beginPath();data.forEach((r,i)=>{i?ctx.lineTo(x(i),y(r.reward)):ctx.moveTo(x(i),y(r.reward));});ctx.strokeStyle='#dcff61';ctx.lineWidth=2;ctx.stroke();
    ctx.lineTo(x(data.length-1),h-22);ctx.lineTo(x(0),h-22);ctx.closePath();const gradient=ctx.createLinearGradient(0,0,0,h);gradient.addColorStop(0,'#dcff6129');gradient.addColorStop(1,'#dcff6100');ctx.fillStyle=gradient;ctx.fill();
    ctx.beginPath();ctx.arc(x(data.length-1),y(data.at(-1).reward),3,0,Math.PI*2);ctx.fillStyle='#dcff61';ctx.fill();
    $('chart-latest').textContent=`${data[0].update} → ${data.at(-1).update} · ${data.length} samples`;
  }else $('chart-latest').textContent='No completed episodes yet';
}
new ResizeObserver(drawChart).observe($('learning-chart'));

async function poll(){
  try{
    state=await api('state');failures=0;$('connection').textContent='Connected';
    const generation=state.watching?.generation;
    if(generation && geometryGeneration!==generation){
      const track=await api('track');if(track){viewer?.setTrack(track);geometryGeneration=track.generation;$('track-distance').textContent=fmt(track.length/1000,2)+' KM';}
    }
    if(state.training?.run && metricsRun!==state.training.run){metricsRun=state.training.run;history=[];}
    updateUI();
  }catch(err){
    failures++;$('connection').textContent='Disconnected';
    if(failures===3)toast('Connection lost. Check that Hybrid Lab is running.',true);
  }finally{setTimeout(poll,100);}
}
async function slowPoll(){
  try{
    if(state.training?.run){history=await api('metrics?run='+encodeURIComponent(state.training.run));drawChart();}
    // Do not replace controls while the user is choosing a checkpoint.
    if(!document.activeElement?.matches('[data-checkpoint]'))await refreshRuns();
  }catch{}finally{setTimeout(slowPoll,2500);}
}

async function init(){
  try{viewer=new Viewer($('scene'));}catch(err){$('scene-loading').textContent='3D viewer unavailable. Enable WebGL 2 in your browser. Training controls remain available.';toast('3D renderer: '+err.message,true);}
  try{
    options=await api('options');
    for(const id of ['watch-track','train-track'])$(id).innerHTML=options.tracks.map(t=>`<option value="${t}">${names[t]||t}</option>`).join('');
    await refreshRuns();
  }catch(err){toast(err.message,true);}
  showPage(['watch','train','runs'].includes(location.hash.slice(1))?location.hash.slice(1):'watch');
  poll();slowPoll();
}
init();

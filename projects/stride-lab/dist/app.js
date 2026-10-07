import {BASELINE,DEFAULT_ENV,VERSION,G,runTrial,decode,PARAMS,clamp} from './physics.js';
import {drawTrack,drawTelemetry,drawLearning} from './render.js';
const $=s=>document.querySelector(s);
const state={event:'sprint',env:{...DEFAULT_ENV},rows:[],active:null,trial:null,index:0,playing:false,speed:1,clock:0,lastTimestamp:0,chart:'force',history:[],training:null,worker:null,robust:null};
let notices;
function notify(message){$('#notification').textContent=message;$('#notification').hidden=false;clearTimeout(notices);notices=setTimeout(()=>$('#notification').hidden=true,5000);}
function escape(value){const e=document.createElement('span');e.textContent=String(value);return e.innerHTML;}
function shortTrial(t){return t.finished?`${t.duration.toFixed(2)} s`:`DNF · ${t.distance.toFixed(1)} m`;}
function evaluate(row){row.trials={};for(const event of ['sprint','hurdles'])row.trials[event]=runTrial(row.policy,event,state.env,true);row.evaluationEnv={...state.env};return row;}
function addRow({id,name,policy,evaluations=0,seed=null,history=[],trainingEnv=null,scope='baseline',generations=0}){
 const row=evaluate({id,name,policy:[...policy],evaluations,seed,history,trainingEnv,scope,generations});state.rows.push(row);renderResults();return row;
}
function select(row,event=state.event,play=false){
 state.active=row;state.event=event;state.trial=row.trials[event];state.index=0;state.clock=0;state.playing=play;
 $('#play').disabled=false;$('#timeline').disabled=false;$('#trial-label').textContent=row.name.toUpperCase();
 for(const b of document.querySelectorAll('[data-event]')){const on=b.dataset.event===event;b.classList.toggle('selected',on);b.setAttribute('aria-pressed',String(on));}
 $('#metric-third-label').textContent=event==='hurdles'?'Hurdles cleared':'Distance';
 $('#robustness').disabled=!!state.training;$('#export').disabled=false;state.robust=null;state.robustJob=null;$('#robustness').textContent='Test robustness';$('#robustness-results').hidden=true;
 render();
}
function render(){
 const t=state.trial,f=t?.frames?.[state.index];drawTrack($('#track'),t,state.index,{forces:$('#show-forces').checked});drawTelemetry($('#telemetry-chart'),t,state.index,state.chart);drawLearning($('#learning-chart'),state.history);
 if(!f)return;
 $('#metric-time').innerHTML=`${f.t.toFixed(2)}<small>s</small>`;$('#metric-speed').innerHTML=`${f.vx.toFixed(2)}<small>m/s</small>`;
 $('#metric-distance').innerHTML=state.event==='hurdles'?`${f.cleared}<small>/ 5</small>`:`${Math.min(100,f.x).toFixed(1)}<small>m</small>`;
 const atEnd=state.index===t.frames.length-1;
 $('#status-label').textContent=atEnd?t.reason:state.playing?'Replay running':'Replay paused';
 $('#status-label').style.color=atEnd&&!t.finished?'var(--orange)':'var(--lime)';
 $('#status-detail').textContent=atEnd?`${t.steps} steps · ${t.peakForce.toFixed(1)} BW peak`:`${t.finished?'Finished in '+t.duration.toFixed(2)+' s':t.reason+' at '+t.distance.toFixed(1)+' m'}`;
 $('#phase-label').textContent=f.stance?'STANCE · GROUND CONTACT':f.jump?'FLIGHT · HURDLE CLEARANCE':'FLIGHT · BALLISTIC';
 $('#play').textContent=state.playing?'Ⅱ':atEnd?'↺':'▶';$('#play').setAttribute('aria-label',state.playing?'Pause replay':atEnd?'Restart replay':'Play replay');
 $('#timeline').value=t.duration?String(f.t/t.duration*1000):'0';$('#replay-time').textContent=`${f.t.toFixed(2)} / ${t.duration.toFixed(2)} s`;
 const captions={force:'Vertical ground reaction force · body weights',height:'Center-of-mass height · meters',energy:'Kinetic (lime) · gravitational (blue) · spring (orange) · kJ'};
 $('#chart-caption').textContent=captions[state.chart];$('#chart-reading').textContent=state.chart==='force'?`${(f.fy/(t.env.mass*G)).toFixed(2)} BW`:state.chart==='height'?`${f.y.toFixed(2)} m`:`${((f.kinetic+f.potential+f.elastic)/1000).toFixed(2)} kJ total`;
}
function seek(time){
 const frames=state.trial?.frames;if(!frames)return;
 let lo=0,hi=frames.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(frames[mid].t<time)lo=mid+1;else hi=mid;}
 state.index=lo;state.clock=time;
}
function renderResults(){
 $('#results-body').innerHTML=state.rows.map(row=>{
  const s=row.trials.sprint,h=row.trials.hurdles;
  const sub=row.scope==='baseline'?'Hand-set reference':`${row.scope==='shared'?'Shared policy':row.scope+' specialist'} · seed ${row.seed} · ${row.generations} generations`;
  return `<tr><td>${escape(row.name)}<span class="row-subtitle">${escape(sub)}</span></td><td>${row.evaluations?row.evaluations.toLocaleString():'—'}</td><td class="${s.finished?'positive':'negative'}">${shortTrial(s)}</td><td class="${h.finished?'positive':'negative'}" title="${escape(h.reason)}">${shortTrial(h)}</td><td>${h.cleared} / ${h.totalHurdles}</td><td><button class="replay-button" data-row="${row.id}" data-replay="sprint">Sprint</button> <button class="replay-button" data-row="${row.id}" data-replay="hurdles">Hurdles</button></td></tr>`;
 }).join('');
 $('#results-footnote').textContent=`All rows evaluated at ${state.env.mass} kg, friction ${state.env.friction.toFixed(2)}, and ${state.env.height.toFixed(2)} m hurdles. DNF = did not finish. Trial counts show training cost; replay/evaluation trials are excluded. Rows with different budgets do not establish transfer.`;
}
function currentConditions(){return {...DEFAULT_ENV,mass:Number($('#mass').value),friction:Number($('#friction').value),height:Number($('#hurdle-height').value)};}
function conditionsChanged(){
 state.env=currentConditions();state.playing=false;
 for(const row of state.rows)evaluate(row);
 renderResults();if(state.active)select(state.active);
 notify('All controllers re-evaluated under the new conditions.');
}
function setTrainingUi(active){
 $('#train').innerHTML=active?'Stop & keep best <span aria-hidden="true">■</span>':'Train a policy <span aria-hidden="true">↗</span>';
 for(const id of ['training-task','generations','hurdle-height','mass','friction','seed','robustness'])$('#'+id).disabled=active;
 $('#run-baseline').disabled=active;
}
function finishTraining(stopped=false){
 const t=state.training;if(!t)return;
 const name=`${t.scope==='shared'?'Shared':t.scope==='sprint'?'Sprint':'Hurdle'} · run ${state.rows.filter(r=>r.scope!=='baseline'&&!r.example).length+1}`;
 if(t.latest){
  const row=addRow({id:`run-${Date.now()}`,name,policy:t.latest.policy,evaluations:t.latest.evaluations,seed:t.seed,history:[...state.history],trainingEnv:{...t.env},scope:t.scope,generations:t.latest.generation});select(row,state.event,false);
 }
 $('#training-status').textContent=stopped?'Stopped · best kept':'Training complete';
 $('#training-detail').textContent=t.latest?`${t.latest.evaluations.toLocaleString()} simulated training trials. Best controller added below.`:'Stopped before a generation completed.';
 state.training=null;setTrainingUi(false);$('#robustness').disabled=!state.active;
 notify(stopped?'Training stopped. Completed results are kept.':'Training complete. Compare the measured results below.');
}
function getWorker(){
 if(state.worker)return state.worker;
 const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});state.worker=worker;
 worker.onmessage=({data})=>{
  if(data.type==='progress'&&state.training&&data.jobId===state.training.jobId){
   state.training.latest=data;state.history.push({generation:data.generation,score:data.score,mean:data.mean,evaluations:data.evaluations});
   $('#training-progress').value=data.generation;$('#training-progress-label').textContent=`${data.generation} / ${state.training.generations}`;
   $('#training-status').textContent='Searching controllers…';$('#learning-caption').textContent=`Best ${data.score.toFixed(1)} · population mean ${data.mean.toFixed(1)}`;drawLearning($('#learning-chart'),state.history);
  }else if(data.type==='done'&&state.training&&data.jobId===state.training.jobId){finishTraining();}
  else if(data.type==='robustness'&&data.jobId===state.robustJob){
   state.robust=data.results;$('#robustness').disabled=false;$('#robustness').textContent='Test robustness';renderRobustness();
  }else if(data.type==='error'){state.training=null;setTrainingUi(false);$('#training-status').textContent='Training error';notify(data.message);}
 };
 worker.onerror=()=>{state.training=null;setTrainingUi(false);$('#training-status').textContent='Training unavailable';notify('The training worker could not start. Reload the page and try again.');worker.terminate();state.worker=null;};
 return worker;
}
function train(){
 if(state.training){getWorker().postMessage({type:'stop'});finishTraining(true);return;}
 const seed=Number($('#seed').value);if(!Number.isSafeInteger(seed)||seed<1||seed>2147483647){notify('Use an integer seed from 1 to 2147483647.');$('#seed').focus();return;}
 const scope=$('#training-task').value,generations=Number($('#generations').value),env={...state.env};
 state.training={scope,generations,env,seed,latest:null,jobId:crypto.randomUUID()};state.robustJob=null;state.history=[];setTrainingUi(true);
 $('#training-progress').max=generations;$('#training-progress').value=0;$('#training-progress-label').textContent=`0 / ${generations}`;$('#training-status').textContent='Starting search…';
 $('#training-detail').textContent=`32 candidates per generation · ${scope==='shared'?'both events':'one event'} · 4 sampled conditions · seed ${seed}`;
 $('#learning-caption').textContent='Evaluating controllers in the simulation.';drawLearning($('#learning-chart'),state.history);
 try{getWorker().postMessage({type:'train',jobId:state.training.jobId,config:{scope,env,seed,population:32,start:BASELINE},generations});}catch(error){state.training=null;setTrainingUi(false);notify(error.message);}
}
function testRobustness(){
 if(!state.active||state.training)return;
 $('#robustness').disabled=true;$('#robustness').textContent='Testing…';
 state.robustJob=crypto.randomUUID();getWorker().postMessage({type:'robustness',jobId:state.robustJob,policy:state.active.policy,env:state.env});
}
function renderRobustness(){
 const list=state.robust;if(!list)return;
 const passes=list.filter(v=>v.trials.every(t=>t.finished)).length;
 const box=$('#robustness-results');box.hidden=false;
 box.innerHTML=`<div class="section-title"><h2>${escape(state.active.name)} · sensitivity check</h2><span class="${passes===list.length?'positive':'negative'} small">Both events finished in ${passes} / ${list.length} conditions</span></div><div class="robustness-grid">${list.map(v=>`<div class="robust-card"><strong>${escape(v.name)}</strong><span>Sprint: ${shortTrial(v.trials[0])}</span><br><span class="${v.trials[1].finished?'positive':'negative'}">Hurdles: ${shortTrial(v.trials[1])}</span></div>`).join('')}</div><p class="small muted" style="margin-top:14px">Deterministic sensitivity tests, not statistical confidence intervals. A changed outcome at half the timestep indicates numerical sensitivity; treat that strategy as provisional.</p>`;
 notify(`Robustness check complete: both events finished in ${passes} of ${list.length} conditions.`);
}
function download(){
 const data={modelVersion:VERSION,exportedAt:new Date().toISOString(),environment:state.env,controllers:state.rows.map(r=>({...r,parameters:decode(r.policy)})),robustness:state.robust,notes:'Simplified actuated SLIP. Gait structure hand-designed; eight parameters optimized with CEM. No human validity or global optimality claim. Full replay telemetry included. Training and evaluation conditions may differ.'};
 const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`stride-lab-${new Date().toISOString().slice(0,10)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Results, policies, and replay telemetry exported.');return {controllers:state.rows.length,modelVersion:VERSION};
}
function notes(){
 $('#notes-content').innerHTML=`<p><strong>A small, inspectable experiment in athletic control.</strong> A 75 kg athlete runs on a 1.05 m spring leg. The swing limb carries 12% of body mass, is moved by a hip actuator bounded at 700 N, and is planted wherever it has actually reached — it is never teleported to the commanded placement. Leg length is a state that retracts and extends in flight. The upper body is drawn for readability; its joint dynamics are not simulated.</p><h3>What is physical</h3><p>Gravity, ground contact, compression-only leg force, damping, bounded axial drive, drag, and and a friction constraint determine the body’s trajectory. Leaving the friction cone slides the foot and dissipates energy rather than ending the trial. The default timestep is 2 ms, and touchdown is solved to the exact crossing time inside a step, so a stance never opens with the leg already compressed. Every trial carries a closed energy ledger: actuator work in, damper, drag and force-cap losses out, and a residual that is discretisation error alone — under 0.03% of actuator work, converging to zero with the timestep. The athlete starts at rest, with a slightly compressed leg; that is an initial condition, counted in the first energy reading. No forward force acts during flight other than drag.</p><h3>What the agent learns</h3><p>The cross-entropy method (CEM) samples 32 controllers per generation and fits the next sample distribution to the best six. It adjusts target speed, stiffness, foot placement, expected stance time, push, jump drive, jump timing, and retraction. Retraction now shortens the leg itself, which costs reach, so hurdle clearance is paid for rather than granted. The gait structure and obstacle reflex are designed by hand. This is parameter-level policy search, not learned human technique or a neural whole-body controller.</p><h3>Events and scoring</h3><p>The sprint is 100 m. The hurdle course is a custom 60 m exercise with five hurdles at 12, 21, 30, 39, and 48 m at a default 0.35 m. It is not an official competition configuration, and the height was lowered from 0.60 m once retraction became a real leg-length change: one planar leg length shared by both limbs cannot represent hurdling technique, and 0.60 m is impassable for this body. Contact is checked against the limb's actual length and angle; the illustrated limbs do not have separate collision shapes.</p><p>Successful trials score <code>250 + 8 × (30 − time)</code>. Failures receive distance progress and hurdle-clearance shaping. Shared policies receive the average of the two event scores. Finishing one event always outranks failing that same event. All trials have a 30 s limit. Shared scores average over both events and all sampled conditions. Mechanical energy is kinetic + gravitational + spring energy; it is not a metabolic estimate.</p><h3>Reproducibility and comparison</h3><p>Every training run starts from the same hand-set baseline. Seed, generation count, conditions, and evaluated-trial counts are included in the export. Training conditions are resampled every generation — mass ±10%, friction ±15%, hurdle height ±0.05 m, and timestep from 2 / 1 / 0.5 ms — with all candidates in a generation sharing one set, and the incumbent re-simulated under it. The returned controller is picked on 8 fixed held-out conditions, never on the generation’s own draw, so the plotted score is a held-out score. A shared 60-generation run uses 16,336 training trials. Specialist runs evaluate one event per candidate; compare total trial budgets before drawing conclusions about transfer.</p><p>Changing physical conditions re-evaluates every controller without retraining it. Robustness checks run at half, quarter and eighth timestep and vary friction, mass, and hurdle height. Failures remain visible. A fast reference trial can still be fragile, and because training now samples similar conditions these checks are a consistency test rather than a fully out-of-sample one.</p><h3>Scope</h3><p>This model omits muscles, tendon anatomy, fatigue, joint torques, 3D balance, limb inertia, and pole vaulting. Contact forces are capped at 5.5 body weights. Positive drive power is limited to 2.4 kW normally and 4.2 kW on hurdle takeoffs. These are experiment settings, not calibrated human performance limits.</p><h3>Background</h3><p><a href="https://pubmed.ncbi.nlm.nih.gov/2625422/" target="_blank" rel="noopener">Blickhan: The spring-mass model for running and hopping</a><br><a href="https://underactuated.csail.mit.edu/simple_legs.html" target="_blank" rel="noopener">MIT: Simple models of walking and running</a><br><a href="https://algorithmsbook.com/files/chapter-10.pdf" target="_blank" rel="noopener">Algorithms for Decision Making: Policy search</a></p><p>Simulation and training run in your browser. Experiment state stays in this session until you export it. Web fonts are loaded separately from Google Fonts.</p>`;
 $('#notes').showModal();
}
for(const button of document.querySelectorAll('[data-event]'))button.addEventListener('click',()=>{state.event=button.dataset.event;if(state.active)select(state.active,state.event,false);});
$('#run-baseline').addEventListener('click',()=>{let row=state.rows.find(r=>r.scope==='baseline');if(!row)row=addRow({id:'baseline',name:'Baseline',policy:BASELINE});select(row,state.event,true);});
$('#play').addEventListener('click',()=>{if(!state.trial)return;if(state.index>=state.trial.frames.length-1)seek(0);state.playing=!state.playing;render();});
$('#timeline').addEventListener('input',()=>{state.playing=false;seek(Number($('#timeline').value)/1000*state.trial.duration);render();});
$('#speed').addEventListener('change',()=>state.speed=Number($('#speed').value));$('#show-forces').addEventListener('change',render);
for(const button of document.querySelectorAll('[data-chart]'))button.addEventListener('click',()=>{state.chart=button.dataset.chart;for(const b of document.querySelectorAll('[data-chart]'))b.classList.toggle('active',b===button);render();});
$('#train').addEventListener('click',train);$('#robustness').addEventListener('click',testRobustness);$('#export').addEventListener('click',download);
$('#conditions-toggle').addEventListener('click',()=>{const open=$('#conditions').hidden;$('#conditions').hidden=!open;$('#conditions-toggle').setAttribute('aria-expanded',String(open));$('#conditions-toggle span').textContent=open?'−':'+';});
for(const [id,format] of [['generations',v=>v],['hurdle-height',v=>Number(v).toFixed(2)+' m'],['mass',v=>v+' kg'],['friction',v=>Number(v).toFixed(2)]]){
 $('#'+id).addEventListener('input',()=>$('#'+id+'-value').textContent=format($('#'+id).value));if(id!=='generations')$('#'+id).addEventListener('change',conditionsChanged);
}
$('#notes-open').addEventListener('click',notes);$('#notes-close').addEventListener('click',()=>$('#notes').close());$('#notes').addEventListener('click',e=>{if(e.target===$('#notes')){const r=$('#notes').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('#notes').close();}});
$('#results-body').addEventListener('click',e=>{const b=e.target.closest('[data-row]');if(!b)return;const row=state.rows.find(r=>r.id===b.dataset.row);if(row)select(row,b.dataset.replay,true);});
window.addEventListener('resize',render);
document.addEventListener('visibilitychange',()=>state.lastTimestamp=0);
function animate(now){
 const delta=state.lastTimestamp?Math.min(.08,(now-state.lastTimestamp)/1000):0;state.lastTimestamp=now;
 if(state.playing&&state.trial&&!document.hidden){seek(state.clock+delta*state.speed);if(state.index>=state.trial.frames.length-1)state.playing=false;render();}
 requestAnimationFrame(animate);
}
const baseline=addRow({id:'baseline',name:'Baseline',policy:BASELINE});select(baseline);requestAnimationFrame(animate);
// The saved example was generated by the same public physics/search module.
try{
 const response=await fetch('example.json');if(response.ok){const example=await response.json();if(example.version===VERSION){const row=addRow({...example,id:'example',name:'Shared · saved example'});row.example=true;state.history=example.history;$('#learning-caption').textContent=`Saved example · ${example.generations} generations · seed ${example.seed} · held-out score`;drawLearning($('#learning-chart'),state.history);}}
}catch{ /* The lab remains fully usable without the optional saved example. */ }
// Progressive enhancement: the page remains functional without WebMCP.
const mc=document.modelContext,lifecycle=new AbortController();
if(mc?.registerTool){
 const register=tool=>{try{Promise.resolve(mc.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{}};
 register({name:'read_experiment_results',title:'Read experiment results',description:'Read measured trials and current physical conditions.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute(input){if(!input||Object.keys(input).length)throw new Error('No parameters accepted');return {version:VERSION,environment:state.env,training:!!state.training,rows:state.rows.map(r=>({name:r.name,trainingTrials:r.evaluations,sprint:shortTrial(r.trials.sprint),hurdles:shortTrial(r.trials.hurdles),cleared:r.trials.hurdles.cleared}))};}});
 register({name:'select_experiment_replay',title:'Select a replay',description:'Choose an existing controller and event and show its paused replay.',inputSchema:{type:'object',properties:{controller:{type:'string'},event:{type:'string',enum:['sprint','hurdles']}},required:['controller','event'],additionalProperties:false},annotations:{readOnlyHint:false},execute(input){if(!input||Object.keys(input).some(k=>!['controller','event'].includes(k))||!['sprint','hurdles'].includes(input.event))throw new Error('Invalid replay request');const row=state.rows.find(r=>r.name===input.controller);if(!row)throw new Error('Unknown controller');select(row,input.event);return {controller:row.name,event:input.event,result:shortTrial(row.trials[input.event])};}});
 window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}

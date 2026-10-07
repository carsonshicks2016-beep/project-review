import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity, Flag, Route, FlaskConical, Gauge, Play, Square, RotateCcw, Download, Box, Eye, Hammer, Plus, X, ChevronRight, AlertCircle} from 'lucide-react';
import ExperimentPage from './ExperimentPage';
import {checkpointDefaults} from './checkpointDefaults';
import './style.css';

type Point={x:number;y:number;z:number};
type Course={inspectionReview?:{url:string;status:string;revision:string};id:string;name:string;family:string;seed:number;rocks:number;scenery?:boolean;suite:string;review:string;legacyObstacleLayout?:boolean;legacyReason?:string;definition:{gates:Point[];length:number;width:number;scenery?:boolean;obstacleObjects?:number;topology?:string;surface?:string;confidence?:string;episodeSeconds?:number;mapPoints?:Point[];landmarks?:{name:string;station:number}[];circuitRevision?:string;generatorVersion?:string;terrainHash?:string;firstSector?:number;sectorCount?:number}};
type Checkpoint={id:string;run:string;step:number|null;prepared:boolean;created:number;build:string};
type Demonstration={id:string;courseId:string;controller:string;transitions:number;created:number};
type Summary={attempts:number;episodes_completed?:number;requested:number|null;complete:boolean;finishes:number;best:number|null;median:number|null;finish_interval:number[]|null;outcomes:Record<string,number>};
type Job={id:string;kind:string;state:string;created:number;updated:number;spec:any;detail:any;summary:Summary};
type State={jobs:Job[];courses:Course[];checkpoints:Checkpoint[];demonstrations:Demonstration[];time:number};
const empty:State={jobs:[],courses:[],checkpoints:[],demonstrations:[],time:0};
const active=(j:Job)=>['queued','starting','running','stopping'].includes(j.state);
const number=(n:number|null|undefined)=>n==null?'—':n.toLocaleString();
const seconds=(n:number|null|undefined)=>n==null?'—':`${n.toFixed(2)} s`;
const canonical=(value:any):any=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])])):value;
const evaluationConditions=(spec:any)=>{const {checkpoint,...conditions}=spec;return JSON.stringify(canonical({...conditions,courses:[...(conditions.courses||[])].sort()}))};
async function api(path:string, body?:unknown){const r=await fetch('/api'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
const appendTelemetry=(previous:any,next:any)=>{
 const history={...(previous.history||{})};
 const caps:Record<string,number>={episodes:1600,scalars:2400,resources:500,checkpoints:250,lifecycle:250};
 for(const event of next.events||[]){const key=event.kind==='episode'?'episodes':event.kind==='resource'?'resources':event.kind==='scalar'?'scalars':event.kind==='checkpoint'?'checkpoints':event.kind==='lifecycle'?'lifecycle':'';if(!key)continue;const rows=history[key]||[];if(!rows.some((row:any)=>row.seq===event.seq))history[key]=[...rows,event].slice(-caps[key]);}
 return {...previous,...next,history};
};

function CircuitElevation({course}:{course:Course}){
 const points=course.definition.mapPoints||[];if(!points.length)return null;
 const lo=Math.min(...points.map(p=>p.y)),hi=Math.max(...points.map(p=>p.y));
 const line=points.map((p,i)=>`${i/(points.length-1)*300},${55-(p.y-lo)/Math.max(1,hi-lo)*45}`).join(' ');
 return <><svg viewBox="0 0 300 65" role="img" aria-label="Full circuit source elevation"><polyline points={line} fill="none" stroke="#42654a" strokeWidth="2"/></svg><small>Source elevation range {(hi-lo).toFixed(1)} m · terrain-derived, not surveyed pavement</small>{course.inspectionReview&&<p><a href={course.inspectionReview.url} target="_blank" rel="noreferrer">Open complete circuit visual review</a><br/><small>Inspection evidence · training approval remains separate</small></p>}<details><summary>Named sections and reconstruction</summary><p>{course.definition.landmarks?.map(l=>l.name).join(' · ')}</p><small>Circuit {course.definition.circuitRevision?.slice(0,16)} · {course.definition.generatorVersion || "Initial reconstruction"} · historical references, estimated dimensions</small></details></>;
}

function CourseMap({course,trajectory=[],cursor=1}:{course?:Course;trajectory?:any[];cursor?:number}){
 const ref=useRef<HTMLCanvasElement>(null);
 useEffect(()=>{const canvas=ref.current;if(!canvas)return;const ctx=canvas.getContext('2d')!;const draw=()=>{const box=canvas.getBoundingClientRect();canvas.width=box.width*2;canvas.height=box.height*2;ctx.scale(2,2);const w=box.width,h=box.height;ctx.fillStyle='#eef1ef';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#dde3df';ctx.lineWidth=1;for(let x=0;x<w;x+=28){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}for(let y=0;y<h;y+=28){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}const points=course?.definition.mapPoints||course?.definition.gates;if(!points?.length)return;const xs=points.map(p=>p.x),zs=points.map(p=>p.z);const minX=Math.min(...xs),minZ=Math.min(...zs);const sx=Math.max(...xs)-minX,sz=Math.max(...zs)-minZ;const scale=Math.min((w-50)/Math.max(sx,1),(h-45)/Math.max(sz,1));const pos=(p:Point)=>[(w-sx*scale)/2+(p.x-minX)*scale,h-22-(p.z-minZ)*scale];ctx.lineJoin='round';ctx.lineCap='round';const line=(color:string,width:number)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{const[x,y]=pos(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()};line('#b6c0b9',10);line('#ffffff',6);const samples=trajectory.slice(0,Math.max(1,Math.floor(trajectory.length*cursor)));for(let i=1;i<samples.length;i++){const a=pos(samples[i-1].position),b=pos(samples[i].position);ctx.strokeStyle=`hsl(${Math.min(150,samples[i].speed*3)},65%,37%)`;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...a as [number,number]);ctx.lineTo(...b as [number,number]);ctx.stroke()}[points[1],points[points.length-2]].forEach((p,i)=>{const[x,y]=pos(p);ctx.fillStyle=i?'#172b23':'#0b966c';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()});if(samples.length){const[x,y]=pos(samples[samples.length-1].position);ctx.fillStyle='#d44645';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()}};draw();const observer=new ResizeObserver(draw);observer.observe(canvas);return()=>observer.disconnect()},[course,trajectory,cursor]);
 return <canvas ref={ref} aria-label={course?`${course.name} course map`:'Course map'} />;
}

function App(){
 const [state,setState]=useState<State>(empty),[tab,setTab]=useState('Overview'),[connected,setConnected]=useState(false),[error,setError]=useState(''),[busy,setBusy]=useState(false),[build,setBuild]=useState<any>(null),[selected,setSelected]=useState<string>(''),[telemetry,setTelemetry]=useState<any>(null),[telemetryError,setTelemetryError]=useState(''),[modal,setModal]=useState('');
 const [checkpointContext,setCheckpointContext]=useState<{checkpoint:string;note:string}|null>(null);
 const [compareIds,setCompareIds]=useState<string[]>([]),[comparison,setComparison]=useState<{jobs:any[];loading:boolean;error:string}>({jobs:[],loading:false,error:''});
 const [run,setRun]=useState({name:'',mode:'specialist',course:'',seed:42,steps:2000000,workers:1,reward:'baseline-v1',startingGear:'neutral',parent:'',demonstrations:[] as string[],imitationSteps:10000,imitationMode:'isolated'});
 const [circuitDialog,setCircuitDialog]=useState(false);
 const [circuitForm,setCircuitForm]=useState({circuit:'nordschleife',firstSector:0,sectorCount:16});
 const [courseForm,setCourseForm]=useState({name:'',seed:42000,family:'gentle',length:1000,rocks:0,scenery:true,suite:'library'});
 const [evaluation,setEvaluation]=useState<{checkpoint:string;courses:string[];attempts:number;seed:number;deterministic:boolean;timeScale:number;startingGear:'neutral'|'first'}>({checkpoint:'',courses:[],attempts:20,seed:2026,deterministic:false,timeScale:10,startingGear:'neutral'});
 useEffect(()=>{api('/build').then(setBuild).catch(e=>setError(e.message));const events=new EventSource('/api/events');events.onmessage=e=>{setState(JSON.parse(e.data));setConnected(true)};events.onerror=()=>setConnected(false);return()=>events.close()},[]);
 useEffect(()=>{const syncRoute=()=>{const route=decodeURIComponent(window.location.hash.slice(1));if(route.startsWith('experiment/')){setSelected(route.slice('experiment/'.length));return}setSelected('');const name=['Overview','Training','Courses','Experiments','Evaluation','Checkpoints'].find(item=>item.toLowerCase()===route);if(name)setTab(name)};window.addEventListener('hashchange',syncRoute);syncRoute();return()=>window.removeEventListener('hashchange',syncRoute)},[]);
 useEffect(()=>{if(!selected){setTelemetry(null);setTelemetryError('');return}let live=true,inFlight=false,cursor:number|null=null;const load=async()=>{if(inFlight)return;inFlight=true;try{const path=`/jobs/${selected}/telemetry${cursor===null?'':`?cursor=${cursor}`}`;const result=await api(path);if(!live)return;if(result.incremental){cursor=result.cursor;setTelemetry((previous:any)=>previous?appendTelemetry(previous,result):previous)}else{cursor=result.cursor;setTelemetry(result)}setTelemetryError('')}catch(e){if(live)setTelemetryError((e as Error).message)}finally{inFlight=false}};load();const timer=setInterval(load,3000);return()=>{live=false;clearInterval(timer)}},[selected]);
 useEffect(()=>{let current=true;if(!compareIds.length){setComparison({jobs:[],loading:false,error:''});return()=>{current=false}}setComparison({jobs:[],loading:true,error:''});Promise.all(compareIds.map(id=>api('/jobs/'+id))).then(jobs=>{if(current)setComparison({jobs,loading:false,error:''})}).catch(e=>{if(current)setComparison({jobs:[],loading:false,error:e.message})});return()=>{current=false}},[compareIds]);
 function openJob(id:string){setSelected(id);setTelemetry(null);setTelemetryError('');window.location.hash=`experiment/${encodeURIComponent(id)}`}
 function switchTab(name:string){setTab(name);setSelected('');setTelemetry(null);window.location.hash=name.toLowerCase()}
 async function act(path:string,body:unknown={}){setBusy(true);setError('');try{const result=await api(path,body);setState(await api('/state'));setModal('');if(result.id)openJob(result.id);return result}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 async function reviewCourse(id:string){setBusy(true);setError('');try{await api(`/courses/${id}/review`);setState(await api('/state'))}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 const runs=state.jobs.filter(j=>j.kind==='training'),evaluations=state.jobs.filter(j=>j.kind==='evaluation');
 const activeJobs=state.jobs.filter(active),library=state.courses.filter(c=>c.suite==='library'&&!c.legacyObstacleLayout);
 function applyCheckpointDefaults(id:string,intent:'viewer'|'evaluation'){
  const checkpoint=state.checkpoints.find(item=>item.id===id);
  if(!checkpoint){setEvaluation({...evaluation,checkpoint:id,courses:[]});setCheckpointContext(null);return}
  const defaults=checkpointDefaults(checkpoint,state.jobs,state.courses,evaluation,intent);
  setEvaluation(defaults.settings);
  setCheckpointContext({checkpoint:id,note:defaults.courseNote});
 }
 function watchCheckpoint(checkpoint:Checkpoint){
  if(!checkpoint.prepared){void act(`/checkpoints/${checkpoint.id}/prepare`);return}
  applyCheckpointDefaults(checkpoint.id,'viewer');
  setModal('viewer');
 }
 function checkpointOriginLabel(checkpoint:Checkpoint){
  const source=runs.find(job=>job.id===checkpoint.run);
  if(!source)return 'Source run unavailable';
  if(source.spec.mode==='specialist')return state.courses.find(course=>course.id===source.spec.course)?.name||'Specialist course unavailable';
  return 'Generalist · procedural courses';
 }
 const selectedEvaluations=comparison.jobs.map(j=>({...j,summary:state.jobs.find(item=>item.id===j.id)?.summary??j.summary}));
 const conditionsMatch=selectedEvaluations.length>1&&selectedEvaluations.every(j=>evaluationConditions(j.spec)===evaluationConditions(selectedEvaluations[0].spec));
 const checkpointBuilds=selectedEvaluations.map(j=>state.checkpoints.find(c=>c.id===j.spec.checkpoint)?.build);
 const buildsMatch=checkpointBuilds.length>1&&checkpointBuilds.every(hash=>!!hash&&hash===checkpointBuilds[0]);
 const evaluationsComplete=selectedEvaluations.length>1&&selectedEvaluations.every(j=>j.state==='completed'&&j.summary?.complete);
 const comparisonQualified=conditionsMatch&&buildsMatch&&evaluationsComplete;
 const comparisonCourses=[...new Set(selectedEvaluations.flatMap(j=>j.spec.courses||[]))];
 const focus=state.courses.find(c=>c.id===run.course)||library[0];
 const nav=[['Overview',Gauge],['Training',Activity],['Courses',Route],['Experiments',FlaskConical],['Evaluation',Flag],['Checkpoints',Box]] as const;
 function jobTable(items:Job[]){return <div className="table-scroll"><table><thead><tr><th>Experiment</th><th>Status</th><th>Mode</th><th>Progress</th><th>Best valid</th><th>Updated</th><th/></tr></thead><tbody>{items.map(j=><tr key={j.id} className={selected===j.id?'selected':''}><td><button className="text-button" onClick={()=>openJob(j.id)}>{j.spec.name||j.kind+' · '+j.id.slice(0,6)}</button><small>{j.id}</small></td><td><span className={'status '+j.state}>{j.state}</span></td><td>{j.spec.mode||j.kind}</td><td>{j.kind==='suite'?`${number(j.detail.generated)}/${number(j.detail.requested??60)} courses`:j.kind==='training'?<><span>{number(j.detail.trainer_step)} / {number(j.spec.steps)} steps</span><small>{number(j.summary?.episodes_completed)} episodes completed</small></>:<><span>{number(j.summary?.attempts)}{j.summary?.requested?` / ${number(j.summary.requested)}`:''} attempts</span><small>{j.summary?.finishes==null?'':`${number(j.summary.finishes)} valid finishes`}</small></>}</td><td>{j.kind==='evaluation'?seconds(j.summary?.best):'—'}</td><td>{new Date(j.updated*1000).toLocaleTimeString()}</td><td className="actions">{active(j)&&j.state!=='stopping'?<button title="Stop and save" aria-label="Stop and save" onClick={()=>act(`/jobs/${j.id}/stop`)}><Square size={15}/></button>:j.kind==='training'&&j.detail.resumable&&['stopped','interrupted','failed'].includes(j.state)?<button title="Resume training" aria-label="Resume training" onClick={()=>act(`/jobs/${j.id}/resume`)}><RotateCcw size={15}/></button>:j.state==='stopping'?<button title="Open stop controls" aria-label="Open stop controls" onClick={()=>openJob(j.id)}><ChevronRight size={15}/></button>:null}</td></tr>)}</tbody></table>{!items.length&&<div className="empty">No experiments yet</div>}</div>}
 return <div className="app"><aside><a className="brand" href="#overview" onClick={()=>switchTab('Overview')}><div className="brand-mark"><Flag size={22}/></div><span>RALLY<span className="brand-sub">RESEARCH LAB</span></span></a><div className="workspace-label">WORKSPACE / LOCAL</div><nav>{nav.map(([name,Icon])=><button key={name} className={tab===name?'active':''} onClick={()=>switchTab(name)}><Icon size={18}/><span>{name}</span>{name==='Training'&&activeJobs.length>0&&<b>{activeJobs.length}</b>}</button>)}</nav><div className="aside-bottom"><span className={'dot '+(connected?'online':'')}/>{connected?'Connected locally':'Reconnecting'}<small>PPO · Unity physics</small></div></aside><main>{!selected&&<header><div><span className="breadcrumb">WORKSPACE <ChevronRight size={12}/> {tab.toUpperCase()}</span><h1>{tab==='Overview'?'Research overview':tab}</h1></div><div className="header-actions"><span className="quiet">1 worker default</span><button className="primary" disabled={busy} onClick={()=>setModal('training')}><Plus size={16}/>New experiment</button></div></header>}{error&&<div role="alert" className="error"><AlertCircle size={17}/>{error}<button title="Dismiss error" onClick={()=>setError('')}><X size={16}/></button></div>}{!selected&&!connected&&<div className="connection">Live data unavailable. Showing the last received state.</div>}
 {!selected&&tab==='Overview'&&<><div className="metrics"><div><span>Active jobs</span><strong>{activeJobs.length}</strong><small>{activeJobs.find(j=>j.state==='running')?.kind||'Idle'}</small></div><div><span>Training experiments</span><strong>{runs.length}</strong><small>{runs.filter(j=>j.state==='completed').length} completed</small></div><div><span>Frozen courses</span><strong>{state.courses.length}</strong><small>{library.length} in library</small></div><div><span>Checkpoints</span><strong>{state.checkpoints.length}</strong><small>{state.checkpoints.filter(c=>c.prepared).length} ready to watch</small></div></div><section className="overview-grid"><div><div className="section-heading"><h2>Experiment activity</h2><span className="live">LIVE</span></div>{jobTable(state.jobs.slice(0,7))}</div><div className="course-preview"><div className="section-heading"><h2>{focus?.name||'Course library'}</h2><Route size={17}/></div><CourseMap course={focus}/><div className="map-caption"><span>{focus?`${focus.definition.length} m · ${focus.definition.width} m road`:'No saved course'}</span><button className="text-button" onClick={()=>switchTab('Courses')}>Open library <ChevronRight size={14}/></button></div></div></section><section className="build-line"><div><h2>Simulation build</h2><span className={'status '+(build?.status==='current'?'completed':'queued')}>{build?.status||'Checking'}</span><small>{build?.source?.hash?.slice(0,16)||'Managed build required'}</small></div><button disabled={busy||activeJobs.some(j=>j.kind==='build')} onClick={()=>act('/build')}><Hammer size={16}/>Prepare player</button><button title="Refresh build status" onClick={()=>api('/build').then(setBuild)}><RotateCcw size={16}/></button></section></>}
 {!selected&&tab==='Training'&&<><div className="section-heading"><h2>Training queue</h2><span>{runs.length} experiments</span></div>{jobTable(runs)}</>}
 {!selected&&tab==='Courses'&&<>
  <div className="section-heading"><h2>Frozen course library</h2><div className="actions">
   <button disabled={busy} onClick={()=>act('/courses/seed-library')}><Plus size={15}/>Starter library</button>
   <button disabled={busy} onClick={()=>act('/suites/create')}>Create benchmark suites</button>
   <button onClick={()=>setModal('course')}><Plus size={15}/>Save course</button>
   <button disabled={busy} onClick={()=>setCircuitDialog(true)}><Plus size={15}/>Import Nordschleife</button>
  </div></div>
  <div className="course-grid">{state.courses.map(c=><article key={c.id}>
   <CourseMap course={c}/><div className="course-info"><h3>{c.name}</h3><span className={'status '+(c.legacyObstacleLayout?'failed':'')}>{c.legacyObstacleLayout?'Legacy layout':c.suite}</span>
    <p>{c.family} · {Math.round(c.definition.length).toLocaleString()} m · {c.definition.surface||`${c.rocks} rocks / 100 m`}</p>
    {c.definition.topology==='circuit'&&<><small>Documented reconstruction · {c.definition.episodeSeconds}s budget · {c.definition.sectorCount===16?'Standing-start full lap':'Fixed-start sector variant'}</small><small>Estimated width, banking and edge geometry · visual review pending</small><CircuitElevation course={c}/><button disabled={busy} onClick={()=>act(`/circuits/${c.id}/preview`,{camera:'ThirdPerson'})}><Play size={14}/>Review Karussell (unranked)</button></>}<small>{c.id.slice(0,16)} · {c.review}</small>
    {c.scenery===false&&c.definition.topology!=='circuit'&&<small>{c.definition.obstacleObjects===0?'Verified: no collidable rocks or roadside props':'No-hazard audit unavailable'}</small>}
    {c.legacyObstacleLayout&&<small className="course-legacy-note">Historical course artifact; retained for prior run records.</small>}
    {c.definition.topology==='circuit'&&c.review!=='reviewed'&&<small>Training validation pending · visual inspection does not approve training</small>}
    {!c.legacyObstacleLayout&&c.definition.topology!=='circuit'&&c.review!=='reviewed'&&<button disabled={busy} onClick={()=>reviewCourse(c.id)}><Flag size={14}/>Mark preview reviewed</button>}
    <button disabled={c.legacyObstacleLayout||c.suite!=='library'||c.review!=='reviewed'} onClick={()=>{setRun({...run,course:c.id,mode:'specialist',name:c.name+' specialist'});setModal('training')}}><Play size={14}/>Train specialist</button>
    <button title="Record a reference drive and add its verified demonstration to training" disabled={busy||c.legacyObstacleLayout||c.suite!=='library'||c.review!=='reviewed'} onClick={()=>act('/demonstrations/record',{course:c.id,seed:run.seed,timeScale:10,startingGear:'first'})}><Play size={14}/>Record reference drive</button>
   </div></article>)}</div>
  {!state.courses.length&&<div className="empty">No saved course. Generate the starter library to begin.</div>}
  {jobTable(state.jobs.filter(j=>['course','circuit','circuit-preview','demonstration','suite'].includes(j.kind)))}
 </>}
 {!selected&&tab==='Experiments'&&<><div className="section-heading"><h2>Experiment ledger</h2><span>Immutable configuration · explicit lineage</span></div>{jobTable(state.jobs.filter(j=>['training','evaluation','demonstration'].includes(j.kind)))}</>}
 {!selected&&tab==='Evaluation'&&<>
  <div className="section-heading"><h2>Time-attack evaluations</h2><button className="primary" onClick={()=>{setCheckpointContext(null);setModal('evaluation')}}><Flag size={15}/>New evaluation</button></div>
  {jobTable(evaluations)}
  <div className="evaluation-results">{evaluations.filter(j=>j.summary.attempts>0).map(j=><article key={j.id}>
   <div className="evaluation-card-heading"><h3>{j.spec.name||j.id.slice(0,8)} <span className={'status '+(j.summary.complete?'completed':'queued')}>{j.summary.complete?'Complete':'Partial'}</span></h3>
    <label className="compare-check"><input type="checkbox" checked={compareIds.includes(j.id)} disabled={!compareIds.includes(j.id)&&compareIds.length>=4} onChange={e=>setCompareIds(ids=>e.target.checked?[...ids,j.id]:ids.filter(id=>id!==j.id))}/><span>Compare</span></label>
   </div>
   <dl><dt>Best valid finish</dt><dd>{seconds(j.summary.best)}</dd><dt>Median successful time</dt><dd>{seconds(j.summary.median)}</dd><dt>Valid finishes</dt><dd>{j.summary.finishes} / {j.summary.attempts}</dd><dt>95% finish interval</dt><dd>{j.summary.finish_interval?.map(x=>(100*x).toFixed(1)+'%').join(' – ')||'—'}</dd></dl>
  </article>)}</div>
  <section className="comparison">
   <div className="section-heading"><div><h2>Checkpoint comparison</h2><small>Select 2–4 evaluations to compare course-by-course.</small></div><span>{compareIds.length} selected</span></div>
   {comparison.error&&<p role="alert" className="error">{comparison.error}</p>}
   {comparison.loading&&<div className="empty">Loading evaluation records…</div>}
   {!comparison.loading&&!compareIds.length&&<div className="empty">Select evaluations above to compare frozen checkpoints.</div>}
   {!comparison.loading&&compareIds.length===1&&<div className="empty">Select one more evaluation. Comparisons require at least two checkpoints.</div>}
   {!comparison.loading&&compareIds.length>1&&<>
    <div className={'comparison-status '+(comparisonQualified?'qualified':'unqualified')}>
     <strong>{comparisonQualified?'Matched complete comparison':'Not a qualified comparison'}</strong>
     <span>{!conditionsMatch?'Evaluation conditions differ. Match courses, attempts, action seed, action mode, start gear, and simulation scale.':!buildsMatch?'Checkpoint player-build identities differ or are unavailable.':!evaluationsComplete?'At least one evaluation is partial, active, or incomplete.':'Conditions, build identity, and requested attempts match.'}</span>
    </div>
    <div className="comparison-conditions">{selectedEvaluations[0]?.spec?.attempts} attempts per course · seed {selectedEvaluations[0]?.spec?.seed} · {selectedEvaluations[0]?.spec?.deterministic?'deterministic':'stochastic'} actions · {selectedEvaluations[0]?.spec?.startingGear} start · {selectedEvaluations[0]?.spec?.timeScale}× simulation</div>
    <div className="table-scroll"><table className="comparison-table"><thead><tr><th>Course</th>{selectedEvaluations.map(j=><th key={j.id}><span>{runs.find(r=>r.id===state.checkpoints.find(c=>c.id===j.spec.checkpoint)?.run)?.spec.name||j.id.slice(0,8)}</span><small>Checkpoint {j.spec.checkpoint.slice(0,12)}</small></th>)}</tr></thead><tbody>
     {comparisonCourses.map(courseId=><tr key={courseId}><th>{state.courses.find(c=>c.id===courseId)?.name||courseId.slice(0,12)}</th>{selectedEvaluations.map(j=>{
      if(!(j.spec.courses||[]).includes(courseId))return <td key={j.id}>Not selected</td>;
      const records=(j.episodes||[]).filter((r:any)=>r.courseId===courseId);
      const successful=records.filter((r:any)=>r.outcome==='Finished'&&r.valid&&Number.isFinite(r.seconds)).map((r:any)=>r.seconds).sort((a:number,b:number)=>a-b);
      const median=successful.length?successful.length%2?successful[(successful.length-1)/2]:(successful[successful.length/2-1]+successful[successful.length/2])/2:null;
      const outcomes=Object.entries(records.reduce((counts:Record<string,number>,r:any)=>({...counts,[r.outcome||'Unknown']:(counts[r.outcome||'Unknown']||0)+1}),{} as Record<string,number>)).map(([name,count])=>`${name} ${count}`).join(' · ');
      return <td key={j.id}><strong>Best {seconds(successful[0]??null)}</strong><small>Median {seconds(median)}</small><small>{successful.length} valid / {records.length} of {j.spec.attempts} attempts</small><small>{outcomes||'No attempt records'}</small></td>;
     })}</tr>)}
    </tbody></table></div>
   </>}
  </section>
 </>}
 {!selected && tab==='Checkpoints' && (
  <>
   <div className="section-heading"><h2>Frozen checkpoints</h2><span>{state.checkpoints.length} models</span></div>
   <div className="table-scroll"><table><thead><tr><th>Identity</th><th>Run</th><th>Trainer step</th><th>Preparation</th><th/></tr></thead>
    <tbody>{state.checkpoints.map(c=><tr key={c.id}>
     <td>{c.id.slice(0,16)}</td>
     <td>{runs.find(j=>j.id===c.run)?.spec.name||c.run}<small>{checkpointOriginLabel(c)}</small></td>
     <td>{number(c.step)}</td>
     <td><span className="status">{c.prepared?'Ready':'Not prepared'}</span></td>
     <td className="actions">
      <button title="Prepare checkpoint" disabled={c.prepared||busy} onClick={()=>act(`/checkpoints/${c.id}/prepare`)}><Box size={16}/></button>
      <button title={c.prepared?'Watch checkpoint':'Prepare checkpoint to watch'} aria-label={c.prepared?'Watch checkpoint':'Prepare checkpoint to watch'} disabled={busy} onClick={()=>watchCheckpoint(c)}><Eye size={16}/></button>
      <button title="Evaluate checkpoint" aria-label="Evaluate checkpoint" disabled={!c.prepared} onClick={()=>{applyCheckpointDefaults(c.id,'evaluation');setModal('evaluation')}}><Flag size={16}/></button>
     </td>
    </tr>)}</tbody>
   </table></div>
   {!state.checkpoints.length&&<div className="empty">No managed checkpoints yet</div>}
  </>
 )}
 {selected&&telemetry&&<ExperimentPage data={telemetry} courses={state.courses} checkpoints={state.checkpoints} connected={connected} busy={busy} error={telemetryError} onBack={()=>switchTab('Experiments')} onAction={(action)=>{void act(`/jobs/${selected}/${action}`)}} onTrajectory={(courseId,attempt)=>api(`/jobs/${selected}/trajectory?course=${encodeURIComponent(courseId)}&attempt=${attempt}`)} />}
 {selected&&!telemetry&&!telemetryError&&<div className="empty">Loading experiment telemetry…</div>}
 {selected&&!telemetry&&telemetryError&&<div className="empty"><p>{telemetryError}</p><button onClick={()=>setTelemetryError('')}>Retry</button></div>}
 {!selected&&<footer>Rally Research Lab <span>{state.time?`Updated ${new Date(state.time*1000).toLocaleTimeString()}`:'Waiting for data'}</span></footer>}</main>
 {circuitDialog&&<div className="modal-backdrop"><form className="modal" onSubmit={async e=>{e.preventDefault();await act('/circuits/import',circuitForm);setCircuitDialog(false)}}><div className="section-heading"><h2>Import Nordschleife</h2><button type="button" onClick={()=>setCircuitDialog(false)}><X size={19}/></button></div><p>Use the preserved 20.832 km source layout with reconstructed asphalt, concrete bowls, kerbs and barriers. Estimated detail is labeled; imported variants require review before training.</p><label>First sector<select value={circuitForm.firstSector} onChange={e=>setCircuitForm({...circuitForm,firstSector:+e.target.value,sectorCount:Math.min(circuitForm.sectorCount,16-+e.target.value)})}>{Array.from({length:16},(_,i)=><option key={i} value={i}>Sector {i+1}</option>)}</select></label><label>Contiguous sectors<select value={circuitForm.sectorCount} onChange={e=>setCircuitForm({...circuitForm,sectorCount:+e.target.value})}>{Array.from({length:16-circuitForm.firstSector},(_,i)=><option key={i} value={i+1}>{i+1===16?'Full lap':`${i+1} sector${i?'s':''}`}</option>)}</select></label><div className="modal-footer"><button type="button" onClick={()=>setCircuitDialog(false)}>Cancel</button><button className="primary" disabled={busy}>Import frozen circuit</button></div></form></div>}
 {modal&&<div className="modal-backdrop"><form className="modal" onSubmit={e=>{e.preventDefault();if(modal==='training')act('/runs',run);else if(modal==='course')act('/courses',courseForm);else act(modal==='viewer'?'/viewers':'/evaluations',evaluation)}}><div className="section-heading"><h2>{modal==='training'?'New experiment':modal==='course'?'Save generated course':modal==='viewer'?'Watch frozen checkpoint':'New evaluation'}</h2><button type="button" title="Close dialog" onClick={()=>setModal('')}><X size={19}/></button></div>
 {modal==='course'&&<label>Stage length (m)<input type="number" min="1000" max="5000" step="500" value={courseForm.length} onChange={e=>setCourseForm({...courseForm,length:+e.target.value})}/></label>}
 {modal==='training'?<><label>Experiment name<input required value={run.name} onChange={e=>setRun({...run,name:e.target.value})}/></label><div className="segmented">{['specialist','generalist'].map(mode=><button type="button" key={mode} className={run.mode===mode?'chosen':''} onClick={()=>setRun({...run,mode})}>{mode}</button>)}</div>{run.mode==='specialist'&&<label>Frozen course<select required value={run.course} onChange={e=>setRun({...run,course:e.target.value})}><option value="">Select course</option>{library.map(c=><option key={c.id} value={c.id}>{c.name}</option>)}</select></label>}<div className="form-grid"><label>Training steps<input type="number" min="1000" max="25000000" step="1000" value={run.steps} onChange={e=>setRun({...run,steps:+e.target.value})}/></label><label>Workers<input type="number" min="1" max="4" value={run.workers} onChange={e=>setRun({...run,workers:+e.target.value})}/></label><label>Random seed<input type="number" min="0" value={run.seed} onChange={e=>setRun({...run,seed:+e.target.value})}/></label><label>Reward version<select value={run.reward} onChange={e=>setRun({...run,reward:e.target.value})}><option value="baseline-v1">Baseline v1</option><option value="time-attack-v1">Time attack v1</option><option value="time-attack-no-slip-v1">Time attack / no slide penalty</option></select></label></div><label>Starting gear<select value={run.startingGear} onChange={e=>setRun({...run,startingGear:e.target.value})}><option value="neutral">Neutral</option><option value="first">First gear</option></select></label><label>Initialize from<select value={run.parent} onChange={e=>setRun({...run,parent:e.target.value})}><option value="">Fresh weights</option>{state.checkpoints.map(c=><option key={c.id} value={c.id}>{c.run.slice(0,8)} · {number(c.step)}</option>)}</select></label></>:modal==='course'?<><label>Name<input required value={courseForm.name} onChange={e=>setCourseForm({...courseForm,name:e.target.value})}/></label><label>Course family<select value={courseForm.family} onChange={e=>setCourseForm({...courseForm,family:e.target.value})}>{['gentle','technical','crests'].map(f=><option key={f}>{f}</option>)}</select></label><label>Seed<input type="number" min="0" value={courseForm.seed} onChange={e=>setCourseForm({...courseForm,seed:+e.target.value})}/></label><label>Rocks per 100 m<input type="number" min="0" max="2" step=".5" value={courseForm.rocks} onChange={e=>setCourseForm({...courseForm,rocks:+e.target.value})}/></label><label className="checkbox"><input type="checkbox" checked={courseForm.scenery} onChange={e=>setCourseForm({...courseForm,scenery:e.target.checked})}/>Collidable roadside trees and boulders</label>{!courseForm.scenery&&courseForm.rocks===0&&<small className="course-legacy-note">Road and roadside hazards disabled. This changes the course identity.</small>}</>:<>
  <label>Checkpoint<select required value={evaluation.checkpoint} onChange={e=>applyCheckpointDefaults(e.target.value,modal==='viewer'?'viewer':'evaluation')}><option value="">Select checkpoint</option>{state.checkpoints.filter(c=>c.prepared).map(c=><option key={c.id} value={c.id}>{c.run.slice(0,8)} · {number(c.step)}</option>)}</select></label>
  <label>Course<select required multiple={modal!=='viewer'} value={modal==='viewer'?evaluation.courses[0]||'':evaluation.courses} onChange={e=>{setEvaluation({...evaluation,courses:Array.from(e.target.selectedOptions).map(o=>o.value)});setCheckpointContext(null)}}><option value="">Select course</option>{state.courses.map(c=><option key={c.id} value={c.id}>{c.name} · {c.suite}</option>)}</select></label>
  {checkpointContext?.checkpoint===evaluation.checkpoint&&<small className="checkpoint-context-note">{checkpointContext.note}</small>}
  <div className={modal==='viewer'?'form-grid viewer-settings':'form-grid'}>
   {modal!=='viewer'&&<label>Attempts per course<input type="number" min="1" max="100" value={evaluation.attempts} onChange={e=>setEvaluation({...evaluation,attempts:+e.target.value})}/></label>}
   <label>{modal==='viewer'?'View seed':'Action seed'}<input type="number" min="0" value={evaluation.seed} onChange={e=>setEvaluation({...evaluation,seed:+e.target.value})}/></label>
  </div>
  <label>Starting gear<select value={evaluation.startingGear} onChange={e=>setEvaluation({...evaluation,startingGear:e.target.value as 'neutral'|'first'})}><option value="neutral">Neutral</option><option value="first">First gear</option></select></label>
  <label className="checkbox"><input type="checkbox" checked={evaluation.deterministic} onChange={e=>setEvaluation({...evaluation,deterministic:e.target.checked})}/>{modal==='viewer'?'Deterministic preview':'Deterministic actions'}</label>
 </>}
 {modal==='training'&&<><label>Reference demonstrations<select multiple value={run.demonstrations} onChange={e=>setRun({...run,demonstrations:Array.from(e.target.selectedOptions).map(o=>o.value)})}>{state.demonstrations.map(d=><option key={d.id} value={d.id}>{state.courses.find(c=>c.id===d.courseId)?.name||d.courseId.slice(0,8)} · {d.controller} · {number(d.transitions)} transitions</option>)}</select></label>{run.demonstrations.length>0&&<><div className="segmented">{[['isolated','Clone first'],['joint','Clone with PPO']].map(([value,label])=><button type="button" key={value} className={run.imitationMode===value?'chosen':''} onClick={()=>setRun({...run,imitationMode:value})}>{label}</button>)}</div><label>Cloning-only step budget<input type="number" min="1000" max="1000000" step="1000" value={run.imitationSteps} onChange={e=>setRun({...run,imitationSteps:+e.target.value})}/></label></>}</>}
 <div className="modal-footer"><button type="button" onClick={()=>setModal('')}>Cancel</button><button className="primary" disabled={busy} type="submit"><Play size={15}/>{busy?'Submitting…':modal==='viewer'?'Open Unity viewer':modal==='course'?'Generate and save':'Queue experiment'}</button></div></form></div>}</div>
}
createRoot(document.getElementById('root')!).render(<App/>);

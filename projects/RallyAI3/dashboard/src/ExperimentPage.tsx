import React, {useEffect, useMemo, useRef, useState} from 'react';
import {ArrowLeft, CircleStop, Download, Play, RotateCcw, TriangleAlert} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts';

type Point={x:number;y:number;z:number};
type Course={id:string;name:string;legacyObstacleLayout?:boolean;definition:{gates:Point[];length:number;width:number}};
type Props={
  data:any;
  courses:Course[];
  checkpoints:any[];
  connected:boolean;
  busy:boolean;
  error:string;
  onBack:()=>void;
  onAction:(action:'stop'|'resume'|'force')=>void;
  onTrajectory:(courseId:string,attempt:number)=>Promise<any[]>;
};

const tones:Record<string,string>={Finished:'#11845d',TimedOut:'#b38728',RolledOver:'#b64842',HitObstacle:'#da6c32',HardImpact:'#9a5144',FellOff:'#7658a5',Stalled:'#69766f',Unknown:'#87928a'};
const number=(value:number|null|undefined,digits=0)=>value==null||!Number.isFinite(value)?'—':value.toLocaleString(undefined,{maximumFractionDigits:digits});
const seconds=(value:number|null|undefined)=>value==null||!Number.isFinite(value)?'—':`${value.toFixed(1)} s`;
const clock=(value:number|null|undefined)=>value==null||!Number.isFinite(value)?'—':`${Math.floor(value/3600).toString().padStart(2,'0')}:${Math.floor(value%3600/60).toString().padStart(2,'0')}:${Math.floor(value%60).toString().padStart(2,'0')}`;
const episodeProgress=(row:any)=>Number(row.waypoints??row.waypointsReached??0)/Math.max(1,Number(row.target??row.waypointTarget??1));
const tickTime=(value:number)=>new Date(value*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});

function StageCanvas({course,episodes=[],trajectory=[],cursor=1}:{course?:Course;episodes?:any[];trajectory?:any[];cursor?:number}){
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{
    const canvas=ref.current;if(!canvas)return;
    const draw=()=>{
      const box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
      const dpr=window.devicePixelRatio||1;canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
      const ctx=canvas.getContext('2d');if(!ctx)return;ctx.setTransform(dpr,0,0,dpr,0,0);
      const w=box.width,h=box.height;ctx.fillStyle='#f5f7f5';ctx.fillRect(0,0,w,h);
      ctx.strokeStyle='#e4e9e5';ctx.lineWidth=1;
      for(let x=18;x<w;x+=28){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
      for(let y=16;y<h;y+=28){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
      if(!course?.definition?.gates?.length){ctx.fillStyle='#7b8780';ctx.font='12px system-ui';ctx.fillText('Course geometry was not saved for this run.',16,24);return}
      const geometry=course.definition,points=geometry.gates;
      const xs=points.map(p=>p.x),zs=points.map(p=>p.z),minX=Math.min(...xs),maxX=Math.max(...xs),minZ=Math.min(...zs),maxZ=Math.max(...zs);
      const dx=Math.max(1,maxX-minX),dz=Math.max(1,maxZ-minZ),scale=Math.min((w-70)/dx,(h-56)/dz);
      const pos=(p:Point):[number,number]=>[(w-dx*scale)/2+(p.x-minX)*scale,h-28-(p.z-minZ)*scale];
      ctx.lineJoin='round';ctx.lineCap='round';
      const path=()=>{ctx.beginPath();points.forEach((point,i)=>{const [x,y]=pos(point);i?ctx.lineTo(x,y):ctx.moveTo(x,y)})};
      path();ctx.strokeStyle='#b4beb7';ctx.lineWidth=Math.max(8,Math.min(24,geometry.width*scale));ctx.stroke();
      path();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.setLineDash([5,5]);ctx.stroke();ctx.setLineDash([]);
      const total=points.slice(1).reduce((sum,p,i)=>sum+Math.hypot(p.x-points[i].x,p.z-points[i].z),0)||1;
      const pointAt=(station:number,offset:number):[number,number]=>{
        let target=Math.max(0,Math.min(1,station/Math.max(1,geometry.length)))*total,walked=0;
        for(let i=1;i<points.length;i++){
          const a=points[i-1],b=points[i],length=Math.hypot(b.x-a.x,b.z-a.z);
          if(walked+length>=target||i===points.length-1){const t=length?(target-walked)/length:0;const x=a.x+(b.x-a.x)*t,z=a.z+(b.z-a.z)*t,nx=length?(b.z-a.z)/length:0,nz=length?-(b.x-a.x)/length:0;return pos({x:x+nx*offset,y:0,z:z+nz*offset})}
          walked+=length;
        }
        return pos(points[points.length-1]);
      };
      const start=pos(points[0]),finish=pos(points[points.length-1]);
      for(const [p,label,color] of [[start,'START','#087957'],[finish,'FINISH','#20362b']] as const){ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.fill();ctx.font='10px system-ui';ctx.fillText(label,p[0]+8,p[1]-7)}
      for(const row of episodes){
        if(row.outcome==='Finished'||row.station==null||!Number.isFinite(Number(row.station)))continue;
        const [x,y]=pointAt(Number(row.station),Number(row.off??0));ctx.fillStyle=tones[row.outcome]||tones.Unknown;
        ctx.globalAlpha=.76;ctx.beginPath();ctx.arc(x,y,3.4,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
      }
      const samples=trajectory.slice(0,Math.max(1,Math.floor(trajectory.length*cursor)));
      for(let i=1;i<samples.length;i++){
        const a=samples[i-1].position,b=samples[i].position;if(!a||!b)continue;
        const p1=pos(a),p2=pos(b);ctx.strokeStyle=`hsl(${Math.max(0,Math.min(145,Number(samples[i].speed||0)*3))},65%,36%)`;ctx.lineWidth=3;
        ctx.beginPath();ctx.moveTo(...p1);ctx.lineTo(...p2);ctx.stroke();
      }
      if(samples.length){const p=pos(samples[samples.length-1].position);ctx.fillStyle='#cf473e';ctx.beginPath();ctx.arc(...p,5,0,Math.PI*2);ctx.fill()}
    };
    draw();const observer=new ResizeObserver(draw);observer.observe(canvas);return()=>observer.disconnect();
  },[course,episodes,trajectory,cursor]);
  return <canvas className="telemetry-map" ref={ref} aria-label={course?`${course.name} route and episode failure locations`:'Recorded episode locations'} />;
}

function ChartFrame({title,aside,children,empty}:{title:string;aside?:React.ReactNode;children:React.ReactNode;empty?:boolean}){
  return <section className="telemetry-chart"><div className="telemetry-chart-heading"><h3>{title}</h3>{aside}</div>{empty?<div className="chart-empty">No recorded data for this view yet.</div>:children}</section>;
}

function ExperimentPage({data,courses,checkpoints,connected,busy,error,onBack,onAction,onTrajectory}:Props){
  const [view,setView]=useState('Overview');
  const [scalarTag,setScalarTag]=useState('');
  const [outcomeFilter,setOutcomeFilter]=useState('All outcomes');
  const [selectedAttempt,setSelectedAttempt]=useState<any>(null);
  const [trajectory,setTrajectory]=useState<any[]>([]);
  const [cursor,setCursor]=useState(0);
  const [playing,setPlaying]=useState(false);
  const [forceConfirm,setForceConfirm]=useState(false);
  const job=data.job,spec=job.spec||{},detail=job.detail||{},history=data.history||{};
  const episodes:any[]=history.episodes||[],scalars:any[]=history.scalars||[],resources:any[]=(history.resources||[]).filter((row:any)=>!row.stale);
  const tags=[...new Set<string>(scalars.map((p:any)=>p.tag).filter(Boolean))].sort();
  const runCourse=courses.find(course=>course.id===spec.course);
  const legacyRun=Boolean(runCourse?.legacyObstacleLayout);
  const summary=data.summary||{};
  const latestResource=resources.length?resources[resources.length-1]:null;
  const cpuResources=resources.filter((row:any)=>['dashboard-sampler','worker'].includes(row.source)&&row.cpu_percent!=null&&Number.isFinite(Number(row.cpu_percent)));
  const latestCpuResource=cpuResources.length?cpuResources[cpuResources.length-1]:null;
  const cpuChartResources=resources.map((row:any)=>({...row,cpu_percent:['dashboard-sampler','worker'].includes(row.source)&&row.cpu_percent!=null&&Number.isFinite(Number(row.cpu_percent))?Number(row.cpu_percent):null}));
  let lastThroughputStep=-1;
  const throughputChartResources=resources.map((row:any)=>{const step=row.trainer_step==null?null:Number(row.trainer_step),advanced=step!=null&&Number.isFinite(step)&&step>lastThroughputStep;if(advanced)lastThroughputStep=step!;const rate=Number(row.steps_per_second);return {...row,steps_per_second:advanced&&Number.isFinite(rate)&&rate>0?rate:null}});
  const lastMeasuredThroughput=resources.slice().reverse().find((row:any)=>Number(row.steps_per_second)>0&&Number.isFinite(Number(row.steps_per_second)));
  const currentStep=detail.trainer_step==null?null:Number(detail.trainer_step),targetSteps=spec.steps==null?null:Number(spec.steps),stepsPerSecond=Number(detail.steps_per_second)>0?Number(detail.steps_per_second):lastMeasuredThroughput?Number(lastMeasuredThroughput.steps_per_second):null;
  const throughputDetail=Number(detail.steps_per_second)>0?'Latest trainer summary interval':lastMeasuredThroughput?`Last measured ${clock(Math.max(0,Date.now()/1000-Number(lastMeasuredThroughput.created)))} ago`:'Waiting for a measurable step interval';
  const active=['queued','starting','running','stopping'].includes(job.state);
  const rewardEpisodes=episodes.filter((row:any)=>row.reward!=null&&Number.isFinite(Number(row.reward)));
  const progressEpisodes=episodes.filter((row:any)=>(row.waypoints??row.waypointsReached)!=null&&(row.target??row.waypointTarget)!=null);
  const recentRewards=rewardEpisodes.slice(-100),recentProgress=progressEpisodes.slice(-100);
  const meanReward=recentRewards.length?recentRewards.reduce((sum:number,row:any)=>sum+Number(row.reward),0)/recentRewards.length:null;
  const meanProgress=recentProgress.length?100*recentProgress.reduce((sum:number,row:any)=>sum+Math.min(1,Math.max(0,episodeProgress(row))),0)/recentProgress.length:null;
  const wallElapsed=Math.max(0,(active?Date.now()/1000:Number(job.updated||Date.now()/1000))-Number(job.created||0));
  const eta=job.state==='running'&&stepsPerSecond!=null&&stepsPerSecond>0&&targetSteps!=null&&currentStep!=null&&targetSteps>currentStep?(targetSteps-currentStep)/stepsPerSecond:null;
  const outcomes=Object.entries(summary.outcomes||{}).map(([name,count])=>({name,count:Number(count),fill:tones[name]||tones.Unknown}));
  const filteredEpisodes=outcomeFilter==='All outcomes'?episodes:episodes.filter(row=>row.outcome===outcomeFilter);
  const rewardRows=episodes.map((row:any,index:number)=>({episode:index+1,reward:row.reward==null?null:Number(row.reward),step:row.step==null?null:Number(row.step),progress:(row.waypoints??row.waypointsReached)==null||(row.target??row.waypointTarget)==null?null:Math.min(1,Math.max(0,episodeProgress(row)))*100,outcome:row.outcome})).filter((row:any)=>row.reward!=null&&Number.isFinite(row.reward));
  const selectedScalars=scalars.filter((point:any)=>point.tag===scalarTag&&point.value!=null&&Number.isFinite(Number(point.value))).sort((a:any,b:any)=>Number(a.step)-Number(b.step));
  const scalarLatest=selectedScalars.length?selectedScalars[selectedScalars.length-1]:null;
  const stale=active&&detail.heartbeat!=null&&Date.now()/1000-Number(detail.heartbeat)>10;
  const allCheckpoints=checkpoints.filter(checkpoint=>checkpoint.run===job.id||(checkpoint.origins||[]).some((origin:any)=>origin.run===job.id)).sort((a,b)=>Number(a.step||0)-Number(b.step||0));
  const log=data.log||'';
  const lifecycle=[...(history.lifecycle||[])].sort((a:any,b:any)=>Number(a.seq)-Number(b.seq));
  const selectedCourse=selectedAttempt?courses.find(course=>course.id===selectedAttempt.courseId):runCourse;

  useEffect(()=>{if(tags.length&&!tags.includes(scalarTag))setScalarTag(tags.includes('Environment/Cumulative Reward')?'Environment/Cumulative Reward':tags[0])},[tags.join('|'),scalarTag]);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setCursor(value=>{if(value>=1){setPlaying(false);return 1}return Math.min(1,value+.004)}),40);return()=>clearInterval(timer)},[playing]);
  const loadAttempt=async(record:any)=>{if(!record.courseId||!record.attempt)return;try{const samples=await onTrajectory(record.courseId,Number(record.attempt));setSelectedAttempt(record);setTrajectory(samples);setCursor(0);setPlaying(samples.length>0);setView('Driving')}catch{/* the parent displays the API error */}};

  return <div className="experiment-page">
    <div className="experiment-topline"><button className="back-button" onClick={onBack}><ArrowLeft size={16}/>Experiments</button><span className={'status '+job.state}>{job.state}</span><span className="experiment-id">{job.id}</span><span className="freshness"><i className={'dot '+(connected?'online':'')}/>{connected?'Live updates':`Last known state${stale?' · stale':''}`}</span></div>
    <div className="experiment-heading">
      <div className="experiment-title"><span className="breadcrumb">{job.kind.toUpperCase()} / {spec.mode||'managed job'}</span><h1>{spec.name||job.kind}</h1><p>{spec.mode==='specialist'?runCourse?.name||`Frozen course ${String(spec.course||'').slice(0,12)}`:spec.mode==='generalist'?'Procedural course distribution':job.kind==='evaluation'?`${(spec.courses||[]).length} course evaluation`:job.kind}</p></div>
      <div className="experiment-actions">
        {active&&job.state!=='stopping'&&<button disabled={busy} onClick={()=>onAction('stop')}><CircleStop size={16}/>Stop and save</button>}
        {job.state==='stopping'&&<button className="danger-outline" disabled={busy} onClick={()=>setForceConfirm(true)}><TriangleAlert size={15}/>Force terminate</button>}
        {job.kind==='training'&&detail.resumable&&['stopped','interrupted','failed'].includes(job.state)&&<button title={legacyRun?'This historical run uses a course with stale template obstacles. Start fresh on a corrected course.':'Resume training'} disabled={busy||legacyRun} onClick={()=>onAction('resume')}><RotateCcw size={15}/>Resume</button>}
        <a className="icon-link" title="Export episode records as CSV" href={`/api/jobs/${job.id}/export?format=csv`}><Download size={17}/></a>
      </div>
    </div>
    {forceConfirm&&<div className="force-confirm" role="alert"><strong>Force termination can lose recent unsaved trainer progress.</strong><span>Graceful stop has already been requested.</span><div><button onClick={()=>setForceConfirm(false)}>Keep waiting</button><button className="danger" disabled={busy} onClick={()=>{setForceConfirm(false);onAction('force')}}>Terminate process group</button></div></div>}
    {detail.error&&<div className="telemetry-error" role="alert">Run error: {detail.error}</div>}
    {legacyRun&&<div className="telemetry-error" role="status">Historical course layout: stale template obstacles were included regardless of the requested rock density. Existing results are preserved; this run cannot be resumed.</div>}
    {error&&<div className="telemetry-error" role="alert">{error}</div>}
    {!connected&&<div className="connection">Live updates disconnected. This view retains the last received telemetry.</div>}
    {stale&&<div className="stale-banner">The supervisor heartbeat is older than 10 seconds. Resource readings are marked with their sample time.</div>}
    <div className="experiment-metadata"><span><b>Created</b>{new Date(Number(job.created)*1000).toLocaleString()}</span><span><b>Elapsed wall time</b>{clock(wallElapsed)}</span><span><b>Seed</b>{spec.seed==null?'—':number(spec.seed)}</span><span><b>Workers</b>{spec.workers==null?'—':number(spec.workers)}</span><span><b>Reward</b>{spec.reward||'—'}</span><span><b>Parent checkpoint</b>{spec.parent?String(spec.parent).slice(0,12):'Fresh weights'}</span><span><b>Build</b>{data.manifest?.build?.source?.hash?.slice(0,12)||data.manifest?.build?.hash?.slice(0,12)||'Identity unavailable'}</span></div>
    <nav className="telemetry-tabs" aria-label="Experiment telemetry">{['Overview','Learning','Driving','Resources','Logs & Configuration'].map(name=><button key={name} className={view===name?'active':''} onClick={()=>setView(name)}>{name}</button>)}</nav>

    {view==='Overview'&&<>
      <div className="telemetry-kpis">
        <Metric label="Trainer steps" value={job.kind==='training'?`${number(currentStep)} / ${number(targetSteps)}`:'Not a training run'} detail={job.kind==='training'&&eta!=null?`~${clock(eta)} estimated remaining`:'Aggregate trainer steps'} />
        <Metric label={job.kind==='training'?'Episodes completed':'Evaluation attempts'} value={summary.episodes_completed==null&&summary.attempts==null?'—':number(summary.episodes_completed??summary.attempts)} detail={summary.requested!=null?`${number(summary.attempts)} recorded / ${number(summary.requested)} requested`:'Recorded episode log'} />
        <Metric label="Mean episode reward" value={number(meanReward,2)} detail={`Last ${recentRewards.length} episodes with reward · learning signal`} />
        <Metric label="Mean stage progress" value={meanProgress==null?'—':`${meanProgress.toFixed(1)}%`} detail={`Last ${recentProgress.length} episodes with progress`} />
        <Metric label="Throughput" value={job.kind==='training'&&stepsPerSecond!=null&&stepsPerSecond>0?`${number(stepsPerSecond)} steps/s`:'—'} detail={throughputDetail} />
        <Metric label="Compute snapshot" value={latestResource?.memory_bytes!=null?`${(Number(latestResource.memory_bytes)/1073741824).toFixed(2)} GB`:'—'} detail={latestResource?`CPU ${latestCpuResource?`${number(Number(latestCpuResource.cpu_percent),0)}%`:'unavailable'} · ${number(latestResource.children)} child processes · ${tickTime(Number(latestResource.created))}`:'No resource sample recorded'} />
      </div>
      <div className="telemetry-grid two">
        <ChartFrame title="Episode reward" aside={<span>Raw episode values · not lap times</span>} empty={!rewardRows.length}><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={rewardRows}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="episode" tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip labelFormatter={value=>`Episode ${value}`}/><Line dataKey="reward" name="Reward" stroke="#087957" dot={false} isAnimationActive={false}/></LineChart></ResponsiveContainer></div></ChartFrame>
        <ChartFrame title="Episode outcomes" aside={<span>{number(summary.episodes_completed??summary.attempts)} total</span>} empty={!outcomes.length}><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={outcomes} layout="vertical" margin={{left:18,right:15}}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number" tick={{fontSize:10}} allowDecimals={false}/><YAxis type="category" dataKey="name" width={90} tick={{fontSize:10}}/><Tooltip/><Bar dataKey="count" name="Episodes">{outcomes.map((item:any)=><Cell key={item.name} fill={item.fill}/>)}</Bar></BarChart></ResponsiveContainer></div></ChartFrame>
      </div>
      <div className="telemetry-grid two">
        <ChartFrame title="Stage progress by episode" aside={<span>Fraction of course waypoints reached</span>} empty={!rewardRows.some((row:any)=>row.progress!=null&&Number.isFinite(row.progress))}><div className="chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rewardRows}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="episode" tick={{fontSize:10}}/><YAxis domain={[0,100]} unit="%" tick={{fontSize:10}}/><Tooltip/><Area dataKey="progress" name="Stage progress" stroke="#287fa0" fill="#cfe8ec" isAnimationActive={false}/></AreaChart></ResponsiveContainer></div></ChartFrame>
        <ChartFrame title={spec.mode==='generalist'?'Failure station and lateral offset':'Course failure locations'} aside={<span>Training diagnostics · not recorded trajectories</span>} empty={!episodes.some((row:any)=>row.outcome!=='Finished'&&row.station!=null&&Number.isFinite(Number(row.station)))}>
          {spec.mode==='generalist'?<div className="chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{left:8,right:16,bottom:8}}><CartesianGrid strokeDasharray="3 3"/><XAxis type="number" dataKey="station" name="Station" unit=" m" tick={{fontSize:10}}/><YAxis type="number" dataKey="off" name="Lateral offset" unit=" m" tick={{fontSize:10}}/><Tooltip cursor={{strokeDasharray:'3 3'}}/><ReferenceLine y={0} stroke="#97a39b"/>{Object.keys(tones).filter(name=>name!=='Finished'&&name!=='Unknown').map(name=><Scatter key={name} name={name} data={episodes.filter((row:any)=>row.outcome===name&&row.station!=null&&Number.isFinite(Number(row.station))).map((row:any)=>({station:Number(row.station),off:Number(row.off||0),seed:row.seed}))} fill={tones[name]} />)}</ScatterChart></ResponsiveContainer></div>:<StageCanvas course={runCourse} episodes={episodes}/>}</ChartFrame>
      </div>
      <div className="telemetry-note">Training reward, episode duration, and stage progress describe learning behavior. They are not valid time-attack results; official times come from completed evaluation attempts.</div>
    </>}

    {view==='Learning'&&<>
      <div className="learning-toolbar"><div><h2>Trainer scalars</h2><small>Only metrics present in this run’s TensorBoard event files are offered.</small></div>{tags.length>0&&<label className="scalar-select">Scalar<select value={scalarTag} onChange={event=>setScalarTag(event.target.value)}>{tags.map(tag=><option key={tag}>{tag}</option>)}</select></label>}</div>
      <div className="telemetry-grid two">
        <ChartFrame title={scalarTag||'Trainer scalar'} aside={<span>Source step · TensorBoard</span>} empty={!selectedScalars.length}><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={selectedScalars}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="step" type="number" domain={['dataMin','dataMax']} tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip labelFormatter={value=>`Trainer step ${number(Number(value))}`}/><Line dataKey="value" name={scalarTag} stroke="#087957" dot={false} isAnimationActive={false}/></LineChart></ResponsiveContainer></div></ChartFrame>
        <ChartFrame title="Reward by episode" aside={<span>Episode log</span>} empty={!rewardRows.length}><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={rewardRows}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="episode" tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip labelFormatter={value=>`Episode ${value}`}/><Line dataKey="reward" stroke="#bd792f" dot={false} isAnimationActive={false}/></LineChart></ResponsiveContainer></div></ChartFrame>
      </div>
      <div className="learning-facts"><span>Selected latest value <b>{scalarLatest?number(Number(scalarLatest.value),4):'—'}</b></span><span>At trainer step <b>{scalarLatest?number(Number(scalarLatest.step)):'—'}</b></span><span>Recorded tags <b>{number(tags.length)}</b></span><span>Recent outcome rates come from episode records, not inferred reward</span></div>
      <div className="telemetry-grid two"><ChartFrame title="Outcome counts" empty={!outcomes.length}><div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={outcomes}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name" tick={{fontSize:9}}/><YAxis tick={{fontSize:10}} allowDecimals={false}/><Tooltip/><Bar dataKey="count">{outcomes.map((item:any)=><Cell key={item.name} fill={item.fill}/>)}</Bar></BarChart></ResponsiveContainer></div></ChartFrame><ChartFrame title="Progress fraction" empty={!rewardRows.length}><div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={rewardRows}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="episode" tick={{fontSize:10}}/><YAxis domain={[0,100]} unit="%" tick={{fontSize:10}}/><Tooltip/><Line dataKey="progress" name="Stage progress" stroke="#287fa0" dot={false} isAnimationActive={false}/></LineChart></ResponsiveContainer></div></ChartFrame></div>
    </>}

    {view==='Driving'&&<>
      {job.kind==='evaluation'?<>
        <div className="driving-toolbar"><div><h2>Evaluation attempts</h2><small>Replay uses the exact recorded inference trajectory; no live Unity viewer is launched.</small></div><label className="scalar-select">Outcome<select value={outcomeFilter} onChange={event=>setOutcomeFilter(event.target.value)}><option>All outcomes</option>{Object.keys(summary.outcomes||{}).map(name=><option key={name}>{name}</option>)}</select></label></div>
        <div className="episode-table-wrap"><table className="episode-table"><thead><tr><th>Course</th><th>Attempt</th><th>Outcome</th><th>Simulation time</th><th>Validity</th><th>Trajectory</th></tr></thead><tbody>{filteredEpisodes.map((record:any,index:number)=><tr key={`${record.courseId}-${record.attempt}-${index}`}><td>{courses.find(course=>course.id===record.courseId)?.name||String(record.courseId||'—').slice(0,14)}</td><td>{number(record.attempt)}</td><td><span className="outcome-label" style={{color:tones[record.outcome]||tones.Unknown}}>{record.outcome||'Unknown'}</span></td><td>{seconds(record.seconds)}</td><td>{record.valid?'Valid finish':'Not a valid finish'}</td><td>{record.courseId&&record.attempt?<button className="small-action" onClick={()=>loadAttempt(record)}><Play size={13}/>Replay</button>:'—'}</td></tr>)}</tbody></table>{!filteredEpisodes.length&&<div className="chart-empty">No attempts match this filter.</div>}</div>
        {trajectory.length>0&&<div className="replay-panel"><div className="telemetry-chart-heading"><h3>Recorded trajectory · {selectedCourse?.name||selectedAttempt?.courseId}</h3><span>{selectedAttempt?.outcome} · attempt {selectedAttempt?.attempt}</span></div><StageCanvas course={selectedCourse} trajectory={trajectory} cursor={cursor}/><div className="playback-controls"><button title={playing?'Pause replay':'Play replay'} onClick={()=>setPlaying(!playing)}><Play size={15}/></button><input aria-label="Replay position" type="range" min="0" max="1" step=".001" value={cursor} onChange={event=>{setPlaying(false);setCursor(+event.target.value)}}/></div></div>}
      </>:<>
        <div className="driving-toolbar"><div><h2>Episode forensics</h2><small>Each marker is the episode’s recorded terminal state, not a continuous training replay.</small></div><label className="scalar-select">Outcome<select value={outcomeFilter} onChange={event=>setOutcomeFilter(event.target.value)}><option>All outcomes</option>{Object.keys(summary.outcomes||{}).map(name=><option key={name}>{name}</option>)}</select></label></div>
        <div className="telemetry-grid two"><ChartFrame title={spec.mode==='generalist'?'Failure by procedural station':'Failure locations on frozen course'} aside={<span>Selected run · {number(filteredEpisodes.length)} shown</span>} empty={!filteredEpisodes.length}><StageCanvas course={runCourse} episodes={filteredEpisodes}/></ChartFrame><section className="episode-inspector"><h3>Episode diagnostics</h3><div className="episode-scroll"><table className="episode-table"><thead><tr><th>Ep.</th><th>Outcome</th><th>Progress</th><th>Seed</th><th>Failure</th></tr></thead><tbody>{filteredEpisodes.slice(-200).reverse().map((record:any,index:number)=><tr key={`${record.session||''}-${record.worker||''}-${record.ep??index}`}><td>{number(record.ep??episodes.length-index)}</td><td style={{color:tones[record.outcome]||tones.Unknown}}>{record.outcome||'Unknown'}</td><td>{number(record.waypoints??record.waypointsReached)} / {number(record.target??record.waypointTarget)}</td><td>{number(record.seed)}</td><td>{record.station!=null&&Number.isFinite(Number(record.station))?`${number(Number(record.station),0)} m · ${number(Number(record.off),1)} m off`:'—'}</td></tr>)}</tbody></table>{!filteredEpisodes.length&&<div className="chart-empty">Episode diagnostics appear when an episode is written.</div>}</div></section></div>
      </>}
      {job.kind==='training'&&<div className="telemetry-note">The training player does not record continuous trajectories. Failure station, offset, outcome, and seed come from episode-end forensics. Recorded trajectory replay is available for evaluation attempts only.</div>}
    </>}

    {view==='Resources'&&<>
      <div className="resource-summary"><Metric label="CPU at latest verified sample" value={latestCpuResource?`${number(Number(latestCpuResource.cpu_percent),0)}%`:'—'} detail={latestCpuResource?`Sampled ${tickTime(Number(latestCpuResource.created))}`:stale?'Last supervisor heartbeat is stale':'Waiting for a comparable process sample'} /><Metric label="Resident memory" value={latestResource?.memory_bytes==null?(detail.memory_bytes==null?'—':`${(Number(detail.memory_bytes)/1073741824).toFixed(2)} GB`):`${(Number(latestResource.memory_bytes)/1073741824).toFixed(2)} GB`} detail="Trainer and child processes combined"/><Metric label="Child processes" value={latestResource?.children==null?(detail.children==null?'—':number(detail.children)):number(Number(latestResource.children))} detail={`Configured workers: ${spec.workers==null?'—':number(spec.workers)}`}/><Metric label="Sample cadence" value={resources.length>1?`${number((Number(resources[resources.length-1].created)-Number(resources[resources.length-2].created)),0)} s`:'—'} detail="Low-rate process sampling; no thermal sensor claim"/></div>
      <div className="telemetry-grid two"><ChartFrame title="CPU use" aside={<span>Verified process-tree intervals · percent</span>} empty={!cpuResources.length}><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><AreaChart data={cpuChartResources}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="created" tickFormatter={tickTime} tick={{fontSize:9}}/><YAxis unit="%" tick={{fontSize:10}}/><Tooltip labelFormatter={value=>tickTime(Number(value))}/><Area dataKey="cpu_percent" name="CPU" stroke="#087957" fill="#c9e9da" isAnimationActive={false}/></AreaChart></ResponsiveContainer></div></ChartFrame>
        <ChartFrame title="Resident memory" aside={<span>Combined process tree · GB</span>} empty={!resources.some((row:any)=>row.memory_bytes!=null&&Number.isFinite(Number(row.memory_bytes)))}><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={resources.map((row:any)=>({...row,memory_gb:row.memory_bytes==null?null:Number(row.memory_bytes)/1073741824}))}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="created" tickFormatter={tickTime} tick={{fontSize:9}}/><YAxis unit=" GB" tick={{fontSize:10}}/><Tooltip labelFormatter={value=>tickTime(Number(value))}/><Line dataKey="memory_gb" name="Memory" stroke="#c16d37" dot={false} isAnimationActive={false}/></LineChart></ResponsiveContainer></div></ChartFrame>
      </div>
      <ChartFrame title="Trainer throughput" aside={<span>Reported rate when trainer steps advance</span>} empty={!throughputChartResources.some((row:any)=>row.steps_per_second!=null)}><div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={throughputChartResources}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="created" tickFormatter={tickTime} tick={{fontSize:9}}/><YAxis unit=" steps/s" tick={{fontSize:10}}/><Tooltip labelFormatter={value=>tickTime(Number(value))}/><Line dataKey="steps_per_second" name="Throughput" stroke="#287fa0" dot={false} connectNulls isAnimationActive={false}/></LineChart></ResponsiveContainer></div></ChartFrame>
      <div className="telemetry-note">Resource history is sampled at low frequency to keep local overhead modest. It reports CPU and memory use only; no fan-speed or temperature sensor is inferred.</div>
    </>}

    {view==='Logs & Configuration'&&<>
      <div className="telemetry-grid two">
        <section className="telemetry-chart"><div className="telemetry-chart-heading"><h3>Resolved run configuration</h3><span>{data.manifest?.baseline?.id||'Saved run specification'}</span></div><pre>{JSON.stringify({spec,baseline:data.manifest?.baseline,build:data.manifest?.build?.source?.hash||data.manifest?.build?.hash,trainer:data.manifest?.trainer,contract:data.manifest?.contract},null,2)}</pre></section>
        <section className="telemetry-chart"><div className="telemetry-chart-heading"><h3>Run timeline</h3><span>{lifecycle.length} recorded transitions</span></div><div className="timeline">{lifecycle.slice().reverse().map((event:any)=><div className="timeline-item" key={event.seq}><i/><div><b>{event.state||event.kind}</b><span>{event.from?`${event.from} → ${event.state}`:event.created?'Initial state recorded':''}</span><small>{tickTime(Number(event.created))}</small></div></div>)}{!lifecycle.length&&<div className="chart-empty">No lifecycle transitions have been recorded for this experiment yet.</div>}</div></section>
      </div>
      <section className="telemetry-chart checkpoint-history"><div className="telemetry-chart-heading"><h3>Frozen checkpoints</h3><span>{allCheckpoints.length} known for this run</span></div>{allCheckpoints.length?<div className="checkpoint-strip">{allCheckpoints.map(checkpoint=><a key={checkpoint.id} href="#checkpoints" onClick={event=>event.preventDefault()}><b>{number(checkpoint.step)} steps</b><span>{String(checkpoint.id).slice(0,14)}</span><small>{checkpoint.prepared?'Prepared for inference':'Inference bundle not prepared'}</small></a>)}</div>:<div className="chart-empty">No checkpoint manifest is associated with this run yet.</div>}</section>
      <details className="process-log"><summary>Process logs</summary><pre>{log||'No process output has been recorded.'}</pre></details>
      <details className="process-log"><summary>Recent episode records · {number(episodes.length)}</summary><pre>{JSON.stringify(episodes.slice(-20),null,2)}</pre></details>
    </>}
    <div className="experiment-footer"><span>Updated {data.server_time?new Date(Number(data.server_time)*1000).toLocaleTimeString():new Date(Number(job.updated)*1000).toLocaleTimeString()}</span><span>Data cursor {number(data.cursor)}</span></div>
  </div>;
}

function Metric({label,value,detail}:{label:string;value:string;detail:string}){
  return <div className="telemetry-metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

export default ExperimentPage;

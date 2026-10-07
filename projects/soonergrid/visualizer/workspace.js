'use strict';
(() => {
const $ = id => document.getElementById(id);
const D = window.WORKSPACE;
if (!D) { $('loadError').hidden=false; $('loadError').textContent='Dashboard data is unavailable. Run python3 build_workspace.py from the project folder, then reload.'; return; }
const fmt=(n,d=0)=>Number.isFinite(n)?n.toLocaleString('en-US',{maximumFractionDigits:d}):'—';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const runs=D.research?.runs||[];
let current=null,index=0,playing=false,lastTick=0;
let view={zoom:1,x:0,y:0},drag=null,moved=false;
const canvas=$('map'),ctx=canvas.getContext('2d');
const edges=D.network.edges;
let bounds={minX:Infinity,minY:Infinity,maxX:-Infinity,maxY:-Infinity};
for(const e of edges) for(const [x,y] of e.points){bounds.minX=Math.min(bounds.minX,x);bounds.maxX=Math.max(bounds.maxX,x);bounds.minY=Math.min(bounds.minY,y);bounds.maxY=Math.max(bounds.maxY,y);}
function surface(c){const r=c.getBoundingClientRect(),d=Math.min(window.devicePixelRatio||1,2);if(c.width!==Math.round(r.width*d)||c.height!==Math.round(r.height*d)){c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);}const g=c.getContext('2d');g.setTransform(d,0,0,d,0,0);return [g,r.width,r.height];}
function projection(w,h){const scale=Math.min((w-35)/(bounds.maxX-bounds.minX),(h-35)/(bounds.maxY-bounds.minY))*view.zoom;return ([x,y])=>[(x-(bounds.minX+bounds.maxX)/2)*scale+w/2+view.x,h/2-(y-(bounds.minY+bounds.maxY)/2)*scale+view.y];}
function frame(){return current?.playback_frames[index];}
function drawMap(){
  const [g,w,h]=surface(canvas),project=projection(w,h);
  g.fillStyle='#0f172a';
  g.fillRect(0,0,w,h);
  const states=frame()?.corridor_states||{},layer=$('layer').value;

  // 1. Draw Individual Road Links
  for(const e of edges){
    if(e.points.length<2)continue;
    const p=project(e.points[0]);
    g.beginPath();
    g.moveTo(...p);
    for(let j=1;j<e.points.length;j++)g.lineTo(...project(e.points[j]));
    const s=states[e.id];
    const major=['motorway','motorway_link','trunk','trunk_link','primary','primary_link'].includes(e.type);
    const secondary=['secondary','secondary_link'].includes(e.type);

    if(layer==='network'){
      g.strokeStyle=major?'#38bdf8':secondary?'#818cf8':'#334155';
      g.lineWidth=major?2.0:secondary?1.2:0.6;
    }else if(s){
      const speedRatio=s.speed_mph/Math.max(1,e.speed);
      const densityRatio=(s.density_pct||0)/100;
      const queueM=s.queue_m||0;

      if(layer==='density'){
        if(densityRatio>0.75||queueM>80){
          g.strokeStyle='#ef4444'; g.lineWidth=3.2; // Severe jam
        }else if(densityRatio>0.45||queueM>25){
          g.strokeStyle='#f97316'; g.lineWidth=2.5; // Heavy queue
        }else if(densityRatio>0.20){
          g.strokeStyle='#eab308'; g.lineWidth=2.0; // Moderate
        }else{
          g.strokeStyle='#22c55e'; g.lineWidth=major?1.8:1.2; // Clear
        }
      }else{
        // Speed layer
        if(speedRatio<0.25||queueM>80){
          g.strokeStyle='#ef4444'; g.lineWidth=3.2; // Severe gridlock (<25% speed)
        }else if(speedRatio<0.50||queueM>25){
          g.strokeStyle='#f97316'; g.lineWidth=2.5; // Slowing queue (<50% speed)
        }else if(speedRatio<0.75){
          g.strokeStyle='#eab308'; g.lineWidth=2.0; // Moderate delay (<75% speed)
        }else{
          g.strokeStyle='#22c55e'; g.lineWidth=major?1.8:1.2; // Free flow
        }
      }
    }else{
      // Free-flow / normal network segment
      g.strokeStyle=major?'#166534':secondary?'#134e4a':'#1e293b';
      g.lineWidth=major?1.4:secondary?0.9:0.45;
    }
    g.stroke();
  }

  // 2. Draw Memorial Stadium (Owen Field)
  const stadium=project([0,0]);
  g.beginPath();
  g.arc(...stadium,7,0,Math.PI*2);
  g.fillStyle='#f59e0b';
  g.fill();
  g.strokeStyle='#ffffff';
  g.lineWidth=2;
  g.stroke();
  g.fillStyle='#f8fafc';
  g.font='bold 10px system-ui';
  g.fillText('OWEN FIELD (STADIUM)',stadium[0]+12,stadium[1]+3);

  // 3. Draw Gateways
  for(const gw of D.network.gateways){
    const x=(gw.lon+97.4423)*111320*Math.cos(35.2059*Math.PI/180),y=(gw.lat-35.2059)*110950;
    const p=project([x,y]);
    g.beginPath();
    g.rect(p[0]-3.5,p[1]-3.5,7,7);
    g.fillStyle='#38bdf8';
    g.fill();
    g.strokeStyle='#0284c7';
    g.lineWidth=1.5;
    g.stroke();
    g.fillStyle='#94a3b8';
    g.font='8px system-ui';
    g.fillText(gw.id.replace('GW_',''),p[0]+6,p[1]+3);
  }

  // 4. Draw Key Norman Intersections (Signal & Chokepoint Nodes)
  if(D.network.intersections){
    for(const inter of D.network.intersections){
      const x=(inter.lon+97.4423)*111320*Math.cos(35.2059*Math.PI/180),y=(inter.lat-35.2059)*110950;
      const p=project([x,y]);
      
      let isCongested=false, isQueued=false;
      for(const [eid,s] of Object.entries(states)){
        if(eid.includes(String(inter.node_id)) || (inter.id==='INT_LINDSEY_BERRY' && eid.includes('berry'))){
          if(s.queue_m>60 || s.speed_mph<10){ isCongested=true; break; }
          else if(s.queue_m>20 || s.speed_mph<18){ isQueued=true; }
        }
      }

      g.beginPath();
      if(isCongested){
        g.arc(...p,9,0,Math.PI*2);
        g.fillStyle='rgba(239,68,68,0.35)';
        g.fill();
        g.strokeStyle='#ef4444';
        g.lineWidth=2;
        g.stroke();
        g.beginPath();
        g.arc(...p,4,0,Math.PI*2);
        g.fillStyle='#ef4444';
        g.fill();
      }else if(isQueued){
        g.arc(...p,7,0,Math.PI*2);
        g.fillStyle='rgba(245,158,11,0.3)';
        g.fill();
        g.strokeStyle='#f59e0b';
        g.lineWidth=1.8;
        g.stroke();
        g.beginPath();
        g.arc(...p,3.5,0,Math.PI*2);
        g.fillStyle='#f59e0b';
        g.fill();
      }else{
        g.arc(...p,3.5,0,Math.PI*2);
        g.fillStyle='#06b6d4';
        g.fill();
        g.strokeStyle='#083344';
        g.lineWidth=1.2;
        g.stroke();
      }

      const shortName=inter.name.replace('Lindsey St & ','').replace(' (Campus Corner)','').replace(' (Lloyd Noble)','');
      g.fillStyle='#cbd5e1';
      g.font='8px system-ui';
      g.fillText(shortName,p[0]+6,p[1]-4);
    }
  }
}

function drawChart(){const [g,w,h]=surface($('chart'));g.clearRect(0,0,w,h);const frames=current?.playback_frames||[];if(!frames.length)return;const paired=runs.find(r=>r.shock===current.shock&&r.policy!==current.policy&&r.config.demand_scale===current.config.demand_scale);const comparison=paired?.playback_frames||[];const max=Math.max(40,...frames.map(f=>f.lindsey_speed_mph),...comparison.map(f=>f.lindsey_speed_mph));g.strokeStyle='#edf0ea';g.lineWidth=1;for(let j=1;j<4;j++){g.beginPath();g.moveTo(20,j*h/4);g.lineTo(w-20,j*h/4);g.stroke();}const px=i=>20+i/Math.max(1,frames.length-1)*(w-40),py=v=>h-12-v/max*(h-22);g.beginPath();frames.forEach((f,i)=>i?g.lineTo(px(i),py(f.lindsey_speed_mph)):g.moveTo(px(i),py(f.lindsey_speed_mph)));g.strokeStyle='#39766a';g.lineWidth=1.8;g.stroke();g.lineTo(w-20,h-12);g.lineTo(20,h-12);g.closePath();g.fillStyle='#39766a10';g.fill();if(comparison.length){g.beginPath();comparison.forEach((f,i)=>i?g.lineTo(px(i),py(f.lindsey_speed_mph)):g.moveTo(px(i),py(f.lindsey_speed_mph)));g.strokeStyle='#bc9376';g.setLineDash([3,3]);g.lineWidth=1.2;g.stroke();g.setLineDash([]);}g.beginPath();g.moveTo(px(index),5);g.lineTo(px(index),h-10);g.strokeStyle='#b88662';g.lineWidth=1;g.stroke();}
function updateFrame(){$('mapTooltip').hidden=true;const f=frame();$('timeline').value=index;if(f){const minutes=Math.round(f.time_hr*60);$('clock').textContent=`${String(Math.floor(minutes/60)).padStart(2,'0')}:${String(minutes%60).padStart(2,'0')}`;$('phase').textContent=f.phase;$('speedValue').textContent=fmt(f.lindsey_speed_mph,1);}drawMap();drawChart();}
function choose(){$('mapTooltip').hidden=true;playing=false;$('play').textContent='▶';current=runs.find(r=>r.shock===$('scenario').value&&r.policy===$('policy').value);index=0;const frames=current?.playback_frames||[];$('timeline').max=Math.max(0,frames.length-1);$('timeline').disabled=!frames.length;$('play').disabled=!frames.length;const s=current?.summary||{},c=s.conservation||{};
$('metrics').innerHTML=[['SYSTEM TIME',s.total_system_time_including_boundary_veh_hrs,'veh·h','Includes boundary waiting'],['MODELED EXITS',c.exited_veh,'veh','Absorbed at model boundaries'],['UNFINISHED DEMAND',(c.remaining_veh??NaN)+(c.boundary_queue_veh??NaN),'veh','On-road + awaiting admission'],['CONSERVATION ERROR',c.max_absolute_residual_veh,'veh','Largest absolute step residual']].map(([label,n,unit,detail])=>`<article class="metric"><label>${label}</label><strong>${label==='CONSERVATION ERROR'&&n<.001?'&lt; 0.001':fmt(n)}</strong><span class="unit">${unit}</span><small>${detail}</small></article>`).join('');
$('accounting').innerHTML=[['Generated demand',c.generated_veh],['Exited network',c.exited_veh],['Remaining on roads',c.remaining_veh],['Waiting at boundary',c.boundary_queue_veh]].map(([k,v])=>`<div class="ledger-row"><span>${k}</span><b>${fmt(v,1)}</b></div>`).join('');
$('runInfo').textContent=current?`${frames.length} samples · Δt ${current.config.dt_s}s\nCAV 0% · compliance 60%`:'No completed experiment available';
if(!current){$('phase').textContent='Geometry only · no completed run';$('speedValue').textContent='—';$('clock').textContent='—';}updateFrame();}
const names={NOMINAL:'Nominal game day',SHOCK_COLLISION:'Lindsey collision',SHOCK_THUNDERSTORM:'Thunderstorm',SHOCK_OVERTIME:'Overtime'};
const views={overview:['Network observatory','Explore traffic dynamics. Inspect the evidence. Understand the limits.'],experiments:['Experiment ledger','Matched conditions, explicit accounting, and reproducible run records.'],methods:['Methods & evidence','The assumptions behind the model and the limits of the available evidence.'],archive:['Project archive','Original datasets and viewers, preserved with their evidence classification.']};
function show(name){if(!views[name])return;document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!==name);document.querySelectorAll('.nav').forEach(e=>e.classList.toggle('active',e.dataset.view===name));$('pageTitle').innerHTML=views[name][0]+'<span>.</span>';$('subtitle').textContent=views[name][1];history.replaceState(null,'','#'+name);if(name==='overview')requestAnimationFrame(updateFrame);}
document.querySelectorAll('.nav').forEach(b=>b.onclick=()=>show(b.dataset.view));document.querySelector('[data-open-methods]').onclick=()=>show('methods');window.addEventListener('hashchange',()=>show(location.hash.slice(1)));
$('scenario').onchange=choose;$('policy').onchange=choose;$('layer').onchange=()=>{$('legend').innerHTML=$('layer').value==='density'?'Congested <i></i> Available':$('layer').value==='network'?'Road hierarchy':'Slow <i></i> Free flow';drawMap();};$('timeline').oninput=()=>{index=Number($('timeline').value);updateFrame();};$('play').onclick=()=>{if(index===Number($('timeline').max))index=0;playing=!playing;$('play').textContent=playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',playing?'Pause simulation':'Play simulation');};
function tick(now){if(playing&&now-lastTick>700/Number($('rate').value)){lastTick=now;if(index<Number($('timeline').max)){index++;updateFrame();}else{playing=false;$('play').textContent='▶';}}setTimeout(()=>tick(performance.now()),150);}tick(performance.now());
$('zoomIn').onclick=()=>{view.zoom=Math.min(8,view.zoom*1.3);drawMap();};$('zoomOut').onclick=()=>{view.zoom=Math.max(.6,view.zoom/1.3);drawMap();};$('resetMap').onclick=()=>{$('mapTooltip').hidden=true;view={zoom:1,x:0,y:0};drawMap();};canvas.onpointerdown=e=>{moved=false;drag=[e.clientX-view.x,e.clientY-view.y];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(drag){moved=true;view.x=e.clientX-drag[0];view.y=e.clientY-drag[1];drawMap();}};canvas.onpointerup=e=>{drag=null;if(moved)return;const rect=canvas.getBoundingClientRect(),project=projection(rect.width,rect.height),x=e.clientX-rect.left,y=e.clientY-rect.top;if(D.network.intersections){for(const inter of D.network.intersections){const ix=(inter.lon+97.4423)*111320*Math.cos(35.2059*Math.PI/180),iy=(inter.lat-35.2059)*110950;const p=project([ix,iy]);if(Math.hypot(x-p[0],y-p[1])<16){const states=frame()?.corridor_states||{};let qMax=0,minSpd=99;for(const [eid,s] of Object.entries(states)){if(eid.includes(String(inter.node_id))||(inter.id==='INT_LINDSEY_BERRY'&&eid.includes('berry'))){if(s.queue_m>qMax)qMax=s.queue_m;if(s.speed_mph<minSpd)minSpd=s.speed_mph;}} $('mapTooltip').hidden=false;$('mapTooltip').innerHTML=`<strong>📍 ${esc(inter.name)}</strong><br>Signalized Arterial Node<br>Peak Queue: <b>${fmt(qMax,0)} m</b> · Approach Speed: <b>${minSpd<90?fmt(minSpd,1)+' mph':'Free flow'}</b><br><small>Norman Signalized Corridor · Node #${inter.node_id}</small>`;return;}}}let nearest=null,distance=12;for(const edge of edges){for(let i=1;i<edge.points.length;i++){const a=project(edge.points[i-1]),b=project(edge.points[i]),dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy||1))),d=Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy);if(d<distance){distance=d;nearest=edge;}}}if(nearest){const f=frame()?.corridor_states[nearest.id];$('mapTooltip').hidden=false;$('mapTooltip').innerHTML=`<strong>${esc(nearest.name||'Unnamed road')}</strong><br>`+(f?`${fmt(f.speed_mph,1)} mph · ${fmt(f.density_pct,1)}% occupancy · queue ${fmt(f.queue_m,0)} m`:`Free flow (${fmt(nearest.speed)} mph) · clear corridor`)+`<br><small>${esc(nearest.type)} · assumed free speed ${fmt(nearest.speed)} mph</small>`;}else $('mapTooltip').hidden=true;};canvas.onpointercancel=()=>drag=null;canvas.addEventListener('wheel',e=>{e.preventDefault();view.zoom=Math.max(.6,Math.min(8,view.zoom*(e.deltaY<0?1.1:.9)));drawMap();},{passive:false});
$('runTable').innerHTML=runs.length?runs.map(r=>{const s=r.summary,c=s.conservation;return `<tr><td>${esc(names[r.shock]||r.shock)}<small>${esc(r.policy)} · demand ×${r.config.demand_scale}</small></td><td>${r.config.dt_s} s</td><td>${fmt(s.total_system_time_including_boundary_veh_hrs,1)}</td><td>${fmt(c.exited_veh)}</td><td>${fmt(c.remaining_veh+c.boundary_queue_veh)}</td><td>${c.max_absolute_residual_veh.toExponential(2)}</td></tr>`;}).join(''):'<tr><td colspan="6">No completed runs. Generate a benchmark and rebuild the workspace.</td></tr>';
$('runManifest').innerHTML=D.research?`<p class="fine">Generated ${esc(D.research.created_utc)} · Python ${esc(D.research.environment.python)} · NetworkX ${esc(D.research.environment.networkx)}</p><p class="fine">Source fingerprint</p><code>${esc(D.research.source_sha256)}</code><p class="fine">${esc(D.research.design)}</p><a href="../data/research_benchmark.json" download>Download complete run artifact ↗</a>`:'<p>No experiment artifact available.</p>';
$('inventory').innerHTML=D.inventory.map(f=>`<tr><td><a href="../data/${encodeURIComponent(f.name)}" download>${esc(f.name)}</a></td><td>${esc(f.kind)}</td><td>${fmt(f.bytes/1048576,1)} MB</td><td><small>${esc(f.sha256)}</small></td></tr>`).join('');
const cells=D.illustration.sensitivity;let heat='<div class="heatmap"><div>CAV ↓<br>Compliance →</div>'+[20,40,60,80,100].map(v=>`<div>${v}%</div>`).join('');for(const p of [0,.25,.5,.75,1]){heat+=`<div>${p*100}%</div>`;for(const c of [.2,.4,.6,.8,1]){const v=cells.find(r=>r.cav_penetration===p&&r.driver_compliance===c);heat+=`<div class="cell" title="Assumed delay, vehicle-hours" style="background:hsl(143 18% ${94-(v?.delay_percentage||0)*.5}%)">${fmt(v?.delay_veh_hrs)}</div>`;}}$('sensitivity').innerHTML=heat+'</div><p class="fine">Formula-derived delay · vehicle-hours</p>';
$('archiveLinks').innerHTML=[['01','Road network','network_viewer.html','OSM geometry with heuristic road attributes.'],['02','Game-day demand','demand_viewer.html','Assumed temporal profiles and parking-choice parameters.'],['03','Baseline playback','simulation_viewer.html','Historical simulation. Pre-fix engine, unverified provenance.'],['04','Policy comparison','autonomous_viewer.html','Historical comparison. Do not interpret as validated savings.'],['05','Analytical command center','research_dashboard.html','Prescribed playback curves and assumed policy effects.'],['06','Resilience illustration','resilience_viewer.html','Prescribed scenarios and analytical sensitivity scaling.']].map(([n,t,url,desc])=>`<article class="archive-card"><span class="eyebrow">ARCHIVE / ${n}</span><strong>${t}</strong><p>${desc}</p><a href="${url}">Open original viewer ↗</a></article>`).join('');
$('downloadCsv').onclick=()=>{const rows=[['scenario','policy','dt_s','demand_scale','system_time_veh_h','generated_veh','exited_veh','remaining_veh','boundary_queue_veh','max_residual_veh'],...runs.map(r=>{const s=r.summary,c=s.conservation;return [r.shock,r.policy,r.config.dt_s,r.config.demand_scale,s.total_system_time_including_boundary_veh_hrs,c.generated_veh,c.exited_veh,c.remaining_veh,c.boundary_queue_veh,c.max_absolute_residual_veh];})];const url=URL.createObjectURL(new Blob([rows.map(row=>row.join(',')).join('\n')],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='soonergrid-experiments.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('buildStamp').textContent='ARTIFACT BUILT '+new Date(D.built_utc).toISOString().slice(0,10);new ResizeObserver(()=>{if(!$('overview').hidden){drawMap();drawChart();}}).observe($('map').parentElement);choose();show(views[location.hash.slice(1)]?location.hash.slice(1):'overview');
})();

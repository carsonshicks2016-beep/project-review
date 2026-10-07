import * as THREE from './vendor/three.module.js';
import {createWorld} from './scene.mjs';
import {createFrameLoop,prepareImportedSession,encodeDriverLink,decodeDriverLink,overviewFraming,FRAME_BUDGET_MS,MAX_CATCHUP_SECONDS,MAX_FRAME_DELTA_SECONDS} from './runtime.mjs';
import * as engine from './engine.mjs';
import {makeTrack,Trainer,newCar,tickCar,rng,clamp,DT,POPULATION,readDriver,Comparison,TRACE_STRIDE} from './engine.mjs';
const $=id=>document.getElementById(id);
let track=makeTrack(2077,18,3),trainer=new Trainer(track,99,null,{recordLines:true}),running=false,mode='train',replayCar=null,cameraMode='orbit',speedMultiplier=20,simulationDebt=0;
let lineGroup=null,lineStamp='',comparison=null,comparisonRunning=false,compareStamp='',lastRender=0;
let trackFog=[550,1100],sectorFocusReady=false,weightStamp='';
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
let angle=.68,pitch=.83,zoom=440,drag=null,lastTime=0,lastUI=0,world,carModels=[],lastGeneration=1;
const viewport=$('viewport'),scene=new THREE.Scene();scene.background=new THREE.Color('#293627');scene.fog=new THREE.Fog('#293627',550,1100);
let renderer;
try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;viewport.prepend(renderer.domElement);$('loading').remove();}catch(e){$('loading').textContent='This simulation needs WebGL. Please enable hardware acceleration and reload.';throw e;}
const camera=new THREE.PerspectiveCamera(43,1,.2,1400);const sun=new THREE.DirectionalLight('#fff5d4',3.2);sun.position.set(-100,220,60);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-220,right:220,top:220,bottom:-220,near:1,far:500});sun.shadow.normalBias=1;scene.add(sun);scene.add(new THREE.HemisphereLight('#dce7c7','#393929',2));
function buildWorld(){
 if(world){scene.remove(world);world.userData.dispose();}
 world=createWorld(track);scene.add(world);sectorFocusReady=false;configureWorldCamera();updateTrackInfo();drawProfile();refreshLines();
}
function formatTime(seconds){
 if(seconds===null||seconds===undefined)return '—';
 if(seconds<60)return seconds.toFixed(2)+' s';
 return Math.floor(seconds/60)+':'+(seconds%60).toFixed(2).padStart(5,'0');
}
function configureWorldCamera(){
 const extent=Math.max(track.terrainWidth,track.terrainDepth);
 camera.far=Math.max(1400,extent*7);camera.updateProjectionMatrix();
 trackFog=[Math.max(550,extent*1.8),Math.max(1100,extent*5)];
 scene.fog.near=trackFog[0];scene.fog.far=trackFog[1];
 configureShadows(null);
}
function configureShadows(car,extentOverride){
 const extent=extentOverride??(car?220:Math.max(track.terrainWidth,track.terrainDepth));
 const x=car?.x??0,z=car?.z??0,y=car?.y??0;
 sun.position.set(x-extent*.4,y+extent*.85,z+extent*.3);sun.target.position.set(x,y,z);
 if(!sun.target.parent)scene.add(sun.target);
 Object.assign(sun.shadow.camera,{left:-extent*.6,right:extent*.6,top:extent*.6,bottom:-extent*.6,near:1,far:extent*3});sun.shadow.camera.updateProjectionMatrix();
}
function updateTrackInfo(){
 $('length').textContent=Math.round(track.length);$('ascent').textContent=Math.round(track.ascent);
 $('distance-label').textContent=(track.length/1000).toFixed(2)+' km';
 $('track-label').textContent='RIDGELINE '+track.seed+' · '+(track.length/1000).toFixed(1)+' KM';
 $('circuit-name').textContent='Ridgeline · '+(track.length/1000).toFixed(1)+' km';
 $('time-limit').textContent='Attempt limit '+formatTime(track.episodeLimit)+' · 4 timed sectors';
 const heights=track.points.map(p=>p.y);$('elevation-span').textContent=Math.round(Math.max(...heights)-Math.min(...heights))+' m range';
}
const followTarget=new THREE.Vector3(),sectorFocus=new THREE.Vector3(),sectorWanted=new THREE.Vector3();
// Overview will not pull back past the point where the road stops reading. A 12 m road needs at
// least MIN_ROAD_FRAME_FRACTION of the frame to look like a road rather than a scratch, which caps
// the visible ground at width/fraction metres. Courses that fit inside that footprint keep the
// whole-lap diorama exactly as before; longer ones orbit one sector and leave the lap to the minimap.
function framingFor(leader){
 const framing=overviewFraming({roadWidth:track.width,terrainWidth:track.terrainWidth,terrainDepth:track.terrainDepth,aspect:camera.aspect,fov:camera.fov,zoom});
 return framing.sector?{...framing,focus:leader??track.points[0]}:framing;
}
function renderScene(){let leader=mode==='replay'?replayCar:trainer.leader;const cars=mode==='replay'?[replayCar]:trainer.population;
 const framing=framingFor(leader),followActive=cameraMode==='follow'&&!!leader,usingSector=!followActive&&framing.sector;
 world.userData.updateCars(cars,leader);world.userData.updateEffects(cars,running);
 configureShadows(followActive?leader:(framing.sector?framing.focus:null),usingSector?framing.groundWidth*.75:undefined);
 // Fog has to clear the whole framed sector, not start at the car, or the visible ground hazes out.
 if(usingSector){scene.fog.near=framing.distance+framing.groundWidth;scene.fog.far=framing.distance+framing.groundWidth*5;}
 else{scene.fog.near=trackFog[0];scene.fog.far=trackFog[1];}
 if(followActive){followTarget.set(leader.x,leader.y+1,leader.z);let target=new THREE.Vector3(leader.x-Math.cos(leader.heading)*32,leader.y+23,leader.z-Math.sin(leader.heading)*32);camera.position.lerp(target,reducedMotion?1:.07);camera.lookAt(followTarget);}
 else if(usingSector){const d=framing.distance;
  sectorWanted.set(framing.focus.x,framing.focus.y,framing.focus.z);
  if(sectorFocusReady)sectorFocus.lerp(sectorWanted,reducedMotion?1:.08);else{sectorFocus.copy(sectorWanted);sectorFocusReady=true;}
  camera.position.set(sectorFocus.x+Math.sin(angle)*Math.cos(pitch)*d,sectorFocus.y+Math.sin(pitch)*d,sectorFocus.z+Math.cos(angle)*Math.cos(pitch)*d);
  camera.lookAt(sectorFocus);}
 else{const distance=framing.distance;camera.position.set(Math.sin(angle)*Math.cos(pitch)*distance,Math.sin(pitch)*distance,Math.cos(angle)*Math.cos(pitch)*distance);camera.lookAt(0,-2,0);}
 renderer.render(scene,camera);}
function fitCanvas(canvas){const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2);canvas.width=Math.max(1,box.width*ratio);canvas.height=Math.max(1,box.height*ratio);let ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);return [ctx,box.width,box.height];}
function drawProfile(){const [ctx,w,h]=fitCanvas($('profile'));const heights=track.points.map(p=>p.y),min=Math.min(...heights)-3,max=Math.max(...heights)+3;ctx.strokeStyle='#313d2b';ctx.lineWidth=1;for(let y of [.25,.75]){ctx.beginPath();ctx.moveTo(0,h*y);ctx.lineTo(w,h*y);ctx.stroke();}ctx.beginPath();track.points.forEach((p,i)=>{const x=i/(track.n-1)*w,y=h-8-(p.y-min)/(max-min)*(h-17);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.strokeStyle='#bddb7c';ctx.lineWidth=1.5;ctx.stroke();ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();const grad=ctx.createLinearGradient(0,0,0,h);grad.addColorStop(0,'#82994a55');grad.addColorStop(1,'#82994a03');ctx.fillStyle=grad;ctx.fill();}
function drawChart(){
 const [ctx,w,h]=fitCanvas($('chart'));ctx.strokeStyle='#303a2a';ctx.setLineDash([3,5]);for(let y of [3,h/2,h-8]){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}ctx.setLineDash([]);
 const history=trainer.history.slice(-70),laps=history.filter(p=>p.lap),isLap=laps.length>0,values=isLap?laps.map(p=>p.lap):[0,100];
 const lo=isLap?Math.floor(Math.min(...values)-2):0,hi=isLap?Math.ceil(Math.max(...values)+2):100;
 const labels=document.querySelectorAll('.chart-y span');[hi,(hi+lo)/2,lo].forEach((v,i)=>labels[i].textContent=isLap?v.toFixed(0)+'s':v+'%');
 $('curve-caption').textContent=isLap?'Fastest lap per generation · lower is better':'Best completion per generation';
 $('chart-empty').style.display=history.length?'none':'flex';if(!history.length){$('chart').setAttribute('aria-label','Best track completion per generation');$('chart-range').textContent='GEN 01 →';return;}
 const points=history.map((p,i)=>({x:i/Math.max(1,history.length-1)*(w-5)+2,value:isLap?p.lap:p.completion})).filter(p=>p.value!==null);
 ctx.strokeStyle='#d6fa59';ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{const y=h-8-(p.value-lo)/(hi-lo)*(h-12);i?ctx.lineTo(p.x,y):ctx.moveTo(p.x,y);});ctx.stroke();
 for(const p of points){ctx.beginPath();ctx.arc(p.x,h-8-(p.value-lo)/(hi-lo)*(h-12),history.length<20?2.5:1.2,0,Math.PI*2);ctx.fillStyle='#d6fa59';ctx.fill();}
 $('chart').setAttribute('aria-label',isLap?'Fastest lap time per generation in seconds':'Best track completion per generation');
 $('chart-range').textContent='GEN '+String(history[0].generation).padStart(2,'0')+' → '+String(history.at(-1).generation).padStart(2,'0');
}
// Diversity gets its own plot sharing only the generation axis: it is a genome distance, not a
// percentage or a lap time, so it must not borrow the learning curve's y-scale.
function drawDiversity(){
 const [ctx,w,h]=fitCanvas($('diversity'));
 const values=trainer.history.slice(-70).map(p=>p.diversity).filter(Number.isFinite);
 $('diversity-now').textContent=values.length?values.at(-1).toFixed(2):'—';
 if(values.length<2){$('diversity').setAttribute('aria-label','Population diversity appears once training begins.');return;}
 const peak=Math.max(...values)||1;
 const x=i=>i/(values.length-1)*(w-3)+1.5,y=v=>h-3-v/peak*(h-9);
 ctx.strokeStyle='#2b331f';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,h-1.5);ctx.lineTo(w,h-1.5);ctx.stroke();
 ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));
 ctx.strokeStyle='#7e8c6a';ctx.lineWidth=1.5;ctx.stroke();
 ctx.lineTo(x(values.length-1),h);ctx.lineTo(x(0),h);ctx.closePath();
 ctx.fillStyle='rgba(126,140,106,.16)';ctx.fill();
 $('diversity').setAttribute('aria-label','Population diversity went from '+values[0].toFixed(2)+' to '+values.at(-1).toFixed(2)+' across '+values.length+' generations.');
}
// The genome as a picture. The same numbers sit in a visually hidden table so the panel is not
// colour-only, and the cells carry titles for pointer users.
function drawWeights(){
 const weights=trainer.champion?.weights,body=$('weight-table').querySelector('tbody');
 if(!weights){$('weight-grid').innerHTML='';body.innerHTML='';return;}
 const peak=Math.max(...weights.map(Math.abs))||1;
 $('weight-grid').innerHTML=weights.map((v,i)=>{
  const strength=(.12+Math.min(1,Math.abs(v)/peak)*.88).toFixed(3);
  const tint=v>=0?'214,250,89':'242,145,110';
  return '<i title="'+(i<12?'Steering':'Throttle')+' · '+engine.INPUT_NAMES[i%12]+' · '+v.toFixed(3)+'">'+
   '<span style="background:rgba('+tint+','+strength+')"></span></i>';
 }).join('');
 body.innerHTML='<tr><td></td>'+engine.INPUT_NAMES.map(n=>'<th scope="col">'+n+'</th>').join('')+'</tr>'+
  [0,1].map(row=>'<tr><th scope="row">'+(row?'Throttle':'Steering')+'</th>'+
   weights.slice(row*12,row*12+12).map(v=>'<td>'+v.toFixed(3)+'</td>').join('')+'</tr>').join('');
}
function renderAblation(result){
 const peak=Math.max(.01,...result.rows.filter(r=>r.finished).map(r=>Math.max(0,r.penalty??0)));
 const rows=[...result.rows].sort((a,b)=>a.finished!==b.finished?(a.finished?1:-1):(b.penalty??0)-(a.penalty??0));
 $('ablation-rows').innerHTML=rows.map(r=>{
  const width=r.finished?Math.max(1,Math.max(0,r.penalty??0)/peak*100):100;
  const cost=r.finished?(r.penalty>=0?'+':'')+r.penalty.toFixed(2)+' s':'DNF '+Math.round(r.progress*100)+'%';
  return '<div class="ablation-row'+(r.finished?'':' broken')+'"><span>'+r.input+'</span>'+
   '<span class="bar"><b style="width:'+width.toFixed(1)+'%"></b></span><span class="cost">'+cost+'</span></div>';
 }).join('');
 $('ablation-empty').style.display='none';
}
$('ablate').onclick=()=>{
 if(!trainer.champion?.lap){toast('Complete a lap first — ablation replays the champion.');return;}
 $('ablate').disabled=true;$('ablate').textContent='Running…';
 // Thirteen deterministic replays. Yielding once lets the button repaint before they run.
 setTimeout(()=>{
  try{renderAblation(engine.ablate(track,trainer.champion.weights));toast('Each bar is the lap time lost without that sense.');}
  catch(error){toast('Ablation failed: '+error.message);}
  finally{$('ablate').textContent='Run ablation';$('ablate').disabled=!trainer.champion?.lap;}
 },16);
};
function syncUI(){const c=mode==='replay'?replayCar:trainer.leader,progress=clamp(c.bestProgress/track.length*100,0,100);$('speed').textContent=Math.round(c.speed*3.6);$('generation').textContent=String(trainer.generation).padStart(3,'0');$('generation-inline').textContent=String(trainer.generation).padStart(3,'0');$('best-lap').textContent=formatTime(trainer.bestLap);$('completion').textContent=progress.toFixed(0)+'%';$('progress-bar').style.width=progress+'%';$('lap-progress').setAttribute('aria-valuenow',Math.round(progress));$('throttle-bar').style.width=Math.abs(c.throttle)*100+'%';$('throttle-bar').style.background=c.throttle<0?'#f2916e':'#d6fa59';$('steer-bar').style.left=(50+c.steer*90)+'%';$('alive').textContent=mode==='replay'?1:trainer.population.filter(c=>!c.dead&&!c.finished).length;$('episode').textContent=formatTime(c.time);$('replay').disabled=!trainer.bestLap;$('export').disabled=!trainer.champion;$('share').disabled=!trainer.champion;$('laps-count').textContent=trainer.totalLaps+' completed laps';if(trainer.generation!==lastGeneration){lastGeneration=trainer.generation;drawChart();drawDiversity();setState();}if(lineStamp!==track.seed+':'+(trainer.champion?.generation??'none'))refreshLines();
 const stamp=track.seed+':'+track.surface+':'+(trainer.champion?.generation??'none');
 if(weightStamp!==stamp){weightStamp=stamp;drawWeights();}
 $('ablate').disabled=!trainer.champion?.lap;drawMinimap(c);updateSectors(c); }
function setState(){
 const replay=mode==='replay';
 $('start-text').textContent=running?(replay?'Pause replay':'Pause training'):(replay?'Resume replay':trainer.elapsed?'Resume training':'Start training');
 $('start-icon').textContent=running?'Ⅱ':'▶';
 $('status').textContent=comparisonRunning?'Comparing algorithms':running?(replay?'Replaying best lap':trainer.exploring?'Exploring after a plateau':'Training in progress'):(replay?'Replay paused':trainer.elapsed?'Session paused':'Ready to train');
 const framed=cameraMode==='orbit'&&framingFor(null).sector?' · SECTOR':'';
 $('view-label').textContent=(replay?'BEST LAP REPLAY':'LIVE SIMULATION')+framed;
 $('orbit').setAttribute('aria-pressed',cameraMode==='orbit');$('follow').setAttribute('aria-pressed',cameraMode==='follow');
}
function toast(message){$('toast').textContent=message;$('toast').classList.add('visible');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('visible'),3500);}
$('start').onclick=()=>{if(!running&&cameraMode==='orbit'&&track.length>2000){cameraMode='follow';$('follow').classList.add('selected');$('orbit').classList.remove('selected');}comparisonRunning=false;updateComparisonButton();clearFault();if(mode==='replay'&&(replayCar.finished||replayCar.dead))replayCar=newCar(track,trainer.champion.weights);running=!running;setState();};$('speed-select').onchange=e=>{speedMultiplier=+e.target.value;simulationDebt=0;};
$('elevation').oninput=e=>$('elevation-value').textContent=e.target.value+' m';
function clearDriverHash(){if(location.hash.startsWith('#driver='))history.replaceState(null,'',location.pathname+location.search);}
$('generate').onclick=()=>{let seed=+$('seed').value;if(!Number.isInteger(seed)||seed<1||seed>999999){toast('Choose a whole-number seed from 1 to 999999.');return;}if(seed===track.seed&&+$('elevation').value===track.elevation&&+$('course-length').value===track.lengthKm&&$('surface').value===track.surface){seed=Math.floor(Math.random()*999998)+1;$('seed').value=seed;}clearDriverHash();const weights=trainer.champion?.weights;track=makeTrack(seed,+$('elevation').value,$('course-length').value==='legacy'?null:+$('course-length').value,$('surface').value);trainer=new Trainer(track,seed+99,weights,{recordLines:true});running=false;mode='train';$('replay').textContent='↻ Replay best';lastGeneration=0;simulationDebt=0;clearComparison();clearFault();buildWorld();drawChart();drawDiversity();syncUI();setState();toast(weights?'New terrain. Your best driver is ready to adapt.':'New terrain generated. Ready to train.');};
$('replay').onclick=()=>{world?.userData.clearEffects?.();comparisonRunning=false;updateComparisonButton();clearFault();if(mode==='replay'){mode='train';$('replay').textContent='↻ Replay best';running=false;}else if(trainer.champion?.lap){replayCar=newCar(track,trainer.champion.weights);mode='replay';running=true;$('replay').textContent='← Back to training';toast('Replaying the fastest driver at real time.');}simulationDebt=0;setState();};
$('export').onclick=()=>{if(!trainer.champion)return;const payload={format:'rally-neural-policy-v1',createdAt:new Date().toISOString(),algorithm:'elitist-neuroevolution',inputs:['heading_8m','heading_18m','heading_35m','heading_60m','lateral_offset','speed','slip','curvature','grade','load_delta','previous_steer','bias'],outputs:['steering','throttle_brake'],activation:'tanh',weights:trainer.champion.weights,track:{seed:track.seed,elevation:track.elevation,lengthKm:track.lengthKm,surface:track.surface},bestLapSeconds:trainer.bestLap,generation:trainer.champion.generation};let url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='rally-driver-'+track.seed+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Driver weights and track settings exported.');};
function applyDriver(data){
 const next=prepareImportedSession(data,track,engine);
 // Construct and evaluate before replacing the session; invalid data leaves it intact.
 track=next.track;trainer=next.trainer;running=false;mode='train';replayCar=null;
 simulationDebt=0;lastGeneration=0;clearComparison();clearFault();
 $('seed').value=track.seed;$('elevation').value=track.elevation;
 const lengthValue=track.lengthKm===null?'legacy':String(track.lengthKm);
 if(![...$('course-length').options].some(o=>o.value===lengthValue))$('course-length').add(new Option(track.lengthKm===null?'Legacy 0.82 km':track.lengthKm+' km',lengthValue));
 $('course-length').value=lengthValue;
 $('elevation-value').textContent=track.elevation+' m';$('replay').textContent='↻ Replay best';
 buildWorld();drawChart();drawDiversity();syncUI();setState();
 toast(next.probe.finished?'Driver loaded. Verified lap: '+formatTime(next.probe.time)+'.':'Driver loaded. Train to improve its track completion.');
}
$('import').onclick=()=>$('import-file').click();
$('import-file').onchange=async e=>{
 const file=e.target.files[0];e.target.value='';if(!file)return;
 if(file.size>1e6){toast('That file is too large to be a driver export.');return;}
 try{applyDriver(JSON.parse(await file.text()));}catch(error){toast(error.message||'That file is not valid driver JSON.');}
};
$('share').onclick=async()=>{
 if(!trainer.champion)return;
 const base=new URL(location.href);base.hash=encodeDriverLink({weights:trainer.champion.weights,track});
 $('driver-link').value=base.href;
 try{await navigator.clipboard.writeText(base.href);toast('Driver link copied. The recipient needs access to this app.');}
 catch{$('link-dialog').showModal();$('driver-link').select();}
};
$('close-link').onclick=()=>$('link-dialog').close();
$('copy-link').onclick=async()=>{
 try{await navigator.clipboard.writeText($('driver-link').value);$('link-dialog').close();toast('Driver link copied.');}
 catch{$('driver-link').focus();$('driver-link').select();toast('Use your browser’s Copy command to copy the selected link.');}
};
function loadHash(){try{const data=decodeDriverLink(location.hash);if(data)applyDriver(data);}catch(error){toast(error.message);}}
addEventListener('hashchange',loadHash);
$('reset').onclick=()=>{clearDriverHash();clearFault();world?.userData.clearEffects?.();running=false;mode='train';trainer=new Trainer(track,99,null,{recordLines:true});simulationDebt=0;lastGeneration=0;drawChart();drawDiversity();syncUI();setState();$('replay').textContent='↻ Replay best';toast('Learning reset. A fresh population is ready.');};
for(let id of ['orbit','follow'])$(id).onclick=()=>{cameraMode=id;$(id).classList.add('selected');$(id==='orbit'?'follow':'orbit').classList.remove('selected');setState();};
$('about').onclick=()=>$('about-dialog').showModal();$('close-about').onclick=$('got-it').onclick=()=>$('about-dialog').close();$('about-dialog').addEventListener('click',e=>{if(e.target===$('about-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
viewport.addEventListener('pointerdown',e=>{if(e.target!==renderer.domElement)return;drag=[e.clientX,e.clientY];viewport.setPointerCapture(e.pointerId);});viewport.addEventListener('pointermove',e=>{if(!drag)return;angle-=(e.clientX-drag[0])*.006;pitch=clamp(pitch+(e.clientY-drag[1])*.005,.22,1.35);drag=[e.clientX,e.clientY];cameraMode='orbit';$('orbit').classList.add('selected');$('follow').classList.remove('selected');setState();});viewport.addEventListener('pointerup',()=>drag=null);viewport.addEventListener('pointercancel',()=>drag=null);viewport.addEventListener('wheel',e=>{e.preventDefault();zoom=clamp(zoom+e.deltaY*.25,180,740);},{passive:false});

function drawMinimap(car){
 const [ctx,w,h]=fitCanvas($('minimap')),padding=12,scale=Math.min((w-padding*2)/track.terrainWidth,(h-padding*2)/track.terrainDepth);
 const xy=p=>[w/2+p.x*scale,h/2+p.z*scale];
 ctx.strokeStyle='#a4b78b';ctx.lineWidth=2;ctx.beginPath();
 track.points.forEach((p,i)=>{const [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.stroke();
 ctx.font='10px monospace';ctx.fillStyle='#e7ecd8';
 for(let sector=0;sector<4;sector++){const p=engine.sample(track,track.length*sector/4),[x,y]=xy(p);ctx.fillText(sector===0?'S/F':String(sector+1),x+3,y-3);}
 const [x,y]=xy(car);ctx.fillStyle='#d6fa59';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();
 $('minimap').setAttribute('aria-label','Leader at '+Math.round(clamp(car.progress/track.length*100,0,100))+' percent of the '+(track.length/1000).toFixed(1)+' kilometre circuit.');
}
function updateSectors(car){
 const stamp=car.sectorTimes.join(':')+'|'+trainer.bestSectors.join(':');
 if(updateSectors.stamp!==stamp){
  updateSectors.stamp=stamp;
  $('sectors').innerHTML=car.sectorTimes.map((time,index)=>{
   const best=trainer.bestSectors[index],delta=time!==null&&best!==null?time-best:null;
   const state=time===null?'':best===null||Math.abs(delta)<.02?'session-best':delta<0?'faster':'slower';
   const detail=delta===null?'Best '+formatTime(best):Math.abs(delta)<.02?'Matches session best':(delta>0?'+':'')+delta.toFixed(2)+' s vs best';
   return '<div class="'+state+'">SECTOR '+(index+1)+'<strong>'+formatTime(time)+'</strong><small>'+detail+'</small></div>';
  }).join('');
 }
 const bestSectors=trainer.bestSectors;
 $('lap-summary').textContent=bestSectors.every(t=>t!==null)?'Ideal sector total '+formatTime(bestSectors.reduce((a,b)=>a+b,0))+' · actual best lap '+formatTime(trainer.bestLap)+' · ideal combines different attempts.':'Each sector is '+(track.length/4000).toFixed(2)+' km. Complete all four to set a lap.';
 $('grade-value').textContent='Grade '+(track.points[car.index].grade*100).toFixed(0)+'%';
 $('load-value').textContent='Load '+car.load.toFixed(1)+'×';
}
function refreshLines(){
 if(lineGroup){scene.remove(lineGroup);lineGroup.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
 lineGroup=new THREE.Group();lineGroup.visible=$('show-lines').checked;scene.add(lineGroup);
 const lines=trainer.lines;
 lines.forEach((line,index)=>{
  const trace=line.trace,positions=[],colors=[],indices=[];
  for(let offset=0;offset<trace.length;offset+=TRACE_STRIDE){
   const next=offset+TRACE_STRIDE<trace.length?offset+TRACE_STRIDE:Math.max(0,offset-TRACE_STRIDE);
   let dx=trace[next]-trace[offset],dz=trace[next+2]-trace[offset+2];
   if(next<offset){dx=-dx;dz=-dz;}const length=Math.hypot(dx,dz)||1;
   const color=new THREE.Color('#67c5ee').lerp(new THREE.Color('#d6fa59'),clamp(trace[offset+4]/(120/3.6),0,1));
   const lineWidth=Math.max(.4,track.length/8000);for(const side of [-1,1]){positions.push(trace[offset]-dz/length*lineWidth*side,trace[offset+1]+.7,trace[offset+2]+dx/length*lineWidth*side);colors.push(color.r,color.g,color.b);}
   const vertex=offset/TRACE_STRIDE*2;
   if(offset+TRACE_STRIDE<trace.length)indices.push(vertex,vertex+1,vertex+2,vertex+1,vertex+3,vertex+2);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));geometry.setIndex(indices);
  const material=new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:index===lines.length-1?.95:.14+.08*index,depthWrite:false});
  const mesh=new THREE.Mesh(geometry,material);mesh.renderOrder=index+1;lineGroup.add(mesh);
 });
 const best=lines.at(-1);
 $('line-summary').textContent=best?'Gen '+best.generation+' · '+formatTime(best.lap)+' · '+lines.length+' champion line'+(lines.length===1?'':'s'):'A completed champion lap will draw the first line.';
 lineStamp=track.seed+':'+(trainer.champion?.generation??'none');
}
$('show-lines').onchange=()=>{if(lineGroup)lineGroup.visible=$('show-lines').checked;};
const algorithmNames=['Genetic search','Hill climber','Random search'];
const algorithmColors=['#d6fa59','#67c5ee','#cf9bf5'];
function updateComparisonButton(){
 $('compare').textContent=comparisonRunning?'Pause comparison':comparison&&!comparison.done?'Resume comparison':'Run comparison';
 $('comparison-budget').disabled=comparisonRunning;
}
function clearComparison(){comparison=null;comparisonRunning=false;compareStamp='';updateComparisonButton();drawComparison();}
$('comparison-budget').onchange=clearComparison;
$('compare').onclick=()=>{
 if(comparisonRunning){comparisonRunning=false;}
 else{
  if(!comparison||comparison.done)comparison=new Comparison(track,{generations:+$('comparison-budget').value,seed:99});
  running=false;comparisonRunning=true;clearFault();
 }
 updateComparisonButton();setState();drawComparison();
};
function drawComparison(){
 const [ctx,w,h]=fitCanvas($('comparison-chart')),left=48,right=w-12,top=10,bottom=h-28;
 ctx.font='12px monospace';ctx.fillStyle='#b8c4af';ctx.strokeStyle='#34412e';
 const trainers=comparison?.trainers??[],laps=trainers.flatMap(t=>t.history.map(p=>p.bestLap).filter(x=>x!==null));
 const lo=laps.length?Math.floor(Math.min(...laps)-2):0,hi=laps.length?Math.ceil(Math.max(...laps)+2):60;
 for(let i=0;i<3;i++){
  const y=top+(bottom-top)*i/2;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/2).toFixed(0)+' s',0,y+4);
 }
 const budget=(comparison?.generations??+$('comparison-budget').value)*POPULATION;
 ctx.fillText('0',left,bottom+20);ctx.textAlign='right';ctx.fillText(budget+' evaluated drivers',right,bottom+20);ctx.textAlign='left';
 trainers.forEach((trainer,index)=>{
  ctx.strokeStyle=algorithmColors[index];ctx.lineWidth=2;ctx.beginPath();let started=false;
  for(const result of trainer.history){if(result.bestLap===null)continue;let x=left+result.evaluations/budget*(right-left),y=bottom-(result.bestLap-lo)/(hi-lo)*(bottom-top);started?ctx.lineTo(x,y):ctx.moveTo(x,y);started=true;}
  ctx.stroke();
 });
 $('comparison-empty').style.display=laps.length?'none':'grid';
 $('comparison-empty').textContent=comparison?'Waiting for the first completed lap…':'Compare genetic search, hill climbing and random search on this circuit.';
 if(comparison?.done&&!laps.length)$('comparison-empty').textContent='No completed laps in this budget. Completion results are below.';
 $('comparison-results').innerHTML=algorithmNames.map((name,index)=>{
  const t=trainers[index],completion=t?Math.min(100,(t.champion?.progress??0)/track.length*100):0;
  return '<tr><th scope="row">'+name+'</th><td>'+(t?.evaluations??0)+'</td><td>'+(t?.bestLap?formatTime(t.bestLap):'No lap yet')+'</td><td>'+completion.toFixed(1)+'%</td></tr>';
 }).join('');
 $('comparison-status').textContent=comparison?'Track '+track.seed+' · random seed 99 · '+(comparison.done?'complete':comparisonRunning?'running':'paused')+'. Equal evaluation budget; compute time per attempt differs.':'No comparison run yet. Runs independently of your live session.';
}
function reportFailure(error){
 running=false;comparisonRunning=false;simulationDebt=0;
 $('runtime-message').textContent='The view encountered a problem. Training is paused and your driver is still available to export. Retry the view or reset learning.';
 $('runtime-error').hidden=false;setState();updateComparisonButton();console.error('Rally runtime error:',error);
}
function clearFault(){frameLoop.recover();$('runtime-error').hidden=true;lastTime=0;simulationDebt=0;}
$('retry-render').onclick=()=>{clearFault();try{buildWorld();resize();renderScene();}catch(error){reportFailure(error);}};
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();reportFailure(new Error('WebGL context lost'));});
renderer.domElement.addEventListener('webglcontextrestored',()=>{clearFault();resize();toast('The view has recovered. Resume training when ready.');});
viewport.addEventListener('keydown',e=>{
 let handled=true;
 if(e.key==='ArrowLeft')angle+=.08;else if(e.key==='ArrowRight')angle-=.08;
 else if(e.key==='ArrowUp')pitch=clamp(pitch+.06,.22,1.35);else if(e.key==='ArrowDown')pitch=clamp(pitch-.06,.22,1.35);
 else if(e.key==='+'||e.key==='=')zoom=clamp(zoom-20,180,740);else if(e.key==='-')zoom=clamp(zoom+20,180,740);else handled=false;
 if(handled){e.preventDefault();cameraMode='orbit';$('orbit').classList.add('selected');$('follow').classList.remove('selected');setState();}
});
document.addEventListener('visibilitychange',()=>{lastTime=0;simulationDebt=0;});
function resize(){
 const box=viewport.getBoundingClientRect();if(!box.width||!box.height)return;
 renderer.setSize(box.width,box.height);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();drawChart();drawDiversity();drawProfile();drawComparison();setState();
}
new ResizeObserver(()=>{try{resize();}catch(error){reportFailure(error);}}).observe(viewport);
function stepFrame(time){
 const delta=lastTime?Math.min((time-lastTime)/1000,MAX_FRAME_DELTA_SECONDS):0;lastTime=time;
 const budget=performance.now()+FRAME_BUDGET_MS;
 if(comparisonRunning){
  while(!comparison.done&&performance.now()<budget)comparison.step();
  if(comparison.done){comparisonRunning=false;updateComparisonButton();setState();toast('Comparison complete. Results use the same evaluation budget.');}
 }else if(running){
  simulationDebt=Math.min(simulationDebt+delta*(mode==='replay'?1:speedMultiplier),MAX_CATCHUP_SECONDS);
  while(simulationDebt>=DT&&performance.now()<budget){
   if(mode==='replay'){
    tickCar(track,replayCar);
    if(replayCar.finished||replayCar.dead){running=false;setState();toast(replayCar.finished?'Best lap complete: '+formatTime(replayCar.time)+'.':'Replay ended.');break;}
   }else trainer.step();
   simulationDebt-=DT;
  }
 }
 // Avoid redrawing the stationary scene at 60 fps while paused or benchmarking.
 if(running||drag||time-lastRender>=200){renderScene();lastRender=time;}
 if(time-lastUI>100){
  syncUI();lastUI=time;
  const stamp=comparison?.trainers.map(t=>t.generation).join(':')??'';
  if(stamp!==compareStamp){compareStamp=stamp;drawComparison();}
 }
}
const frameLoop=createFrameLoop({step:stepFrame,onError:reportFailure,schedule:requestAnimationFrame,cancel:cancelAnimationFrame});
try{buildWorld();resize();syncUI();loadHash();}catch(error){reportFailure(error);}
frameLoop.start();

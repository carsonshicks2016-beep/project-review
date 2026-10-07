from pathlib import Path
p=Path('dist/app.mjs');s=p.read_text()
s=s.replace("import {createWorld} from './scene.mjs';", "import {createWorld} from './scene.mjs';\nimport {createFrameLoop,prepareImportedSession,encodeDriverLink,decodeDriverLink,FRAME_BUDGET_MS,MAX_CATCHUP_SECONDS,MAX_FRAME_DELTA_SECONDS} from './runtime.mjs';\nimport * as engine from './engine.mjs';")
s=s.replace('POPULATION,readDriver}', 'POPULATION,readDriver,Comparison,TRACE_STRIDE}')
s=s.replace('new Trainer(track)', "new Trainer(track,99,null,{recordLines:true})").replace('new Trainer(track,seed+99,weights)', "new Trainer(track,seed+99,weights,{recordLines:true})")
s=s.replace("let angle=.68", "let lineGroup=null,lineStamp='',comparison=null,comparisonRunning=false,compareStamp='',lastRender=0;\nconst reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;\nlet angle=.68")
s=s.replace('camera.position.lerp(target,.07)', 'camera.position.lerp(target,reducedMotion?1:.07)')
s=s.replace("$('progress-bar').style.width=progress+'%';", "$('progress-bar').style.width=progress+'%';$('lap-progress').setAttribute('aria-valuenow',Math.round(progress));")
s=s.replace("$('export').disabled=!trainer.champion;", "$('export').disabled=!trainer.champion;$('share').disabled=!trainer.champion;")
s=s.replace("if(trainer.generation!==lastGeneration){lastGeneration=trainer.generation;drawChart();} }", "if(trainer.generation!==lastGeneration){lastGeneration=trainer.generation;drawChart();setState();}if(lineStamp!==track.seed+':'+(trainer.champion?.generation??'none'))refreshLines(); }")
a=s.index('function setState()');b=s.index('function toast',a)
s=s[:a]+'''function setState(){
 const replay=mode==='replay';
 $('start-text').textContent=running?(replay?'Pause replay':'Pause training'):(replay?'Resume replay':trainer.elapsed?'Resume training':'Start training');
 $('start-icon').textContent=running?'Ⅱ':'▶';
 $('status').textContent=comparisonRunning?'Comparing algorithms':running?(replay?'Replaying best lap':trainer.exploring?'Exploring after a plateau':'Training in progress'):(replay?'Replay paused':trainer.elapsed?'Session paused':'Ready to train');
 $('view-label').textContent=replay?'BEST LAP REPLAY':'LIVE SIMULATION';
 $('orbit').setAttribute('aria-pressed',cameraMode==='orbit');$('follow').setAttribute('aria-pressed',cameraMode==='follow');
}
''' + s[b:]
s=s.replace("$('start').onclick=()=>{", "$('start').onclick=()=>{comparisonRunning=false;updateComparisonButton();clearFault();")
s=s.replace("simulationDebt=0;buildWorld();drawChart();syncUI();setState();toast(weights?", "simulationDebt=0;clearComparison();clearFault();buildWorld();drawChart();syncUI();setState();toast(weights?")
s=s.replace("$('replay').onclick=()=>{", "$('replay').onclick=()=>{comparisonRunning=false;updateComparisonButton();clearFault();")
a=s.index("$('import').onclick");b=s.index("$('reset').onclick",a)
s=s[:a]+'''function applyDriver(data){
 const next=prepareImportedSession(data,track,engine);
 // Construct and evaluate before replacing the session; invalid data leaves it intact.
 track=next.track;trainer=next.trainer;running=false;mode='train';replayCar=null;
 simulationDebt=0;lastGeneration=0;clearComparison();clearFault();
 $('seed').value=track.seed;$('elevation').value=track.elevation;
 $('elevation-value').textContent=track.elevation+' m';$('replay').textContent='↻ Replay best';
 buildWorld();drawChart();syncUI();setState();
 toast(next.probe.finished?'Driver loaded. Verified lap: '+next.probe.time.toFixed(2)+' s.':'Driver loaded. Train to improve its track completion.');
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
''' + s[b:]
s=s.replace("$('reset').onclick=()=>{running=false;", "$('reset').onclick=()=>{clearFault();running=false;")
s=s.replace("$(id==='orbit'?'follow':'orbit').classList.remove('selected');};", "$(id==='orbit'?'follow':'orbit').classList.remove('selected');setState();};")
s=s.replace("$('follow').classList.remove('selected');});", "$('follow').classList.remove('selected');setState();});")
a=s.index('function resize()')
s=s[:a]+'''
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
   for(const side of [-1,1]){positions.push(trace[offset]-dz/length*.4*side,trace[offset+1]+.38,trace[offset+2]+dx/length*.4*side);colors.push(color.r,color.g,color.b);}
   const vertex=offset/TRACE_STRIDE*2;
   if(offset+TRACE_STRIDE<trace.length)indices.push(vertex,vertex+1,vertex+2,vertex+1,vertex+3,vertex+2);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));geometry.setIndex(indices);
  const material=new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:index===lines.length-1?.95:.14+.08*index,depthWrite:false});
  const mesh=new THREE.Mesh(geometry,material);mesh.renderOrder=index+1;lineGroup.add(mesh);
 });
 const best=lines.at(-1);
 $('line-summary').textContent=best?'Gen '+best.generation+' · '+best.lap.toFixed(2)+' s · '+lines.length+' champion line'+(lines.length===1?'':'s'):'A completed champion lap will draw the first line.';
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
  return '<tr><th scope="row">'+name+'</th><td>'+(t?.evaluations??0)+'</td><td>'+(t?.bestLap?t.bestLap.toFixed(2)+' s':'No lap yet')+'</td><td>'+completion.toFixed(1)+'%</td></tr>';
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
 renderer.setSize(box.width,box.height);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();drawChart();drawProfile();drawComparison();
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
    if(replayCar.finished||replayCar.dead){running=false;setState();toast(replayCar.finished?'Best lap complete: '+replayCar.time.toFixed(2)+' seconds.':'Replay ended.');break;}
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
'''
p.write_text(s)

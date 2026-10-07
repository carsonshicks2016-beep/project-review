from pathlib import Path
p=Path('dist/index.html');s=p.read_text()
s=s.replace('RIDGELINE 2077','RIDGELINE 2077 · 3 KM').replace('<strong>Ridgeline Circuit</strong>','<strong id="circuit-name">Ridgeline · 3 km</strong>')
s=s.replace('<div class="speed-hud">','<canvas id="minimap" role="img" aria-label="Circuit map showing the leader and four sectors"></canvas><div class="speed-hud">')
s=s.replace('</i></div></div><div class="legend">','</i></div><div class="car-conditions"><span id="grade-value">Grade 0%</span><span id="load-value">Load 1.0×</span></div></div><div class="legend">')
s=s.replace('<div class="control-row"><label for="seed">', '<div class="control-row"><label for="course-length">Course length</label><select id="course-length"><option value="3" selected>3 km · Club circuit</option><option value="6">6 km · Long circuit</option><option value="10">10 km · Endurance</option></select></div><div class="control-row"><label for="seed">')
s=s.replace('<div class="track-stats">','<div class="track-stats">').replace('<p class="fine">Keeps the best driver; starts a new session.</p>', '<p class="fine">Keeps the best driver; starts a new session.</p><p class="fine" id="time-limit">Longer courses receive longer attempt limits.</p>')
s=s.replace('<div class="bottom-grid">','''<section class="sector-panel" aria-label="Sector timing"><div class="section-heading"><h2>Sector timing</h2><span>Current leader / session best · four equal-distance sectors</span></div><div id="sectors" class="sectors"><div>Sector 1 <strong>—</strong></div><div>Sector 2 <strong>—</strong></div><div>Sector 3 <strong>—</strong></div><div>Sector 4 <strong>—</strong></div></div><p id="lap-summary">Complete a sector to set the first split.</p></section>
<div class="bottom-grid">''')
s=s.replace('One track. 48 drivers. A faster generation.','Longer roads. 48 drivers. A faster generation.')
s=s.replace('Closing it ends the session.','Closing it ends the session.')
s=s.replace('The racing line shows up to four improved champion laps, colored by speed.', 'The racing line shows up to four improved champion laps, colored by speed. Choose 3, 6 or 10 km circuits; each has four timed sectors. Attempt limits scale with course length. The minimap tracks the current leader, and the Follow car camera keeps the driving visible on long courses.')
p.write_text(s)
p=Path('dist/style.css');s=p.read_text()+'''
#minimap{position:absolute;left:24px;bottom:48px;width:155px;height:130px;z-index:2;background:#172414ba;border:1px solid #a7bf8155;pointer-events:none}.car-conditions{display:flex;justify-content:space-between;margin-top:10px;font:11px var(--mono);color:#d0dbc4}.sector-panel{border:1px solid var(--line);background:#161b17;padding:20px 24px;margin-top:20px}.sector-panel .section-heading>span{font-size:12px;color:var(--muted)}.sectors{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px}.sectors>div{border-left:3px solid #526443;padding:4px 14px;font-size:14px;color:#bcc8b0}.sectors strong{display:block;font:500 30px var(--display);color:#edf4e2;margin:8px 0 6px}.sectors small{display:block;font:12px var(--mono);color:#aaba9a}.sectors .session-best{border-color:#cf9bf5}.sectors .session-best strong{color:#d6b3f0}.sectors .slower{border-color:#eb9a81}.sectors .faster{border-color:#d6fa59}#lap-summary{font-size:13px;color:#a7b69b;margin:18px 0 0}.stats strong{font-size:34px}#course-length{max-width:162px;font-family:var(--body);font-size:13px}.line-toolbar{row-gap:9px}.simulation-footer{gap:8px}.legend{font-size:11px}@media(max-width:760px){#minimap{width:120px;height:100px;left:16px;bottom:46px}.sector-panel .section-heading{align-items:start;flex-direction:column}.sectors{grid-template-columns:1fr 1fr;gap:18px}.sectors strong{font-size:29px}.stats strong{font-size:32px}.speed-hud{width:145px}}
''';p.write_text(s)
p=Path('dist/app.mjs');s=p.read_text()
s=s.replace('let track=makeTrack(),','let track=makeTrack(2077,18,3),')
s=s.replace('world=createWorld(track);scene.add(world);updateTrackInfo();', 'world=createWorld(track);scene.add(world);configureWorldCamera();updateTrackInfo();')
a=s.index('function updateTrackInfo()');b=s.index('const followTarget',a)
s=s[:a]+'''function formatTime(seconds){
 if(seconds===null||seconds===undefined)return '—';
 if(seconds<60)return seconds.toFixed(2)+' s';
 return Math.floor(seconds/60)+':'+(seconds%60).toFixed(2).padStart(5,'0');
}
function configureWorldCamera(){
 const extent=Math.max(track.terrainWidth,track.terrainDepth);
 camera.far=Math.max(1400,extent*7);camera.updateProjectionMatrix();
 scene.fog.near=Math.max(550,extent*1.8);scene.fog.far=Math.max(1100,extent*5);
 configureShadows(null);
}
function configureShadows(car){
 const extent=car?220:Math.max(track.terrainWidth,track.terrainDepth);
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
''' + s[b:]
s=s.replace('const distance=zoom*Math.max(1,1.25/camera.aspect)', 'const distance=zoom*Math.max(1,1.25/camera.aspect)*Math.max(track.terrainWidth/405,track.terrainDepth/348)')
s=s.replace('world.userData.updateCars(cars,leader);','world.userData.updateCars(cars,leader);configureShadows(cameraMode===\'follow\'?leader:null);')
s=s.replace("$('best-lap').innerHTML=trainer.bestLap?trainer.bestLap.toFixed(2)+'<small> s</small>':'—<small> s</small>';", "$('best-lap').textContent=formatTime(trainer.bestLap);")
s=s.replace("$('episode').textContent=c.time.toFixed(1)+' s';", "$('episode').textContent=formatTime(c.time);")
s=s.replace("refreshLines(); }", "refreshLines();drawMinimap(c);updateSectors(c); }")
s=s.replace("$('start').onclick=()=>{comparisonRunning=false;", "$('start').onclick=()=>{if(!running&&cameraMode==='orbit'&&track.length>2000){cameraMode='follow';$('follow').classList.add('selected');$('orbit').classList.remove('selected');}comparisonRunning=false;")
s=s.replace("&&+$('elevation').value===track.elevation)", "&&+$('elevation').value===track.elevation&&+$('course-length').value===track.lengthKm)")
s=s.replace("track=makeTrack(seed,+$('elevation').value);", "track=makeTrack(seed,+$('elevation').value,$('course-length').value==='legacy'?null:+$('course-length').value);")
s=s.replace('track:{seed:track.seed,elevation:track.elevation}', 'track:{seed:track.seed,elevation:track.elevation,lengthKm:track.lengthKm}')
s=s.replace("$('seed').value=track.seed;$('elevation').value=track.elevation;", "$('seed').value=track.seed;$('elevation').value=track.elevation;\n const lengthValue=track.lengthKm===null?'legacy':String(track.lengthKm);\n if(![...$('course-length').options].some(o=>o.value===lengthValue))$('course-length').add(new Option(track.lengthKm===null?'Legacy 0.82 km':track.lengthKm+' km',lengthValue));\n $('course-length').value=lengthValue;")
s=s.replace("next.probe.time.toFixed(2)+' s.'", "formatTime(next.probe.time)+'.'")
s=s.replace("best.lap.toFixed(2)+' s · '", "formatTime(best.lap)+' · '")
s=s.replace("t.bestLap.toFixed(2)+' s'", "formatTime(t.bestLap)")
s=s.replace("replayCar.time.toFixed(2)+' seconds.'", "formatTime(replayCar.time)+'.'")
s=s.replace("for(const side of [-1,1]){positions.push(trace[offset]-dz/length*.4*side,trace[offset+1]+.7,trace[offset+2]+dx/length*.4*side);", "const lineWidth=Math.max(.4,track.length/8000);for(const side of [-1,1]){positions.push(trace[offset]-dz/length*lineWidth*side,trace[offset+1]+.7,trace[offset+2]+dx/length*lineWidth*side);")
# New panels rely only on actual episode measurements.
a=s.index('function refreshLines()')
s=s[:a]+'''function drawMinimap(car){
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
''' +s[a:]
p.write_text(s)

from pathlib import Path
p=Path('dist/engine.mjs');s=p.read_text().replace('makeTrack(seed=2077,elevation=18)','makeTrack(seed=2077,elevation=18,lengthKm=null)')
s=s.replace(' const random=rng(seed)', " if(lengthKm!==null&&(!Number.isFinite(lengthKm)||lengthKm<.5||lengthKm>12))throw Error('Course length must be between 0.5 and 12 km.');\n const random=rng(seed)",1)
s=s.replace(' const n=480,points=[];let length=0;', ' const n=lengthKm===null?480:Math.ceil(lengthKm*1000/1.7),points=[];let length=0;')
s=s.replace('const r=112+19*Math.sin(3*a+phase)+13*Math.sin(2*a+phase2)+7*Math.sin(5*a+phase);','const r=112+19*Math.sin(3*a+phase)+13*Math.sin(2*a+phase2)+7*Math.sin(5*a+phase)+(lengthKm===null?0:5*Math.sin(11*a+phase2)+3*Math.sin(17*a+phase));')
needle=' for(let i=0;i<n;i++){let p=points[i],q=points[(i+1)%n];p.angle='
s=s.replace(needle,''' if(lengthKm!==null){
  const perimeter=points.reduce((sum,p,i)=>sum+Math.hypot(points[(i+1)%n].x-p.x,points[(i+1)%n].z-p.z),0),scale=lengthKm*1000/perimeter;
  for(const p of points){p.x*=scale;p.z*=scale;p.y=height(p.x,p.z);}
 }
 for(let i=0;i<n;i++){let p=points[i],q=points[(i+1)%n];p.angle=''')
s=s.replace('return {seed,elevation,height,points,length,width:12,n,ascent:', '''return {seed,elevation,lengthKm,height,points,length,width:12,n,
  episodeLimit:lengthKm===null?MAX_EPISODE_SECONDS:Math.max(MAX_EPISODE_SECONDS,Math.ceil(length/6)),
  terrainWidth:lengthKm===null?405:Math.max(...points.map(p=>Math.abs(p.x)))*2+100,
  terrainDepth:lengthKm===null?348:Math.max(...points.map(p=>Math.abs(p.z)))*2+100,
  ascent:''')
s=s.replace('Math.ceil(MAX_EPISODE_SECONDS/DT/TRACE_EVERY_STEPS)', 'Math.ceil(track.episodeLimit/DT/TRACE_EVERY_STEPS)')
s=s.replace('traceLength:0,score:0','traceLength:0,sectorTimes:[null,null,null,null],lastSectorTime:0,score:0')
s=s.replace(' // Only a champion replay records', ''' for(let sector=0;sector<4;sector++){
  if(c.sectorTimes[sector]===null&&c.progress>=track.length*(sector+1)/4){
   c.sectorTimes[sector]=c.time-c.lastSectorTime;c.lastSectorTime=c.time;
  }
 }
 // Only a champion replay records''')
s=s.replace('(MAX_EPISODE_SECONDS-c.time)','(track.episodeLimit-c.time)').replace('c.time>MAX_EPISODE_SECONDS','c.time>track.episodeLimit')
s=s.replace(" return {weights:[...w],seed:known?t.seed:null,elevation:known?t.elevation:null};", " if(t.lengthKm!=null&&(!Number.isFinite(t.lengthKm)||t.lengthKm<.5||t.lengthKm>12))throw Error('Invalid course length in driver.');\n return {weights:[...w],seed:known?t.seed:null,elevation:known?t.elevation:null,lengthKm:known?(t.lengthKm??null):null};")
s=s.replace('this.lines=[];this.stagnantGenerations', 'this.lines=[];this.bestSectors=[null,null,null,null];this.stagnantGenerations')
s=s.replace('progress:car.bestProgress,generation};', 'progress:car.bestProgress,sectors:[...car.sectorTimes],generation};')
s=s.replace('this.bestLap=this.champion.lap;', 'this.bestLap=this.champion.lap;this.bestSectors=[...car.sectorTimes];')
s=s.replace('progress:best.bestProgress,generation:this.generation};', 'progress:best.bestProgress,sectors:[...best.sectorTimes],generation:this.generation};')
s=s.replace('  if(finished.length)this.bestLap=', '''  for(const car of ranked)for(let sector=0;sector<4;sector++)if(car.sectorTimes[sector]!==null)this.bestSectors[sector]=Math.min(this.bestSectors[sector]??Infinity,car.sectorTimes[sector]);
  if(finished.length)this.bestLap=''')
p.write_text(s)
p=Path('dist/runtime.mjs');s=p.read_text().replace('driver.elevation??currentTrack.elevation);','driver.elevation??currentTrack.elevation,driver.seed!==null?driver.lengthKm:currentTrack.lengthKm);')
s=s.replace(' const bytes=new Uint8Array(205),view=new DataView(bytes.buffer);',''' const long=track.lengthKm!=null;
 if(long&&(!Number.isFinite(track.lengthKm)||track.lengthKm<.5||track.lengthKm>12))throw Error('Invalid course length.');
 const bytes=new Uint8Array(long?213:205),view=new DataView(bytes.buffer);''').replace('view.setUint8(0,1);','view.setUint8(0,long?2:1);')
s=s.replace('weights.forEach((w,i)=>view.setFloat64(13+i*8,w,true));','if(long)view.setFloat64(205,track.lengthKm,true);\n weights.forEach((w,i)=>view.setFloat64(13+i*8,w,true));')
s=s.replace("!/^[A-Za-z0-9_-]{274}$/.test(encoded)","!/^([A-Za-z0-9_-]{274}|[A-Za-z0-9_-]{284})$/.test(encoded)")
s=s.replace("replaceAll('_','/')+'=='", "replaceAll('_','/')+'='.repeat((4-encoded.length%4)%4)")
s=s.replace("if(bytes.length!==205||bytes[0]!==1)","if(!((bytes.length===205&&bytes[0]===1)||(bytes.length===213&&bytes[0]===2)))")
s=s.replace(' // Validate fully before replacing', ' if(bytes[0]===2)data.track.lengthKm=view.getFloat64(205,true);\n // Validate fully before replacing')
p.write_text(s)

// Pure, deterministic simulation and neuroevolution; also runs directly in Node.
export const DT=1/30, INPUTS=12, POPULATION=48;
// Keep these units and values stable: exported policies depend on the physics and reward.
export const MAX_EPISODE_SECONDS=180, NEAREST_SEGMENT_WINDOW=12;
export const MAX_STEER_RADIANS=.5, STEERING_RESPONSE=.28;
export const INCOMPLETE_REWARD=9000, FINISH_REWARD=10000, LAP_SECOND_REWARD=40;
export const TRACE_STRIDE=5, TRACE_EVERY_STEPS=4;
export const ALGORITHMS=['genetic','hill','random'];
export const SURFACES={dry:1,wet:.75};
// Input order matches observe(); used by the ablation study and the weight inspector.
export const INPUT_NAMES=['heading 8 m','heading 18 m','heading 35 m','heading 60 m','lateral offset','speed','slip angle','curvature','grade','tire load','prev steering','bias'];
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
export function rng(seed){let n=seed>>>0;return()=>{n+=0x6D2B79F5;let t=n;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-9,r())))*Math.cos(2*Math.PI*r());}
export function makeTrack(seed=2077,elevation=18,lengthKm=null,surface='dry'){
 if(lengthKm!==null&&(!Number.isFinite(lengthKm)||lengthKm<.5||lengthKm>12))throw Error('Course length must be between 0.5 and 12 km.');
 if(!(surface in SURFACES))throw Error('Surface must be dry or wet.');
 const random=rng(seed),phase=random()*6.28,phase2=random()*6.28;
 const heightScale=lengthKm===null?1:Math.max(1,lengthKm/3);
 // Relief amplitude tracks the horizontal stretch. Scaling only the wavelength spread the same
 // hills thinner, so long courses read flat and climbed no more than a 3 km one. Because amplitude
 // and wavelength scale together, grade per metre is unchanged: lap times and learned policies
 // carry over, while total ascent and visible relief now grow with the course.
 const height=(worldX,worldZ)=>{const x=worldX/heightScale,z=worldZ/heightScale;return elevation*heightScale*(.46*Math.sin(x*.021+phase)*Math.cos(z*.016)+.27*Math.sin(z*.033+phase2)+.14*Math.sin((x+z)*.054));};
 // The radius before length scaling stays above 65 m (73 m in legacy mode). Increasing polar angle and a
 // positive radius keep the generated centerline simple (no crossings).
 // nearest() deliberately follows this ordered route, not arbitrary custom circuits.
 const n=lengthKm===null?480:Math.ceil(lengthKm*1000/1.7),points=[];let length=0;
 for(let i=0;i<n;i++){const a=i/n*Math.PI*2;const r=112+19*Math.sin(3*a+phase)+13*Math.sin(2*a+phase2)+7*Math.sin(5*a+phase)+(lengthKm===null?0:5*Math.sin(11*a+phase2)+3*Math.sin(17*a+phase));let x=Math.cos(a)*r*1.22,z=Math.sin(a)*r*.9;points.push({x,z,y:height(x,z),s:0});}
 if(lengthKm!==null){
  const perimeter=points.reduce((sum,p,i)=>sum+Math.hypot(points[(i+1)%n].x-p.x,points[(i+1)%n].z-p.z),0),scale=lengthKm*1000/perimeter;
  for(const p of points){p.x*=scale;p.z*=scale;p.y=height(p.x,p.z);}
 }
 for(let i=0;i<n;i++){let p=points[i],q=points[(i+1)%n];p.angle=Math.atan2(q.z-p.z,q.x-p.x);p.ds=Math.hypot(q.x-p.x,q.z-p.z);p.s=length;length+=p.ds;p.grade=(q.y-p.y)/p.ds;}
 for(let i=0;i<n;i++){let p=points[i],prev=points[(i+n-1)%n];p.curvature=wrap(p.angle-prev.angle)/p.ds;p.vertical=(p.grade-prev.grade)/p.ds;}
 return {seed,elevation,lengthKm,heightScale,surface,grip:SURFACES[surface],height,points,length,width:12,n,
  episodeLimit:lengthKm===null?MAX_EPISODE_SECONDS:Math.max(MAX_EPISODE_SECONDS,Math.ceil(length/6)),
  terrainWidth:lengthKm===null?405:Math.max(...points.map(p=>Math.abs(p.x)))*2+100,
  terrainDepth:lengthKm===null?348:Math.max(...points.map(p=>Math.abs(p.z)))*2+100,
  ascent:points.reduce((v,p,i)=>v+Math.max(0,points[(i+1)%n].y-p.y),0)};
}
export function sample(track,s){s=((s%track.length)+track.length)%track.length;let lo=0,hi=track.n-1;while(lo<hi){let m=Math.ceil((lo+hi)/2);if(track.points[m].s<=s)lo=m;else hi=m-1;}const p=track.points[lo],q=track.points[(lo+1)%track.n],u=(s-p.s)/p.ds;return {...p,x:p.x+(q.x-p.x)*u,z:p.z+(q.z-p.z)*u,y:p.y+(q.y-p.y)*u,index:lo};}
export function nearest(track,c){let best=Infinity,result=null;for(let k=-NEAREST_SEGMENT_WINDOW;k<=NEAREST_SEGMENT_WINDOW;k++){let i=(c.index+k+track.n)%track.n,p=track.points[i],q=track.points[(i+1)%track.n],dx=q.x-p.x,dz=q.z-p.z,u=clamp(((c.x-p.x)*dx+(c.z-p.z)*dz)/(p.ds*p.ds),0,1),x=p.x+dx*u,z=p.z+dz*u,d=(c.x-x)**2+(c.z-z)**2;if(d<best){best=d;result={...p,index:i,s:p.s+u*p.ds,lateral:((c.z-z)*dx-(c.x-x)*dz)/p.ds,distance:Math.sqrt(d)};}}return result;}
export function newCar(track,weights,id=0,{recordTrace=false}={}){let p=track.points[0];return {id,weights,x:p.x,z:p.z,y:p.y,heading:p.angle,vx:Math.cos(p.angle)*2,vz:Math.sin(p.angle)*2,speed:2,index:0,progress:0,bestProgress:0,time:0,dead:false,finished:false,steer:0,throttle:0,load:1,lateral:0,offTime:0,stuck:0,offRoad:false,slide:0,inputs:Array(INPUTS).fill(0),trace:recordTrace?new Float32Array((Math.ceil(track.episodeLimit/DT/TRACE_EVERY_STEPS)+2)*TRACE_STRIDE):null,traceLength:0,sectorTimes:[null,null,null,null],lastSectorTime:0,score:0};}
export function observe(track,c,road){const v=Math.hypot(c.vx,c.vz),slip=wrap(Math.atan2(c.vz,c.vx)-c.heading),look=[8,18,35,60].map(d=>{let p=sample(track,road.s+d);return clamp(wrap(Math.atan2(p.z-c.z,p.x-c.x)-c.heading),-1.5,1.5);});const ahead=sample(track,road.s+25);return [...look,clamp(road.lateral/6,-2,2),v/32,slip,Math.min(2,Math.max(Math.abs(road.curvature),Math.abs(ahead.curvature))*25),road.grade*4,c.load-1,c.steer,1];}
export function policy(w,inputs){return [0,1].map(j=>Math.tanh(inputs.reduce((s,x,i)=>s+x*w[j*INPUTS+i],0)));}
export function tickCar(track,c){if(c.dead||c.finished)return;const road=nearest(track,c);c.index=road.index;c.lateral=road.lateral;c.inputs=observe(track,c,road);let [steering,power]=policy(c.weights,c.inputs);c.steer+=(steering*MAX_STEER_RADIANS-c.steer)*STEERING_RESPONSE;c.throttle=power;let speed=Math.hypot(c.vx,c.vz);c.load=clamp(1+speed*speed*road.vertical/9.81,.25,1.6);const off=road.distance>track.width/2,grip=(off?.48:1.12)*track.grip*9.81*c.load;
 const forwardX=Math.cos(c.heading),forwardZ=Math.sin(c.heading),lateralV=-c.vx*forwardZ+c.vz*forwardX;
 const yaw=speed/2.7*Math.tan(c.steer);c.heading=wrap(c.heading+yaw*DT);
 const accel=(power>=0?power*7:power*13)*track.grip-.006*speed*speed-.35-9.81*road.grade-(off?2.5:0);
 // Demanded cornering force before the tires run out. The overshoot drives dust and tire marks.
 const demand=yaw*speed-lateralV*4,lateralA=clamp(demand,-grip,grip);
 c.offRoad=off;c.slide=grip>0?clamp((Math.abs(demand)-grip)/grip,0,1):0;
 c.vx+=(forwardX*accel-forwardZ*lateralA)*DT;c.vz+=(forwardZ*accel+forwardX*lateralA)*DT;
 if(c.vx*forwardX+c.vz*forwardZ<0){c.vx*=.5;c.vz*=.5;}c.x+=c.vx*DT;c.z+=c.vz*DT;c.speed=Math.hypot(c.vx,c.vz);c.y=track.height(c.x,c.z);c.time+=DT;
 let delta=road.s-((c.progress%track.length+track.length)%track.length);if(delta<-track.length/2)delta+=track.length;if(delta>track.length/2)delta-=track.length;c.progress+=delta;c.bestProgress=Math.max(c.bestProgress,c.progress);c.stuck=c.speed<1?c.stuck+DT:0;c.offTime=off?c.offTime+DT:Math.max(0,c.offTime-DT);
 for(let sector=0;sector<4;sector++){
  if(c.sectorTimes[sector]===null&&c.progress>=track.length*(sector+1)/4){
   c.sectorTimes[sector]=c.time-c.lastSectorTime;c.lastSectorTime=c.time;
  }
 }
 // Only a champion replay records a packed trace; population evaluations allocate none.
 if(c.trace&&Math.round(c.time/DT)%TRACE_EVERY_STEPS===0){
  const offset=c.traceLength*TRACE_STRIDE;
  c.trace[offset]=c.x;c.trace[offset+1]=c.y;c.trace[offset+2]=c.z;
  c.trace[offset+3]=c.time;c.trace[offset+4]=c.speed;c.traceLength++;
 }
 if(c.progress>=track.length && c.time>10){c.finished=true;c.score=FINISH_REWARD+(track.episodeLimit-c.time)*LAP_SECOND_REWARD;}
 else {c.score=c.bestProgress/track.length*INCOMPLETE_REWARD-c.offTime*20;if(road.distance>16||c.offTime>3||c.time>track.episodeLimit||c.stuck>5||c.progress< -10){c.dead=true;}}
}
// Validates a parsed driver export. Track settings are optional; null means "keep the current track".
export function readDriver(data){
 const w=data&&data.weights;
 if(!Array.isArray(w)||w.length!==INPUTS*2||!w.every(v=>typeof v==='number'&&Number.isFinite(v)))throw Error('Expected a driver export with '+INPUTS*2+' numeric weights.');
 const t=data.track||{},known=Number.isInteger(t.seed)&&t.seed>=1&&t.seed<=999999&&Number.isFinite(t.elevation)&&t.elevation>=0&&t.elevation<=36;
 if(t.lengthKm!=null&&(!Number.isFinite(t.lengthKm)||t.lengthKm<.5||t.lengthKm>12))throw Error('Invalid course length in driver.');
 if(t.surface!=null&&!(t.surface in SURFACES))throw Error('Invalid surface in driver.');
 return {weights:[...w],seed:known?t.seed:null,elevation:known?t.elevation:null,lengthKm:known?(t.lengthKm??null):null,
  surface:known?(t.surface??'dry'):'dry'};
}
export function prior(r){const w=Array(24).fill(0);let steer=[.1,.65,.1,0,-.22,0,-.5,0,0,0,-.1,0],throttle=[0,0,0,0,0,-.7,0,-.75,-.12,.05,0,.85];for(let i=0;i<12;i++){w[i]=steer[i]+normal(r)*.13;w[12+i]=throttle[i]+normal(r)*.2;}return w;}
export function evaluateDriver(track,weights,{recordTrace=false}={}){
 const car=newCar(track,weights,0,{recordTrace});
 while(!car.dead&&!car.finished)tickCar(track,car);
 if(car.trace)car.trace=car.trace.slice(0,car.traceLength*TRACE_STRIDE);
 return car;
}

export class Trainer{
 constructor(track,seed=99,champion=null,{algorithm='genetic',recordLines=false,escapePlateaus=true}={}){
  if(!ALGORITHMS.includes(algorithm))throw Error('Unknown search algorithm.');
  this.track=track;this.random=rng(seed);this.algorithm=algorithm;
  this.recordLines=recordLines;this.escapePlateaus=escapePlateaus;
  this.generation=1;this.history=[];this.champion=null;this.bestLap=null;
  this.totalLaps=0;this.elapsed=0;this.lastResult=null;this.evaluations=0;
  this.lines=[];this.bestSectors=[null,null,null,null];this.stagnantGenerations=0;this.exploring=false;this.simulationSteps=0;
  // All algorithms begin with the same seeded population for a fair comparison.
  this.population=Array.from({length:POPULATION},(_,i)=>newCar(track,champion?champion.map(x=>x+(i?normal(this.random)*.12:0)):prior(this.random),i));
 }
 step(){
  for(const car of this.population){if(!car.dead&&!car.finished)this.simulationSteps++;tickCar(this.track,car);}
  this.elapsed+=DT;
  if(this.population.every(car=>car.dead||car.finished))this.evolve();
 }
 adopt(weights,generation=0){
  const car=evaluateDriver(this.track,weights,{recordTrace:this.recordLines});
  this.champion={weights:[...weights],score:car.score,trace:car.trace,lap:car.finished?car.time:null,progress:car.bestProgress,sectors:[...car.sectorTimes],generation};
  this.bestLap=this.champion.lap;this.bestSectors=[...car.sectorTimes];
  if(car.trace&&car.finished)this.lines=[{trace:car.trace,generation,lap:car.time}];
  return car;
 }
 evolve(){
  const ranked=[...this.population].sort((a,b)=>b.score-a.score),best=ranked[0],finished=ranked.filter(car=>car.finished);
  const spread=diversity(this.population);
  this.totalLaps+=finished.length;this.evaluations+=POPULATION;
  const improved=!this.champion||best.score>this.champion.score;
  this.stagnantGenerations=improved?0:this.stagnantGenerations+1;
  if(improved){
   const trace=this.recordLines&&best.finished?evaluateDriver(this.track,best.weights,{recordTrace:true}).trace:null;
   this.champion={weights:[...best.weights],score:best.score,trace,lap:best.finished?best.time:null,progress:best.bestProgress,sectors:[...best.sectorTimes],generation:this.generation};
   if(trace){this.lines.push({trace,generation:this.generation,lap:best.time});this.lines=this.lines.slice(-4);}
  }
  for(const car of ranked)for(let sector=0;sector<4;sector++)if(car.sectorTimes[sector]!==null)this.bestSectors[sector]=Math.min(this.bestSectors[sector]??Infinity,car.sectorTimes[sector]);
  if(finished.length)this.bestLap=Math.min(this.bestLap??Infinity,...finished.map(car=>car.time));
  this.lastResult={generation:this.generation,lap:finished.length?finished[0].time:null,bestLap:this.bestLap,diversity:spread,completion:Math.min(100,best.bestProgress/this.track.length*100),finishers:finished.length,evaluations:this.evaluations};
  this.history.push(this.lastResult);this.generation++;
  // Reheat mutation only after 25 generations without improvement. Preserve the
  // champion and original first-20-generation behavior; don't silently reset progress.
  this.exploring=this.algorithm!=='random'&&this.escapePlateaus&&this.stagnantGenerations>=25;
  const elite=ranked.slice(0,8),sigma=Math.max(this.exploring?.12:.035,.14*Math.pow(.987,this.generation));
  this.population=Array.from({length:POPULATION},(_,i)=>{
   let weights;
   if(this.algorithm==='random')weights=prior(this.random);
   else if(this.algorithm==='hill')weights=i===0?[...this.champion.weights]:this.champion.weights.map(w=>w+normal(this.random)*sigma);
   else if(i===0)weights=[...this.champion.weights];
   else if(i<6)weights=[...elite[i-1].weights];
   else if(i>43&&(this.generation%8===0||this.exploring))weights=prior(this.random);
   else{
    const a=elite[Math.floor(this.random()*elite.length)].weights,b=elite[Math.floor(this.random()*elite.length)].weights;
    weights=a.map((w,j)=>(this.random()<.15?b[j]:w)+(this.random()<.8?normal(this.random)*sigma:0));
   }
   return newCar(this.track,weights,i);
  });
  this.elapsed=0;
 }
 get leader(){return this.population.reduce((a,b)=>b.score>a.score?b:a,this.population[0]);}
}

// Zeroing an input's weights is exactly equivalent to zeroing the input for a linear+tanh policy,
// so one deterministic replay per input measures what that sense is actually worth.
export function ablate(track,weights){
 const base=evaluateDriver(track,weights);
 const rows=INPUT_NAMES.map((input,index)=>{
  const car=evaluateDriver(track,weights.map((w,j)=>j%INPUTS===index?0:w));
  return {input,index,finished:car.finished,lap:car.finished?car.time:null,
   penalty:car.finished&&base.finished?car.time-base.time:null,
   progress:Math.min(1,car.bestProgress/track.length)};
 });
 return {base:{finished:base.finished,lap:base.finished?base.time:null},rows};
}
// Mean pairwise genome distance. Watching this collapse is what explains a lap-time plateau.
export function diversity(population){
 let sum=0,pairs=0;
 for(let a=0;a<population.length;a++)for(let b=a+1;b<population.length;b++){
  const wa=population[a].weights,wb=population[b].weights;let d=0;
  for(let i=0;i<wa.length;i++){const gap=wa[i]-wb[i];d+=gap*gap;}
  sum+=Math.sqrt(d);pairs++;
 }
 return pairs?sum/pairs:0;
}
// Round-robin scheduling gives each algorithm the same completed episode budget.
// The comparison is independent of the live training session and uses a fixed seed.
export class Comparison{
 constructor(track,{seed=99,generations=20}={}){
  this.track=track;this.seed=seed;this.generations=generations;this.cursor=0;
  this.trainers=ALGORITHMS.map(algorithm=>new Trainer(track,seed,null,{algorithm}));
 }
 get done(){return this.trainers.every(t=>t.generation>this.generations);}
 step(){
  for(let count=0;count<this.trainers.length;count++){
   const trainer=this.trainers[this.cursor++%this.trainers.length];
   if(trainer.generation<=this.generations){trainer.step();return;}
  }
 }
}

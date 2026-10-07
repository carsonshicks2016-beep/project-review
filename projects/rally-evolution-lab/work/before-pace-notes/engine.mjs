// Pure, deterministic simulation and neuroevolution; also runs directly in Node.
export const DT=1/30, INPUTS=12, POPULATION=48;
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
export function rng(seed){let n=seed>>>0;return()=>{n+=0x6D2B79F5;let t=n;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-9,r())))*Math.cos(2*Math.PI*r());}
export function makeTrack(seed=2077,elevation=18){
 const random=rng(seed),phase=random()*6.28,phase2=random()*6.28;
 const height=(x,z)=>elevation*(.46*Math.sin(x*.021+phase)*Math.cos(z*.016)+.27*Math.sin(z*.033+phase2)+.14*Math.sin((x+z)*.054));
 const n=480,points=[];let length=0;
 for(let i=0;i<n;i++){const a=i/n*Math.PI*2;const r=112+19*Math.sin(3*a+phase)+13*Math.sin(2*a+phase2)+7*Math.sin(5*a+phase);let x=Math.cos(a)*r*1.22,z=Math.sin(a)*r*.9;points.push({x,z,y:height(x,z),s:0});}
 for(let i=0;i<n;i++){let p=points[i],q=points[(i+1)%n];p.angle=Math.atan2(q.z-p.z,q.x-p.x);p.ds=Math.hypot(q.x-p.x,q.z-p.z);p.s=length;length+=p.ds;p.grade=(q.y-p.y)/p.ds;}
 for(let i=0;i<n;i++){let p=points[i],prev=points[(i+n-1)%n];p.curvature=wrap(p.angle-prev.angle)/p.ds;p.vertical=(p.grade-prev.grade)/p.ds;}
 return {seed,elevation,height,points,length,width:12,n,ascent:points.reduce((v,p,i)=>v+Math.max(0,points[(i+1)%n].y-p.y),0)};
}
export function sample(track,s){s=((s%track.length)+track.length)%track.length;let lo=0,hi=track.n-1;while(lo<hi){let m=Math.ceil((lo+hi)/2);if(track.points[m].s<=s)lo=m;else hi=m-1;}const p=track.points[lo],q=track.points[(lo+1)%track.n],u=(s-p.s)/p.ds;return {...p,x:p.x+(q.x-p.x)*u,z:p.z+(q.z-p.z)*u,y:p.y+(q.y-p.y)*u,index:lo};}
export function nearest(track,c){let best=Infinity,result=null;for(let k=-12;k<=12;k++){let i=(c.index+k+track.n)%track.n,p=track.points[i],q=track.points[(i+1)%track.n],dx=q.x-p.x,dz=q.z-p.z,u=clamp(((c.x-p.x)*dx+(c.z-p.z)*dz)/(p.ds*p.ds),0,1),x=p.x+dx*u,z=p.z+dz*u,d=(c.x-x)**2+(c.z-z)**2;if(d<best){best=d;result={...p,index:i,s:p.s+u*p.ds,lateral:((c.z-z)*dx-(c.x-x)*dz)/p.ds,distance:Math.sqrt(d)};}}return result;}
export function newCar(track,weights,id=0){let p=track.points[0];return {id,weights,x:p.x,z:p.z,y:p.y,heading:p.angle,vx:Math.cos(p.angle)*2,vz:Math.sin(p.angle)*2,speed:2,index:0,progress:0,bestProgress:0,time:0,dead:false,finished:false,steer:0,throttle:0,load:1,lateral:0,offTime:0,stuck:0,inputs:Array(INPUTS).fill(0),trace:[],score:0};}
export function observe(track,c,road){const v=Math.hypot(c.vx,c.vz),slip=wrap(Math.atan2(c.vz,c.vx)-c.heading),look=[8,18,35,60].map(d=>{let p=sample(track,road.s+d);return clamp(wrap(Math.atan2(p.z-c.z,p.x-c.x)-c.heading),-1.5,1.5);});const ahead=sample(track,road.s+25);return [...look,clamp(road.lateral/6,-2,2),v/32,slip,Math.min(2,Math.max(Math.abs(road.curvature),Math.abs(ahead.curvature))*25),road.grade*4,c.load-1,c.steer,1];}
export function policy(w,inputs){return [0,1].map(j=>Math.tanh(inputs.reduce((s,x,i)=>s+x*w[j*INPUTS+i],0)));}
export function tickCar(track,c){if(c.dead||c.finished)return;const road=nearest(track,c);c.index=road.index;c.lateral=road.lateral;c.inputs=observe(track,c,road);let [steering,power]=policy(c.weights,c.inputs);c.steer+=(steering*.5-c.steer)*.28;c.throttle=power;let speed=Math.hypot(c.vx,c.vz);c.load=clamp(1+speed*speed*road.vertical/9.81,.25,1.6);const off=road.distance>track.width/2,grip=(off?.48:1.12)*9.81*c.load;
 const forwardX=Math.cos(c.heading),forwardZ=Math.sin(c.heading),lateralV=-c.vx*forwardZ+c.vz*forwardX;
 const yaw=speed/2.7*Math.tan(c.steer);c.heading=wrap(c.heading+yaw*DT);
 const accel=(power>=0?power*7:power*13)-.006*speed*speed-.35-9.81*road.grade-(off?2.5:0);
 const lateralA=clamp(yaw*speed-lateralV*4,-grip,grip);
 c.vx+=(forwardX*accel-forwardZ*lateralA)*DT;c.vz+=(forwardZ*accel+forwardX*lateralA)*DT;
 if(c.vx*forwardX+c.vz*forwardZ<0){c.vx*=.5;c.vz*=.5;}c.x+=c.vx*DT;c.z+=c.vz*DT;c.speed=Math.hypot(c.vx,c.vz);c.y=track.height(c.x,c.z);c.time+=DT;
 let delta=road.s-((c.progress%track.length+track.length)%track.length);if(delta<-track.length/2)delta+=track.length;if(delta>track.length/2)delta-=track.length;c.progress+=delta;c.bestProgress=Math.max(c.bestProgress,c.progress);c.stuck=c.speed<1?c.stuck+DT:0;c.offTime=off?c.offTime+DT:Math.max(0,c.offTime-DT);
 if(Math.round(c.time/DT)%4===0)c.trace.push([c.x,c.y,c.z,c.time]);
 if(c.progress>=track.length && c.time>10){c.finished=true;c.score=10000+(180-c.time)*40;}
 else {c.score=c.bestProgress/track.length*9000-c.offTime*20;if(road.distance>16||c.offTime>3||c.time>180||c.stuck>5||c.progress< -10){c.dead=true;}}
}
// Validates a parsed driver export. Track settings are optional; null means "keep the current track".
export function readDriver(data){
 const w=data&&data.weights;
 if(!Array.isArray(w)||w.length!==INPUTS*2||!w.every(v=>typeof v==='number'&&Number.isFinite(v)))throw Error('Expected a driver export with '+INPUTS*2+' numeric weights.');
 const t=data.track||{},known=Number.isInteger(t.seed)&&t.seed>=1&&t.seed<=999999&&Number.isFinite(t.elevation)&&t.elevation>=0&&t.elevation<=36;
 return {weights:[...w],seed:known?t.seed:null,elevation:known?t.elevation:null};
}
export function prior(r){const w=Array(24).fill(0);let steer=[.1,.65,.1,0,-.22,0,-.5,0,0,0,-.1,0],throttle=[0,0,0,0,0,-.7,0,-.75,-.12,.05,0,.85];for(let i=0;i<12;i++){w[i]=steer[i]+normal(r)*.13;w[12+i]=throttle[i]+normal(r)*.2;}return w;}
export class Trainer{
 constructor(track,seed=99,champion=null){this.track=track;this.random=rng(seed);this.generation=1;this.history=[];this.champion=null;this.bestLap=null;this.totalLaps=0;this.elapsed=0;this.lastResult=null;this.population=Array.from({length:POPULATION},(_,i)=>newCar(track,champion?champion.map((x,j)=>x+(i?normal(this.random)*.12:0)):prior(this.random),i));}
 step(){for(const c of this.population)tickCar(this.track,c);this.elapsed+=DT;if(this.population.every(c=>c.dead||c.finished))this.evolve();}
 evolve(){let ranked=[...this.population].sort((a,b)=>b.score-a.score),best=ranked[0],finished=ranked.filter(c=>c.finished);this.totalLaps+=finished.length;
 if(!this.champion||best.score>this.champion.score){this.champion={weights:[...best.weights],score:best.score,trace:best.trace.map(p=>[...p]),lap:best.finished?best.time:null,progress:best.bestProgress,generation:this.generation};}
 if(finished.length)this.bestLap=Math.min(this.bestLap??Infinity,...finished.map(c=>c.time));
 this.lastResult={generation:this.generation,lap:finished.length?finished[0].time:null,completion:Math.min(100,best.bestProgress/this.track.length*100),finishers:finished.length};this.history.push(this.lastResult);this.generation++;
 const elite=ranked.slice(0,8),sigma=Math.max(.035,.14*Math.pow(.987,this.generation));
 this.population=Array.from({length:POPULATION},(_,i)=>{let weights;if(i===0)weights=[...this.champion.weights];else if(i<6)weights=[...elite[i-1].weights];else if(i>43&&this.generation%8===0)weights=prior(this.random);else{const a=elite[Math.floor(this.random()*elite.length)].weights,b=elite[Math.floor(this.random()*elite.length)].weights;weights=a.map((w,j)=>(this.random()<.15?b[j]:w)+(this.random()<.8?normal(this.random)*sigma:0));}return newCar(this.track,weights,i);});this.elapsed=0;
 }
 get leader(){return this.population.reduce((a,b)=>b.score>a.score?b:a,this.population[0]);}
}

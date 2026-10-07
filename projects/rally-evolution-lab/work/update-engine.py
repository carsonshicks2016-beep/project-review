from pathlib import Path
p=Path('dist/engine.mjs');s=p.read_text()
s=s.replace('export const DT=1/30, INPUTS=12, POPULATION=48;', '''export const DT=1/30, INPUTS=12, POPULATION=48;
// Keep these units and values stable: exported policies depend on the physics and reward.
export const MAX_EPISODE_SECONDS=180, NEAREST_SEGMENT_WINDOW=12;
export const MAX_STEER_RADIANS=.5, STEERING_RESPONSE=.28;
export const INCOMPLETE_REWARD=9000, FINISH_REWARD=10000, LAP_SECOND_REWARD=40;
export const TRACE_STRIDE=5, TRACE_EVERY_STEPS=4;
export const ALGORITHMS=['genetic','hill','random'];''')
s=s.replace(' const n=480,points=[];let length=0;', ''' // The radius is at least 112-19-13-7=73 m. Increasing polar angle and a
 // positive radius keep the generated centerline simple (no crossings).
 // nearest() deliberately follows this ordered route, not arbitrary custom circuits.
 const n=480,points=[];let length=0;''')
s=s.replace('k=-12;k<=12','k=-NEAREST_SEGMENT_WINDOW;k<=NEAREST_SEGMENT_WINDOW')
s=s.replace('export function newCar(track,weights,id=0)', 'export function newCar(track,weights,id=0,{recordTrace=false}={})')
s=s.replace('trace:[],score:0','trace:recordTrace?new Float32Array((Math.ceil(MAX_EPISODE_SECONDS/DT/TRACE_EVERY_STEPS)+2)*TRACE_STRIDE):null,traceLength:0,score:0')
s=s.replace('(steering*.5-c.steer)*.28','(steering*MAX_STEER_RADIANS-c.steer)*STEERING_RESPONSE')
s=s.replace(' if(Math.round(c.time/DT)%4===0)c.trace.push([c.x,c.y,c.z,c.time]);',''' // Only a champion replay records a packed trace; population evaluations allocate none.
 if(c.trace&&Math.round(c.time/DT)%TRACE_EVERY_STEPS===0){
  const offset=c.traceLength*TRACE_STRIDE;
  c.trace[offset]=c.x;c.trace[offset+1]=c.y;c.trace[offset+2]=c.z;
  c.trace[offset+3]=c.time;c.trace[offset+4]=c.speed;c.traceLength++;
 }''')
s=s.replace('10000+(180-c.time)*40','FINISH_REWARD+(MAX_EPISODE_SECONDS-c.time)*LAP_SECOND_REWARD').replace('track.length*9000','track.length*INCOMPLETE_REWARD').replace('c.time>180','c.time>MAX_EPISODE_SECONDS')
start=s.index('export class Trainer')
s=s[:start]+'''export function evaluateDriver(track,weights,{recordTrace=false}={}){
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
  this.lines=[];this.stagnantGenerations=0;this.exploring=false;this.simulationSteps=0;
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
  this.champion={weights:[...weights],score:car.score,trace:car.trace,lap:car.finished?car.time:null,progress:car.bestProgress,generation};
  this.bestLap=this.champion.lap;
  if(car.trace&&car.finished)this.lines=[{trace:car.trace,generation,lap:car.time}];
  return car;
 }
 evolve(){
  const ranked=[...this.population].sort((a,b)=>b.score-a.score),best=ranked[0],finished=ranked.filter(car=>car.finished);
  this.totalLaps+=finished.length;this.evaluations+=POPULATION;
  const improved=!this.champion||best.score>this.champion.score;
  this.stagnantGenerations=improved?0:this.stagnantGenerations+1;
  if(improved){
   const trace=this.recordLines&&best.finished?evaluateDriver(this.track,best.weights,{recordTrace:true}).trace:null;
   this.champion={weights:[...best.weights],score:best.score,trace,lap:best.finished?best.time:null,progress:best.bestProgress,generation:this.generation};
   if(trace){this.lines.push({trace,generation:this.generation,lap:best.time});this.lines=this.lines.slice(-4);}
  }
  if(finished.length)this.bestLap=Math.min(this.bestLap??Infinity,...finished.map(car=>car.time));
  this.lastResult={generation:this.generation,lap:finished.length?finished[0].time:null,bestLap:this.bestLap,completion:Math.min(100,best.bestProgress/this.track.length*100),finishers:finished.length,evaluations:this.evaluations};
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
'''
p.write_text(s)

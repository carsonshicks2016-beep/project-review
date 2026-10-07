/**
 * Stride Lab — planar, actuated spring-loaded inverted pendulum (SLIP) with a limb.
 * SI units throughout. No joint-level human model is implied. The swing limb carries mass,
 * is moved by a bounded hip actuator, and cannot be teleported into place; leg length is a
 * state that retracts and extends in flight, so retraction costs reach; a foot that leaves
 * the friction cone slides rather than ending the trial instantly.
 * Symplectic Euler integration; compression-only spring; fixed no-slip contact.
 * The controller may place a massless swing foot and apply bounded axial force.
 * All airborne CoM acceleration is gravity plus aerodynamic drag.
 *
 * Contact is solved to the exact crossing time inside a step, so a stance never begins
 * with the leg already compressed. Every step also accumulates an explicit energy ledger
 * — actuator in; damper, drag and force-clamp out — leaving only the integrator's own
 * residual, which must converge to zero with the timestep. See `energyResidual`.
 */
export const VERSION='stride-slip-3.0';
export const G=9.81;
export const DT=1/500;
const LEG=1.05,START_Y=0.96;
// The limb is no longer weightless scenery. A fraction of body mass rides at the hip,
// offset horizontally by the swing coordinate, driven by a bounded hip actuator. Leg
// length is a state that only changes in flight, where the spring carries no load.
// LIMB_R: the limb's centre of mass sits at this fraction of the leg, so `swing` is tracked
// in centre-of-mass coordinates and the foot offset is swing/LIMB_R.
// HIP_MAX is the binding constraint, not HIP_W: 700 N at the limb's centre of mass is about
// 294 Nm of hip torque, within elite human range. Results are flat across 500-1200 N.
const SWING=0.12,LIMB_R=0.4,HIP_W=45,HIP_MAX=700,MAX_LEG_RATE=6.0,SLIP_C=6000,SWING_LIMIT=0.85;
// Hurdle height recalibrated with the limbed model: retraction is now a real leg-length
// change that costs reach, and a single planar leg length cannot represent hurdling
// technique. 0.60 m is impassable here, 0.45 m marginal; 0.35 m is a course this body
// can actually attempt. Still not an official competition layout.
export const DEFAULT_ENV={mass:75,friction:0.9,height:0.35,dt:DT};
export const PARAMS=[
 {key:'speed',label:'Target speed',min:3.5,max:11,unit:'m/s'},
 {key:'stiffness',label:'Leg stiffness',min:18000,max:48000,unit:'N/m'},
 {key:'placement',label:'Foot placement gain',min:0.005,max:0.065,unit:'s'},
 {key:'stanceTime',label:'Stance prediction',min:0.08,max:0.18,unit:'s'},
 {key:'push',label:'Axial drive',min:150,max:950,unit:'N'},
 {key:'jumpDrive',label:'Jump drive',min:400,max:2200,unit:'N'},
 {key:'lead',label:'Jump lookahead',min:0.13,max:0.5,unit:'s'},
 {key:'tuck',label:'Flight retraction',min:0.05,max:0.45,unit:'fraction'},
];
// Re-tuned for the limbed model: the previous hand-set reference assumed a weightless leg
// that teleported into place. This is the nearest round-numbered policy that still runs.
export const BASELINE=[0.25,0.35,0.45,0.50,0.45,0.30,0.45,0.50];
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function decode(p){return Object.fromEntries(PARAMS.map((q,i)=>[q.key,q.min+clamp(p[i],0,1)*(q.max-q.min)]));}
export function makeRng(seed=42){let v=seed>>>0;return()=>{v+=0x6D2B79F5;let t=v;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
export function gaussian(rng){return Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(2*Math.PI*rng());}
export function validateEnv(env={}){const e={...DEFAULT_ENV,...env};for(const [k,a,b] of [['mass',40,120],['friction',0.1,1.5],['height',0.1,1.2],['dt',0.0001,0.005]])if(!Number.isFinite(e[k])||e[k]<a||e[k]>b)throw new Error(`Invalid ${k}`);return e;}
export function makeState(policy=BASELINE,event='sprint',env={}){
 if(!Array.isArray(policy)||policy.length!==PARAMS.length||policy.some(x=>!Number.isFinite(x)))throw new Error('Invalid policy');
 if(!['sprint','hurdles'].includes(event))throw new Error('Invalid event');
 const e=validateEnv(env),p=decode(policy);
 // The start is at rest on a slightly compressed leg. That stored energy is an initial
 // condition, present in the first energy reading, not something contact conjures later.
 return {p,policy:[...policy],env:e,event,length:event==='sprint'?100:60,hurdles:event==='hurdles'?[12,21,30,39,48]:[],
 x:0,y:START_Y,vx:0,vy:0,t:0,legLength:LEG,restLength:LEG,foot:-0.16,trail:-0.42,swing:LIMB_R*0.16,swingRate:0,
 stance:true,leg:0,lastTakeoff:0,lastLanding:0,steps:0,
 jump:false,jumpedFor:-1,cleared:0,failed:false,finished:false,reason:'',forceX:0,forceY:0,axial:0,
 elastic:0,work:0,actuator:0,hip:0,dissipated:0,clamped:0,residual:0,contactCompression:0,friction:0,slipDistance:0,
 peakForce:0,peakHeight:START_Y,minHeight:START_Y,minLeg:LEG,plantLag:0,contactTime:0,flightTime:0,
 bodyRadius:0.12,frames:[],nextFrame:0,minClearance:null,slips:0,lowApex:0};
}
function nextHurdle(s){return s.hurdles.findIndex(x=>x>s.x-0.2);}
// Kinetic + gravitational + stored spring energy of the current configuration.
export function mechanical(s){
 const mt=s.env.mass,ml=SWING*mt,mb=mt-ml;let spring=0;
 if(s.stance){const c=Math.max(0,s.legLength-Math.hypot(s.x-s.foot,s.y));spring=0.5*s.p.stiffness*c*c;}
 // The swing limb rides at the body's height and translates horizontally by `swing`.
 const w=s.vx+s.swingRate;
 return 0.5*mt*s.vy*s.vy+0.5*mb*s.vx*s.vx+0.5*ml*w*w+mt*G*s.y+spring;
}
/** One fixed sub-step. Touchdown is decided by `step`, never here. */
function advance(s,dt){
 const {p,env:e}=s,mt=e.mass,ml=SWING*mt,mb=mt-ml,L=s.legLength,oldX=s.x,E0=mechanical(s);
 let fx=0,fy=0,ux=0,uy=0,spring=0,damping=0,motor=0,force=0,ideal=0,slip=0;
 const drag=0.26*s.vx*Math.abs(s.vx);
 if(s.stance){
  const dx=s.x-s.foot,len=Math.hypot(dx,s.y);ux=dx/len;uy=s.y/len;
  const radial=s.vx*ux+s.vy*uy;
  const i=nextHurdle(s),dist=i<0?Infinity:s.hurdles[i]-s.x;
  // A task-conditioned controller can increase axial drive on a pre-obstacle stance.
  if(i>=0&&i!==s.jumpedFor&&dist>0&&dist<Math.max(0.65,s.vx*p.lead+0.3)){
   s.jump=true;s.jumpedFor=i;
  }
  spring=p.stiffness*Math.max(0,L-len);
  damping=-75*radial;
  // Apply drive on extension, and briefly from the stationary start. Bounded positive mechanical power.
  motor=(radial>0||s.t<0.15)?p.push+(s.jump?p.jumpDrive:0):0;
  const powerLimit=s.jump?4200:2400;
  if(radial>0.02)motor=Math.min(motor,powerLimit/radial);
  ideal=spring+damping+motor;
  force=clamp(ideal,0,5.5*mt*G);
  if(len>=L&&radial>0&&s.t-s.lastLanding>0.015){
   // Takeoff: no leg force acts over this sub-step, so the ledger must see none either.
   // The foot just left becomes the limb that has to be swung forward for the next contact.
   s.stance=false;s.lastTakeoff=s.t;s.leg=1-s.leg;s.trail=s.foot-s.x;force=0;ideal=0;damping=0;motor=0;
  }
  if(s.stance){
   fx=force*ux;fy=force*uy;
   // The reaction on a massless point foot must lie along the leg axis, so the friction
   // condition |T| <= mu*N reduces to pure geometry: |x - foot| <= mu*y. Beyond it the
   // contact cannot be static. The foot slides instead of the trial dying instantly;
   // sliding widens the rake, so a real slip still runs away into lost support.
   const excess=Math.abs(dx)-e.friction*s.y;
   if(excess>0&&s.t-s.lastLanding>0.01){
    slip=-Math.sign(dx)*force*excess/(len*SLIP_C);
    s.slipDistance+=Math.abs(slip)*dt;
   }
   s.axial=force;s.contactTime+=dt;
  }else s.axial=0;
 }else{
  s.axial=0;s.flightTime+=dt;
  // Heel up while rising or clearing an obstacle, extend while falling, at a bounded rate.
  // The spring carries no load in flight, so this does no work — but a tucked leg has less
  // reach and cannot be planted until it extends again. That delay is what retraction costs.
  const target=(s.vy>0||s.jump)?s.restLength*(1-p.tuck):s.restLength;
  s.legLength+=clamp(target-s.legLength,-MAX_LEG_RATE*dt,MAX_LEG_RATE*dt);
 }
 // Bounded hip actuator drives the swing limb toward the commanded placement. It reacts on
 // the body, so repositioning the leg is no longer free, and the limb cannot teleport.
 const reach=LIMB_R*SWING_LIMIT*s.legLength;
 let hip=clamp(ml*(HIP_W*HIP_W*(LIMB_R*placement(s)-s.swing)-2*HIP_W*s.swingRate),-HIP_MAX,HIP_MAX);
 if((s.swing>=reach&&hip>0)||(s.swing<=-reach&&hip<0))hip=0;
 s.forceX=fx;s.forceY=fy;
 const ax=(fx-drag-hip)/mb,vx0=s.vx,vy0=s.vy,qd0=s.swingRate;
 s.vx=vx0+ax*dt;s.vy=vy0+(fy/mt-G)*dt;s.swingRate=qd0+(hip/ml-ax)*dt;
 s.x+=s.vx*dt;s.y+=s.vy*dt;s.swing+=s.swingRate*dt;s.foot+=slip*dt;s.t+=dt;
 // Midpoint velocities make the kinetic-energy change of this integrator exact, so the only
 // thing left in the residual is the position update's own discretisation error.
 const vxm=(vx0+s.vx)/2,vym=(vy0+s.vy)/2,qdm=(qd0+s.swingRate)/2,rm=ux*vxm+uy*vym;
 const dAct=motor*rm*dt,dDamp=-damping*rm*dt,dClamp=(ideal-force)*rm*dt,dDrag=drag*vxm*dt;
 const dHip=hip*qdm*dt,dFric=-spring*ux*slip*dt;
 s.actuator+=dAct+dHip;s.hip+=dHip;if(dAct>0)s.work+=dAct;if(dHip>0)s.work+=dHip;
 s.dissipated+=dDamp+dDrag;s.clamped+=dClamp;s.friction+=dFric;
 s.elastic=s.stance?0.5*p.stiffness*Math.max(0,s.legLength-Math.hypot(s.x-s.foot,s.y))**2:0;
 s.residual+=(mechanical(s)-E0)-dAct-dHip+dDamp+dDrag+dClamp+dFric;
 s.peakForce=Math.max(s.peakForce,Math.hypot(fx,fy)/(mt*G));
 s.peakHeight=Math.max(s.peakHeight,s.y);s.minHeight=Math.min(s.minHeight,s.y);
 if(!Number.isFinite(s.y)||s.y<0.5||s.y>5||s.vx < -0.8){s.failed=true;s.reason='Lost support';}
 for(let i=0;i<s.hurdles.length;i++){
  const hx=s.hurdles[i];
  // Lowest point is the foot of the limb at its actual length and angle -- the same geometry
  // the contact test uses. Retraction buys clearance through the leg state itself, at the cost
  // of reach, rather than through a separate envelope. No claim of full limb collision.
  if(Math.abs(s.x-hx)<0.13){
   const q=clamp(s.swing/LIMB_R,-s.legLength,s.legLength);
   const margin=(s.y-Math.sqrt(Math.max(0,s.legLength*s.legLength-q*q)))-e.height;
   s.minClearance=s.minClearance===null?margin:Math.min(s.minClearance,margin);
   if(margin<0){s.failed=true;s.reason=`Hit hurdle ${i+1}`;}
  }
  if(oldX<=hx+0.13&&s.x>hx+0.13&&!s.failed)s.cleared++;
 }
 if(s.x>=s.length&&!s.failed){s.finished=true;s.reason='Finished';}
 if(s.t>=30&&!s.finished){s.failed=true;s.reason='Time limit';}
 return s;
}
// Commanded horizontal offset for the swing foot. The hip tracks it; it is not teleported to.
function placement(s){const p=s.p;return clamp(s.vx*p.stanceTime/2+p.placement*(s.vx-p.speed),-0.30,0.58);}
/**
 * When does the body cross the leg-reach circle downward inside the next dt?
 * Solved against this integrator's own position update, y(h) = y + vy*h - G*h^2, using the
 * leg's current length and the swing limb's actual offset, so advancing by h lands on the
 * circle exactly and the stance starts uncompressed.
 */
function contactSolve(s,dt){
 const L=s.legLength,offset=clamp(s.swing/LIMB_R,-L,L),Y=Math.sqrt(Math.max(0,L*L-offset*offset)),d=s.y-Y;
 if(d>0){
  const disc=s.vy*s.vy+4*G*d;
  if(disc<0)return null;
  const h=(s.vy+Math.sqrt(disc))/(2*G);
  return h>0&&h<=dt?{h,offset,mode:'crossing'}:null;
 }
 // The body apexed inside the reach circle, so the limb cannot be planted where it is with a
 // straight leg at this height. Plant at the reachable distance rather than conjuring a
 // pre-compressed spring. Counted in `lowApexContacts` so the condition stays visible.
 return s.vy<=0?{h:0,offset,mode:'lowApex'}:null;
}
function land(s,c){
 // The limb is planted where it actually is, at the offset it has reached by this instant.
 // The leg can never be longer than the hip-to-foot distance at plant, so it flexes to fit
 // when the body is inside the reach circle and the stance then runs on the shorter leg.
 // min() makes the stance open at exactly zero compression without teleporting the foot.
 const off=clamp(s.swing/LIMB_R,-s.legLength,s.legLength);
 s.foot=s.x+off;
 s.legLength=Math.min(s.legLength,Math.hypot(off,s.y));
 // Recorded so the invariant is checkable from outside: this must stay at zero.
 s.contactCompression=Math.max(s.contactCompression,Math.max(0,s.legLength-Math.hypot(s.x-s.foot,s.y)));
 // The limb coordinate is reassigned to the leg that now trails. Swing kinetic energy depends
 // on the swing rate, not its position, so carrying the rate across makes the handover exactly
 // energy-neutral. Touchdown impact loss is therefore not modelled, rather than mismodelled.
 // How far the plant ended up from what the controller asked for. A model that teleported the
 // foot to the commanded placement would drive this to zero.
 s.plantLag=Math.max(s.plantLag,Math.abs(placement(s)-off));
 s.minLeg=Math.min(s.minLeg,s.legLength);
 s.swing=LIMB_R*clamp(s.trail,-SWING_LIMIT*s.legLength,SWING_LIMIT*s.legLength);
 s.stance=true;s.lastLanding=s.t;s.steps++;s.jump=false;
 if(c.mode==='lowApex')s.lowApex++;
}
export function step(s,dt=s.env.dt){
 if(s.failed||s.finished)return s;
 if(!s.stance){
  const c=contactSolve(s,dt);
  if(c){
   if(c.h>0)advance(s,c.h);
   if(s.failed||s.finished)return s;
   land(s,c);
   const rest=dt-c.h;
   if(rest>1e-12)advance(s,rest);
   return s;
  }
 }
 return advance(s,dt);
}
export function frame(s){
 const {mass:m}=s.env;
 return {t:s.t,x:s.x,y:s.y,vx:s.vx,vy:s.vy,fx:s.forceX,fy:s.forceY,stance:s.stance,foot:s.foot,leg:s.leg,
 jump:s.jump,tuck:s.p.tuck,swing:s.swing,steps:s.steps,cleared:s.cleared,kinetic:0.5*m*(s.vx*s.vx+s.vy*s.vy),potential:m*G*s.y,
 elastic:s.elastic,work:s.work,legLength:s.legLength,lastTakeoff:s.lastTakeoff};
}
export function runTrial(policy=BASELINE,event='sprint',env={},record=true){
 const s=makeState(policy,event,env);const frames=record?[frame(s)]:null;let next=1/60;
 while(!s.failed&&!s.finished){step(s);if(record&&s.t>=next){frames.push(frame(s));next+=1/60;}}
 if(record)frames.push(frame(s));
 const distance=clamp(s.x,0,s.length);
 // Finishing always outranks failure. Dense distance reward supports early exploration.
 const score=s.finished?250+(30-s.t)*8:(distance/s.length)*140+s.cleared*6;
 return {version:VERSION,event,env:{...s.env},policy:[...policy],distance,duration:s.t,finished:s.finished,reason:s.reason,
 cleared:s.cleared,totalHurdles:s.hurdles.length,hurdles:s.hurdles,score,peakForce:s.peakForce,peakHeight:s.peakHeight,
 positiveWork:s.work,actuatorWork:s.actuator,hipWork:s.hip,dissipated:s.dissipated,clampLoss:s.clamped,
 frictionLoss:s.friction,slipDistance:s.slipDistance,energyResidual:s.residual,
 maxContactCompression:s.contactCompression,lowApexContacts:s.lowApex,minHeight:s.minHeight,
 minLegLength:s.minLeg,plantLag:s.plantLag,restLength:s.restLength,slips:s.slips,meanSpeed:distance/s.t,steps:s.steps,minClearance:s.minClearance,frames};
}
/**
 * One randomized training condition. Mass, friction, hurdle height and — critically —
 * the timestep are all resampled, so a controller that only works at one discretization
 * cannot score well.
 */
export function sampleEnv(base,rng){
 const refine=[1,2,4][Math.floor(rng()*3)];
 return validateEnv({mass:base.mass*(0.9+0.2*rng()),friction:clamp(base.friction*(0.85+0.3*rng()),0.1,1.5),
  height:clamp(base.height+(rng()-0.5)*0.1,0.1,1.2),dt:base.dt/refine});
}
export function evaluateUnder(policy,scope,envs,record=false){
 const events=scope==='shared'?['sprint','hurdles']:[scope];
 const trials=[];
 for(const event of events)for(const env of envs)trials.push(runTrial(policy,event,env,record));
 // Equal weight over events and conditions; a weak event or a weak condition still counts.
 return {policy:[...policy],score:trials.reduce((a,t)=>a+t.score,0)/trials.length,trials};
}
export function evaluatePolicy(policy,scope,env={},record=false){return evaluateUnder(policy,scope,[env],record);}
/**
 * A fixed sample of training conditions, drawn from a stream of its own. Held out from the
 * per-generation draw and reused unchanged for the whole run, so scores from different
 * generations are directly comparable and the final answer is not whichever policy the
 * last generation's conditions happened to flatter.
 */
export function validationEnvs(base,seed,count=8){
 const rng=makeRng((seed^0x9E3779B9)>>>0);
 return Array.from({length:count},()=>sampleEnv(base,rng));
}
export function createSearch({scope='shared',env={},seed=42,start=BASELINE,population=32,samples=4,validation=8}={}){
 if(!['shared','sprint','hurdles'].includes(scope))throw new Error('Invalid scope');
 if(!Number.isInteger(samples)||samples<1||samples>16)throw new Error('Invalid samples');
 if(!Number.isInteger(validation)||validation<1||validation>32)throw new Error('Invalid validation');
 const e=validateEnv(env),events=scope==='shared'?2:1,check=validationEnvs(e,seed,validation);
 const seeded=evaluateUnder(start,scope,check);
 return {scope,env:e,seed,population,samples,rng:makeRng(seed),mean:[...start],sigma:PARAMS.map(()=>0.20),
 validation:check,incumbent:seeded,best:seeded,generation:0,evaluations:events*check.length,history:[]};
}
export function generation(search){
 const s=search,events=s.scope==='shared'?2:1;
 // One set of conditions for the whole generation (common random numbers), resampled every
 // generation. The incumbent is re-simulated under them, so elitism compares like with like
 // and a lucky draw cannot be banked permanently.
 const envs=Array.from({length:s.samples},()=>sampleEnv(s.env,s.rng));
 const candidates=[evaluateUnder(s.incumbent.policy,s.scope,envs)];
 for(let n=1;n<s.population;n++){
  const p=s.mean.map((v,i)=>clamp(v+s.sigma[i]*gaussian(s.rng),0,1));
  candidates.push(evaluateUnder(p,s.scope,envs));
 }
 candidates.sort((a,b)=>b.score-a.score);const elite=candidates.slice(0,Math.max(4,Math.floor(s.population*.2)));
 s.incumbent=candidates[0];
 s.mean=s.mean.map((v,i)=>0.2*v+0.8*elite.reduce((a,c)=>a+c.policy[i],0)/elite.length);
 s.sigma=s.sigma.map((v,i)=>Math.max(.025,.2*v+.8*Math.sqrt(elite.reduce((a,c)=>a+(c.policy[i]-s.mean[i])**2,0)/elite.length)));
 // The answer is chosen on the held-out conditions, never on the generation's own draw.
 const checked=evaluateUnder(s.incumbent.policy,s.scope,s.validation);
 if(checked.score>s.best.score)s.best=checked;
 s.generation++;s.evaluations+=s.population*events*s.samples+events*s.validation.length;
 const point={generation:s.generation,score:s.best.score,training:s.incumbent.score,
  mean:candidates.reduce((a,c)=>a+c.score,0)/candidates.length,evaluations:s.evaluations,conditions:s.samples};
 s.history.push(point);return {...point,policy:[...s.best.policy],trials:s.best.trials.map(t=>({...t,frames:undefined}))};
}
export function robustness(policy,env={}){
 const e=validateEnv(env);
 return [
  {name:'Reference',env:e},
  {name:'Half timestep',env:{...e,dt:e.dt/2}},
  {name:'Quarter timestep',env:{...e,dt:e.dt/4}},
  {name:'Eighth timestep',env:{...e,dt:e.dt/8}},
  {name:'Lower friction',env:{...e,friction:Math.max(.1,e.friction-.2)}},
  {name:'Mass −10%',env:{...e,mass:e.mass*.9}},
  {name:'Mass +10%',env:{...e,mass:e.mass*1.1}},
  {name:'Higher hurdles',env:{...e,height:Math.min(1.2,e.height+.1)}}
 ].map(c=>({name:c.name,env:c.env,trials:['sprint','hurdles'].map(v=>runTrial(policy,v,c.env,false))}));
}

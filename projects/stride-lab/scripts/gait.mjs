/**
 * Is it walking, or falling forward in a controlled way?
 *
 * Distance alone cannot tell those apart — a body that leans and shuffles covers ground,
 * and so does one that topples slowly. Locomotion means the feet alternate: each one
 * loads, unloads, and swings while the other carries. So this measures the contact
 * pattern, not just the displacement.
 */
import fs from 'node:fs';
import {kinematics,pointJacobian,NQ} from '../dist/body.js';
import {step,contactPoints,SOLE_POINTS,NC} from '../dist/dynamics.js';
import {reset,observe,act,upright,OBS,ACT,STAND} from '../dist/task.js';
import {makePolicy} from '../dist/policy.js';

export function loadPolicy(file){
 const art=JSON.parse(fs.readFileSync(file,'utf8'));
 const z={mean:Float64Array.from(art.norm.mean),std:Float64Array.from(art.norm.std)};
 return {art,policy:makePolicy(Float64Array.from(art.w),art.obs,art.act,z)};
}
// A penalty contact chatters: the sole force crosses the load threshold many times per
// gait cycle, so counting raw edges counts the integrator, not the body. Debounce it —
// a foot only enters stance after carrying load continuously for DEBOUNCE, and only
// enters swing after being unloaded that long. Real stance and swing last hundreds of
// milliseconds, so 40 ms cannot merge two steps but does reject chatter.
const DEBOUNCE=0.040;
function tracker(){return {state:true,pending:true,since:0,steps:0};}
function feed(f,loaded,t){
 if(loaded!==f.pending){f.pending=loaded;f.since=t;}
 else if(loaded!==f.state&&t-f.since>=DEBOUNCE){f.state=loaded;if(loaded)f.steps++;}
 return f.state;
}
/** One trial, recording which soles carry load over time. */
export function gait(policy,{env={},pose=STAND,seconds=10,control=20,shove=0,shoveAt=3}={}){
 const b=reset(env,pose);
 const obs=new Float64Array(OBS),a=new Float64Array(ACT);
 const steps=Math.round(seconds/b.env.dt);
 const fl=tracker(),fr=tracker();
 let both=0,air=0,samples=0,applied=false,live=0,peakSpeed=0;
 // A planted foot is stationary while it carries load. Sliding under load is skating,
 // which covers ground at walking speed without taking a single step.
 const jac=new Float64Array(2*NQ),SOLE_LINK=[3,3,6,6];
 let loaded=0,slip=0,slipMax=0;
 for(let n=0;n<steps;n++){
  if(n%control===0){observe(b,obs);policy(obs,a);act(b,a);}
  if(shove&&!applied&&b.t>=shoveAt){b.qd[0]+=shove/b.env.mass;applied=true;}
  step(b);
  if(!upright(b))break;
  live=b.t;peakSpeed=Math.max(peakSpeed,b.qd[0]);
  const k=kinematics(b.q,b.qd,b.seg,b.k);contactPoints(b,k);
  for(let i=0;i<SOLE_POINTS.length;i++){
   const c=SOLE_POINTS[i];
   if(b.fn[c]<=1)continue;
   pointJacobian(k,b.seg,SOLE_LINK[i],b.cx[c],b.cy[c],jac);
   let vx=0;for(let j=0;j<NQ;j++)vx+=jac[j]*b.qd[j];
   loaded++;slip+=Math.abs(vx);slipMax=Math.max(slipMax,Math.abs(vx));
  }
  const L=feed(fl,b.fn[SOLE_POINTS[0]]>1||b.fn[SOLE_POINTS[1]]>1,b.t);
  const R=feed(fr,b.fn[SOLE_POINTS[2]]>1||b.fn[SOLE_POINTS[3]]>1,b.t);
  if(L&&R)both++;else if(!L&&!R)air++;
  samples++;
 }
 const dist=b.q[0];
 return {live,fell:!upright(b),distance:dist,speed:dist/Math.max(live,1e-9),peakSpeed,
  stepsL:fl.steps,stepsR:fr.steps,steps:fl.steps+fr.steps,
  doubleSupport:samples?both/samples:0,flight:samples?air/samples:0,
  cadence:live>0?(fl.steps+fr.steps)/live:0,height:b.q[1],pitch:b.q[2],
  slip:loaded?slip/loaded:0,slipMax};
}
export function verdict(r){
 // Alternating debounced stance, both feet contributing, feet planted while loaded.
 const alternating=Math.min(r.stepsL,r.stepsR)>=3&&
   Math.min(r.stepsL,r.stepsR)/Math.max(1,Math.max(r.stepsL,r.stepsR))>0.5;
 if(r.fell&&r.live<3)return 'fell early';
 if(r.distance<0.5)return 'stationary';
 if(r.slip>0.25)return 'skating';
 if(!alternating)return 'shuffling or toppling';
 return r.fell?'walked, then fell':'walked';
}
if(import.meta.url===`file://${process.argv[1]}`){
 const file=process.argv[2]||'outputs/walk.json';
 const {art,policy}=loadPolicy(file);
 console.log(`${file}: ${art.task}, ${art.iters} iterations, return ${art.best.toFixed(2)}\n`);
 const row=(name,r)=>console.log(`  ${name.padEnd(22)} ${verdict(r).padEnd(22)}`+
  `${r.distance.toFixed(2).padStart(6)} m  ${r.speed.toFixed(2)} m/s  `+
  `steps ${String(r.stepsL)}L/${String(r.stepsR)}R  cadence ${r.cadence.toFixed(2)}/s  `+
  `double ${(100*r.doubleSupport).toFixed(0)}%  slip ${r.slip.toFixed(2)} m/s  ${r.live.toFixed(1)}s`);
 console.log('Nominal');
 row('reference',gait(policy));
 console.log('\nOther bodies, floors and timesteps');
 for(const [n,env] of [['mass 60 kg',{mass:60}],['mass 95 kg',{mass:95}],
   ['friction 0.5',{friction:0.5}],['soft floor',{kn:5e3,dn:380}],
   ['timestep 0.35 ms',{dt:3.5e-4}],['timestep 0.25 ms',{dt:2.5e-4}],['timestep 0.15 ms',{dt:1.5e-4}]])
  row(n,gait(policy,{env}));
 console.log('\nShoved while moving');
 for(const s of [40,80])row(`shove ${s} N·s`,gait(policy,{shove:s}));
}

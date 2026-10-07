import fs from 'node:fs';
import path from 'node:path';
import {balance,sampleCondition,validationSet,OBS,ACT,STAND} from '../dist/task.js';
import {makePolicy,train,observeNorm,makeRng} from '../dist/policy.js';

const [itersText='300',seedText='42',out='outputs/balance.json']=process.argv.slice(2);
const iters=Number(itersText),seed=Number(seedText);
if(!Number.isInteger(iters)||iters<1||!Number.isInteger(seed))throw new Error('Expected iterations, seed, output path');

// Conditions are resampled every generation and shared across that generation's probes, so
// their scores are comparable. The champion is chosen on a fixed held-out set it never
// trains against — a policy that only survives the exact start it saw scores badly there.
const SAMPLES=3;
const rng=makeRng(seed^0x5f3759df);
const held=validationSet(seed,12,makeRng);
let conditions=Array.from({length:SAMPLES},()=>sampleCondition(rng));

function mean(xs){return xs.reduce((a,b)=>a+b,0)/xs.length;}
function rollout(w,z,evaluate=false){
 const base=makePolicy(w,OBS,ACT,z);
 if(evaluate)return mean(held.map(c=>balance(base,{condition:c}).reward));
 const policy=(obs,out)=>{observeNorm(z,obs);base(obs,out);};
 return mean(conditions.map(c=>balance(policy,{condition:c}).reward));
}
const t0=Date.now();
const log=[];
const {w,z,best,evals}=train(rollout,{nObs:OBS,nAct:ACT,seed,iters,
 onIterStart:()=>{conditions=Array.from({length:SAMPLES},()=>sampleCondition(rng));},
 onIter:({iter,reward,best,evals,mean})=>{
  log.push({iter,reward,best,evals});
  if(iter===1||iter%20===0||iter===iters)
   console.log(`iter ${String(iter).padStart(4)}  return ${reward.toFixed(1).padStart(8)}  best ${best.toFixed(1).padStart(8)}  probe mean ${mean.toFixed(1).padStart(8)}  ${evals} rollouts`);
 }});
const final=makePolicy(w,OBS,ACT,z);
const check=balance(final);
const heldRuns=held.map(c=>balance(final,{condition:c}));
const stood=heldRuns.filter(r=>!r.fell).length;
console.log(`\nnominal start: ${check.fell?'FELL at '+check.seconds.toFixed(2):'STOOD '+check.seconds.toFixed(2)} s   height ${check.height.toFixed(3)}  pitch ${check.pitch.toFixed(3)}  drift ${check.drift.toFixed(3)} m`);
console.log(`held-out: stood ${stood}/${held.length}   mean upright ${(heldRuns.reduce((a,r)=>a+r.seconds,0)/heldRuns.length).toFixed(2)} s   mean return ${(heldRuns.reduce((a,r)=>a+r.reward,0)/heldRuns.length).toFixed(2)}`);
console.log(`${evals} rollouts in ${((Date.now()-t0)/1000).toFixed(0)} s`);
const artifact={task:'balance',seed,iters,evals,best,samples:SAMPLES,
 stood:!check.fell,seconds:check.seconds,heldOut:{stood,of:held.length},
 obs:OBS,act:ACT,pose:STAND,w:[...w],norm:{mean:[...z.mean],std:[...z.std],n:z.n},history:log.filter((h,i)=>i===0||i===log.length-1||(h.iter%10===0))};
fs.mkdirSync(path.dirname(out),{recursive:true});
fs.writeFileSync(out,JSON.stringify(artifact,null,1));
// The viewer is served from dist/, so the policy has to sit next to it.
const served=new URL('../dist/balance.json',import.meta.url);
fs.writeFileSync(served,JSON.stringify(artifact));
console.log(`Saved ${out} and dist/balance.json`);

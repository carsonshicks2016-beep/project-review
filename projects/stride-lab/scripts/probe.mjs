/**
 * Does the policy balance, or has it just found one pose that happens not to fall over?
 * A static pose survives a clean start and nothing else. Balance means rejecting a shove,
 * a different body, a slipperier floor and a start it was not trained from.
 */
import fs from 'node:fs';
import {step} from '../dist/dynamics.js';
import {reset,observe,act,upright,OBS,ACT,STAND} from '../dist/task.js';
import {makePolicy} from '../dist/policy.js';

export function loadPolicy(file){
 const art=JSON.parse(fs.readFileSync(file,'utf8'));
 const z={mean:Float64Array.from(art.norm.mean),std:Float64Array.from(art.norm.std)};
 return {art,policy:makePolicy(Float64Array.from(art.w),art.obs,art.act,z)};
}
/** Run one trial, optionally shoving the pelvis with a horizontal impulse. */
export function trial(policy,{env={},pose=STAND,shove=0,shoveAt=2.0,seconds=8,control=20}={}){
 const b=reset(env,pose);
 const obs=new Float64Array(OBS),a=new Float64Array(ACT);
 const steps=Math.round(seconds/b.env.dt);
 let live=0,applied=false;
 for(let n=0;n<steps;n++){
  if(n%control===0){observe(b,obs);policy(obs,a);act(b,a);}
  if(shove&&!applied&&b.t>=shoveAt){b.qd[0]+=shove/b.env.mass;applied=true;}
  step(b);
  if(!upright(b))break;
  live=b.t;
 }
 return {live,fell:!upright(b),height:b.q[1],pitch:b.q[2],drift:b.q[0],seconds};
}
/**
 * The cases a memorised trajectory cannot survive but a controller can: being pushed,
 * being a different body, standing on a different floor, being integrated more finely,
 * and starting from somewhere it was never trained.
 */
export function cases(){
 const out=[{group:'clean',name:'reference',opts:{}}];
 for(const s of [20,40,60,80,120,160])out.push({group:'shove',name:`shove ${s} N·s`,opts:{shove:s}});
 for(const [name,env] of [['mass 60 kg',{mass:60}],['mass 95 kg',{mass:95}],
   ['friction 0.5',{friction:0.5}],['friction 0.25',{friction:0.25}],
   ['soft floor 5 kN/m',{kn:5e3,dn:380}],['stiff floor 30 kN/m',{kn:3e4,dn:930}],
   ['timestep 0.25 ms',{dt:2.5e-4}],['timestep 1 ms',{dt:1e-3}]])out.push({group:'world',name,opts:{env}});
 for(const [name,d] of [['leaning forward',{2:0.12}],['leaning back',{2:-0.12}],
   ['deeper knee bend',{4:-0.30,7:-0.30}],['staggered stance',{3:0.25,6:-0.25}]]){
  const pose=[...STAND];for(const k in d)pose[k]+=d[k];
  out.push({group:'pose',name,opts:{pose}});
 }
 return out;
}
export function probe(policy){
 const rows=cases().map(c=>({...c,...trial(policy,c.opts)}));
 const by=g=>rows.filter(r=>r.group===g);
 return {rows,
  passed:rows.filter(r=>!r.fell).length,total:rows.length,
  groups:Object.fromEntries(['clean','shove','world','pose'].map(g=>
   [g,{passed:by(g).filter(r=>!r.fell).length,of:by(g).length}]))};
}

if(import.meta.url===`file://${process.argv[1]}`){
 const file=process.argv[2]||'outputs/balance.json';
 const {art,policy}=loadPolicy(file);
 const r=probe(policy);
 console.log(`${file}: ${art.task}, ${art.iters} iterations, return ${art.best.toFixed(2)}\n`);
 let group=null;
 const titles={clean:'Clean start',shove:'Horizontal impulse at the pelvis, 2 s in',
  world:'A different body and a different floor',pose:'Started from a pose it was not trained on'};
 for(const row of r.rows){
  if(row.group!==group){group=row.group;console.log(`\n${titles[group]}`);}
  console.log(`  ${row.name.padEnd(30)} `+
   (row.fell?('fell at '+row.live.toFixed(2)+' s').padEnd(18):('stood '+row.seconds+' s').padEnd(18))+
   `height ${row.height.toFixed(3)}  pitch ${(row.pitch*57.3).toFixed(1)}°  drift ${row.drift.toFixed(3)} m`);
 }
 console.log(`\nsurvived ${r.passed}/${r.total}   `+
  Object.entries(r.groups).map(([g,v])=>`${g} ${v.passed}/${v.of}`).join('   '));
}

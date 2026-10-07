import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {balance,validationSet,OBS,ACT,STAND} from '../dist/task.js';
import {makePolicy,makeRng} from '../dist/policy.js';
import {loadPolicy,trial,probe} from '../scripts/probe.mjs';

const FILE=new URL('../dist/balance.json',import.meta.url);
const art=JSON.parse(fs.readFileSync(FILE));
const {policy}=loadPolicy(FILE);
const zero=(o,out)=>out.fill(0);

test('the shipped policy is a complete, self-describing artifact',()=>{
 assert.equal(art.task,'balance');
 assert.equal(art.obs,OBS);assert.equal(art.act,ACT);
 assert.equal(art.w.length,OBS*ACT);
 assert.ok(art.w.every(Number.isFinite));
 assert.equal(art.norm.mean.length,OBS);assert.equal(art.norm.std.length,OBS);
 assert.ok(art.norm.std.every(s=>s>=1e-3));
 assert.ok(art.evals>1000&&Number.isFinite(art.best));
 assert.ok(art.history.length>2&&art.history.every((h,i,a)=>i===0||h.best>=a[i-1].best),
  'held-out best must be monotone: the champion is only replaced when it improves');
});

// A policy trained from one deterministic start can score a perfect return and still be an
// open-loop torque sequence. The probe is what separates the two: shoves, other bodies,
// other floors, other timesteps, and poses never trained from.
test('the shipped policy survives the perturbation probe',()=>{
 const r=probe(policy);
 assert.ok(r.passed>=16,`survived only ${r.passed}/${r.total}`);
 // Four poses it never trained from. The memorised policy this replaced scored 0/4 here,
 // falling within half a second of every one; this is the clearest memorisation tell.
 assert.equal(r.groups.pose.passed,r.groups.pose.of,'failed a pose it was not trained from');
 assert.equal(r.groups.clean.passed,1);
 assert.ok(r.groups.world.passed>=7,`only ${r.groups.world.passed}/${r.groups.world.of} bodies and floors`);
 assert.ok(r.groups.shove.passed>=4,`only ${r.groups.shove.passed}/${r.groups.shove.of} shoves`);
});

// The tell that caught the previous policy: it fell when the integrator was refined, which
// means it was exploiting one discretisation rather than controlling a body.
test('refining the timestep does not break the policy',()=>{
 for(const dt of [5e-4,3.5e-4,2.5e-4,1.5e-4,1e-4]){
  const r=trial(policy,{env:{dt}});
  assert.ok(!r.fell,`fell at dt=${dt} after ${r.live.toFixed(2)}s`);
  assert.ok(r.height>0.65,`sagged to ${r.height} at dt=${dt}`);
 }
});

test('it stands across every held-out condition, and an unactuated body does not',()=>{
 const held=validationSet(art.seed,12,makeRng);
 const mine=held.map(c=>balance(policy,{condition:c}));
 const none=held.map(c=>balance(zero,{condition:c}));
 assert.equal(mine.filter(r=>!r.fell).length,held.length);
 assert.equal(none.filter(r=>!r.fell).length,0);
 const up=r=>r.reduce((a,x)=>a+x.seconds,0)/r.length;
 assert.ok(up(mine)>5.9,`mean upright only ${up(mine).toFixed(2)}s`);
 assert.ok(up(none)<0.6);
 // The conditions really do differ: several bodies, floors and timesteps.
 assert.ok(new Set(held.map(c=>c.env.dt)).size>1);
 assert.ok(Math.max(...held.map(c=>c.env.mass))-Math.min(...held.map(c=>c.env.mass))>10);
});

test('it stands still rather than drifting or leaning',()=>{
 const r=trial(policy,{seconds:8});
 assert.ok(!r.fell);
 assert.ok(Math.abs(r.drift)<0.25,`drifted ${r.drift} m`);
 assert.ok(Math.abs(r.pitch)<0.15,`leaning ${r.pitch} rad`);
 assert.ok(r.height>0.70,`crouched to ${r.height} m`);
});

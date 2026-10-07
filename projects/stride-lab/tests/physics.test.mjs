import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {BASELINE,DEFAULT_ENV,DT,G,makeState,step,runTrial,createSearch,generation,robustness,decode,PARAMS,
        sampleEnv,validationEnvs,makeRng,validateEnv} from '../dist/physics.js';
const EXAMPLE=JSON.parse(fs.readFileSync(new URL('../dist/example.json',import.meta.url)));
const CASES=[[BASELINE,'sprint'],[BASELINE,'hurdles'],[EXAMPLE.policy,'sprint'],[EXAMPLE.policy,'hurdles']];

test('airborne motion has gravity and drag, with no ground propulsion',()=>{
 const s=makeState();Object.assign(s,{stance:false,y:2,vy:2,vx:5,t:1});
 const oldVy=s.vy,oldVx=s.vx;step(s);
 assert.equal(s.forceX,0);assert.equal(s.forceY,0);assert.ok(Math.abs(s.vy-(oldVy-G*DT))<1e-12);assert.ok(s.vx<oldVx);
 // The hip keeps working on the swing limb in flight; the leg cannot push on nothing.
 assert.equal(s.actuator-s.hip,0);
});
test('baseline sprint is a complete physical trajectory from rest',()=>{
 const t=runTrial();assert.ok(t.finished);assert.equal(t.frames[0].vx,0);assert.equal(t.frames[0].vy,0);assert.equal(t.distance,100);
 assert.ok(t.frames.some(f=>f.stance));assert.ok(t.frames.some(f=>!f.stance));assert.ok(t.frames.every(f=>Number.isFinite(f.x)&&f.fy>=0));assert.ok(t.positiveWork>0);
 assert.ok(Math.abs(t.duration-23.006)<DT);assert.ok(t.peakForce<=5.5+1e-10);
});
test('a hurdle collision terminates and is never reported as a finish',()=>{
 const t=runTrial(BASELINE,'hurdles');assert.equal(t.finished,false);assert.equal(t.cleared,0);assert.match(t.reason,/Hit hurdle/);assert.ok(t.distance<12);assert.ok(t.minClearance<0);
});

// Contact is solved to the crossing time, so a stance can never open with the spring already
// loaded. A non-zero value here is free energy entering the model at every touchdown.
test('no stance ever begins with a compressed leg',()=>{
 for(const [policy,event] of CASES)for(const dt of [DT*2,DT,DT/2,DT/4]){
  const t=runTrial(policy,event,{dt},false);
  assert.ok(t.maxContactCompression<1e-9,`${event} dt=${dt} opened a stance at ${t.maxContactCompression} m of compression`);
 }
});

// The ledger must account for every joule the body gains: actuator in, damper/drag/clamp out.
// What is left is this integrator's discretisation error, which has to vanish with the timestep.
test('the energy ledger closes and its residual converges with the timestep',()=>{
 for(const [policy,event] of CASES){
  const coarse=runTrial(policy,event,{dt:DT},false),fine=runTrial(policy,event,{dt:DT/4},false);
  const share=t=>Math.abs(t.energyResidual)/Math.abs(t.actuatorWork);
  assert.ok(share(coarse)<0.005,`${event} leaves ${(100*share(coarse)).toFixed(3)}% of actuator work unexplained`);
  assert.ok(Math.abs(fine.energyResidual)<Math.abs(coarse.energyResidual),`${event} residual did not shrink with dt`);
  assert.ok(fine.clampLoss>=-1e-9&&coarse.clampLoss>=-1e-9);
  assert.ok(Number.isFinite(coarse.hipWork)&&Number.isFinite(coarse.frictionLoss));
 }
});

// The limb has mass, a bounded hip actuator and a real position. It is never moved to the
// commanded placement -- the foot is planted wherever the limb has actually reached. A model
// that teleported the swing foot would drive this lag to zero.
test('the swing limb is planted where it is, not where it was commanded',()=>{
 for(const [policy,event] of CASES){
  const t=runTrial(policy,event,{},false);
  assert.ok(t.plantLag>0.02,`${event} planted within ${t.plantLag} m of the command: the limb is being teleported`);
  assert.ok(t.plantLag<1.2);
 }
});

// Retraction is a leg-length state, not a clearance envelope. It buys hurdle clearance by
// shortening the leg, which costs reach and delays the next contact.
test('retraction shortens the leg itself and never lengthens it past rest',()=>{
 for(const [policy,event] of CASES){
  const t=runTrial(policy,event,{},false);
  assert.ok(t.minLegLength<t.restLength,`${event} never retracted`);
  assert.ok(t.minLegLength>0.2*t.restLength);
  assert.ok(t.maxContactCompression<1e-9);
 }
 // A tucked leg reaches less far, so clearance is paid for rather than granted.
 const low=runTrial(BASELINE,'hurdles',{height:0.2},false),high=runTrial(BASELINE,'hurdles',{height:0.6},false);
 assert.ok(low.distance>=high.distance);
});

// Leaving the friction cone slides the foot and dissipates energy; it no longer kills the
// trial outright. Less grip must mean more sliding and more loss, not a binary outcome.
test('a foot that leaves the friction cone slides instead of ending the trial',()=>{
 const runs=[0.9,0.3,0.2,0.12].map(friction=>runTrial(BASELINE,'sprint',{friction},false));
 assert.ok(runs.every(r=>r.finished),'sliding should degrade the run, not end it');
 for(let i=1;i<runs.length;i++){
  assert.ok(runs[i].slipDistance>=runs[i-1].slipDistance,'less grip should not slide less');
  assert.ok(runs[i].frictionLoss>=runs[i-1].frictionLoss-1e-9);
 }
 assert.ok(runs.at(-1).slipDistance>0.05,'the lowest-grip run never slid');
 assert.ok(runs[0].slipDistance===0&&runs[0].frictionLoss===0,'full grip should not slide at all');
 assert.ok(runs.every(r=>r.reason!=='Foot slipped'));
});

// The shipped result must be a property of the model, not of the timestep it was trained at.
test('the saved sprint result does not depend on the timestep',()=>{
 const times=[DT,DT/2,DT/4,DT/8].map(dt=>{const t=runTrial(EXAMPLE.policy,'sprint',{dt},false);assert.ok(t.finished);return t.duration;});
 assert.ok(Math.max(...times)-Math.min(...times)<0.1,`sprint time spread ${Math.max(...times)-Math.min(...times)} s across timesteps`);
});

test('training conditions are randomised, reproducible, and leave the base untouched',()=>{
 const base={...DEFAULT_ENV};
 const a=validationEnvs(base,42),b=validationEnvs(base,42),c=validationEnvs(base,7);
 assert.deepEqual(a,b);assert.notDeepEqual(a,c);assert.deepEqual(base,DEFAULT_ENV);
 assert.ok(a.some(e=>e.dt<DEFAULT_ENV.dt),'no condition refines the timestep');
 for(const e of a){assert.deepEqual(e,validateEnv(e));assert.ok(e.mass>=40&&e.mass<=120&&e.dt>=DEFAULT_ENV.dt/4-1e-12);}
 const rng=makeRng(1);assert.deepEqual(sampleEnv(base,makeRng(1)),sampleEnv(base,makeRng(1)));rng();
});
test('fixed seeds reproduce search, and the answer is chosen on held-out conditions',()=>{
 const a=createSearch(),b=createSearch();
 for(let i=0;i<5;i++)assert.deepEqual(generation(a),generation(b));
 // 8 held-out conditions to seed the champion, then 32 candidates x 2 events x 4 conditions
 // plus one held-out check of the incumbent, every generation.
 assert.equal(a.evaluations,2*8+5*(32*2*4+2*8));
 assert.ok(a.history.every((h,i)=>i===0||h.score>=a.history[i-1].score),'held-out score regressed');
 assert.ok(a.history.some(h=>h.training!==h.score),'training and held-out scores should differ');
});
test('saved shared policy is reproducible and beats the baseline sprint',()=>{
 const s=createSearch({scope:EXAMPLE.scope,seed:EXAMPLE.seed,env:EXAMPLE.trainingEnv});
 for(let i=0;i<EXAMPLE.generations;i++)generation(s);
 assert.deepEqual(s.best.policy,EXAMPLE.policy);assert.equal(s.evaluations,EXAMPLE.evaluations);
 const sprint=runTrial(EXAMPLE.policy,'sprint',DEFAULT_ENV,false);
 assert.ok(sprint.finished);assert.ok(sprint.duration<runTrial(BASELINE,'sprint',{},false).duration);
 // Documented limitation: under randomised conditions the shared controller does not clear the
 // hurdle course. The README records this; a change that fixes it must update both.
 assert.equal(runTrial(EXAMPLE.policy,'hurdles',DEFAULT_ENV,false).finished,false);
});
test('sensitivity tests retain failures, do not mutate the reference environment',()=>{
 const e={...DEFAULT_ENV};const r=robustness(EXAMPLE.policy,e);
 assert.equal(r.length,8);assert.deepEqual(e,DEFAULT_ENV);assert.equal(r[1].env.dt,DT/2);assert.equal(r[3].env.dt,DT/8);
 assert.ok(r.every(v=>v.trials[0].finished),'the sprint should survive every sensitivity case');
 assert.ok(r.some(v=>v.trials.some(t=>!t.finished)));
});
test('invalid parameters and inputs fail explicitly',()=>{
 assert.throws(()=>runTrial([0,0]));assert.throws(()=>runTrial(BASELINE,'vault'));assert.throws(()=>runTrial(BASELINE,'sprint',{mass:0}));
 assert.throws(()=>createSearch({scope:'unknown'}));assert.throws(()=>createSearch({samples:0}));assert.throws(()=>createSearch({validation:99}));
 const d=decode(PARAMS.map(()=>2));assert.equal(d.speed,11);
});

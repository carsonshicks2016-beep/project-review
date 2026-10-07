import assert from 'node:assert/strict';
import test from 'node:test';
import {QJ} from '../dist/body.js';
import {step} from '../dist/dynamics.js';
import {reset,observe,act,upright,balance,TORQUE,OBS,ACT,STAND} from '../dist/task.js';
import {makePolicy,train,makeNorm,observeNorm,freezeNorm,makeRng} from '../dist/policy.js';

const zero=(o,out)=>out.fill(0);

test('the observation is finite, correctly sized, and reports sole loading',()=>{
 const b=reset();const o=observe(b);
 assert.equal(o.length,OBS);
 assert.ok([...o].every(Number.isFinite));
 assert.equal(o[0],b.q[2]);assert.equal(o[18],b.q[1]);
 for(let i=0;i<200;i++)step(b);
 const o2=observe(b);
 assert.ok([21,22,23,24].some(i=>o2[i]===1),'a body resting on the ground should load a sole');
 assert.ok([21,22,23,24].every(i=>o2[i]===0||o2[i]===1));
});

// The policy commands a fraction of each joint's limit and cannot exceed it, whatever it
// asks for. These are peak torques for a 75 kg adult, not free actuators.
test('commanded torques are clamped to human joint limits',()=>{
 const b=reset();
 for(const mag of [0.5,1,50,-50]){
  act(b,new Float64Array(ACT).fill(mag));
  for(let j=0;j<ACT;j++)assert.ok(Math.abs(b.tau[QJ+j])<=TORQUE[j]+1e-9);
 }
 act(b,new Float64Array(ACT).fill(3));
 for(let j=0;j<ACT;j++)assert.equal(b.tau[QJ+j],TORQUE[j]);
 assert.ok(TORQUE.every(t=>t>0&&t<=200));
});

// Nothing holds this body up. With no torques it must collapse almost immediately, which
// is what makes a policy that stands a real result rather than a default.
test('an unactuated body cannot stand',()=>{
 const r=balance(zero);
 assert.ok(r.fell);
 assert.ok(r.seconds<1.0,`stayed up ${r.seconds}s with no torques`);
 const b=reset();
 assert.ok(upright(b));
 b.q[1]=0.4;assert.ok(!upright(b));
 b.q[1]=STAND[1];b.q[2]=1.2;assert.ok(!upright(b));
});

test('observation normalisation tracks the mean and spread of what was seen',()=>{
 const z=makeNorm(3),rng=makeRng(5);
 for(let i=0;i<4000;i++)observeNorm(z,[2+3*(rng()-0.5)*3.46,0,10]);
 freezeNorm(z);
 assert.ok(Math.abs(z.mean[0]-2)<0.15);assert.ok(Math.abs(z.std[0]-3)<0.3);
 assert.ok(Math.abs(z.mean[2]-10)<1e-9);
 assert.ok(z.std[1]>=1e-3&&z.std[2]>=1e-3,'a constant channel must not divide by zero');
});

test('the search is reproducible from its seed and improves a known objective',()=>{
 // A quadratic with a known optimum: cheap enough to assert on, same code path as balance.
 const target=[0.4,-0.9,0.25,0.7];
 const roll=w=>-target.reduce((a,t,i)=>a+(w[i]-t)**2,0);
 const opts={nObs:4,nAct:1,iters:60,dirs:8,keep:4,step:0.15,noise:0.08};
 const a=train(roll,{...opts,seed:7}),b=train(roll,{...opts,seed:7});
 assert.deepEqual([...a.w],[...b.w]);
 assert.equal(a.evals,b.evals);
 assert.ok(a.best>roll(new Float64Array(4)),'search did not improve on the starting point');
 assert.ok(a.best>-0.05,`converged only to ${a.best}`);
 const c=train(roll,{...opts,seed:8});
 assert.notDeepEqual([...a.w],[...c.w]);
});

test('a policy is a pure function of the observation it is given',()=>{
 const w=new Float64Array(OBS*ACT).fill(0.01);
 const p=makePolicy(w,OBS,ACT);
 const o=observe(reset()),x=new Float64Array(ACT),y=new Float64Array(ACT);
 p(o,x);p(o,y);
 assert.deepEqual([...x],[...y]);
 assert.ok([...x].every(Number.isFinite));
});

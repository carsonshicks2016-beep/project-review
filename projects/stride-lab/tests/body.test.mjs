import assert from 'node:assert/strict';
import test from 'node:test';
import {NQ,LINKS,segments,kinBuffers,kinematics} from '../dist/body.js';
import {makeBody,step,energy,contactPoints,limitExcess,NC,DEFAULT_ENV} from '../dist/dynamics.js';

const drop=(dt,audit=true)=>{
 const b=makeBody({dt});b.audit=audit;
 b.q[1]=1.0;b.q[2]=0.15;b.q[3]=0.3;b.q[6]=-0.3;b.q[4]=-0.4;b.q[7]=-0.2;
 while(b.t<3)step(b);
 return b;
};

test('segment masses are an anthropometric partition of the body',()=>{
 for(const m of [55,75,110]){
  const seg=segments(m);
  assert.equal(seg.length,9);
  assert.ok(Math.abs(seg.reduce((a,s)=>a+s.m,0)-m)<1e-9);
  assert.ok(seg.every(s=>s.m>0&&s.I>0&&s.len>0));
 }
});

// The dynamics are exact for this topology. With nothing touching the body but gravity,
// mechanical energy is conserved up to the integrator's own first-order error, which has
// to halve when the timestep halves.
test('free flight conserves energy, and the drift is first order in the timestep',()=>{
 const drift=[1e-3,5e-4,2.5e-4].map(dt=>{
  const b=makeBody({dt});b.audit=true;
  // Pose and rates chosen to keep every joint off its stops through the whole second.
  // Free ankles still graze theirs by a few thousandths of a degree, which is asserted
  // to be negligible rather than chased to zero.
  b.q[1]=50;b.q[2]=0.2;b.q[3]=0.5;b.q[4]=-1.0;b.q[6]=0.4;b.q[9]=0.7;
  b.qd[2]=0.3;b.qd[3]=0.3;b.qd[4]=-0.2;b.qd[6]=0.3;
  const E0=energy(b,kinematics(b.q,b.qd,b.seg,b.k));
  while(b.t<1)step(b);
  const E1=energy(b,kinematics(b.q,b.qd,b.seg,b.k));
  assert.equal(b.work,0);
  assert.ok(b.dissipated<1e-6*Math.abs(E0),`a joint stop dissipated ${b.dissipated} J`);
  return {rel:Math.abs(E1-E0)/Math.abs(E0),abs:Math.abs(E1-E0)};
 });
 assert.ok(drift[0].rel<1e-3,`free-flight drift ${drift[0].rel}`);
 for(let i=1;i<drift.length;i++)assert.ok(drift[i].abs<drift[i-1].abs*0.6,'drift is not first order');
});

// A ragdoll with no torques has to fall over and stay on the floor. Contact points cover
// the pelvis, shoulders, knees, hands and soles; without them a fallen body sinks through.
test('an unactuated ragdoll falls, lands, and stays above the floor',()=>{
 const b=drop(5e-4);
 assert.ok(!b.singular);
 assert.ok(Math.abs(b.q[2])>1.0,'the body never toppled');
 assert.ok(b.q[1]>-0.1,`pelvis fell through the floor to ${b.q[1]}`);
 const k=kinematics(b.q,b.qd,b.seg,b.k);contactPoints(b,k);
 for(let c=0;c<NC;c++)assert.ok(b.cy[c]>-0.05,'a contact point sank through the floor');
 assert.ok(Math.hypot(...b.qd)<1.5,'the body never came to rest');
});

// The impact is where a penalty contact loses energy accounting. The residual must be a
// vanishing discretisation error, not a structural leak, and the resting pose it settles
// into must not depend on the timestep at all.
test('the contact ledger residual converges and the resting pose does not move',()=>{
 const runs=[1e-3,5e-4,2.5e-4,1.25e-4].map(dt=>drop(dt));
 const share=runs.map(b=>Math.abs(b.residual)/b.dissipated);
 // Stiff joint stops cost the first-order integrator accuracy: the residual is several
 // times what it was without them. What must hold is that it still converges to zero.
 assert.ok(share[0]<0.45,`residual ${share[0]} at the coarsest timestep`);
 for(let i=1;i<share.length;i++)assert.ok(share[i]<share[i-1]*0.75,'residual is not converging');
 assert.ok(share.at(-1)<0.05,`residual ${share.at(-1)} at the finest timestep`);
 for(const b of runs.slice(1))assert.ok(Math.abs(b.q[2]-runs[0].q[2])<0.02,'resting pitch moved with dt');
});

test('contact forces only push, and friction stays inside the cone',()=>{
 const b=makeBody({dt:5e-4});
 b.q[1]=1.0;b.q[2]=0.4;b.qd[0]=2.5;
 while(b.t<2){
  step(b);
  for(let c=0;c<NC;c++){
   assert.ok(b.fn[c]>=0,'a contact pulled on the body');
   assert.ok(Math.abs(b.ft[c])<=b.env.friction*b.fn[c]+1e-9,'friction left the cone');
  }
 }
});

test('the simulation is deterministic and rejects invalid conditions',()=>{
 const a=drop(5e-4,false),b=drop(5e-4,false);
 assert.deepEqual([...a.q],[...b.q]);assert.deepEqual([...a.qd],[...b.qd]);
 assert.throws(()=>makeBody({mass:0}));assert.throws(()=>makeBody({dt:1}));
 assert.throws(()=>makeBody({kn:-1}));assert.throws(()=>makeBody({friction:9}));
 assert.equal(NQ,11);assert.equal(LINKS,9);assert.equal(kinBuffers().phi.length,9);
 assert.deepEqual(makeBody().env,{...DEFAULT_ENV});
});

// Without stops the search found an ankle at -151 degrees, folding the foot back past the
// shank with its tip underground, and a shoulder 122 degrees behind the torso.
test('joints cannot be driven far outside their anatomical range',()=>{
 const peak=[200,200,150,200,200,150,60,60];
 for(const dir of [1,-1]){
  const b=makeBody({dt:5e-4});b.q[1]=1.2;
  let worst=0;
  for(let n=0;n<6000;n++){
   for(let j=0;j<8;j++)b.tau[3+j]=dir*peak[j];
   step(b);worst=Math.max(worst,limitExcess(b));
  }
  assert.ok(worst<0.45,`a joint reached ${(worst*57.3).toFixed(1)} degrees past its range`);
 }
});

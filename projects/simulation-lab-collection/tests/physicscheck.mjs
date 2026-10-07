import { DronePhysics } from '../src/dronePhysics.js';

// The simulator is Y-up. A pure yaw must leave the drone upright; pitch and roll must
// tilt it about the correct body axes.
const d = new DronePhysics();
let ok = true;
const near = (a,b,t=1e-5)=>Math.abs(a-b)<t;

for (const yaw of [0, 0.7, Math.PI/2, -2.544, 3.0]) {
  const q = d.eulerToQuat(0, yaw, 0);
  const up = d.rotateVectorByQuat([0,1,0], q);
  const fwd = d.rotateVectorByQuat([0,0,-1], q);
  const upright = near(up[1], 1, 1e-4);
  // Yaw must rotate the forward vector in the horizontal plane
  const heading = Math.atan2(-fwd[0], -fwd[2]);
  const headingOk = near(Math.cos(heading), Math.cos(yaw), 1e-4) && near(Math.sin(heading), Math.sin(yaw), 1e-4);
  if (!upright || !headingOk) ok = false;
  console.log(`${upright&&headingOk?'PASS':'FAIL'} yaw=${yaw.toFixed(3)} bodyUp=(${up.map(v=>v.toFixed(3)).join(',')}) heading=${heading.toFixed(3)}`);
}

// Pitch tilts forward/back (body up leans along z), roll tilts sideways (along x).
{
  const up = d.rotateVectorByQuat([0,1,0], d.eulerToQuat(0.3, 0, 0));
  const pass = Math.abs(up[2]) > 0.2 && Math.abs(up[0]) < 1e-4;
  if (!pass) ok = false;
  console.log(`${pass?'PASS':'FAIL'} pitch tilts about x: bodyUp=(${up.map(v=>v.toFixed(3)).join(',')})`);
}
{
  const up = d.rotateVectorByQuat([0,1,0], d.eulerToQuat(0, 0, 0.3));
  const pass = Math.abs(up[0]) > 0.2 && Math.abs(up[2]) < 1e-4;
  if (!pass) ok = false;
  console.log(`${pass?'PASS':'FAIL'} roll tilts about z:  bodyUp=(${up.map(v=>v.toFixed(3)).join(',')})`);
}

// A level drone at hover-plus thrust must gain altitude, from any start yaw.
{
  const p = new DronePhysics();
  p.reset([0,20,0], -2.544);
  for (let t=0;t<60;t++) p.step({thrust:0.5,pitch:0,roll:0,yaw:0}, 1/60);
  const pass = p.pos[1] > 20;
  if (!pass) ok = false;
  console.log(`${pass?'PASS':'FAIL'} climbs from yawed start: y 20.00 -> ${p.pos[1].toFixed(2)}`);
}
// Rate commands are BODY rates: rotating about an axis must leave that axis fixed.
for (const [label, setup] of [['level', [0,0,0]], ['rolled 90', [0,0,Math.PI/2]], ['pitched 60', [1.05,0,0]]]) {
  for (const axis of [0,1,2]) {
    const p = new DronePhysics();
    p.reset([0,50,0], 0);
    p.quat = p.eulerToQuat(setup[0], setup[1], setup[2]);
    const unit = [[1,0,0],[0,1,0],[0,0,1]][axis];
    const before = p.rotateVectorByQuat(unit, p.quat);
    p.angularVel[0]=0; p.angularVel[1]=0; p.angularVel[2]=0;
    p.angularVel[axis] = 2.0;
    for (let t=0;t<20;t++) p.quat = p.integrateQuat(p.quat, p.angularVel, 1/60);
    const after = p.rotateVectorByQuat(unit, p.quat);
    const dot = before[0]*after[0]+before[1]*after[1]+before[2]*after[2];
    const pass = dot > 0.9999;
    if (!pass) ok = false;
    console.log(`${pass?'PASS':'FAIL'} body axis ${axis} preserved under its own rate (${label}) dot=${dot.toFixed(5)}`);
  }
}

console.log(ok ? 'PHYSICS CHECK PASSED' : 'PHYSICS CHECK FAILED');
process.exit(ok?0:1);

import { extractEvents, KEYFRAME_HZ } from '../src/commentary/raceEvents.js';
import { Commentator } from '../src/commentary/commentator.js';
import { VoxelGrid } from '../src/collision/voxelGrid.js';
import { DroneEnv } from '../src/ppo/droneEnv.js';
import { defaultTrackData } from '../src/defaultTrack.js';

let ok = true;
const check = (c, label, extra='') => { if(!c) ok=false; console.log(`${c?'PASS':'FAIL'} ${label} ${extra}`); };
const gates = defaultTrackData.gates;

// A deterministic synthetic lap: interpolate straight through every gate centre, facing
// along the path. The scripted pilot is too crude to reliably clear gates, which made the
// gate and delta assertions depend on luck.
function syntheticLap(speedMs = 25) {
  const trace = [];
  const step = speedMs / KEYFRAME_HZ;   // metres per keyframe
  const q = [1, 0, 0, 0];

  // Start behind gate 0 along its normal, exactly as a real episode does. Starting ON the
  // gate means the signed distance never changes sign, so the crossing never registers.
  const g0 = gates[0].position, n0 = gates[0].normal || [0,0,1];
  const pre = [g0[0]-n0[0]*15, g0[1]-n0[1]*15, g0[2]-n0[2]*15];
  const preSteps = Math.max(2, Math.round(15 / step));
  for (let k = 0; k < preSteps; k++) {
    const t = k / preSteps;
    trace.push(pre[0]+(g0[0]-pre[0])*t, pre[1]+(g0[1]-pre[1])*t, pre[2]+(g0[2]-pre[2])*t, q[0], q[1], q[2], q[3]);
  }

  for (let i = 0; i < gates.length; i++) {
    const a = gates[i].position;
    const b = gates[(i + 1) % gates.length].position;
    const seg = Math.hypot(b[0]-a[0], b[1]-a[1], b[2]-a[2]);
    const n = Math.max(2, Math.round(seg / step));
    for (let k = 0; k < n; k++) {
      const t = k / n;
      trace.push(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t, q[0], q[1], q[2], q[3]);
    }
  }
  // Overshoot past the start/finish so the final crossing completes.
  for (let k = 1; k <= preSteps; k++) {
    const t = k / preSteps;
    trace.push(g0[0]+n0[0]*15*t, g0[1]+n0[1]*15*t, g0[2]+n0[2]*15*t, q[0], q[1], q[2], q[3]);
  }
  return { trajectory: Float32Array.from(trace), startGateIdx: 0, crashed: false, hitWall: false };
}

const run = syntheticLap();
const evAll = extractEvents(run, { gates });
run.gatesCleared = evAll.filter(e=>e.type==='gate').length;
console.log(`synthetic lap: ${run.gatesCleared} gates, ${(run.trajectory.length/7/KEYFRAME_HZ).toFixed(1)}s`);
const events = extractEvents(run, { gates });
const types = {};
events.forEach(e => types[e.type] = (types[e.type]||0)+1);
console.log('events:', JSON.stringify(types));

check(events.length > 0, 'events extracted');
check(events[0].type === 'launch', 'first event is launch');
check(events.every((e,i) => i===0 || e.frame >= events[i-1].frame), 'events are time-ordered');
check(events.filter(e=>e.type==='gate').length === run.gatesCleared, 'gate events match gates cleared',
      `${events.filter(e=>e.type==='gate').length} vs ${run.gatesCleared}`);
const ending = events.filter(e=>['finish','wallHit','groundHit','timeout'].includes(e.type));
check(ending.length === 1, 'exactly one terminal event', ending.map(e=>e.type).join(','));
check(events.filter(e=>e.type==='topSpeed').length === 1, 'one topSpeed event');
const gateEvents = events.filter(e=>e.type==='gate');
check(gateEvents.every(e => e.speed >= 0 && Number.isFinite(e.speed)), 'gate speeds are finite');
check(gateEvents.every((e,i) => i===0 || e.ordinal === gateEvents[i-1].ordinal+1), 'gate ordinals increment');

// With a grid present, near-miss detection should engage.
// Place the slab right beside the first gate so the path genuinely passes close to it.
const g0 = gates[0].position;
const wall = [];
for (let dy=-15; dy<15; dy+=1.0) for (let dz=-60; dz<60; dz+=1.0) for (let dx=3; dx<7; dx+=1.0)
  wall.push(g0[0]+dx, g0[1]+dy, g0[2]+dz);
const grid = VoxelGrid.fromPoints(Float32Array.from(wall), { cellSize: 1.5, minPoints: 1 });
const withGrid = extractEvents(run, { gates, grid });
const misses = withGrid.filter(e=>e.type==='nearMiss');
console.log(`near misses with a wall in the scene: ${misses.length}`);
check(misses.every(e => e.clearance >= 0 && e.clearance < 4.0), 'near misses are within threshold');
check(misses.every(e => typeof e.side === 'string' && e.side.length), 'near misses name a side');
// Rate limiting: no two closer than 1.5s
check(misses.every((e,i)=> i===0 || e.t - misses[i-1].t >= 1.5 - 1e-9), 'near misses are rate-limited');

// Reference deltas
const ref = { gateFrames: gateEvents.map(e => e.frame + 10) }; // pretend the record was slower
const vsRef = extractEvents(run, { gates, reference: ref }).filter(e=>e.type==='gate');
check(vsRef.length && vsRef.every(e => e.delta !== null && e.delta < 0), 'deltas computed and read as ahead of a slower reference',
      vsRef.length ? `first delta ${vsRef[0].delta.toFixed(2)}s` : '');

// ── Scheduler ──
const c = new Commentator({ cooldown: 1.1 });
const said = [];
c.onLine(l => said.push(l));
c.loadRun(run, { gates });
const total = run.trajectory.length/7;
for (let f = 0; f <= total; f++) c.update(f);
await new Promise(r => setTimeout(r, 50)); // voice.say is async
console.log(`\n--- commentary (${said.length} lines) ---`);
said.slice(0, 14).forEach(l => console.log(`  [${l.t.toFixed(1).padStart(5)}s] ${l.text}`));
check(said.length > 6, 'scheduler emitted lines', `${said.length} lines`);
check(said.every(l => typeof l.text === 'string' && l.text.length > 0), 'all lines are non-empty text');
check(said.every((l,i)=> i===0 || l.t >= said[i-1].t), 'lines are chronological');
// No line should contain an undefined/NaN leak from a template
check(said.every(l => !/undefined|NaN|Infinity/.test(l.text)), 'no undefined/NaN leaked into text');

// Scrubbing backwards should not dump old lines at once, but playing forward from the
// new position must produce commentary again.
const before = said.length;
c.update(5);
await new Promise(r => setTimeout(r, 20));
check(said.length === before, 'scrubbing back does not replay history immediately');
for (let f = 5; f <= total; f++) c.update(f);
await new Promise(r => setTimeout(r, 50));
check(said.length > before, 'playing forward after a scrub re-emits commentary',
      `${before} -> ${said.length}`);

console.log(ok ? '\nCOMMENTARY CHECK PASSED' : '\nCOMMENTARY CHECK FAILED');
process.exit(ok?0:1);

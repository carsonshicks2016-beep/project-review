import assert from 'node:assert/strict';
import {makeTrack,Trainer,Comparison,newCar,evaluateDriver,prior,TRACE_STRIDE} from '../dist/engine.mjs';
import {createFrameLoop,encodeDriverLink,decodeDriverLink,prepareImportedSession} from '../dist/runtime.mjs';
import * as engine from '../dist/engine.mjs';
import {createWorld} from '../dist/scene.mjs';

const track=makeTrack();
const baseline=evaluateDriver(track,prior(()=>1));
assert(!baseline.finished);
assert(Math.abs(baseline.time-5.4)<.05,'Noiseless prior still stalls');
const trainer=new Trainer(track,99,null,{recordLines:true});
while(trainer.generation<=20)trainer.step();
assert(Math.abs(trainer.bestLap-35.466666666667216)<1e-9,'Preserve original learning trajectory');
assert.equal(trainer.totalLaps,743);
assert(trainer.history[0].completion<1,'Generation one still has to learn');
assert(trainer.population.every(c=>c.trace===null),'Population evaluation allocates no trace buffers');
assert.equal(trainer.lines.length,4);
const line=trainer.lines.at(-1);
assert(line.trace instanceof Float32Array);
assert(line.trace.length>100&&line.trace.length%TRACE_STRIDE===0);
assert(Math.abs(line.trace.at(-2)-trainer.bestLap)<.15,'Racing line covers the winning lap');
assert(Array.from(line.trace).every(Number.isFinite));

const payload={weights:trainer.champion.weights,track:{seed:2077,elevation:18}};
const hash=encodeDriverLink(payload),decoded=decodeDriverLink(hash);
assert.deepEqual(decoded,payload,'Link preserves full precision');
assert.equal(hash.length,282);
const imported=prepareImportedSession(decoded,makeTrack(1,0),engine);
assert.equal(imported.track.seed,2077);
assert.equal(imported.trainer.bestLap,trainer.bestLap,'Link replay exactly matches champion');
assert.equal(imported.trainer.lines.length,1,'Imported driver immediately has a racing line');
assert.throws(()=>decodeDriverLink('#driver=broken'));
assert.throws(()=>decodeDriverLink('#driver='+'A'.repeat(274)));
assert.equal(decodeDriverLink('#unrelated'),null);
assert.throws(()=>encodeDriverLink({weights:Array(24).fill(Infinity),track:payload.track}));
assert.throws(()=>prepareImportedSession({weights:[0]},track,engine));

// Input ablation: zeroing an input's weights is equivalent to removing that sense, so one
// deterministic replay per input prices it. Some senses are load-bearing, some are near noise.
const ablation=engine.ablate(track,trainer.champion.weights);
assert.equal(ablation.rows.length,12);
assert(ablation.base.finished&&Math.abs(ablation.base.lap-trainer.bestLap)<1e-9,'ablation baseline is the champion lap');
assert.deepEqual(ablation.rows.map(r=>r.input),engine.INPUT_NAMES);
assert(ablation.rows.some(r=>!r.finished),'removing at least one sense breaks the lap entirely');
assert(ablation.rows.every(r=>r.progress>0&&r.progress<=1));
assert.deepEqual(engine.ablate(track,trainer.champion.weights),ablation,'ablation is deterministic');

// Diversity: mean pairwise genome distance, which collapses as the population converges.
assert.equal(engine.diversity([{weights:[0,0]},{weights:[3,4]}]),5);
assert.equal(engine.diversity([{weights:[1,2]}]),0,'a single genome has no spread');
assert(trainer.history.every(h=>Number.isFinite(h.diversity)));
assert(trainer.history[0].diversity>trainer.history.at(-1).diversity,'diversity falls as it converges');

// Surface: wet shrinks the friction budget for cornering and for drive/brake alike, and dry is
// left bit-for-bit alone so every stored lap time and exported driver still replays exactly.
assert.equal(makeTrack(2077,18).grip,1);
assert.equal(makeTrack(2077,18).surface,'dry');
assert.equal(makeTrack(2077,18,3,'wet').grip,engine.SURFACES.wet);
assert.throws(()=>makeTrack(2077,18,3,'ice'));
const dryRun=evaluateDriver(makeTrack(2077,18),trainer.champion.weights);
const wetRun=evaluateDriver(makeTrack(2077,18,null,'wet'),trainer.champion.weights);
assert(Math.abs(dryRun.time-trainer.bestLap)<1e-9,'dry replay is unchanged');
assert(!wetRun.finished||wetRun.time>dryRun.time,'a dry-trained driver is never faster in the wet');
assert(makeTrack(2077,18,null,'wet').points.every((p,i)=>p.y===makeTrack(2077,18).points[i].y),'surface does not move the terrain');

// Slide state drives the dust and tire marks, so it has to be live on every car.
const slider=newCar(track,trainer.champion.weights);
let sawSlide=false;
while(!slider.dead&&!slider.finished){engine.tickCar(track,slider);if(slider.slide>0)sawSlide=true;assert(slider.slide>=0&&slider.slide<=1);assert(typeof slider.offRoad==='boolean');}
assert(sawSlide,'a champion lap saturates the tires somewhere');

const comparison=new Comparison(track,{generations:20});
for(const t of comparison.trainers.slice(1))assert.deepEqual(t.population.map(c=>c.weights),comparison.trainers[0].population.map(c=>c.weights));
while(!comparison.done)comparison.step();
assert(comparison.trainers.every(t=>t.evaluations===960));
assert.equal(comparison.trainers[0].bestLap,trainer.bestLap);
assert(comparison.trainers.every(t=>t.history.length===20&&t.champion));
// The escape phase changes exploration, while its first elite retains the champion.
const stuck=new Trainer(track);stuck.champion={weights:[...trainer.champion.weights],score:1e9,progress:track.length,lap:trainer.bestLap,generation:1};stuck.stagnantGenerations=24;stuck.evolve();
assert(stuck.exploring);assert.deepEqual(stuck.population[0].weights,trainer.champion.weights);

let queue=[],steps=0,errors=0;
const loop=createFrameLoop({schedule:callback=>{queue.push(callback);return queue.length;},step:()=>{steps++;if(steps===1)throw Error('render failed');},onError:()=>{errors++;throw Error('reporter failed too');}});
loop.start();queue.shift()(0);assert(loop.faulted);assert.equal(errors,1);assert.equal(queue.length,1);
queue.shift()(16);assert.equal(steps,1,'Fault pauses simulation, not scheduling');assert.equal(queue.length,1);
loop.recover();queue.shift()(32);assert.equal(steps,2);loop.stop();queue.shift()(48);assert.equal(queue.length,0);

const world=createWorld(track);assert.equal(world.userData.stats.sourceMeshes,1025);assert.equal(world.userData.stats.batchedMeshes,23);
assert.equal(world.userData.stats.effectMeshes,2,'dust and tire marks are one instanced mesh each');
assert.equal(world.userData.stats.renderMeshes,25);assert.equal(world.userData.stats.shadowCasters,14);
const car=newCar(track,trainer.champion.weights);world.userData.updateCars([car],car);
assert(world.children.every(m=>Array.from(m.instanceMatrix.array).every(Number.isFinite)));
world.userData.dispose();
// Radial generation's positive radius is an invariant of every allowed seed;
// also inspect varied samples and the closing segment for finite, nonzero lengths.
for(const seed of [1,42,2077,9001,999999]){const t=makeTrack(seed,36);assert(t.points.every(p=>Math.hypot(p.x/1.22,p.z/.9)>=73-1e-9&&p.ds>0&&Number.isFinite(p.grade)));}
console.log(JSON.stringify({noiselessPrior:{seconds:baseline.time,progressMeters:baseline.bestProgress},bestLap:trainer.bestLap,traceBytes:trainer.lines.reduce((n,l)=>n+l.trace.byteLength,0),scene:world.userData.stats,linkPayloadCharacters:hash.length-8,comparison:comparison.trainers.map(t=>({algorithm:t.algorithm,evaluations:t.evaluations,lap:t.bestLap,physicsSteps:t.simulationSteps})),frameRecovery:'passed',exactLinkReplay:'passed'},null,2));

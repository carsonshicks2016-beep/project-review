import assert from 'node:assert/strict';
import {makeTrack,Trainer,newCar,tickCar,readDriver} from '../dist/engine.mjs';
const a=makeTrack(2077,18),b=makeTrack(2077,18),c=makeTrack(2088,18);
assert.deepEqual(a.points,b.points);assert.notDeepEqual(a.points,c.points);assert.equal(makeTrack(1,0).ascent,0);assert(a.ascent>20);
const t=new Trainer(a);const started=Date.now();let initial=null;
while(t.generation<=20){t.step();if(t.bestLap!==null&&initial===null)initial=t.bestLap;}
assert(t.bestLap!==null,'At least one driver must complete a lap');assert(t.history.some(x=>x.finishers>0));assert(t.bestLap<initial,'Evolution improves the first completed lap');
let champion=newCar(a,t.champion.weights);while(!champion.dead&&!champion.finished)tickCar(a,champion);assert(champion.finished);assert(Math.abs(champion.time-t.bestLap)<.0001,'Champion replay reproduces best time');
console.log(JSON.stringify({generations:t.history.length,initialLap:initial,bestLap:t.bestLap,totalLaps:t.totalLaps,firstCompletion:t.history.find(x=>x.finishers)?.generation,seconds:(Date.now()-started)/1000,history:t.history.map(x=>({generation:x.generation,lap:x.lap,completion:x.completion}))},null,2));

// Driver import: a JSON round-trip rebuilds the same track and replays the same lap.
const file=JSON.parse(JSON.stringify({format:'rally-neural-policy-v1',weights:t.champion.weights,track:{seed:a.seed,elevation:a.elevation},bestLapSeconds:t.bestLap}));
const loaded=readDriver(file);
assert.deepEqual(loaded.weights,t.champion.weights,'Import preserves weights exactly');
assert.equal(loaded.seed,2077);assert.equal(loaded.elevation,18);
const rebuilt=makeTrack(loaded.seed,loaded.elevation),imported=newCar(rebuilt,loaded.weights);
while(!imported.dead&&!imported.finished)tickCar(rebuilt,imported);
assert(imported.finished&&Math.abs(imported.time-t.bestLap)<.0001,'Imported driver reproduces its lap time');
assert.equal(readDriver({weights:t.champion.weights}).seed,null,'Missing track settings keep the current track');
for(const bad of [null,{},{weights:[]},{weights:Array(24).fill(0).concat(0)},{weights:Array(23).fill(0)},{weights:Array(24).fill(NaN)},{weights:Array(24).fill('0')}])assert.throws(()=>readDriver(bad));

for(const [seed,elevation] of [[1,0],[42,36],[9001,18]]){const tr=new Trainer(makeTrack(seed,elevation));while(tr.generation<=20)tr.step();assert(tr.bestLap!==null,`Track ${seed} can be learned`);assert(tr.population.every(c=>Number.isFinite(c.x)&&Number.isFinite(c.z)));console.log({seed,elevation,bestLap:tr.bestLap});}
const reverse=newCar(a,Array(24).fill(0));reverse.heading+=Math.PI;reverse.vx=-reverse.vx;reverse.vz=-reverse.vz;reverse.weights[23]=2;while(!reverse.dead&&!reverse.finished)tickCar(a,reverse);assert(!reverse.finished,'Crossing start backwards is not a completed lap');

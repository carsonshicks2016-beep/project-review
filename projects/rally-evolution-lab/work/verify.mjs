import assert from 'node:assert/strict';
import {makeTrack,Trainer,newCar,tickCar} from '../dist/engine.mjs';
const a=makeTrack(2077,18),b=makeTrack(2077,18),c=makeTrack(2088,18);
assert.deepEqual(a.points,b.points);assert.notDeepEqual(a.points,c.points);assert.equal(makeTrack(1,0).ascent,0);assert(a.ascent>20);
const t=new Trainer(a);const started=Date.now();let initial=null;
while(t.generation<=20){t.step();if(t.generation===2&&initial===null)initial=t.bestLap;}
assert(t.bestLap!==null,'At least one driver must complete a lap');assert(t.history.some(x=>x.finishers>0));assert(t.bestLap<=initial||initial===null,'Evolution retains fastest lap');
let champion=newCar(a,t.champion.weights);while(!champion.dead&&!champion.finished)tickCar(a,champion);assert(champion.finished);assert(Math.abs(champion.time-t.bestLap)<.0001,'Champion replay reproduces best time');
console.log(JSON.stringify({generations:t.history.length,initialLap:initial,bestLap:t.bestLap,totalLaps:t.totalLaps,firstCompletion:t.history.find(x=>x.finishers)?.generation,seconds:(Date.now()-started)/1000,history:t.history.map(x=>({generation:x.generation,lap:x.lap,completion:x.completion}))},null,2));

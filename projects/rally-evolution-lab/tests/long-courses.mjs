import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as engine from '../dist/engine.mjs';
import {encodeDriverLink,decodeDriverLink,prepareImportedSession,overviewFraming,MIN_ROAD_FRAME_FRACTION} from '../dist/runtime.mjs';
for(const km of [3,6,10]){
 const track=engine.makeTrack(2077,18,km);
 assert(Math.abs(track.length-km*1000)<1e-7);
 assert(track.n>=km*500&&track.episodeLimit>180);
 assert(track.points.every(p=>Math.abs(p.x)<track.terrainWidth/2-40&&Math.abs(p.z)<track.terrainDepth/2-40));
 assert.deepEqual(engine.makeTrack(2077,18,km).points,track.points);
}

// Overview framing: courses that fit keep the whole-lap diorama, longer ones frame a sector so the
// road never shrinks below MIN_ROAD_FRAME_FRACTION of the view.
const view={aspect:1.7,fov:43,zoom:440};
const framing=t=>overviewFraming({roadWidth:t.width,terrainWidth:t.terrainWidth,terrainDepth:t.terrainDepth,...view});
const visibleGround=f=>2*f.distance*Math.tan(view.fov*Math.PI/360)*view.aspect;
const legacyView=framing(engine.makeTrack());
assert.equal(legacyView.sector,false,'the legacy circuit still frames the whole lap');
assert(Math.abs(legacyView.distance-440)<1e-9,'legacy overview distance is unchanged');
for(const km of [3,6,12]){
 const long=engine.makeTrack(2077,18,km),f=framing(long);
 assert(f.sector,km+' km frames a sector instead of the whole lap');
 assert(long.width/visibleGround(f)>=MIN_ROAD_FRAME_FRACTION-1e-9,'road stays legible at '+km+' km');
}

// Relief grows with the course instead of spreading the same hills thinner, and because amplitude
// and wavelength scale together the grade per metre is unchanged, so learned policies transfer.
const relief3=engine.makeTrack(2077,18,3),relief12=engine.makeTrack(2077,18,12);
const span=t=>Math.max(...t.points.map(p=>p.y))-Math.min(...t.points.map(p=>p.y));
const steepest=t=>Math.max(...t.points.map(p=>Math.abs(p.grade)));
assert(relief12.ascent>relief3.ascent*3.5,'a 12 km course climbs far more than a 3 km one');
assert(span(relief12)>span(relief3)*3.5,'and has proportionally taller relief');
assert(Math.abs(steepest(relief12)-steepest(relief3))<.02,'while grade per metre is preserved');
assert.equal(engine.makeTrack(2077,18,3).ascent,relief3.ascent,'3 km courses are unaffected');

const track=engine.makeTrack(2077,18,3),trainer=new engine.Trainer(track,99,null,{recordLines:true});
while(trainer.generation<=30)trainer.step();
assert(trainer.bestLap!==null);assert(Math.abs(trainer.bestLap-114.63333333332939)<1e-8);
const car=engine.evaluateDriver(track,trainer.champion.weights);
assert(car.finished);assert(car.sectorTimes.every(t=>t>0));assert(Math.abs(car.sectorTimes.reduce((a,b)=>a+b,0)-car.time)<1e-8);
const data={weights:trainer.champion.weights,track:{seed:2077,elevation:18,lengthKm:3}},hash=encodeDriverLink(data);
assert.equal(hash.length,292);assert.deepEqual(decodeDriverLink(hash),data);
const loaded=prepareImportedSession(decodeDriverLink(hash),engine.makeTrack(),engine);
assert.equal(loaded.trainer.bestLap,trainer.bestLap);assert.equal(loaded.track.lengthKm,3);
assert.throws(()=>engine.makeTrack(1,18,100));assert.throws(()=>engine.readDriver({weights:car.weights,track:{seed:1,elevation:18,lengthKm:NaN}}));
fs.mkdirSync('work',{recursive:true});fs.writeFileSync('work/long-driver.json',JSON.stringify(data));
console.log(JSON.stringify({lengthKm:3,bestLap:trainer.bestLap,sectors:car.sectorTimes,linkReplay:'exact',maps:[3,6,10]},null,2));

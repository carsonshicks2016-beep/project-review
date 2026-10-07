import test from 'node:test';
import assert from 'node:assert/strict';
import {cachedActivity} from '../../static/neural-activity.js';
const chunks=[{start:0,end:.1},{start:.1,end:.2},{start:.2,end:.3}];
test('delayed future chunk cannot restore highlights after a backward seek',async()=>{
 const cache=new Map([[0,{indices:[7],times:[.04]}]]);
 let resolve;const request=new Promise(r=>resolve=r).then(data=>cache.set(2,data));
 assert.equal(cachedActivity(chunks,.25,cache),null);
 assert.deepEqual(cachedActivity(chunks,.05,cache).rates,{7:10});
 resolve({indices:[9],times:[.24]});await request;
 assert.deepEqual(cachedActivity(chunks,.05,cache).rates,{7:10});
 assert.deepEqual(cachedActivity(chunks,.01,cache).rates,{});
});
test('missing boundary chunks produce loading rather than partial or stale activity',()=>{
 const cache=new Map([[1,{indices:[9],times:[.12]}]]);
 assert.equal(cachedActivity(chunks,.15,cache),null);
 cache.set(0,{indices:[7],times:[.07]});
 assert.deepEqual(cachedActivity(chunks,.15,cache).rates,{7:10,9:10});
 assert.deepEqual(cachedActivity(chunks,.12,cache).lastSpikes,{7:.07,9:.12});
});

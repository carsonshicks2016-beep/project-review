import test from 'node:test';import assert from 'node:assert/strict';
import {eventActivity,activityChunkIndices} from '../../static/neural-activity.js';
test('100ms rate windows retain neuron identity and exclude the left boundary',()=>{const data=[{indices:[4,4,17,4],times:[.1,.15,.2,.2001]}];assert.deepEqual(eventActivity(data,.2),{rates:{4:10,17:10},lastSpikes:{4:.15,17:.2}});});
test('boundary and fractional-pose timestamps select both adjacent chunks',()=>{const chunks=[{start:0,end:.1},{start:.1001,end:.2},{start:.2001,end:.3}];assert.deepEqual(activityChunkIndices(chunks,.133333),[0,1]);assert.deepEqual(activityChunkIndices(chunks,.3),[1,2]);});
test('scrubbing backward recomputes activity instead of retaining future spikes',()=>{const data=[{indices:[2,2],times:[.04,.19]}];assert.deepEqual(eventActivity(data,.2).rates,{2:10});assert.deepEqual(eventActivity(data,.05).rates,{2:10});assert.deepEqual(eventActivity(data,0).rates,{});});

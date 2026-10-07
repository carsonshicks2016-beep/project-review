import {test} from 'node:test';
import assert from 'node:assert/strict';
import {recordingIndex} from '../../static/replay.js';
const frames = times => times.map(time => ({world:{time}}));
test('uses recorded cadence rather than assuming 100ms',()=>{
 const r=frames([.1,.6,1.1]);
 assert.equal(recordingIndex(r,.4),0);
 assert.equal(recordingIndex(r,.5),1);
 assert.equal(recordingIndex(r,1),2);
});
test('supports nonzero start, irregular samples and final sample',()=>{
 const r=frames([20,20.1,20.8,22]);
 assert.equal(recordingIndex(r,0),0);
 assert.equal(recordingIndex(r,.5),1);
 assert.equal(recordingIndex(r,1),2);
 assert.equal(recordingIndex(r,100),3);
 assert.equal(recordingIndex([],0),0);
});

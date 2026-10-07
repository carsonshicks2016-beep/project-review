import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {followPosition} from '../../static/recording-camera.js';

test('open arena preserves the existing follow viewpoint',()=>{
 assert.deepEqual(followPosition(new THREE.Vector3(2,3,1)).toArray(),[-6,-8,8]);
});
test('follow camera changes sides when a wall obscures the recorded fly',()=>{
 const target=new THREE.Vector3(11.6,0,1.1),box=new THREE.Box3(new THREE.Vector3(8.85,-6.15,-.15),new THREE.Vector3(11.15,6.15,4.15));
 const camera=followPosition(target,[box]);
 const delta=target.clone().sub(camera),ray=new THREE.Ray(camera,delta.clone().normalize()),hit=ray.intersectBox(box,new THREE.Vector3());
 assert.ok(!hit||hit.distanceTo(camera)>=delta.length()-.1);
 assert.deepEqual(target.toArray(),[11.6,0,1.1]);
});

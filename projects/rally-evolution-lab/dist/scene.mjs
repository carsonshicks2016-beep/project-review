import * as THREE from './vendor/three.module.js';
import {rng,POPULATION} from './engine.mjs';
export function createWorld(track){
 let world=new THREE.Group(),carModels=[];
 const terrainWidth=track.terrainWidth,terrainDepth=track.terrainDepth;
 // One road surface, shared. Verge sits 2 cm under the road so it still draws behind it, and
 // painted markings sit 1 cm over it. Previously these were .32/.46/.50, which left a 14 cm
 // ledge at the verge and floated the start grid 4 cm above the road it is painted on.
 const ROAD_SURFACE=.46,VERGE_SURFACE=ROAD_SURFACE-.02,PAINT_LIFT=.012;
 let terrainFloor=Infinity;
const mat=(color,extra={})=>new THREE.MeshStandardMaterial({color,roughness:1,...extra});
const wet=track.surface==='wet';
const grass=mat('#ffffff',{vertexColors:true,flatShading:true}),roadMat=wet?mat('#6d6350',{roughness:.42}):mat('#b5a071'),edgeMat=wet?mat('#585640'):mat('#827c52'),trunkMat=mat('#4e5037'),treeMats=['#334e30','#3b5632','#435c34'].map(c=>mat(c)),darkMat=mat('#151d18'),glassMat=mat('#243c39',{roughness:.4}),whiteMat=mat('#ecefd6');
function mesh(geo,material,x=0,y=0,z=0,parent=world){let o=new THREE.Mesh(geo,material);o.position.set(x,y,z);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
function ribbon(width,lift,material){const v=[],idx=[];for(let i=0;i<=track.n;i++){let p=track.points[i%track.n];for(const side of [-1,1]){let x=p.x-Math.sin(p.angle)*width*side/2,z=p.z+Math.cos(p.angle)*width*side/2;v.push(x,track.height(x,z)+lift,z);}if(i<track.n){let j=i*2;idx.push(j,j+1,j+2,j+1,j+3,j+2);}}const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(v,3));g.setIndex(idx);g.computeVertexNormals();let m=mesh(g,material);m.material.side=THREE.DoubleSide;return m;}
 const geometry=new THREE.PlaneGeometry(terrainWidth,terrainDepth,track.lengthKm===null?90:Math.min(360,Math.ceil(terrainWidth/(3.5*track.heightScale))),track.lengthKm===null?78:Math.min(320,Math.ceil(terrainDepth/(3.5*track.heightScale))));geometry.rotateX(-Math.PI/2);const pos=geometry.attributes.position,colors=[];const random=rng(track.seed+300);for(let i=0;i<pos.count;i++){let x=pos.getX(i),z=pos.getZ(i);const groundY=track.height(x,z)-.06;pos.setY(i,groundY);if(groundY<terrainFloor)terrainFloor=groundY;const c=new THREE.Color().setHSL(.235+(random()-.5)*.025,.16+random()*.09,.27+random()*.08+track.height(x,z)*.0015/track.heightScale);colors.push(c.r,c.g,c.b);}geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));geometry.computeVertexNormals();mesh(geometry,grass);ribbon(16,VERGE_SURFACE,edgeMat);ribbon(12,ROAD_SURFACE,roadMat);
 // The skirt and backdrop follow the real terrain minimum; tall relief on long courses used to
 // sink straight through the fixed -28 / -29 planes.
 const skirtBottom=Math.min(-28,terrainFloor-14);
 // Exposed sides make the uneven landscape readable as a terrain tile.
 const rim=[];
 for(let i=0;i<=90;i++)rim.push([-terrainWidth/2+i/90*terrainWidth,-terrainDepth/2]);
 for(let i=0;i<=80;i++)rim.push([terrainWidth/2,-terrainDepth/2+i/80*terrainDepth]);
 for(let i=0;i<=90;i++)rim.push([terrainWidth/2-i/90*terrainWidth,terrainDepth/2]);
 for(let i=0;i<=80;i++)rim.push([-terrainWidth/2,terrainDepth/2-i/80*terrainDepth]);
 let vertices=[];for(let i=0;i<rim.length;i++){let p=rim[i],q=rim[(i+1)%rim.length];vertices.push(p[0],track.height(...p),p[1],q[0],track.height(...q),q[1],p[0],skirtBottom,p[1],q[0],track.height(...q),q[1],q[0],skirtBottom,q[1],p[0],skirtBottom,p[1]);}let sides=new THREE.BufferGeometry();sides.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));sides.computeVertexNormals();mesh(sides,mat('#3c4330',{side:THREE.DoubleSide}));
 const base=mesh(new THREE.PlaneGeometry(terrainWidth*5,terrainDepth*5),mat('#293627'),0,skirtBottom-1,0);base.rotation.x=-Math.PI/2;
 for(let i=0;i<(track.lengthKm===null?205:Math.min(2200,Math.round(track.lengthKm*220)));i++){let x=(random()-.5)*(terrainWidth-20),z=(random()-.5)*(terrainDepth-21);let distance=Math.min(...track.points.filter((_,i)=>i%5===0).map(p=>Math.hypot(x-p.x,z-p.z)));if(distance<20)continue;let h=5+random()*10,y=track.height(x,z);mesh(new THREE.CylinderGeometry(.5,.8,1,5),trunkMat,x,y+h*.2,z).scale.y=h*.4;mesh(new THREE.ConeGeometry(1,1,5),treeMats[i%3],x,y+h*.65,z).scale.set(h*.3,h*.85,h*.3);mesh(new THREE.ConeGeometry(1,1,5),treeMats[(i+1)%3],x,y+h*.95,z).scale.set(h*.23,h*.68,h*.23);}
 for(let i=0;i<track.n;i+=12){let p=track.points[i];for(let s of [-1,1]){let x=p.x-Math.sin(p.angle)*8*s,z=p.z+Math.cos(p.angle)*8*s;mesh(new THREE.CylinderGeometry(.16,.16,1.8,5),i%24===0?whiteMat:mat('#9cac7f'),x,track.height(x,z)+VERGE_SURFACE+.9,z);}}
 const start=track.points[0];const gate=new THREE.Group();gate.position.set(start.x,start.y+VERGE_SURFACE,start.z);gate.rotation.y=-start.angle;world.add(gate);for(let side of [-1,1])mesh(new THREE.BoxGeometry(.6,7,.6),darkMat,0,3.5,side*7.5,gate);mesh(new THREE.BoxGeometry(.7,1.5,16),darkMat,0,7,0,gate);for(let i=0;i<16;i++)mesh(new THREE.BoxGeometry(.73,.5,.5),i%2?darkMat:whiteMat,0,7.1,(i-7.5)*.8,gate);
 for(let row=0;row<2;row++)for(let i=0;i<12;i++){const localX=row*.8-0.4,localZ=i-5.5;let x=start.x+Math.cos(start.angle)*localX-Math.sin(start.angle)*localZ,z=start.z+Math.sin(start.angle)*localX+Math.cos(start.angle)*localZ;let tile=mesh(new THREE.PlaneGeometry(.8,1),((i+row)%2)?darkMat:whiteMat,x,track.height(x,z)+ROAD_SURFACE+PAINT_LIFT,z);tile.rotation.set(-Math.PI/2,0,-start.angle);}
 for(let i=0;i<POPULATION;i++){const car=new THREE.Group(),bodyMat=mat('#d6fa59');car.userData.body=bodyMat;mesh(new THREE.BoxGeometry(4.2,.85,1.85),bodyMat,0,.95,0,car);mesh(new THREE.BoxGeometry(1.95,.72,1.58),glassMat,-.2,1.68,0,car);mesh(new THREE.BoxGeometry(1.65,.15,1.64),bodyMat,-.3,2.08,0,car);mesh(new THREE.BoxGeometry(.5,.14,2.2),darkMat,-1.85,1.8,0,car);mesh(new THREE.BoxGeometry(.13,.28,1.45),whiteMat,2.13,1.02,0,car);for(let x of [-1.3,1.3])for(let z of [-1.02,1.02]){let wheel=mesh(new THREE.CylinderGeometry(.46,.46,.36,8),darkMat,x,.55,z,car);wheel.rotation.x=Math.PI/2;}car.scale.setScalar(1.1);world.add(car);carModels.push(car);}


 // Instance repeated geometry/material pairs. CPU-side source objects are used only
 // during construction; rendering submits each batch once, including the population.
 const sourceCount=(()=>{let n=0;world.traverse(o=>{if(o.isMesh)n++;});return n;})();
 for(const car of carModels)world.remove(car);
 world.updateMatrixWorld(true);
 const staticSources=[];
 world.traverse(o=>{if(o.isMesh)staticSources.push({mesh:o,matrix:o.matrixWorld.clone()});});
 const carSources=[];
 carModels.forEach((car,index)=>{car.scale.setScalar(1);car.updateMatrixWorld(true);car.traverse(o=>{if(o.isMesh)carSources.push({mesh:o,matrix:o.matrixWorld.clone(),car:index});});});
 const geometryKey=g=>JSON.stringify([g.type,g.parameters??g.uuid]);
 function batch(sources,dynamic){
  const groups=new Map();
  for(const source of sources){
   const m=source.mesh.material,key=geometryKey(source.mesh.geometry)+'|'+m.color.getHexString()+'|'+m.side+'|'+m.vertexColors;
   if(!groups.has(key))groups.set(key,[]);groups.get(key).push(source);
  }
  return [...groups.values()].map(items=>{
   const first=items[0].mesh,body=dynamic&&first.material.color.getHexString()==='d6fa59';
   const material=first.material.clone();if(body)material.color.set('#ffffff');
   const batch=new THREE.InstancedMesh(first.geometry,material,items.length);
   batch.castShadow=dynamic||first.geometry.type==='ConeGeometry'||(first.geometry.type==='CylinderGeometry'&&first.geometry.parameters.radiusTop===.5)||first.geometry.type==='BoxGeometry';
   batch.receiveShadow=true;batch.frustumCulled=!dynamic;
   items.forEach((item,i)=>{batch.setMatrixAt(i,item.matrix);if(body)batch.setColorAt(i,new THREE.Color('#d6fa59'));});
   if(dynamic)batch.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
   else{batch.computeBoundingSphere();batch.computeBoundingBox();}
   return {mesh:batch,items,body};
  });
 }
 const staticBatches=batch(staticSources,false),carBatches=batch(carSources,true);
 world.clear();for(const b of [...staticBatches,...carBatches])world.add(b.mesh);
 // Free duplicate construction geometry and materials after each unique batch is retained.
 const retained=new Set([...staticBatches,...carBatches].map(b=>b.mesh.geometry));
 const oldGeometries=new Set(),oldMaterials=new Set();
 for(const item of [...staticSources,...carSources]){oldGeometries.add(item.mesh.geometry);oldMaterials.add(item.mesh.material);}
 for(const geometry of oldGeometries)if(!retained.has(geometry))geometry.dispose();
 for(const material of oldMaterials)material.dispose();
 const transform=new THREE.Object3D(),matrix=new THREE.Matrix4(),hidden=new THREE.Matrix4().makeScale(0,0,0);
 const leaderColor=new THREE.Color('#d6fa59'),populationColor=new THREE.Color('#8bada0');
 world.userData.updateCars=(cars,leader)=>{
  const transforms=cars.map(c=>{
   if(c.dead)return hidden;
   transform.position.set(c.x,c.y+.42,c.z);transform.rotation.set(0,-c.heading,Math.atan(track.points[c.index].grade));
   transform.scale.setScalar(c===leader?1.32:1);transform.updateMatrix();return transform.matrix.clone();
  });
  for(const batch of carBatches){
   batch.items.forEach((item,i)=>{
    matrix.multiplyMatrices(transforms[item.car]??hidden,item.matrix);batch.mesh.setMatrixAt(i,matrix);
    if(batch.body)batch.mesh.setColorAt(i,cars[item.car]===leader?leaderColor:populationColor);
   });
   batch.mesh.instanceMatrix.needsUpdate=true;if(batch.body)batch.mesh.instanceColor.needsUpdate=true;
  }
 };
 // Gravel plumes and tire marks. Both are single instanced meshes added after batching, so the
 // whole effect layer costs two draw calls. Dust fades by tinting toward the fog colour, which is
 // what InstancedMesh can do without per-instance alpha.
 const DUST_MAX=260,MARK_MAX=560,DUST_LIFE=1.15,EFFECT_STEP=1/60;
 const dustColor=new THREE.Color('#cdbf97'),fadeColor=new THREE.Color('#3a4636'),scratch=new THREE.Color();
 const dustMesh=new THREE.InstancedMesh(new THREE.PlaneGeometry(2.4,2.4),
  new THREE.MeshBasicMaterial({color:'#ffffff',transparent:true,opacity:.46,depthWrite:false}),DUST_MAX);
 const markMesh=new THREE.InstancedMesh(new THREE.PlaneGeometry(1.05,2.9),
  new THREE.MeshBasicMaterial({color:wet?'#241f18':'#332c21',transparent:true,opacity:wet?.3:.4,depthWrite:false}),MARK_MAX);
 for(const m of [dustMesh,markMesh]){
  m.castShadow=false;m.receiveShadow=false;m.frustumCulled=false;
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  world.add(m);
 }
 const hiddenMatrix=new THREE.Matrix4().makeScale(0,0,0);
 for(let i=0;i<DUST_MAX;i++){dustMesh.setMatrixAt(i,hiddenMatrix);dustMesh.setColorAt(i,dustColor);}
 for(let i=0;i<MARK_MAX;i++)markMesh.setMatrixAt(i,hiddenMatrix);
 const dust=Array.from({length:DUST_MAX},()=>({age:Infinity,x:0,y:0,z:0,vx:0,vy:0,vz:0,spin:0}));
 const effectTransform=new THREE.Object3D();
 let dustCursor=0,markCursor=0,markCooldown=new Map();
 world.userData.updateEffects=(cars,emit=true)=>{
  let spawned=0;
  for(const car of (emit?cars:[])){
   if(!car||car.dead||car.finished)continue;
   const intensity=Math.max(car.slide??0,car.offRoad?.55:0);
   if(intensity<=.18||car.speed<6)continue;
   // Cap emissions so a whole sliding field cannot drain the pool in one frame.
   if(spawned<7&&Math.random()<intensity*.85){
    spawned++;
    const p=dust[dustCursor++%DUST_MAX];
    const back=-Math.cos(car.heading)*2.1,backZ=-Math.sin(car.heading)*2.1;
    p.age=0;p.x=car.x+back;p.y=car.y+.35;p.z=car.z+backZ;
    p.vx=back*1.6+(Math.random()-.5)*3.4;p.vz=backZ*1.6+(Math.random()-.5)*3.4;
    p.vy=2.4+Math.random()*2.6*intensity;p.spin=Math.random()*Math.PI;
   }
   const last=markCooldown.get(car.id)??-1;
   if(car.slide>.3&&!car.offRoad&&last!==Math.round(car.time*20)){
    markCooldown.set(car.id,Math.round(car.time*20));
    effectTransform.position.set(car.x,car.y+ROAD_SURFACE+PAINT_LIFT*2,car.z);
    effectTransform.rotation.set(-Math.PI/2,0,-car.heading);
    effectTransform.scale.set(1,Math.min(2.2,.8+car.speed/22),1);
    effectTransform.updateMatrix();
    markMesh.setMatrixAt(markCursor++%MARK_MAX,effectTransform.matrix);
    markMesh.instanceMatrix.needsUpdate=true;
   }
  }
  for(let i=0;i<DUST_MAX;i++){
   const p=dust[i];
   if(p.age===Infinity)continue;
   p.age+=EFFECT_STEP;
   if(p.age>=DUST_LIFE){p.age=Infinity;dustMesh.setMatrixAt(i,hiddenMatrix);continue;}
   p.x+=p.vx*EFFECT_STEP;p.y+=p.vy*EFFECT_STEP;p.z+=p.vz*EFFECT_STEP;
   p.vy-=7*EFFECT_STEP;p.vx*=.965;p.vz*=.965;
   const life=p.age/DUST_LIFE;
   effectTransform.position.set(p.x,p.y,p.z);
   effectTransform.rotation.set(0,p.spin,0);
   effectTransform.scale.setScalar(.5+life*2.3);
   effectTransform.updateMatrix();
   dustMesh.setMatrixAt(i,effectTransform.matrix);
   dustMesh.setColorAt(i,scratch.copy(dustColor).lerp(fadeColor,life));
  }
  dustMesh.instanceMatrix.needsUpdate=true;
  if(dustMesh.instanceColor)dustMesh.instanceColor.needsUpdate=true;
 };
 world.userData.clearEffects=()=>{
  for(const p of dust)p.age=Infinity;
  markCooldown.clear();
  for(let i=0;i<DUST_MAX;i++)dustMesh.setMatrixAt(i,hiddenMatrix);
  for(let i=0;i<MARK_MAX;i++)markMesh.setMatrixAt(i,hiddenMatrix);
  dustMesh.instanceMatrix.needsUpdate=true;markMesh.instanceMatrix.needsUpdate=true;
 };
 const effectMeshes=[dustMesh,markMesh].length;
 world.userData.stats={sourceMeshes:sourceCount,renderMeshes:world.children.length,
  batchedMeshes:world.children.length-effectMeshes,effectMeshes,
  shadowCasters:world.children.filter(o=>o.castShadow).length};
 world.userData.dispose=()=>{
  const geometries=new Set(),materials=new Set();world.traverse(o=>{if(o.isMesh){geometries.add(o.geometry);materials.add(o.material);if(o.isInstancedMesh)o.dispose();}});
  for(const geometry of geometries)geometry.dispose();for(const material of materials)material.dispose();
 };
 return world;
}

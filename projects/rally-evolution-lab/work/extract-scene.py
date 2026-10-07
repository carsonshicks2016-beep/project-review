from pathlib import Path
s=Path('dist/app.mjs').read_text()
start=s.index('const mat=');end=s.index('function updateTrackInfo')
old=s[start:end]
material_end=old.index('function mesh(')
materials=old[:material_end]
helpers=old[material_end:old.index('function buildWorld()')]
body=old[old.index(' const geometry='):]
body=body.replace(' updateTrackInfo();drawProfile();}', '')
body=body.replace("mesh(new THREE.CylinderGeometry(.5,.8,h*.4,5),trunkMat,x,y+h*.2,z);", "mesh(new THREE.CylinderGeometry(.5,.8,1,5),trunkMat,x,y+h*.2,z).scale.y=h*.4;")
body=body.replace("mesh(new THREE.ConeGeometry(h*.3,h*.85,5),treeMats[i%3],x,y+h*.65,z);", "mesh(new THREE.ConeGeometry(1,1,5),treeMats[i%3],x,y+h*.65,z).scale.set(h*.3,h*.85,h*.3);")
body=body.replace("mesh(new THREE.ConeGeometry(h*.23,h*.68,5),treeMats[(i+1)%3],x,y+h*.95,z);", "mesh(new THREE.ConeGeometry(1,1,5),treeMats[(i+1)%3],x,y+h*.95,z).scale.set(h*.23,h*.68,h*.23);")
new="import * as THREE from './vendor/three.module.js';\nimport {rng,POPULATION} from './engine.mjs';\nexport function createWorld(track){\n let world=new THREE.Group(),carModels=[];\n"+materials+helpers+body
new+='''
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
   transform.position.set(c.x,c.y+.12,c.z);transform.rotation.set(0,-c.heading,Math.atan(track.points[c.index].grade));
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
 world.userData.stats={sourceMeshes:sourceCount,renderMeshes:world.children.length,shadowCasters:world.children.filter(o=>o.castShadow).length};
 world.userData.dispose=()=>{
  const geometries=new Set(),materials=new Set();world.traverse(o=>{if(o.isMesh){geometries.add(o.geometry);materials.add(o.material);if(o.isInstancedMesh)o.dispose();}});
  for(const geometry of geometries)geometry.dispose();for(const material of materials)material.dispose();
 };
 return world;
}
'''
Path('dist/scene.mjs').write_text(new)
s=s[:start]+'''function buildWorld(){
 if(world){scene.remove(world);world.userData.dispose();}
 world=createWorld(track);scene.add(world);updateTrackInfo();drawProfile();refreshLines();
}
'''+s[end:]
a=s.index(' carModels.forEach');b=s.index(" if(cameraMode==='follow'",a)
s=s[:a]+' world.userData.updateCars(cars,leader);\n'+s[b:]
s=s.replace("import * as THREE from './vendor/three.module.js';", "import * as THREE from './vendor/three.module.js';\nimport {createWorld} from './scene.mjs';")
Path('dist/app.mjs').write_text(s)

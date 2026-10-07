import * as THREE from 'three';

// Observer camera only: geometry and recorded dynamics remain unchanged.
export function followPosition(target,bounds=[]){
  const initial=new THREE.Vector3(-8,-11,7),ray=new THREE.Ray(),hit=new THREE.Vector3();
  const clear=position=>{
    const delta=target.clone().sub(position),length=delta.length();ray.set(position,delta.normalize());
    return !bounds.some(box=>box.containsPoint(position)||(ray.intersectBox(box,hit)&&hit.distanceTo(position)<length-.1));
  };
  for(const height of [7,14,28])for(const angle of [0,Math.PI/2,-Math.PI/2,Math.PI,Math.PI/4,-Math.PI/4,Math.PI*.75,-Math.PI*.75]){
    const offset=initial.clone().applyAxisAngle(new THREE.Vector3(0,0,1),angle);offset.z=height;
    const candidate=target.clone().add(offset);if(clear(candidate))return candidate;
  }
  return target.clone().add(new THREE.Vector3(0,-.01,35));
}

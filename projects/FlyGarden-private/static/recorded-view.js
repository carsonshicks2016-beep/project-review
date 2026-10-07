import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {followPosition} from './recording-camera.js';

// Shared by interactive playback and offline export: recorded poses are the only motion source.
export class RecordedView {
  constructor(canvas,{interactive=true,camera='follow'}={}) {
    this.canvas=canvas;this.cameraMode=camera;this.interactive=interactive;
    this.renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});this.renderer.setPixelRatio(1);this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.autoClear=false;
    this.arena=new THREE.Scene();this.arena.background=new THREE.Color('#28382b');this.arena.add(new THREE.HemisphereLight(0xe5efc5,0x314531,2));
    const sun=new THREE.DirectionalLight(0xfff2d4,3);sun.position.set(-25,-20,70);this.arena.add(sun);
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(84,64),new THREE.MeshStandardMaterial({color:0x455640,roughness:1}));this.arena.add(ground);
    const grid=new THREE.GridHelper(80,20,0x75846a,0x526349);grid.rotation.x=Math.PI/2;grid.position.z=.015;this.arena.add(grid);
    this.arenaCamera=new THREE.PerspectiveCamera(42,1,.1,500);this.arenaCamera.up.set(0,0,1);
    this.brain=new THREE.Scene();this.brain.background=new THREE.Color('#101c24');this.brainCamera=new THREE.PerspectiveCamera(40,1,.1,5000);this.brainCamera.up.set(0,-1,0);
    this.fly=new THREE.Group();this.objects=new THREE.Group();this.arena.add(this.fly,this.objects);this.flyMeshes=new Map();this.foods=new Map();this.lines=new Map();this.shapes=[];this.activityMode='rate';this.activeOnly=false;this.selected=null;
    if(interactive){
      this.controls=new OrbitControls(this.brainCamera,canvas);this.controls.enableDamping=true;this.controls.enablePan=false;this.controls.minDistance=20;this.controls.maxDistance=2200;this.controls.enabled=false;
      this.arenaControls=new OrbitControls(this.arenaCamera,canvas);this.arenaControls.enableDamping=true;this.arenaControls.minDistance=1;this.arenaControls.maxDistance=180;this.arenaControls.enabled=false;
      const route=e=>{const r=canvas.getBoundingClientRect(),brain=e.clientX-r.left>r.width*.55;this.controls.enabled=brain;this.arenaControls.enabled=!brain&&this.cameraMode==='free';};
      canvas.addEventListener('pointerdown',route,{capture:true});canvas.addEventListener('wheel',route,{capture:true});
      let down=null;canvas.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});canvas.addEventListener('click',e=>{if(down&&Math.hypot(e.clientX-down[0],e.clientY-down[1])<5)this.pick(e);});
    }
  }
  setCameraMode(mode){
    if(mode==='free'&&this.cameraMode!=='free'&&this.interactive){
      this.arenaControls.dispose();this.arenaControls=new OrbitControls(this.arenaCamera,this.canvas);this.arenaControls.enableDamping=true;this.arenaControls.minDistance=1;this.arenaControls.maxDistance=180;this.arenaControls.target.copy(this.arenaTarget||new THREE.Vector3());this.arenaControls.update();
    }
    this.cameraMode=mode;if(this.arenaControls)this.arenaControls.enabled=false;
  }
  cameraView(){return {position:this.arenaCamera.position.toArray(),target:(this.cameraMode==='free'?this.arenaControls?.target:this.arenaTarget||new THREE.Vector3()).toArray()};}
  restoreCamera(saved){
    if(!saved||![saved.position,saved.target].every(v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite)))return false;
    this.setCameraMode('free');this.arenaCamera.position.fromArray(saved.position);this.arenaControls?.target.fromArray(saved.target);this.arenaTarget=new THREE.Vector3(...saved.target);this.arenaCamera.lookAt(this.arenaTarget);this.arenaControls?.update();return true;
  }
  clear(group){group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});group.clear();}
  setGeometry(data){this.clear(this.fly);this.flyMeshes.clear();const cache=new Map();for(const [key,m] of Object.entries(data.meshes||{})){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(m.vertices,3));g.setIndex(m.faces);g.computeVertexNormals();cache.set(key,g);}for(const g of data.geoms||[]){const mesh=new THREE.Mesh(cache.get(g.mesh),new THREE.MeshStandardMaterial({color:new THREE.Color(...g.color.slice(0,3)),roughness:.65,side:THREE.DoubleSide}));mesh.matrixAutoUpdate=false;this.fly.add(mesh);this.flyMeshes.set(g.id,mesh);}}
  setArena(arena){this.clear(this.objects);this.foods.clear();this.obstacleBounds=[];const box=(w,h,z,color)=>new THREE.Mesh(new THREE.BoxGeometry(w,h,z),new THREE.MeshStandardMaterial({color,roughness:.9}));
    for(const b of arena.blocks){const m=box(b.w,b.h,b.z,b.kind==='terrain'?0xa69c77:b.kind==='shelter'?0x809077:0x69765e);m.position.set(b.x,b.y,b.z/2);this.objects.add(m);m.updateMatrixWorld(true);this.obstacleBounds.push(new THREE.Box3().setFromObject(m).expandByScalar(.15));}
    for(const f of arena.foods){const m=new THREE.Mesh(new THREE.SphereGeometry(.95,20,12),new THREE.MeshStandardMaterial({color:f.odor?0xb9a5e8:0xc5e28c,roughness:.75}));m.position.set(f.x,f.y,.75);this.objects.add(m);this.foods.set(f.id,m);}
    this.predator=new THREE.Mesh(new THREE.SphereGeometry(1.6,20,12),new THREE.MeshStandardMaterial({color:0xbc715d,roughness:.65}));this.predator.scale.set(1.5,.8,.6);this.objects.add(this.predator);
  }
  makeShape(row,full=false){const s=row.shape,v=s.vertices,e=s.edges;const pairs=[];
    if(full||s.lod==='coarse'){for(let i=0;i<e.length;i+=2)pairs.push([e[i],e[i+1]]);}
    else {
      // Collapse degree-two chains, preserving every branch point and terminal.
      const adjacency=Array.from({length:v.length/3},()=>[]);for(let i=0;i<e.length;i+=2){adjacency[e[i]].push(e[i+1]);adjacency[e[i+1]].push(e[i]);}
      const visited=new Set();const edgeKey=(a,b)=>a<b?`${a}:${b}`:`${b}:${a}`;
      for(let a=0;a<adjacency.length;a++){if(adjacency[a].length===2)continue;for(const neighbor of adjacency[a]){if(visited.has(edgeKey(a,neighbor)))continue;let previous=a,current=neighbor,anchor=a,steps=1;visited.add(edgeKey(a,neighbor));while(true){if(adjacency[current].length!==2||steps%4===0){pairs.push([anchor,current]);anchor=current;}if(adjacency[current].length!==2)break;const next=adjacency[current][0]===previous?adjacency[current][1]:adjacency[current][0];if(visited.has(edgeKey(current,next)))break;visited.add(edgeKey(current,next));previous=current;current=next;steps++;}}}
      // Preserve closed loops instead of silently omitting them.
      for(let i=0;i<e.length;i+=2)if(!visited.has(edgeKey(e[i],e[i+1])))pairs.push([e[i],e[i+1]]);
    }
    const positions=new Float32Array(pairs.length*6);let cursor=0;for(const pair of pairs)for(const index of pair)for(let d=0;d<3;d++)positions[cursor++]=v[index*3+d];
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.computeBoundingSphere();const material=new THREE.LineBasicMaterial({color:0x101b22,transparent:true,opacity:.18});const line=new THREE.LineSegments(geometry,material);line.userData=row;this.brain.add(line);this.lines.set(row.id,line);return line;
  }
  setNeurons(rows){this.clear(this.brain);this.lines.clear();this.selected=null;this.shapes=rows;for(const row of rows)this.makeShape(row);this.fitBrain();}
  addNeuron(row){if(this.lines.has(row.id))return;this.shapes.push(row);this.makeShape(row);}
  fitBrain(){const bounds=new THREE.Box3();for(const line of this.lines.values())bounds.expandByObject(line);if(bounds.isEmpty()){this.brainCamera.position.set(0,0,-500);this.brainCamera.lookAt(0,0,0);return;}this.center=bounds.getCenter(new THREE.Vector3());const size=bounds.getSize(new THREE.Vector3()),span=Math.max(...size.toArray(),30);this.span=span;const aspect=Math.max(.15,(this.canvas.clientWidth||1920)*.45/(this.canvas.clientHeight||1080)),distance=Math.max(size.y,size.x/aspect,30)/(2*Math.tan(20*Math.PI/180))*1.35+size.z*.5;this.brainCamera.position.copy(this.center).add(new THREE.Vector3(0,0,-distance));this.brainCamera.lookAt(this.center);this.controls?.target.copy(this.center);}
  select(id){const old=this.selected;this.selected=id;for(const key of [old,id]){const line=this.lines.get(key);if(!line)continue;const row=line.userData;this.brain.remove(line);line.geometry.dispose();line.material.dispose();this.makeShape(row,key===id);}this.onSelect?.(this.lines.get(id)?.userData);}
  pick(e){const r=this.canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width;if(x<.55)return;const pointer=new THREE.Vector2((x-.55)/.45*2-1,-(e.clientY-r.top)/r.height*2+1);const ray=new THREE.Raycaster();ray.params.Line.threshold=(this.span||200)/500;ray.setFromCamera(pointer,this.brainCamera);const hit=ray.intersectObjects([...this.lines.values()].filter(o=>o.visible))[0];if(hit)this.select(hit.object.userData.id);}
  draw(state,time,rates={},lastSpike={},nextState=null){const w=this.canvas.clientWidth||1920,h=this.canvas.clientHeight||850;if(this.canvas.width!==w||this.canvas.height!==h)this.renderer.setSize(w,h,false);
    const split=Math.floor(w*.55),matrix=new THREE.Matrix4();const duration=(nextState?.world?.time||time)-(state.world?.time||0),alpha=nextState&&duration>0?Math.max(0,Math.min(1,(time-state.world.time)/duration)):0;const nextGeoms=new Map((nextState?.visuals||[]).map(g=>[g.id,g]));for(const g of state.visuals||[]){const m=this.flyMeshes.get(g.id);if(!m)continue;const r=g.r,p=g.p;matrix.set(r[0],r[1],r[2],p[0],r[3],r[4],r[5],p[1],r[6],r[7],r[8],p[2],0,0,0,1);const next=nextGeoms.get(g.id);if(next&&alpha>0){const q0=new THREE.Quaternion().setFromRotationMatrix(matrix),nr=next.r,rotation=new THREE.Matrix4().set(nr[0],nr[1],nr[2],0,nr[3],nr[4],nr[5],0,nr[6],nr[7],nr[8],0,0,0,0,1),q1=new THREE.Quaternion().setFromRotationMatrix(rotation);matrix.compose(new THREE.Vector3(...g.p).lerp(new THREE.Vector3(...next.p),alpha),q0.slerp(q1,alpha),new THREE.Vector3(1,1,1));}m.matrix.copy(matrix);}
    for(const f of state.arena?.foods||[]){const m=this.foods.get(f.id);if(m){m.visible=f.units>0;m.scale.setScalar(.65+.12*f.units);}}
    const predator=state.arena?.predator;if(predator){this.predator.visible=predator.enabled;this.predator.position.set(predator.x,predator.y,1.2);}
    const position=new THREE.Vector3(...(state.body?.position||[0,0,0]));if(nextState?.body?.position)position.lerp(new THREE.Vector3(...nextState.body.position),alpha);if(this.cameraMode!=='free'){this.arenaTarget=this.cameraMode==='overhead'?new THREE.Vector3():position;this.arenaCamera.position.copy(this.cameraMode==='overhead'?new THREE.Vector3(0,-.01,90):followPosition(position,this.obstacleBounds));this.arenaCamera.lookAt(this.arenaTarget);}else this.arenaControls?.update();
    let visible=0;for(const row of this.shapes){const line=this.lines.get(row.id);const rate=rates[row.index]||0;
      const age=time-(lastSpike[row.index]??-Infinity);
      const gain=this.activityMode==='flash'?(age>=0?Math.max(0,1-age/.015):0):Math.min(1,rate/500);
      line.visible=!this.activeOnly||gain>0||row.id===this.selected;if(line.visible)visible++;
      const color=new THREE.Color().setHSL(.58-gain*.58,.7,.035+gain*.585);line.material.color.copy(color);
      // Selection retains the activity color; it must not look like continuous firing.
      line.material.opacity=gain>0?.55+gain*.4:(row.id===this.selected?.4:.18);
    }
    this.controls?.update();this.arenaCamera.aspect=split/h;this.arenaCamera.updateProjectionMatrix();this.brainCamera.aspect=(w-split)/h;this.brainCamera.updateProjectionMatrix();this.renderer.setScissorTest(true);this.renderer.setViewport(0,0,split,h);this.renderer.setScissor(0,0,split,h);this.renderer.clear();this.renderer.render(this.arena,this.arenaCamera);this.renderer.setViewport(split,0,w-split,h);this.renderer.setScissor(split,0,w-split,h);this.renderer.clear();this.renderer.render(this.brain,this.brainCamera);this.renderer.setScissorTest(false);return visible;
  }
}

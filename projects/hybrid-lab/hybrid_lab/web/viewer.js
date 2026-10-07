import * as THREE from './vendor/three.module.js';

const clamp = THREE.MathUtils.clamp;
const lerp = THREE.MathUtils.lerp;
const world = (p, h = 0) => new THREE.Vector3(p[0], h, -p[1]);

export class Viewer {
  constructor(element) {
    this.element = element;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#72847e');
    this.scene.fog = new THREE.FogExp2('#72847e', .0018);
    this.camera = new THREE.PerspectiveCamera(52, 1, .15, 6000);
    this.renderer = new THREE.WebGLRenderer({antialias:true, powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    element.appendChild(this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight('#e6f1e7','#33442a', 2.5));
    this.sun = new THREE.DirectionalLight('#fff1cd', 3.3);
    this.sun.position.set(80, 140, -60);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, {left:-55,right:55,top:55,bottom:-55,near:1,far:450});
    this.sun.shadow.bias = -.0004;
    this.scene.add(this.sun, this.sun.target);
    this.trackGroup = new THREE.Group();
    this.scene.add(this.trackGroup);
    this.car = this.makeCar();
    this.scene.add(this.car);
    this.target = null;
    this.cameraMode = 'chase';
    this.distance = 19;
    this.orbitAngle = .7;
    this.orbitElevation = .55;
    this.smoothYaw = 0;
    this.snapped = false;
    this.rays = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:'#e0ff76',transparent:true,opacity:.55}));
    this.scene.add(this.rays);
    this.rays.visible = false;
    this.trailPoints = [];
    this.trails = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:'#131915',transparent:true,opacity:.58}));
    this.scene.add(this.trails);
    this.lastTrail = null;
    let pointer = null;
    element.addEventListener('pointerdown', e => {pointer={x:e.clientX,y:e.clientY};element.setPointerCapture(e.pointerId);});
    element.addEventListener('pointerup',()=>pointer=null);
    element.addEventListener('pointercancel',()=>pointer=null);
    element.addEventListener('pointermove',e=>{
      if(!pointer) return;
      this.setCamera('orbit');
      this.orbitAngle -= (e.clientX-pointer.x)*.007;
      this.orbitElevation=clamp(this.orbitElevation+(e.clientY-pointer.y)*.005,.12,1.45);
      pointer={x:e.clientX,y:e.clientY};
    });
    element.addEventListener('wheel',e=>{e.preventDefault();this.distance=clamp(this.distance+e.deltaY*.02,7,120);},{passive:false});
    element.style.touchAction='none';
    new ResizeObserver(()=>this.resize()).observe(element);
    this.resize();
    this.lastTime=performance.now();
    this.renderer.setAnimationLoop(t=>this.animate(t));
  }

  material(color, roughness=.7, metalness=0) {return new THREE.MeshStandardMaterial({color,roughness,metalness});}
  box(w,h,d,material,x=0,y=0,z=0,parent=this.trackGroup) {
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),material);
    mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;
  }

  makeCar() {
    const car=new THREE.Group();
    this.paint=this.material('#d9e76a',.28,.5);
    const carbon=this.material('#202b2b',.48,.25), rubber=this.material('#111718'), rim=this.material('#bdc6c3',.3,.8);
    const glass=new THREE.MeshStandardMaterial({color:'#263b3b',metalness:.5,roughness:.12});
    // Bespoke coupe body, built from cross sections along the longitudinal axis.
    const sections=[[-2.2,.52,.7],[-1.85,.77,.94],[-.75,.88,.98],[.8,.76,.93],[1.75,.7,.87],[2.2,.55,.78]];
    const pos=[],indices=[];
    for(const [x,y,w] of sections) pos.push(x,.35,-w,x,y,-w*.96,x,y,w*.96,x,.35,w);
    for(let i=0;i<sections.length-1;i++) for(let j=0;j<4;j++){
      const a=i*4+j,b=i*4+(j+1)%4,c=(i+1)*4+j,d=(i+1)*4+(j+1)%4;
      indices.push(a,c,b,b,c,d);
    }
    indices.push(0,1,2,0,2,3,20,22,21,20,23,22);
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geo.setIndex(indices);geo.computeVertexNormals();
    const body=new THREE.Mesh(geo,this.paint);body.castShadow=true;body.receiveShadow=true;car.add(body);
    this.box(4.45,.12,1.9,carbon,0,.34,0,car);
    this.box(1.25,.06,1.39,this.paint,-.37,1.3,0,car);
    const cabinPos=[-1.25,.78,-.8,-.9,1.28,-.64,.25,1.28,-.64,.98,.76,-.8,
                    -1.25,.78,.8,-.9,1.28,.64,.25,1.28,.64,.98,.76,.8];
    const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.Float32BufferAttribute(cabinPos,3));
    cg.setIndex([0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,2,3,7,2,7,6]);cg.computeVertexNormals();
    const cabin=new THREE.Mesh(cg,glass);cabin.material.side=THREE.DoubleSide;car.add(cabin);
    this.box(.055,.49,1.38,this.paint,-.28,1.02,0,car);
    this.box(.32,.03,1.66,carbon,1.35,.77,0,car);
    const light = new THREE.MeshStandardMaterial({color:'#f1ffed',emissive:'#e2f5b5',emissiveIntensity:2});
    const red=new THREE.MeshStandardMaterial({color:'#d8573e',emissive:'#bd2713',emissiveIntensity:1.5});
    for(const side of [-1,1]){
      this.box(.07,.1,.5,light,2.18,.58,side*.48,car);
      this.box(.07,.11,.59,red,-2.19,.58,side*.48,car);
      this.box(.22,.10,.22,carbon,.1,1,side*.91,car);
      this.box(.18,.34,.06,carbon,-1.7,1,side*.65,car);
      this.box(.15,.12,.23,carbon,-2.2,.35,side*.65,car);
    }
    this.box(.45,.065,2.08,carbon,-1.8,1.19,0,car);
    this.wheels=[];
    for(const x of [-1.28,1.27]) for(const z of [-.92,.92]) {
      const pivot=new THREE.Group();pivot.position.set(x,.35,z);car.add(pivot);
      const wheel=new THREE.Group();pivot.add(wheel);
      const tire=new THREE.Mesh(new THREE.CylinderGeometry(.35,.35,.24,24),rubber);tire.rotation.x=Math.PI/2;wheel.add(tire);tire.castShadow=true;
      const hub=new THREE.Mesh(new THREE.CylinderGeometry(.24,.24,.253,16),rim);hub.rotation.x=Math.PI/2;wheel.add(hub);
      const cap=new THREE.Mesh(new THREE.CylinderGeometry(.12,.12,.26,12),carbon);cap.rotation.x=Math.PI/2;wheel.add(cap);
      for(let a=0;a<5;a++){
        const spoke=this.box(.42,.045,.26,carbon,0,0,0,wheel);spoke.rotation.z=a*Math.PI/5;
      }
      this.wheels.push({pivot,wheel,front:x>0});
    }
    // Hood stripe and door-number plate are geometry, requiring no remote assets.
    this.box(1.2,.015,.12,carbon,1.35,.76,-.22,car);
    this.box(.9,.22,.015,carbon,-.15,.65,-.975,car);
    this.box(.9,.22,.015,carbon,-.15,.65,.975,car);
    return car;
  }

  setCamera(mode) {
    this.cameraMode=mode;
    document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('selected',b.dataset.camera===mode));
  }

  setTrack(track) {
    if (!track || this.generation===track.generation) return;
    this.generation=track.generation;this.track=track;
    this.trackGroup.traverse(o=>{o.geometry?.dispose();if(o.material){(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>{m.map?.dispose();m.dispose();});}});
    this.trackGroup.clear();this.clearTrails();this.snapped=false;
    const n=track.center.length;
    const asphalt=this.material('#454d49',.97), white=this.material('#d4d7bf'), orange=this.material('#d19566');
    const roadPos=[],roadIdx=[],kerbPos=[],kerbColors=[];
    this.edges=[];
    for(let i=0;i<n;i++) {
      const h=track.elevation[i],bank=track.bank[i];
      const half=Math.hypot(track.left[i][0]-track.center[i][0],track.left[i][1]-track.center[i][1]);
      const l=world(track.left[i],h+bank*half),r=world(track.right[i],h-bank*half);
      this.edges.push([l,r]);roadPos.push(...l,...r);
    }
    for(let i=0;i<n;i++) {
      const j=(i+1)%n;roadIdx.push(i*2,j*2,i*2+1,i*2+1,j*2,j*2+1);
      for(let side=0;side<2;side++){
        const a=this.edges[i][side],b=this.edges[j][side];
        const ca=world(track.center[i],track.elevation[i]),cb=world(track.center[j],track.elevation[j]);
        const ao=a.clone().add(a.clone().sub(ca).normalize().multiplyScalar(.75));
        const bo=b.clone().add(b.clone().sub(cb).normalize().multiplyScalar(.75));
        const color=new THREE.Color(Math.floor(i/3)%2?'#d6d6ba':'#b56549');
        for(const p of [a,b,ao,ao,b,bo]){kerbPos.push(p.x,p.y+.045,p.z);kerbColors.push(color.r,color.g,color.b);}
      }
    }
    const rg=new THREE.BufferGeometry();rg.setAttribute('position',new THREE.Float32BufferAttribute(roadPos,3));rg.setIndex(roadIdx);rg.computeVertexNormals();
    const road=new THREE.Mesh(rg,asphalt);road.material.side=THREE.DoubleSide;road.receiveShadow=true;this.trackGroup.add(road);
    const kg=new THREE.BufferGeometry();kg.setAttribute('position',new THREE.Float32BufferAttribute(kerbPos,3));kg.setAttribute('color',new THREE.Float32BufferAttribute(kerbColors,3));kg.computeVertexNormals();
    const kerb=new THREE.Mesh(kg,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:1}));kerb.receiveShadow=true;this.trackGroup.add(kerb);
    for(let side=0;side<2;side++){
      const edge=this.edges.map(pair=>pair[side].clone().add(new THREE.Vector3(0,.06,0)));
      edge.push(edge[0]);const g=new THREE.BufferGeometry().setFromPoints(edge);
      this.trackGroup.add(new THREE.Line(g,new THREE.LineBasicMaterial({color:'#e3e4cb'})));
    }
    const xs=track.center.map(p=>p[0]),zs=track.center.map(p=>-p[1]);
    const minX=Math.min(...xs)-180,maxX=Math.max(...xs)+180,minZ=Math.min(...zs)-180,maxZ=Math.max(...zs)+180;
    this.bounds={minX,maxX,minZ,maxZ};
    this.trackCenter=new THREE.Vector3((minX+maxX)/2,Math.max(...track.elevation),(minZ+maxZ)/2);
    this.trackExtent=Math.max(maxX-minX-250,maxZ-minZ-250);
    // Height-aware ground follows the road's altitude and leaves the ribbon exposed.
    const ground=new THREE.PlaneGeometry(maxX-minX,maxZ-minZ,65,65);ground.rotateX(-Math.PI/2);
    ground.translate((minX+maxX)/2,0,(minZ+maxZ)/2);
    const gp=ground.attributes.position,colors=[];
    for(let i=0;i<gp.count;i++){
      const x=gp.getX(i),z=gp.getZ(i);let closest=Infinity,h=0;
      for(let j=0;j<n;j++){
        const d=(x-xs[j])**2+(z-zs[j])**2;
        if(d<closest){closest=d;h=track.elevation[j];}
      }
      const variation=Math.sin(x*.052)*Math.cos(z*.034);
      const y=h-2.5+variation*Math.min(3,Math.sqrt(closest)/35);
      gp.setY(i,y);
      const c=new THREE.Color('#607151');c.multiplyScalar(.88+variation*.08);
      colors.push(c.r,c.g,c.b);
    }
    ground.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));ground.computeVertexNormals();
    const field=new THREE.Mesh(ground,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1}));field.receiveShadow=true;this.trackGroup.add(field);
    // Instanced pines around the circuit: custom scenery with bounded draw calls.
    const treeCount=Math.floor(n/3),trunks=new THREE.InstancedMesh(new THREE.CylinderGeometry(.25,.42,3,5),this.material('#555345'),treeCount);
    const foliage=new THREE.InstancedMesh(new THREE.ConeGeometry(3.1,11,7),this.material('#344e3a'),treeCount);
    const dummy=new THREE.Object3D();
    for(let k=0;k<treeCount;k++){
      const i=k*3,side=k%2,edge=this.edges[i][side],center=world(track.center[i]);
      const away=edge.clone().sub(center);away.y=0;away.normalize();
      const point=edge.clone().add(away.multiplyScalar(14+(Math.sin(k*13.1)+1)*18));
      const s=.8+(Math.sin(k*8.7)+1)*.3;
      dummy.position.copy(point).add(new THREE.Vector3(0,-1,0));dummy.scale.set(s,s,s);dummy.updateMatrix();trunks.setMatrixAt(k,dummy.matrix);
      dummy.position.y+=5;dummy.updateMatrix();foliage.setMatrixAt(k,dummy.matrix);
    }
    trunks.castShadow=foliage.castShadow=true;this.trackGroup.add(trunks,foliage);
    const p0=world(track.center[0],track.elevation[0]+.06),p1=world(track.center[1]);
    const yaw=Math.atan2(-(p1.z-p0.z),p1.x-p0.x);
    const start=new THREE.Group();start.position.copy(p0);start.rotation.y=yaw;this.trackGroup.add(start);
    const width=this.edges[0][0].distanceTo(this.edges[0][1]);
    for(let i=0;i<12;i++)for(let j=0;j<2;j++)this.box(.65,.025,width/12,i%2===j?white:asphalt,j*.65,.02,-width/2+(i+.5)*width/12,start);
    const beam=this.material('#384a3b');
    this.box(.4,6,.4,beam,0,3,-width/2-1,start);this.box(.4,6,.4,beam,0,3,width/2+1,start);
    this.box(.5,.65,width+2.6,beam,0,6,0,start);
    const labelCanvas=document.createElement('canvas');labelCanvas.width=1024;labelCanvas.height=128;
    const ctx=labelCanvas.getContext('2d');ctx.fillStyle='#202b24';ctx.fillRect(0,0,1024,128);ctx.fillStyle='#dcff61';ctx.font='bold 62px sans-serif';ctx.textAlign='center';ctx.fillText('HYBRID  /  LAB',512,86);
    const tex=new THREE.CanvasTexture(labelCanvas);tex.colorSpace=THREE.SRGBColorSpace;
    const sign=new THREE.Mesh(new THREE.PlaneGeometry(width*.72,1.1),new THREE.MeshBasicMaterial({map:tex,side:THREE.DoubleSide}));sign.position.set(-.27,6.15,0);sign.rotation.y=-Math.PI/2;start.add(sign);
    // Paddock blocks and braking markers give useful speed/depth references.
    for(let k=0;k<5;k++){
      this.box(5,3.5,7,this.material('#949b84'),-8-k*6,1,-width/2-13,start);
      this.box(5.2,.2,7.2,beam,-8-k*6,2.8,-width/2-13,start);
    }
    for(let i=20;i<n;i+=35){
      const edge=this.edges[i][0];this.box(.4,1.3,.4,orange,edge.x,edge.y+.65,edge.z);
    }
    document.getElementById('scene-loading').hidden=true;
  }

  clearTrails() {this.trailPoints=[];this.lastTrail=null;this.trails.geometry.dispose();this.trails.geometry=new THREE.BufferGeometry();}

  setFrame(watching) {
    if(!watching?.frame)return;
    const f=watching.frame;
    if(this.episode!==watching.episode || (this.target && Math.hypot(f.x-this.target.x,f.y-this.target.y)>30)) {
      this.snapped=false;this.clearTrails();
    }
    this.episode=watching.episode;this.target=f;this.paused=watching.paused;this.rate=watching.rate;
    const colors={supra:'#d9e76a',rx7:'#cf7754',skyline:'#a2cbd2'};
    this.paint.color.set(colors[watching.car]||colors.supra);
    const positions=[];
    for(const p of f.beams||[])positions.push(f.x,f.z+.8,-f.y,p[0],f.z+.4,-p[1]);
    this.rays.geometry.dispose();this.rays.geometry=new THREE.BufferGeometry();
    this.rays.geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    const now=[f.x,f.z+.07,-f.y];
    if(this.lastTrail && !watching.paused && Math.abs(f.slip)>6 && !f.offtrack && f.speed>15){
      for(const side of [-.78,.78]){
        const dx=Math.sin(f.yaw)*side,dz=Math.cos(f.yaw)*side;
        this.trailPoints.push(this.lastTrail[0]+dx,this.lastTrail[1],this.lastTrail[2]+dz,now[0]+dx,now[1],now[2]+dz);
      }
      if(this.trailPoints.length>14400)this.trailPoints.splice(0,this.trailPoints.length-14400);
      this.trails.geometry.dispose();this.trails.geometry=new THREE.BufferGeometry();
      this.trails.geometry.setAttribute('position',new THREE.Float32BufferAttribute(this.trailPoints,3));
    }
    this.lastTrail=now;
    this.drawMap(f);
  }

  drawMap(f) {
    if(!this.track)return;
    const canvas=document.getElementById('minimap'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    const t=this.track,b=this.bounds,scale=Math.min((w-18)/(b.maxX-b.minX-320),(h-18)/(b.maxZ-b.minZ-320));
    const cx=(b.maxX+b.minX)/2,cz=(b.maxZ+b.minZ)/2;
    const point=p=>[(p[0]-cx)*scale+w/2,(-p[1]-cz)*scale+h/2];
    ctx.clearRect(0,0,w,h);ctx.beginPath();
    t.center.forEach((p,i)=>{const[x,y]=point(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.strokeStyle='#b8c8b399';ctx.lineWidth=3;ctx.stroke();
    const[x,y]=point([f.x,f.y]);ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fillStyle='#dcff61';ctx.fill();
  }

  resize() {
    const w=this.element.clientWidth,h=this.element.clientHeight;if(!w||!h)return;
    this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();
  }

  animate(time) {
    const dt=clamp((time-this.lastTime)/1000,0,.1);this.lastTime=time;
    if(this.target && this.track){
      const f=this.target,a=this.snapped?1-Math.exp(-dt*15):1;
      const pos=new THREE.Vector3(f.x,f.z,-f.y);
      this.car.position.lerp(pos,a);
      this.smoothYaw+=Math.atan2(Math.sin(f.yaw-this.smoothYaw),Math.cos(f.yaw-this.smoothYaw))*a;
      this.car.rotation.set(0,this.smoothYaw,0);
      this.car.rotateZ(lerp(0,f.pitch,a));this.car.rotateX(-f.roll);
      for(const wheel of this.wheels){if(!this.paused)wheel.wheel.rotation.z-=(f.speed/3.6)*dt*(this.rate||1)/.35;wheel.pivot.rotation.y=wheel.front?f.steer*.52:0;}
      const heading=new THREE.Vector3(Math.cos(this.smoothYaw),0,-Math.sin(this.smoothYaw));
      const aim=this.car.position.clone().add(new THREE.Vector3(0,1,0));
      let cameraTarget,look=aim.clone();
      if(this.cameraMode==='overhead'){
        cameraTarget=this.trackCenter.clone().add(new THREE.Vector3(0,this.trackExtent*1.05,this.trackExtent*.3));look=this.trackCenter;
      }else if(this.cameraMode==='hood'){
        cameraTarget=aim.clone().addScaledVector(heading,.8).add(new THREE.Vector3(0,.12,0));look=aim.clone().addScaledVector(heading,70);
      }else if(this.cameraMode==='orbit'){
        cameraTarget=aim.clone().add(new THREE.Vector3(Math.cos(this.orbitAngle)*this.distance,Math.sin(this.orbitElevation)*this.distance,Math.sin(this.orbitAngle)*this.distance));
      }else{
        cameraTarget=aim.clone().addScaledVector(heading,-this.distance).add(new THREE.Vector3(0,this.distance*.46,0));look.addScaledVector(heading,9);
      }
      this.camera.position.lerp(cameraTarget,this.snapped?1-Math.exp(-dt*6):1);
      this.camera.lookAt(look);
      this.sun.position.copy(this.car.position).add(new THREE.Vector3(80,140,-60));this.sun.target.position.copy(this.car.position);
      this.snapped=true;
    }
    this.renderer.render(this.scene,this.camera);
  }
}

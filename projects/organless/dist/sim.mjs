export const SIZE=17, COUNT=SIZE*SIZE, INPUTS=8, PARAMS=24;
export const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
export function rng(seed=1){let s=seed>>>0;return()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
export const XY=Array.from({length:COUNT},(_,i)=>[i%SIZE-8,Math.floor(i/SIZE)-8]);
export const NEIGHBORS=XY.map(([x,y],i)=>[x>-8?i-1:-1,x<8?i+1:-1,y>-8?i-SIZE:-1,y<8?i+SIZE:-1].filter(j=>j>=0));
export class Tissue {
 constructor(weights,seed=1){
  if(!weights||weights.length!==PARAMS||!weights.every(Number.isFinite))throw Error('Expected 24 finite weights');
  this.weights=Float64Array.from(weights);this.seed=seed;this.tick=0;this.x=0;this.y=0;this.vx=0;this.vy=0;this.frozen=false;this.intake=0;this.distance=0;this.transitions=0;
  this.alive=new Uint8Array(COUNT);this.energy=new Float64Array(COUNT);this.roles=new Float64Array(COUNT*3);this.nextRoles=new Float64Array(COUNT*3);this.sx=new Float64Array(COUNT);this.sy=new Float64Array(COUNT);this.nextE=new Float64Array(COUNT);this.nextX=new Float64Array(COUNT);this.nextY=new Float64Array(COUNT);this.exposure=new Float64Array(COUNT);this.food=new Float64Array(COUNT);this.inputs=new Float64Array(INPUTS);this.changed=new Float64Array(COUNT);
  const r=rng(seed);for(let i=0;i<COUNT;i++){const[x,y]=XY[i];if(x*x+y*y<=49){this.alive[i]=1;this.energy[i]=.45+r()*.16;for(let k=0;k<3;k++)this.roles[i*3+k]=1/3;}}
  const angle=r()*Math.PI*2;this.target={x:Math.cos(angle)*20,y:Math.sin(angle)*20};this.initialCount=this.alive.reduce((a,b)=>a+b,0);this.damageSnapshot=null;
 }
 setTarget(x,y){if(!Number.isFinite(x)||!Number.isFinite(y))throw Error('Invalid target');this.target={x:clamp(x,-80,80),y:clamp(y,-80,80)};}
 step(){
  let fx=0,fy=0,n=0,uptake=0;
  for(let i=0;i<COUNT;i++){
   if(!this.alive[i])continue;n++;let neighbors=0,meanE=0,meanX=0,meanY=0;
   for(const j of NEIGHBORS[i])if(this.alive[j]){neighbors++;meanE+=this.energy[j];meanX+=this.sx[j];meanY+=this.sy[j];}
   meanE/=Math.max(1,neighbors);meanX/=Math.max(1,neighbors);meanY/=Math.max(1,neighbors);
   const exposure=1-neighbors/4;this.exposure[i]=exposure;
   const dx=this.target.x-this.x-XY[i][0],dy=this.target.y-this.y-XY[i][1],d=Math.hypot(dx,dy);
   const food=.04+.96*Math.exp(-d*d/350);this.food[i]=food;
   const e=this.energy[i],sig=Math.hypot(this.sx[i],this.sy[i]);
   const input=this.inputs;input[0]=1;input[1]=e;input[2]=meanE;input[3]=exposure;input[4]=food*exposure;input[5]=sig;input[6]=Math.abs(e-meanE);input[7]=Math.max(0,meanE-e);
   if(!this.frozen){let logits=[0,0,0];for(let k=0;k<3;k++)for(let f=0;f<INPUTS;f++)logits[k]+=this.weights[k*INPUTS+f]*input[f];const max=Math.max(...logits);let sum=0;for(let k=0;k<3;k++){logits[k]=Math.exp(logits[k]-max);sum+=logits[k];}
    const old=this.dominant(i);for(let k=0;k<3;k++)this.nextRoles[i*3+k]=.8*this.roles[i*3+k]+.2*logits[k]/sum;
    const next=this.nextRoles.subarray(i*3,i*3+3);if(old!==next.indexOf(Math.max(...next))){this.transitions++;this.changed[i]=this.tick;}
   }
   if(this.frozen)this.nextRoles.set(this.roles.subarray(i*3,i*3+3),i*3);
   const sense=this.roles[i*3],transport=this.roles[i*3+1],motor=this.roles[i*3+2];
   const source=sense*exposure;
   this.nextX[i]=clamp(.68*this.sx[i]+(.12+.12*transport)*meanX+.3*source*dx/Math.max(d,1),-1,1);
   this.nextY[i]=clamp(.68*this.sy[i]+(.12+.12*transport)*meanY+.3*source*dy/Math.max(d,1),-1,1);
   let flow=0;for(const j of NEIGHBORS[i])if(this.alive[j])flow+=.095*(.03+transport+this.roles[j*3+1])*(this.energy[j]-e);
   const feed=.22*sense*exposure*food*(1-e);uptake+=feed;
   this.nextE[i]=clamp(e+flow+feed-.0012-.0024*motor-.0006*sense);
   fx+=motor*e*this.sx[i];fy+=motor*e*this.sy[i];
  }
  [this.roles,this.nextRoles]=[this.nextRoles,this.roles];
  [this.energy,this.nextE]=[this.nextE,this.energy];[this.sx,this.nextX]=[this.nextX,this.sx];[this.sy,this.nextY]=[this.nextY,this.sy];
  this.vx=.85*this.vx+.15*fx/Math.max(1,n)*2.8;this.vy=.85*this.vy+.15*fy/Math.max(1,n)*2.8;
  this.x+=this.vx;this.y+=this.vy;this.distance+=Math.hypot(this.vx,this.vy);this.intake+=uptake;this.tick++;
 }
 dominant(i){const a=this.roles[i*3],b=this.roles[i*3+1],c=this.roles[i*3+2];return a>=b&&a>=c?0:b>=c?1:2;}
 damage(predicate){const hits=[];for(let i=0;i<COUNT;i++)if(this.alive[i]&&predicate(XY[i][0],XY[i][1],i))hits.push(i);if(!hits.length)return 0;let removed=0;this.damageSnapshot={tick:this.tick,roles:Array.from(this.roles),alive:Array.from(this.alive),metrics:this.metrics()};for(const i of hits){this.alive[i]=0;this.energy[i]=this.sx[i]=this.sy[i]=this.nextE[i]=this.nextX[i]=this.nextY[i]=0;this.roles.fill(0,i*3,i*3+3);this.nextRoles.fill(0,i*3,i*3+3);removed++;}return removed;}
 cut(x,y,r=2){return this.damage((cx,cy)=>(cx-x)**2+(cy-y)**2<=r*r);}
 clone(){const t=new Tissue(this.weights,this.seed);for(const key of Object.keys(this)){const v=this[key];t[key]=ArrayBuffer.isView(v)?v.slice():v&&typeof v==='object'?structuredClone(v):v;}return t;}
 metrics(){let n=0,energy=0,deficient=0,roles=[0,0,0],counts=[0,0,0],switched=0,shift=0;for(let i=0;i<COUNT;i++)if(this.alive[i]){n++;energy+=this.energy[i];deficient+=this.energy[i]<.15?1:0;counts[this.dominant(i)]++;for(let k=0;k<3;k++)roles[k]+=this.roles[i*3+k];if(this.damageSnapshot){const old=this.damageSnapshot.roles.slice(i*3,i*3+3);if(old.indexOf(Math.max(...old))!==this.dominant(i))switched++;for(let k=0;k<3;k++)shift+=Math.abs(old[k]-this.roles[i*3+k]);}}
  return{cells:n,energy:energy/Math.max(n,1),roles:roles.map(v=>v/Math.max(n,1)),counts,deficient,switched,roleShift:shift/Math.max(2*n,1),speed:Math.hypot(this.vx,this.vy),intake:this.intake,distance:this.distance,targetDistance:Math.hypot(this.x-this.target.x,this.y-this.target.y),tick:this.tick};}
}
export function trial(weights,seed=1,{steps=420,damage=true,frozen=false}={}){const t=new Tissue(weights,seed);for(let s=0;s<steps;s++){if(s===160&&damage){t.frozen=frozen;t.damage((x,y)=>x>2&&y<3);}t.step();}const m=t.metrics();return{fitness:2*(20-m.targetDistance)+30*m.energy-15*m.deficient/Math.max(1,m.cells),...m};}

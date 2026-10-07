/**
 * Equations of motion for the 9-link body, plus ground contact and an energy ledger.
 *
 * M(q) q̈ = τ + Σ Jᵀ F − bias(q, q̇).  The mass matrix is assembled from segment
 * Jacobians; the bias is the centripetal acceleration each segment has when every joint
 * acceleration is zero, propagated down the tree. Ground contact is a penalty spring and
 * damper with regularised Coulomb friction. The spring force is applied whenever a point
 * is below ground, so its stored energy is always returned, and the damper is limited so
 * the contact can never pull — the ledger has no discarded-force term to hide in.
 *
 * Set `audit` to accumulate the energy ledger. It costs roughly half the step budget, so
 * training runs with it off and verification runs with it on.
 */
import {G,NQ,LINKS,QJ,JOINTS,LIMITS,ATTACH,segments,kinBuffers,kinematics,pointJacobian} from './body.js';

export const DT=1/2000;
// Floor stiffness trades the energy ledger against penetration. At 10 kN/m a standing
// body sinks 2.4 mm — about what a running track deflects — and the ledger closes to
// roughly 4% through a full-body impact at the default timestep, tightening with dt.
export const DEFAULT_ENV={mass:75,friction:0.9,dt:DT,kn:1e4,dn:540,kt:1.0e4};
const SOLE=0.07,HEEL=0.02,TOE=0.20;
/**
 * Where the body can touch the ground. Not just the soles: a body that has to learn
 * balance spends most of training falling over, so the pelvis, shoulders, knees and
 * hands all need to land on something. Without them a fallen ragdoll sinks through the
 * floor and the failure carries no information.
 */
const TOUCH=[
 {link:0,t:0,   sole:0,   name:'pelvis'},  {link:0,t:0.60,sole:0,   name:'shoulder'},
 {link:1,t:0.42,sole:0,   name:'kneeL'},   {link:4,t:0.42,sole:0,   name:'kneeR'},
 {link:3,t:HEEL,sole:SOLE,name:'heelL'},   {link:3,t:TOE, sole:SOLE,name:'toeL'},
 {link:6,t:HEEL,sole:SOLE,name:'heelR'},   {link:6,t:TOE, sole:SOLE,name:'toeR'},
 {link:7,t:0.60,sole:0,   name:'handL'},   {link:8,t:0.60,sole:0,   name:'handR'},
];
export const NC=TOUCH.length,TOUCH_NAMES=TOUCH.map(c=>c.name);
export const SOLE_POINTS=[4,5,6,7];
export const SOLE_LINKS=SOLE_POINTS.map(i=>TOUCH[i].link);

/**
 * Soft joint stops. A joint driven past its anatomical range meets a spring and damper,
 * not a wall: the restoring torque is applied whenever the joint is beyond the limit so
 * its stored energy is always returned, and the damper is capped so a stop can never
 * drive the joint further out. KLIM reaches a joint's peak actuator torque about 15
 * degrees past the limit, so the stops are firm without being stiffer than the
 * integrator can carry.
 */
const STOP_MARGIN=0.15;           // a joint's own peak torque holds it this far past (8.6 deg)
const STOP_DAMP=0.12;              // damping capped at this fraction of inertia per timestep
const PEAK=[200,200,150,200,200,150,60,60];
/**
 * Stop stiffness and damping are sized per joint. A single pair cannot serve both a hip,
 * which must arrest a whole swinging leg, and an ankle, whose foot is a hundred times
 * lighter — one setting is either soft enough to fold the foot back on itself or stiff
 * enough to break the integrator. Stiffness holds against that joint's own actuator;
 * damping is critical for the inertia it has to stop, capped for stability.
 */
export function stopGains(seg,dt){
 const K=new Float64Array(JOINTS),D=new Float64Array(JOINTS);
 for(let j=0;j<JOINTS;j++){
  // Inertia of everything distal to this joint, about it, in the neutral pose.
  let I=0;
  for(let n=0;n<LINKS;n++){
   if(!((seg[n].mask>>(QJ+j))&1))continue;
   // Distance from this joint out to the link's centre of mass, walking up the chain.
   let r=seg[n].com,c=n,p=seg[n].parent;
   while(p>=0&&((seg[p].mask>>(QJ+j))&1)){r+=ATTACH[c]*seg[p].len;c=p;p=seg[p].parent;}
   I+=seg[n].I+seg[n].m*r*r;
  }
  I=Math.max(I,1e-3);
  K[j]=PEAK[j]/STOP_MARGIN;
  D[j]=Math.min(2*Math.sqrt(K[j]*I),STOP_DAMP*I/dt);
 }
 return {K,D};
}
function applyLimits(b){
 const {rhs,q,qd,stopK,stopD}=b;
 let out=0,pe=0;
 for(let j=0;j<JOINTS;j++){
  const i=QJ+j,lo=LIMITS[j][0],hi=LIMITS[j][1];
  const e=q[i]>hi?q[i]-hi:(q[i]<lo?q[i]-lo:0);
  if(e===0)continue;
  const ts=-stopK[j]*e;                       // restoring: opposes the excess
  let td=-stopD[j]*qd[i];
  td=e>0?Math.min(td,-ts):Math.max(td,-ts);   // never push further past the stop
  rhs[i]+=ts+td;
  out+=-td*qd[i];
  pe+=0.5*stopK[j]*e*e;
 }
 b.limitPE=pe;
 return out;
}
export function limitEnergy(b){
 let pe=0;
 for(let j=0;j<JOINTS;j++){
  const i=QJ+j,lo=LIMITS[j][0],hi=LIMITS[j][1];
  const e=b.q[i]>hi?b.q[i]-hi:(b.q[i]<lo?b.q[i]-lo:0);
  if(e)pe+=0.5*b.stopK[j]*e*e;
 }
 return pe;
}
/** How far any joint is outside its range, in radians. Should stay small. */
export function limitExcess(b){
 let worst=0;
 for(let j=0;j<JOINTS;j++){
  const i=QJ+j,lo=LIMITS[j][0],hi=LIMITS[j][1];
  worst=Math.max(worst,b.q[i]>hi?b.q[i]-hi:(b.q[i]<lo?lo-b.q[i]:0));
 }
 return worst;
}
const stopsFor=e=>{const g=stopGains(segments(e.mass),e.dt);return {stopK:g.K,stopD:g.D};};
export function makeBody(env={}){
 const e={...DEFAULT_ENV,...env};
 for(const [k,a,b] of [['mass',30,150],['friction',0.05,2],['dt',1e-5,4e-3],['kn',1e3,1e7],['dn',10,1e5],['kt',10,1e6]])
  if(!Number.isFinite(e[k])||e[k]<a||e[k]>b)throw new Error(`Invalid ${k}`);
 return {env:e,seg:segments(e.mass),audit:false,
  q:new Float64Array(NQ),qd:new Float64Array(NQ),qdd:new Float64Array(NQ),tau:new Float64Array(NQ),
  M:new Float64Array(NQ*NQ),rhs:new Float64Array(NQ),jac:new Float64Array(2*NQ),
  k:kinBuffers(),k2:kinBuffers(),L:new Float64Array(NQ*NQ),sv:new Float64Array(NQ),
  aax:new Float64Array(LINKS),aay:new Float64Array(LINKS),qd0:new Float64Array(NQ),
  cx:new Float64Array(NC),cy:new Float64Array(NC),fn:new Float64Array(NC),ft:new Float64Array(NC),
  grf:new Float64Array(2),t:0,steps:0,limitPE:0,...stopsFor(e),
  work:0,dissipated:0,springPE:0,residual:0,singular:false};
}
/** Contact point world coordinates into the body's own buffers. */
export function contactPoints(b,k){
 for(let c=0;c<NC;c++){
  const {link:i,t,sole}=TOUCH[c],a=k.phi[i],sn=Math.sin(a),cs=Math.cos(a);
  b.cx[c]=k.ax[i]+t*sn-sole*cs;
  b.cy[c]=k.ay[i]-t*cs-sole*sn;
 }
}
function assemble(b,k){
 const {seg,M,rhs,jac,aax,aay}=b,n=NQ;
 M.fill(0);rhs.fill(0);
 for(let i=0;i<LINKS;i++){
  const p=seg[i].parent;
  if(p<0){aax[i]=0;aay[i]=0;continue;}
  const t=ATTACH[i]*seg[p].len,w2=k.om[p]*k.om[p];
  aax[i]=aax[p]-t*w2*Math.sin(k.phi[p]);aay[i]=aay[p]+t*w2*Math.cos(k.phi[p]);
 }
 for(let i=0;i<LINKS;i++){
  const s=seg[i],w2=k.om[i]*k.om[i],m=s.m,mask=s.mask;
  const fx=-m*(aax[i]-s.com*w2*Math.sin(k.phi[i]));
  const fy=-m*(G+aay[i]+s.com*w2*Math.cos(k.phi[i]));
  pointJacobian(k,seg,i,k.cx[i],k.cy[i],jac);
  for(let j=0;j<n;j++){
   const jx=jac[j],jy=jac[n+j];
   if(jx===0&&jy===0)continue;
   rhs[j]+=jx*fx+jy*fy;
   const bj=(mask>>j)&1;
   for(let l=j;l<n;l++){
    let v=m*(jx*jac[l]+jy*jac[n+l]);
    if(bj&&((mask>>l)&1))v+=s.I;
    if(v!==0){M[j*n+l]+=v;if(l!==j)M[l*n+j]+=v;}
   }
  }
 }
}
function applyContacts(b,k){
 const {seg,env:e,rhs,jac,qd}=b,n=NQ;
 let out=0;b.grf[0]=0;b.grf[1]=0;
 contactPoints(b,k);
 for(let c=0;c<NC;c++){
  b.fn[c]=0;b.ft[c]=0;
  const y=b.cy[c];
  if(y>=0)continue;
  pointJacobian(k,seg,TOUCH[c].link,b.cx[c],y,jac);
  let vx=0,vy=0;
  for(let j=0;j<n;j++){vx+=jac[j]*qd[j];vy+=jac[n+j]*qd[j];}
  const fs=-e.kn*y;                             // conservative: matched by springPE
  const fd=Math.max(-fs,-e.dn*vy);              // dissipative, never pulls
  const fn=fs+fd;
  const mu=e.friction*fn;
  const ft=Math.max(-mu,Math.min(mu,-e.kt*vx));
  for(let j=0;j<n;j++)rhs[j]+=jac[j]*ft+jac[n+j]*fn;
  out+=-fd*vy-ft*vx;
  b.fn[c]=fn;b.ft[c]=ft;b.grf[0]+=ft;b.grf[1]+=fn;
 }
 return out;
}
function solve(b){
 const {M,rhs,qdd,L,sv}=b,n=NQ;
 for(let i=0;i<n;i++)for(let j=0;j<=i;j++){
  let s=M[i*n+j];
  for(let k2=0;k2<j;k2++)s-=L[i*n+k2]*L[j*n+k2];
  if(i===j){if(s<=1e-12)return false;L[i*n+j]=Math.sqrt(s);}
  else L[i*n+j]=s/L[j*n+j];
 }
 for(let i=0;i<n;i++){let s=rhs[i];for(let j=0;j<i;j++)s-=L[i*n+j]*sv[j];sv[i]=s/L[i*n+i];}
 for(let i=n-1;i>=0;i--){let s=sv[i];for(let j=i+1;j<n;j++)s-=L[j*n+i]*qdd[j];qdd[i]=s/L[i*n+i];}
 return true;
}
function springEnergy(b,k){
 contactPoints(b,k);
 let s=0;for(let c=0;c<NC;c++)if(b.cy[c]<0)s+=0.5*b.env.kn*b.cy[c]*b.cy[c];
 return s;
}
/** Kinetic + gravitational + contact-spring energy. */
export function energy(b,k){
 const {seg,jac,qd}=b;let ke=0,pe=0;
 for(let i=0;i<LINKS;i++){
  pointJacobian(k,seg,i,k.cx[i],k.cy[i],jac);
  let vx=0,vy=0;for(let j=0;j<NQ;j++){vx+=jac[j]*qd[j];vy+=jac[NQ+j]*qd[j];}
  ke+=0.5*seg[i].m*(vx*vx+vy*vy)+0.5*seg[i].I*k.om[i]*k.om[i];
  pe+=seg[i].m*G*k.cy[i];
 }
 return ke+pe+b.springPE+b.limitPE;
}
export function step(b,dt=b.env.dt){
 const k=kinematics(b.q,b.qd,b.seg,b.k);
 let E0=0;
 if(b.audit){b.springPE=springEnergy(b,k);b.limitPE=limitEnergy(b);E0=energy(b,k);}
 assemble(b,k);
 const out=applyContacts(b,k)+applyLimits(b);
 for(let j=QJ;j<NQ;j++)b.rhs[j]+=b.tau[j];
 if(!solve(b)){b.singular=true;return b;}
 const {qd0,qd,q,qdd,tau}=b;
 if(b.audit)qd0.set(qd);
 for(let j=0;j<NQ;j++)qd[j]+=qdd[j]*dt;
 for(let j=0;j<NQ;j++)q[j]+=qd[j]*dt;
 b.t+=dt;b.steps++;
 if(b.audit){
  let w=0;for(let j=QJ;j<NQ;j++)w+=tau[j]*0.5*(qd0[j]+qd[j])*dt;
  b.work+=w;b.dissipated+=out*dt;
  const k1=kinematics(q,qd,b.seg,b.k2);
  b.springPE=springEnergy(b,k1);b.limitPE=limitEnergy(b);
  b.residual+=(energy(b,k1)-E0)-w+out*dt;
 }
 return b;
}

/**
 * Stride Lab — planar 9-link articulated body.
 *
 * A floating-base tree: pelvis (x, y, pitch) plus eight revolute joints — two hips, two
 * knees, two ankles, two shoulders — for eleven degrees of freedom. Segment masses,
 * lengths and radii of gyration are Winter's anthropometric fractions, so a 75 kg body
 * is distributed the way a person is rather than lumped at a point.
 *
 * Dynamics are exact for this topology, not a spring abstraction. Each step builds the
 * mass matrix M(q) from segment Jacobians, forms the velocity-dependent bias by
 * propagating centripetal acceleration through the tree at zero joint acceleration, and
 * solves M q̈ = τ + J'F for the accelerations. Nothing is prescribed: the body falls over
 * unless something holds it up.
 */
export const G=9.81;
export const NQ=11;                 // px, py, pitch, hipL, kneeL, ankleL, hipR, kneeR, ankleR, shoulderL, shoulderR
export const DOF=['px','py','pitch','hipL','kneeL','ankleL','hipR','kneeR','ankleR','shoulderL','shoulderR'];
export const JOINTS=8,QJ=3;         // eight actuated joints, starting at index 3

// Winter's segment fractions for a 1.80 m, 75 kg body. `frac` is share of body mass,
// `len` segment length, `com` centre of mass from the proximal joint, `gyr` radius of
// gyration about the centre of mass as a fraction of length.
const SEG=[
 {name:'torso', frac:0.578,len:0.60,com:0.30,gyr:0.50,off:Math.PI,   parent:-1,chain:[2]},
 {name:'thighL',frac:0.100,len:0.42,com:0.18,gyr:0.323,off:0,        parent:-1,chain:[2,3]},
 {name:'shankL',frac:0.0465,len:0.42,com:0.18,gyr:0.302,off:0,       parent:1, chain:[2,3,4]},
 {name:'footL', frac:0.0145,len:0.22,com:0.09,gyr:0.475,off:Math.PI/2,parent:2,chain:[2,3,4,5]},
 {name:'thighR',frac:0.100,len:0.42,com:0.18,gyr:0.323,off:0,        parent:-1,chain:[2,6]},
 {name:'shankR',frac:0.0465,len:0.42,com:0.18,gyr:0.302,off:0,       parent:4, chain:[2,6,7]},
 {name:'footR', frac:0.0145,len:0.22,com:0.09,gyr:0.475,off:Math.PI/2,parent:5,chain:[2,6,7,8]},
 {name:'armL',  frac:0.050,len:0.60,com:0.26,gyr:0.368,off:0,        parent:0, chain:[2,9]},
 {name:'armR',  frac:0.050,len:0.60,com:0.26,gyr:0.368,off:0,        parent:0, chain:[2,10]},
];
export const LINKS=SEG.length;
/**
 * Anatomical joint ranges, in radians, for coordinates 3..10. Without these the search
 * finds poses no body can hold: a balance policy trained without them put one ankle at
 * -151 degrees, folding the foot back past the shank with its tip underground, and
 * rotated a shoulder 122 degrees behind the torso.
 *
 * hip      flexes forward, a little extension behind
 * knee     flexes one way only; it cannot hyperextend
 * ankle    dorsiflexion up, plantarflexion down, from the neutral right angle
 * shoulder swings forward through overhead, limited behind
 */
export const LIMITS=[
 [-0.35,2.09],[-2.44,0.02],[-0.87,0.35],   // left  hip, knee, ankle
 [-0.35,2.09],[-2.44,0.02],[-0.87,0.35],   // right hip, knee, ankle
 [-1.05,3.14],[-1.05,3.14],                // shoulders
];

// Where each link's proximal joint sits on its parent, as a fraction of the parent's
// length. -1 means it hangs directly off the pelvis, the floating base itself.
const ATTACH=[0,0,1,1,0,1,1,1,1];
export function segments(mass=75){
 return SEG.map(s=>({...s,m:s.frac*mass,I:s.frac*mass*(s.gyr*s.len)**2,
  mask:s.chain.reduce((a,c)=>a|(1<<c),0)}));
}
const dirX=a=>Math.sin(a),dirY=a=>-Math.cos(a);   // a link at angle 0 points straight down
export {ATTACH};

export function kinBuffers(){
 return {phi:new Float64Array(LINKS),om:new Float64Array(LINKS),
  ax:new Float64Array(LINKS),ay:new Float64Array(LINKS),
  cx:new Float64Array(LINKS),cy:new Float64Array(LINKS)};
}
/** Forward kinematics: absolute angles, proximal points, centres of mass, angular rates. */
export function kinematics(q,qd,seg,k){
 const {phi,om,ax,ay,cx,cy}=k;
 for(let i=0;i<LINKS;i++){
  const s=seg[i],ch=s.chain;let a=s.off,w=0;
  for(let c=0;c<ch.length;c++){a+=q[ch[c]];w+=qd[ch[c]];}
  phi[i]=a;om[i]=w;
  const p=s.parent,sn=Math.sin(a),cs=Math.cos(a);
  if(p<0){ax[i]=q[0];ay[i]=q[1];}
  else{const t=ATTACH[i]*seg[p].len;ax[i]=ax[p]+t*Math.sin(phi[p]);ay[i]=ay[p]-t*Math.cos(phi[p]);}
  cx[i]=ax[i]+s.com*sn;cy[i]=ay[i]-s.com*cs;
 }
 return k;
}
/** World point at distance `t` along link `i`, and the row of its Jacobian. */
export function pointOn(k,seg,i,t){
 return [k.ax[i]+t*dirX(k.phi[i]),k.ay[i]+t*dirY(k.phi[i])];
}
/**
 * Jacobian of a world point attached to link `i`: column j is how the point moves per
 * unit of q[j]. Translation columns are unit; every rotational coordinate on the link's
 * chain swings the point about that joint's own anchor.
 */
export function pointJacobian(k,seg,i,px,py,out){
 out.fill(0);out[0]=1;out[NQ+1]=1;
 for(const j of seg[i].chain){
  // Anchor of coordinate j: pitch turns the whole body about the pelvis; a joint turns
  // everything distal to it about that joint's proximal point.
  let axj,ayj;
  if(j===2){axj=k.ax[0];ayj=k.ay[0];}
  else{const link=jointLink(j);axj=k.ax[link];ayj=k.ay[link];}
  out[j]=-(py-ayj);out[NQ+j]=(px-axj);
 }
}
// Which link is driven by joint coordinate j (index 3..10).
const JOINT_LINK=[0,0,0, 1,2,3, 4,5,6, 7,8];
function jointLink(j){return JOINT_LINK[j];}
export {JOINT_LINK};

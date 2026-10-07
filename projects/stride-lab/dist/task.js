/**
 * Tasks for the articulated body: what the policy sees, what it can do, and what counts
 * as success or as falling over.
 *
 * Nothing here prescribes a gait or a posture. The policy gets joint angles, rates, the
 * torso's attitude, the pelvis state and which soles are loaded; it returns eight joint
 * torques inside human-scale limits. Balance and coordination are whatever comes out of
 * that, or do not.
 */
import {NQ,QJ,kinematics,pointJacobian} from './body.js';
import {makeBody,step,contactPoints,NC,SOLE_POINTS,SOLE_LINKS} from './dynamics.js';

// Peak joint torques for a 75 kg adult, in newton-metres. The policy commands a fraction
// of each; it cannot ask for more than a person could produce.
export const TORQUE=[200,200,150,200,200,150,60,60];
export const OBS=25,ACT=8;
export const STAND=[0,0.905,0, 0.05,-0.10,0.05, 0.05,-0.10,0.05, 0.1,0.1];

export function reset(env={},pose=STAND){
 const b=makeBody(env);
 b.q.set(pose);
 return b;
}
/**
 * Observation: torso attitude, every joint angle and rate, pelvis height and velocity,
 * and which soles carry load. Angles and rates are raw — the trainer normalises them
 * from what it actually sees rather than from constants chosen by hand.
 */
export function observe(b,out=new Float64Array(OBS)){
 const k=kinematics(b.q,b.qd,b.seg,b.k);
 out[0]=b.q[2];out[1]=b.qd[2];
 for(let j=0;j<8;j++){out[2+j]=b.q[QJ+j];out[10+j]=b.qd[QJ+j];}
 out[18]=b.q[1];out[19]=b.qd[0];out[20]=b.qd[1];
 contactPoints(b,k);
 for(let i=0;i<4;i++)out[21+i]=b.fn[SOLE_POINTS[i]]>1?1:0;
 return out;
}
export function act(b,a){
 for(let j=0;j<8;j++)b.tau[QJ+j]=TORQUE[j]*Math.max(-1,Math.min(1,a[j]));
}
/** Upright enough to still be trying. Below this the episode is over. */
export function upright(b){return b.q[1]>0.62&&Math.abs(b.q[2])<0.9&&!b.singular;}

/**
 * One training condition: a body, a floor, a starting pose and the shoves it will take.
 *
 * Training from a single deterministic start does not produce a controller. It produces a
 * torque sequence that happens to work once — it can score a perfect return and still fall
 * over when the timestep changes, which means the result is a property of the
 * discretisation rather than of the policy. Every episode therefore gets its own body,
 * floor, pose and disturbances.
 */
export function sampleCondition(rng,base={}){
 const u=(a,b)=>a+(b-a)*rng();
 const pose=[...STAND];
 pose[1]+=u(-0.012,0.012);
 pose[2]+=u(-0.10,0.10);
 for(let j=3;j<11;j++)pose[j]+=u(-0.12,0.12);
 const kn=1e4*u(0.5,2.5);
 const env={mass:75*u(0.85,1.15),friction:0.9*u(0.55,1.35),
  kn,dn:540*Math.sqrt(kn/1e4),dt:[5e-4,3.5e-4,2.5e-4][Math.floor(rng()*3)],...base};
 const vel={0:u(-0.25,0.25),1:u(-0.10,0.10),2:u(-0.35,0.35)};
 const shoves=[];
 const n=Math.floor(rng()*3);                     // zero, one or two disturbances
 for(let i=0;i<n;i++)shoves.push({t:u(0.6,5.0),impulse:u(-70,70)});
 return {env,pose,vel,shoves};
}
/** A fixed set of conditions, drawn from their own stream and never trained on. */
export function validationSet(seed,count=12,rngFactory){
 const rng=rngFactory((seed^0x9E3779B9)>>>0);
 return Array.from({length:count},()=>sampleCondition(rng));
}
const NOMINAL={env:{},pose:STAND,vel:null,shoves:[]};

function launch(condition){
 const c=condition||NOMINAL;
 const b=reset(c.env,c.pose||STAND);
 if(c.vel)for(const k in c.vel)b.qd[k]=Number(c.vel[k]);
 return {b,shoves:(c.shoves||[]).map(s=>({...s,done:false}))};
}
function disturb(b,shoves){
 for(const s of shoves)if(!s.done&&b.t>=s.t){b.qd[0]+=s.impulse/b.env.mass;s.done=true;}
}
/**
 * Stand still. Reward is time on your feet, minus drifting, leaning, sinking and effort.
 * There is no target trajectory to imitate — only a cost for falling and for flailing.
 */
export function balance(policy,{condition=null,seconds=6,control=20,effort=0.0015}={}){
 const {b,shoves}=launch(condition);
 const obs=new Float64Array(OBS),a=new Float64Array(ACT);
 const steps=Math.round(seconds/b.env.dt);
 let R=0,alive=0;
 for(let n=0;n<steps;n++){
  if(n%control===0){observe(b,obs);policy(obs,a);act(b,a);}
  disturb(b,shoves);
  step(b);alive++;
  if(!upright(b))break;
  if(n%control===0){
   let e=0;for(let j=0;j<8;j++)e+=a[j]*a[j];
   // Bounded below so one diverging rollout cannot dominate the search's reward scale.
   const r=1-3*b.q[2]*b.q[2]-2*(0.88-b.q[1])**2-0.6*b.qd[0]*b.qd[0]-0.8*b.q[0]*b.q[0]-effort*e*20;
   R+=(Number.isFinite(r)?Math.max(-5,r):-5)*control*b.env.dt;
  }
 }
 return {reward:R,seconds:b.t,fell:!upright(b),body:b,
  height:b.q[1],pitch:b.q[2],drift:b.q[0]};
}

/**
 * Travel. Reward is distance covered per second on your feet, minus leaning, sinking and
 * effort. There is still no gait to imitate and no phase machine: whatever coordination
 * appears has to come out of the search. Falling ends the episode, so speed is only worth
 * anything if the body can keep doing it.
 */
export function walk(policy,{condition=null,seconds=8,control=20,effort=0.0015,target=1.4,
                            slipWeight=6,slipFree=0.05}={}){
 const {b,shoves}=launch(condition);
 const obs=new Float64Array(OBS),a=new Float64Array(ACT);
 const jac=new Float64Array(2*NQ);
 const steps=Math.round(seconds/b.env.dt);
 let R=0;
 for(let n=0;n<steps;n++){
  if(n%control===0){observe(b,obs);policy(obs,a);act(b,a);}
  disturb(b,shoves);
  step(b);
  if(!upright(b))break;
  if(n%control===0){
   let e=0;for(let j=0;j<8;j++)e+=a[j]*a[j];
   // How fast is a foot moving while it carries load? A planted foot is still. Paying for
   // pelvis velocity alone buys skating: the friction model allows sustained slip at the
   // cone limit, and sliding reaches target speed without ever taking a step.
   const k=kinematics(b.q,b.qd,b.seg,b.k);
   contactPoints(b,k);
   let loaded=0,slip=0;
   for(let i=0;i<SOLE_POINTS.length;i++){
    const c=SOLE_POINTS[i];
    if(b.fn[c]<=1)continue;
    pointJacobian(k,b.seg,SOLE_LINKS[i],b.cx[c],b.cy[c],jac);
    let vx=0;for(let j=0;j<NQ;j++)vx+=jac[j]*b.qd[j];
    loaded++;slip+=Math.abs(vx);
   }
   const drag=loaded?Math.max(0,slip/loaded-slipFree):0;
   // Credit speed up to the target and no further, so the body is not paid for diving.
   const v=Math.min(b.qd[0],target);
   const r=0.4+1.6*v-2.5*b.q[2]*b.q[2]-2*(0.88-b.q[1])**2-effort*e*20-slipWeight*drag;
   R+=(Number.isFinite(r)?Math.max(-5,r):-5)*control*b.env.dt;
  }
 }
 return {reward:R,seconds:b.t,fell:!upright(b),body:b,
  distance:b.q[0],speed:b.q[0]/Math.max(b.t,1e-9),height:b.q[1],pitch:b.q[2]};
}

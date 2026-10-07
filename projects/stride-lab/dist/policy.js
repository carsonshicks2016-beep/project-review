/**
 * A linear feedback policy and the Augmented Random Search that trains it.
 *
 * The policy is a single matrix: observation in, joint torques out. No phase machine, no
 * scripted stride, no reference motion. Linear policies are enough for locomotion when
 * observations are normalised by what the agent has actually seen (Mania, Guy and Recht,
 * 2018), and the search needs nothing but forward simulation, so it stays reproducible
 * from a seed.
 */
export function makeRng(seed=42){let v=seed>>>0;return()=>{v+=0x6D2B79F5;let t=v;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
export function gaussian(rng){return Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(2*Math.PI*rng());}

/** Running mean and variance of the observations, so the policy sees comparable scales. */
export function makeNorm(n){return {n:0,mean:new Float64Array(n),m2:new Float64Array(n),std:new Float64Array(n).fill(1)};}
export function observeNorm(z,x){
 z.n++;
 for(let i=0;i<x.length;i++){const d=x[i]-z.mean[i];z.mean[i]+=d/z.n;z.m2[i]+=d*(x[i]-z.mean[i]);}
}
export function freezeNorm(z){
 for(let i=0;i<z.std.length;i++)z.std[i]=Math.max(1e-3,Math.sqrt(z.m2[i]/Math.max(1,z.n-1)));
}
export function makePolicy(w,nObs,nAct,z=null){
 const x=new Float64Array(nObs);
 return (obs,out)=>{
  for(let i=0;i<nObs;i++)x[i]=z?(obs[i]-z.mean[i])/z.std[i]:obs[i];
  for(let j=0;j<nAct;j++){
   let s=0,r=j*nObs;
   for(let i=0;i<nObs;i++)s+=w[r+i]*x[i];
   out[j]=s;
  }
 };
}
/**
 * Augmented Random Search, V2-t: probe symmetric perturbations, keep the directions with
 * the strongest response, and step along their reward-weighted average scaled by the
 * spread of the kept returns.
 */
export function train(rollout,{nObs,nAct,seed=42,dirs=16,keep=8,step:alpha=0.02,noise=0.03,
                               iters=300,onIter=null,onIterStart=null,init=null}={}){
 const n=nObs*nAct,rng=makeRng(seed);
 const w=new Float64Array(n),z=makeNorm(nObs);
 // Warm start: carry a policy and its observation statistics over from an earlier task, so
 // a harder objective does not have to rediscover the easier one underneath it.
 if(init){
  if(init.w)w.set(init.w);
  if(init.norm){z.mean.set(init.norm.mean);z.std.set(init.norm.std);
   z.n=Math.max(2,init.norm.n|0);
   for(let i=0;i<nObs;i++)z.m2[i]=init.norm.std[i]*init.norm.std[i]*(z.n-1);}
 }
 const plus=new Float64Array(n),minus=new Float64Array(n),d=[];
 for(let i=0;i<dirs;i++)d.push(new Float64Array(n));
 let best=-Infinity,bestW=new Float64Array(n),evals=0;
 for(let it=1;it<=iters;it++){
  // Fresh conditions for this generation, shared by every probe so their scores compare.
  if(onIterStart)onIterStart(it);
  freezeNorm(z);
  const res=[];
  for(let i=0;i<dirs;i++){
   const di=d[i];
   for(let j=0;j<n;j++)di[j]=gaussian(rng);
   for(let j=0;j<n;j++){plus[j]=w[j]+noise*di[j];minus[j]=w[j]-noise*di[j];}
   const rp=rollout(plus,z),rm=rollout(minus,z);evals+=2;
   res.push({i,rp,rm,max:Math.max(rp,rm)});
  }
  res.sort((a,b)=>b.max-a.max);
  const top=res.slice(0,keep);
  const rs=top.flatMap(t=>[t.rp,t.rm]);
  const mean=rs.reduce((a,b)=>a+b,0)/rs.length;
  const sd=Math.sqrt(rs.reduce((a,b)=>a+(b-mean)**2,0)/rs.length)||1;
  for(const t of top){const c=alpha*(t.rp-t.rm)/(keep*sd),di=d[t.i];
   for(let j=0;j<n;j++)w[j]+=c*di[j];}
  const cur=rollout(w,z,true);evals++;
  if(cur>best){best=cur;bestW.set(w);}
  if(onIter)onIter({iter:it,reward:cur,best,evals,mean});
 }
 return {w:bestW,z,best,evals};
}

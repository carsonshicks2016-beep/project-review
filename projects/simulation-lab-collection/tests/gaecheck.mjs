import { computeGAE, normalizeAdvantages } from '../src/ppo/gae.js';

let ok = true;
const eq = (a,b,tol,label) => {
  const d = Math.abs(a-b);
  const pass = d < tol;
  if (!pass) { ok = false; console.log(`  FAIL ${label}: ${a} vs ${b}`); }
  return pass;
};

// 1. lambda=1, no terminals => advantage must equal the discounted Monte-Carlo return
//    minus V(s_t). This is the defining property of GAE(1).
{
  const T=8, gamma=0.9;
  const rewards=[], values=[], dones=[];
  for(let t=0;t<T;t++){ rewards.push(Math.random()*2-1); values.push(Math.random()*2-1); dones.push(false); }
  const lastValue = 0.37;
  const { advantages } = computeGAE({rewards,values,dones,lastValue,gamma,lambda:1});
  let maxErr=0;
  for(let t=0;t<T;t++){
    let mc=0, disc=1;
    for(let k=t;k<T;k++){ mc += disc*rewards[k]; disc*=gamma; }
    mc += disc*lastValue;           // bootstrap the truncated tail
    maxErr = Math.max(maxErr, Math.abs(advantages[t] - (mc - values[t])));
  }
  console.log(`GAE(1) == MC return - V   maxErr=${maxErr.toExponential(2)}`);
  eq(maxErr,0,1e-12,'gae lambda=1');
}

// 2. lambda=0 => advantage is exactly the one-step TD residual.
{
  const T=6, gamma=0.95;
  const rewards=[], values=[], dones=[];
  for(let t=0;t<T;t++){ rewards.push(Math.random()); values.push(Math.random()); dones.push(false); }
  const lastValue=0.5;
  const { advantages } = computeGAE({rewards,values,dones,lastValue,gamma,lambda:0});
  let maxErr=0;
  for(let t=0;t<T;t++){
    const nv = t===T-1 ? lastValue : values[t+1];
    maxErr = Math.max(maxErr, Math.abs(advantages[t] - (rewards[t] + gamma*nv - values[t])));
  }
  console.log(`GAE(0) == TD residual     maxErr=${maxErr.toExponential(2)}`);
  eq(maxErr,0,1e-12,'gae lambda=0');
}

// 3. A terminal must cut the bootstrap: nothing after a done may influence advantage.
{
  const rewards=[1,1,1,1], values=[0,0,0,0], dones=[false,true,false,false];
  const a = computeGAE({rewards,values,dones,lastValue:1000,gamma:0.99,lambda:0.95}).advantages;
  console.log(`terminal cuts bootstrap:  A[0]=${a[0].toFixed(4)} A[1]=${a[1].toFixed(4)}`);
  eq(a[1],1,1e-12,'done truncates');            // r=1 only, no bootstrap
  eq(a[0],1+0.99*0.95*1,1e-12,'pre-done');      // does not see lastValue
}

// 4. returns == advantages + values
{
  const rewards=[0.5,-0.2,0.3], values=[0.1,0.2,0.3], dones=[false,false,false];
  const { advantages, returns } = computeGAE({rewards,values,dones,lastValue:0.4});
  let e=0; for(let t=0;t<3;t++) e=Math.max(e,Math.abs(returns[t]-(advantages[t]+values[t])));
  console.log(`returns == A + V          maxErr=${e.toExponential(2)}`);
  eq(e,0,1e-12,'returns identity');
}

// 5. normalization => zero mean, unit std
{
  const a = new Float64Array(1000);
  for(let i=0;i<1000;i++) a[i]=Math.random()*17-3;
  normalizeAdvantages(a);
  let m=0; for(const v of a) m+=v; m/=a.length;
  let s=0; for(const v of a) s+=(v-m)*(v-m); s=Math.sqrt(s/a.length);
  console.log(`normalized mean=${m.toExponential(2)} std=${s.toFixed(6)}`);
  eq(m,0,1e-9,'norm mean'); eq(s,1,1e-6,'norm std');
}

console.log(ok ? 'GAE CHECK PASSED' : 'GAE CHECK FAILED');
process.exit(ok?0:1);

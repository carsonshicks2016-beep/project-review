import { logProb, logProbGrads, entropy, entropyGrads, gaussianSample } from '../src/ppo/policy.js';

const D = 4;
const mean = new Float64Array(D), logStd = new Float64Array(D), action = new Float64Array(D);
for (let i=0;i<D;i++){ mean[i]=Math.random()*2-1; logStd[i]=Math.random()-1.2; action[i]=Math.random()*2-1; }

const dMean = new Float64Array(D), dLogStd = new Float64Array(D);
logProbGrads(action, mean, logStd, dMean, dLogStd);

const h=1e-6;
let maxErr=0;
for (let i=0;i<D;i++){
  let o=mean[i];
  mean[i]=o+h; const lp1=logProb(action,mean,logStd);
  mean[i]=o-h; const lp2=logProb(action,mean,logStd);
  mean[i]=o;
  const num=(lp1-lp2)/(2*h);
  maxErr=Math.max(maxErr, Math.abs(num-dMean[i])/Math.max(1e-9,Math.abs(num)+Math.abs(dMean[i])));
}
console.log('d logProb / d mean    relErr =', maxErr.toExponential(2));
let e2=0;
for (let i=0;i<D;i++){
  let o=logStd[i];
  logStd[i]=o+h; const lp1=logProb(action,mean,logStd);
  logStd[i]=o-h; const lp2=logProb(action,mean,logStd);
  logStd[i]=o;
  const num=(lp1-lp2)/(2*h);
  e2=Math.max(e2, Math.abs(num-dLogStd[i])/Math.max(1e-9,Math.abs(num)+Math.abs(dLogStd[i])));
}
console.log('d logProb / d logStd  relErr =', e2.toExponential(2));
const dH=new Float64Array(D); entropyGrads(logStd,dH);
let e3=0;
for (let i=0;i<D;i++){
  let o=logStd[i];
  logStd[i]=o+h; const a=entropy(logStd);
  logStd[i]=o-h; const b=entropy(logStd);
  logStd[i]=o;
  e3=Math.max(e3, Math.abs((a-b)/(2*h)-dH[i]));
}
console.log('d entropy / d logStd  absErr =', e3.toExponential(2));

// The density must integrate to 1: Monte-Carlo check that exp(logProb) is a real pdf
// by confirming the sample mean/std match the parameters.
const N=400000; const acc=new Float64Array(D), acc2=new Float64Array(D);
const s=new Float64Array(D);
for(let n=0;n<N;n++){ gaussianSample(mean,logStd,s); for(let i=0;i<D;i++){acc[i]+=s[i];acc2[i]+=s[i]*s[i];} }
let mErr=0,sErr=0;
for(let i=0;i<D;i++){
  const m=acc[i]/N, v=acc2[i]/N-m*m;
  mErr=Math.max(mErr,Math.abs(m-mean[i]));
  sErr=Math.max(sErr,Math.abs(Math.sqrt(v)-Math.exp(logStd[i])));
}
console.log(`sampler: max |mean err|=${mErr.toExponential(2)}  max |std err|=${sErr.toExponential(2)}`);
const ok = maxErr<1e-7 && e2<1e-7 && e3<1e-7 && mErr<0.02 && sErr<0.02;
console.log(ok ? 'POLICY CHECK PASSED' : 'POLICY CHECK FAILED');

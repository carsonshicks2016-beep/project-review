import { MLP } from '../src/ppo/network.js';

// Standard vector-norm gradient check: ||num-ana|| / ||num+ana||.
// Per-element relative error is meaningless for entries whose true gradient is ~0.
function check(layerSizes, label, h) {
  const net = new MLP(layerSizes);
  const inDim = layerSizes[0], outDim = layerSizes[layerSizes.length-1];
  const acts = layerSizes.map(n => new Float64Array(n));
  const p64 = Float64Array.from(net.params);
  const x = new Float64Array(inDim);
  for (let i=0;i<inDim;i++) x[i] = Math.random()*2-1;
  const target = new Float64Array(outDim);
  for (let i=0;i<outDim;i++) target[i] = Math.random()*2-1;

  const loss = (p) => { const y = net.forward(x, acts, p); let s=0;
    for (let i=0;i<outDim;i++){const d=y[i]-target[i]; s+=0.5*d*d;} return s; };

  const y = net.forward(x, acts, p64);
  const dOut = new Float64Array(outDim);
  for (let i=0;i<outDim;i++) dOut[i]=y[i]-target[i];
  const ana = new Float64Array(net.numParams);
  net.backward(acts, dOut, ana, p64);

  let dn=0, sn=0;
  for (let i=0;i<net.numParams;i++){
    const o=p64[i];
    p64[i]=o+h; const lp=loss(p64);
    p64[i]=o-h; const lm=loss(p64);
    p64[i]=o;
    const num=(lp-lm)/(2*h);
    dn += (num-ana[i])**2;
    sn += (num+ana[i])**2;
  }
  return Math.sqrt(dn)/Math.max(1e-30, Math.sqrt(sn));
}

console.log('h-sweep on [14,64,64,4] (expect a U: truncation down, roundoff up)');
for (const h of [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]) {
  console.log(`  h=${h.toExponential(0)}  relErr=${check([14,64,64,4],'',h).toExponential(2)}`);
}

console.log('\nvector-norm check at h=1e-5:');
let ok = true;
for (const ls of [[3,4,2],[14,16,12,4],[14,64,64,1],[14,64,64,4],[5,8,8,8,3],[22,64,64,4]]) {
  const e = check(ls,'',1e-5);
  const pass = e < 1e-8;
  if (!pass) ok = false;
  console.log(`  ${pass?'PASS':'FAIL'} [${ls}]  relErr=${e.toExponential(2)}`);
}
console.log(ok ? '\nGRADIENT CHECK PASSED' : '\nGRADIENT CHECK FAILED');

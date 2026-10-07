export const C = 343;
export const SAMPLE_RATE = 24000;
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const smoothstep = x => { x=clamp(x,0,1); return x*x*(3-2*x); };
export function regime(m){return m<0.8?'SUBSONIC':m<1.2?'TRANSONIC':m<5?'SUPERSONIC':'HYPERSONIC'}
export function coneAngle(m){return m>1?Math.asin(1/m):null}
export function boomTime(m,h){return m>1?h/C*Math.sqrt(1-1/(m*m)):null}
export function timeWindow(m,h){const b=h/C;return {start:-Math.max(3,2.5*b/Math.max(.6,m)),end:Math.max(7,4*b+2)}}
// Stable quadratic solution of (1-M²)τ² - 2tτ + t² - (h/c)² = 0.
// Squaring also introduces advanced roots: retain only causal solutions.
export function retardedTimes(t,m,h){
 const a=1-m*m,b=h/C;
 if(Math.abs(a)<1e-10){return t>0?[(t*t-b*b)/(2*t)]:[]}
 const disc=m*m*t*t+a*b*b;
 if(disc<0 || (m>1 && t<=0))return [];
 const d=Math.sqrt(disc),q=t+(t<0?-d:d),roots=q===0?[(t-d)/a,(t+d)/a]:[q/a,(t*t-b*b)/q];
 return roots.filter((tau,i)=>Number.isFinite(tau)&&tau<=t+1e-9&&Math.abs(tau+Math.hypot(m*tau,b)-t)<1e-7&&(!i||Math.abs(tau-roots[0])>1e-8)).sort((x,y)=>x-y);
}
export function doppler(tau,m,h){const r=Math.hypot(m*tau,h/C);return 1/Math.abs(1+m*m*tau/r)}
export function nWave(t,arrival,peak=60,duration=.18,rise=.002){
 if(arrival===null)return 0;
 const u=t-arrival;
 if(u<0||u>duration)return 0;
 const a=2*rise/duration,z=12/(6+9*a+Math.sqrt((6+9*a)**2-192*a));
 const peakFactor=(1-a*z)*smoothstep(z);
 return peak/peakFactor*(1-2*u/duration)*smoothstep(u/rise)*smoothstep((duration-u)/rise);
}
function sincKernel(cutoff,length){const out=[];let sum=0;for(let k=0;k<length;k++){const x=k-(length-1)/2;const y=(x===0?2*cutoff:Math.sin(2*Math.PI*cutoff*x)/(Math.PI*x))*(.5-.5*Math.cos(2*Math.PI*k/(length-1)));out.push(y);sum+=y}return out.map(x=>x/sum)}
let sourceLevels;
function getSource(){
 if(sourceLevels)return sourceLevels;
 const n=1<<18,raw=new Float32Array(n),base=new Float32Array(n);let seed=271828;
 for(let i=0;i<n;i++){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;raw[i]=(seed>>>0)/2147483648-1}
 const kernel=sincKernel(2200/SAMPLE_RATE,31),tones=[105,210,480].map(f=>Math.round(f*n/SAMPLE_RATE));
 for(let i=0;i<n;i++){let s=0;for(let k=0;k<kernel.length;k++)s+=raw[(i+k-15+n)%n]*kernel[k];base[i]=s*1.8+.10*Math.sin(2*Math.PI*tones[0]*i/n)+.06*Math.sin(2*Math.PI*tones[1]*i/n)+.035*Math.sin(2*Math.PI*tones[2]*i/n)}
 sourceLevels=[base];const down=sincKernel(.22,31);
 while(sourceLevels.at(-1).length>32){const prev=sourceLevels.at(-1),next=new Float32Array(prev.length/2);for(let i=0;i<next.length;i++){let s=0;for(let k=0;k<down.length;k++)s+=prev[(2*i+k-15+prev.length)%prev.length]*down[k];next[i]=s}sourceLevels.push(next)}return sourceLevels;
}
function periodicSample(array,index){const len=array.length;index=((index%len)+len)%len;const i=Math.floor(index),f=index-i;return array[i]*(1-f)+array[(i+1)%len]*f}
function jetSample(tau,rate,levels){
 // A low-pass mip pyramid prevents a rapidly compressed source aliasing.
 const lod=clamp(Math.log2(Math.max(1,rate)),0,levels.length-1),lo=Math.floor(lod),hi=Math.min(lo+1,levels.length-1),f=lod-lo;
 return periodicSample(levels[lo],tau*SAMPLE_RATE/2**lo)*(1-f)+periodicSample(levels[hi],tau*SAMPLE_RATE/2**hi)*f;
}
export function renderAudio(config,sampleRate=SAMPLE_RATE){
 const {mach:m,altitude:h,mode='jet',pressure=60,duration=.18}=config;
 const {start,end}=timeWindow(m,h),arrival=boomTime(m,h),len=Math.ceil((end-start)*sampleRate),samples=new Float32Array(len),levels=mode==='jet'?getSource():null;
 let peak=0,energy=0;
 for(let i=0;i<len;i++){
  const t=start+i/sampleRate;let p=0;
  if(mode!=='boom')for(const tau of retardedTimes(t,m,h)){
   const r=Math.hypot(m*C*tau,h),rate=doppler(tau,m,h),jac=Math.sqrt(1/(rate*rate)+.08*.08);
   const gate=arrival===null?1:smoothstep((t-arrival)/.06);
   let source=0;
   if(mode==='tone'){const freq=200*rate;const aa=1-smoothstep((freq-sampleRate*.35)/(sampleRate*.1));source=.35*Math.sin(2*Math.PI*200*tau)*aa}
   else source=jetSample(tau,rate,levels);
   p+=source*(8000/r)/jac*gate;
  }
  p+=nWave(t,arrival,pressure,duration);
  // Fixed nonlinear playback mapping. No per-pass normalization or SPL claim.
  const fade=smoothstep((t-start)/.035)*smoothstep((end-t)/.08);
  const s=.95*Math.tanh(p/200)*fade;samples[i]=s;peak=Math.max(peak,Math.abs(s));energy+=s*s;
 }
 return {samples,start,end,arrival,sampleRate,peak,rms:Math.sqrt(energy/len)};
}

import {Tissue,trial,rng,PARAMS} from '../dist/sim.mjs';
import {writeFileSync,mkdirSync} from 'node:fs';
const random=rng(92726);const normal=()=>Math.sqrt(-2*Math.log(Math.max(1e-12,random())))*Math.cos(2*Math.PI*random());
const generations=Number(process.argv[2]||70),population=18;let mean=Array(PARAMS).fill(0),sd=Array(PARAMS).fill(1.3),best=mean.slice(),bestScore=-Infinity;const history=[];
const seeds=[11,37];const score=w=>seeds.reduce((s,seed)=>s+trial(w,seed,{steps:340}).fitness,0)/seeds.length;
const baseline=score(mean);const start=Date.now();
for(let gen=0;gen<generations;gen++){
 const candidates=Array.from({length:population},(_,i)=>{const w=i===0?best.slice():mean.map((v,k)=>v+sd[k]*normal());return{w,score:score(w)};}).sort((a,b)=>b.score-a.score);
 if(candidates[0].score>bestScore){best=candidates[0].w;bestScore=candidates[0].score;}
 const elites=candidates.slice(0,5);for(let k=0;k<PARAMS;k++){const m=elites.reduce((s,v)=>s+v.w[k],0)/elites.length;const v=elites.reduce((s,x)=>s+(x.w[k]-m)**2,0)/elites.length;mean[k]=.25*mean[k]+.75*m;sd[k]=Math.max(.10,.45*sd[k]+.55*Math.sqrt(v));}
 history.push({generation:gen+1,fitness:bestScore,mean:candidates.reduce((s,v)=>s+v.score,0)/population});
 if(gen%5===0||gen===generations-1){console.log(JSON.stringify({generation:gen+1,fitness:bestScore,baseline,seconds:(Date.now()-start)/1000}));writeFileSync('dist/model.json',JSON.stringify({version:1,architecture:'8 inputs → 3 softmax roles',method:'Cross-entropy evolutionary search',seed:92726,weights:best,baseline,history,trainingSeeds:seeds,generations:gen+1,population},null,2));}
}
const heldout=[101,203,307,409,503,607];const evaluation=heldout.map(seed=>({seed,trained:trial(best,seed),uniform:trial(Array(PARAMS).fill(0),seed),frozen:trial(best,seed,{frozen:true})}));
const model=JSON.parse((await import('node:fs')).readFileSync('dist/model.json','utf8'));model.evaluation=evaluation;model.elapsedSeconds=(Date.now()-start)/1000;writeFileSync('dist/model.json',JSON.stringify(model,null,2));console.log(JSON.stringify({evaluation}));

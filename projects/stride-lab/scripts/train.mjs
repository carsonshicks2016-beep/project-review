import fs from 'node:fs';
import path from 'node:path';
import {VERSION,DEFAULT_ENV,createSearch,generation,robustness,runTrial} from '../dist/physics.js';
const [scope='shared',count='20',seedText='42',output='outputs/training-result.json']=process.argv.slice(2);
const generations=Number(count),seed=Number(seedText);
if(!Number.isInteger(generations)||generations<1||generations>10000||!Number.isInteger(seed))throw new Error('Expected scope, positive generation count, integer seed, output path');
const search=createSearch({scope,seed});
for(let i=0;i<generations;i++){const g=generation(search);if(i===0||(i+1)%5===0||i===generations-1)console.log(`Generation ${g.generation}: held-out ${g.score.toFixed(2)}, this generation ${g.training.toFixed(2)}, ${g.evaluations} trials`);}
const result={version:VERSION,scope,seed,generations,trainingEnv:DEFAULT_ENV,randomized:true,samples:search.samples,validation:search.validation.length,policy:search.best.policy,evaluations:search.evaluations,history:search.history,trials:['sprint','hurdles'].map(e=>runTrial(search.best.policy,e,DEFAULT_ENV,true)),robustness:robustness(search.best.policy)};
fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2));console.log(`Saved ${output}`);
for(const t of result.trials)console.log(`${t.event}: ${t.reason} — ${t.duration.toFixed(3)} s, ${t.distance.toFixed(2)} m`);

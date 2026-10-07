import {createSearch,generation,robustness} from '../dist/physics.js';
const s=createSearch();for(let i=0;i<20;i++)generation(s);
console.log(JSON.stringify(robustness(s.best.policy).map(r=>({name:r.name,trials:r.trials.map(t=>({event:t.event,finished:t.finished,time:t.duration,why:t.reason,distance:t.distance}))})),null,2));

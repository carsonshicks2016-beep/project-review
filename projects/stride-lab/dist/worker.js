import {createSearch,generation,robustness,runTrial} from './physics.js';
let active=null;
self.onmessage=({data})=>{
 try{
  if(data.type==='train'){
   const search=createSearch(data.config),token={};active=token;
   const max=Math.max(1,Math.min(60,Math.floor(data.generations)));
   function tick(){
    if(active!==token)return;
    try{
     const result=generation(search);self.postMessage({type:'progress',jobId:data.jobId,...result});
     if(search.generation<max)setTimeout(tick,0);
     else{active=null;self.postMessage({type:'done',jobId:data.jobId,policy:search.best.policy,history:search.history,evaluations:search.evaluations});}
    }catch(error){active=null;self.postMessage({type:'error',jobId:data.jobId,message:error.message});}
   }tick();
  }else if(data.type==='stop'){active=null;}
  else if(data.type==='robustness'){self.postMessage({type:'robustness',jobId:data.jobId,results:robustness(data.policy,data.env)});}
 }catch(error){self.postMessage({type:'error',jobId:data.jobId,message:error.message});}
};

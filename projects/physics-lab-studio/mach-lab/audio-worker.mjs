import {renderAudio} from './physics.mjs';
self.onmessage=({data})=>{try{const result=renderAudio(data.config);self.postMessage({id:data.id,...result},[result.samples.buffer])}catch(error){self.postMessage({id:data.id,error:error.message})}};

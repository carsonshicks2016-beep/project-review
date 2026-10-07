// A testable animation boundary: every callback re-arms, even if rendering or
// the error reporter throws. Faulted simulation stays paused until explicit retry.
export const FRAME_BUDGET_MS=13, MAX_CATCHUP_SECONDS=2, MAX_FRAME_DELTA_SECONDS=.1;
export function createFrameLoop({step,onError,schedule,cancel=()=>{}}){
 let active=false,faulted=false,handle=null;
 function frame(time){
  if(!active)return;
  try{if(!faulted)step(time);}
  catch(error){faulted=true;try{onError(error);}catch{ /* scheduling must survive reporting errors */ }}
  finally{if(active)handle=schedule(frame);}
 }
 return {
  start(){if(active)return;active=true;handle=schedule(frame);},
  stop(){active=false;cancel(handle);},
  recover(){faulted=false;},
  get faulted(){return faulted;}
 };
}
// Overview will not pull back past the point where the road stops reading: a road needs at least
// MIN_ROAD_FRAME_FRACTION of the frame to look like a road rather than a scratch, which caps the
// visible ground at roadWidth/fraction metres. Courses whose terrain fits inside that footprint keep
// the whole-lap diorama; longer ones frame one sector and leave the full lap to the minimap.
export const MIN_ROAD_FRAME_FRACTION=.02, OVERVIEW_ZOOM=440;
export function overviewFraming({roadWidth,terrainWidth,terrainDepth,aspect,fov,zoom}){
 const groundWidth=roadWidth/MIN_ROAD_FRAME_FRACTION;
 if(Math.max(terrainWidth,terrainDepth)<=groundWidth)
  return {sector:false,groundWidth,distance:zoom*Math.max(1,1.25/aspect)*Math.max(terrainWidth/405,terrainDepth/348)};
 const legible=groundWidth/(2*Math.tan(fov*Math.PI/360)*aspect);
 return {sector:true,groundWidth,distance:legible*(zoom/OVERVIEW_ZOOM)};
}
export function prepareImportedSession(data,currentTrack,engine){
 const driver=engine.readDriver(data);
 const track=engine.makeTrack(driver.seed??currentTrack.seed,driver.elevation??currentTrack.elevation,driver.seed!==null?driver.lengthKm:currentTrack.lengthKm,driver.seed!==null?driver.surface:currentTrack.surface);
 const trainer=new engine.Trainer(track,track.seed+99,driver.weights,{recordLines:true});
 const probe=trainer.adopt(driver.weights);
 return {track,trainer,probe,known:driver.seed!==null};
}
// Float64 deliberately preserves replay exactly. Float32's smaller links can
// perturb a marginal driver's trajectory. Version + seed + elevation accompany weights.
export function encodeDriverLink(data){
 const weights=data.weights,track=data.track;
 if(!Array.isArray(weights)||weights.length!==24||!weights.every(Number.isFinite))throw Error('Expected 24 finite weights.');
 if(!Number.isInteger(track?.seed)||track.seed<1||track.seed>999999||!Number.isFinite(track.elevation)||track.elevation<0||track.elevation>36)throw Error('Invalid driver track.');
 const long=track.lengthKm!=null;
 if(long&&(!Number.isFinite(track.lengthKm)||track.lengthKm<.5||track.lengthKm>12))throw Error('Invalid course length.');
 if(track.surface!=null&&track.surface!=='dry'&&track.surface!=='wet')throw Error('Invalid surface.');
 // Version 3 only appears when the surface is wet, so existing dry links keep their exact bytes.
 const wet=track.surface==='wet',version=wet?3:(long?2:1);
 const bytes=new Uint8Array(wet?214:(long?213:205)),view=new DataView(bytes.buffer);
 view.setUint8(0,version);view.setUint32(1,track.seed,true);view.setFloat64(5,track.elevation,true);
 if(version>=2)view.setFloat64(205,long?track.lengthKm:0,true);
 if(wet)view.setUint8(213,1);
 weights.forEach((w,i)=>view.setFloat64(13+i*8,w,true));
 return '#driver='+btoa(String.fromCharCode(...bytes)).replaceAll('+','-').replaceAll('/','_').replace(/=+$/,'');
}
export function decodeDriverLink(hash){
 if(!hash.startsWith('#driver='))return null;
 const encoded=hash.slice(8);
 if(!/^([A-Za-z0-9_-]{274}|[A-Za-z0-9_-]{284}|[A-Za-z0-9_-]{286})$/.test(encoded))throw Error('This driver link is incomplete or invalid.');
 let binary;try{binary=atob(encoded.replaceAll('-','+').replaceAll('_','/')+'='.repeat((4-encoded.length%4)%4));}catch{throw Error('Invalid driver link.');}
 const bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
 if(!((bytes.length===205&&bytes[0]===1)||(bytes.length===213&&bytes[0]===2)||(bytes.length===214&&bytes[0]===3)))throw Error('Unsupported driver link version.');
 const view=new DataView(bytes.buffer),data={weights:Array.from({length:24},(_,i)=>view.getFloat64(13+i*8,true)),track:{seed:view.getUint32(1,true),elevation:view.getFloat64(5,true)}};
 if(bytes[0]>=2){const km=view.getFloat64(205,true);if(km!==0)data.track.lengthKm=km;}
 if(bytes[0]===3)data.track.surface=view.getUint8(213)===1?'wet':'dry';
 // Validate fully before replacing any current session.
 encodeDriverLink(data);return data;
}
export function comparisonSeries(trainers){
 return trainers.map(t=>({algorithm:t.algorithm,points:t.history.map(h=>({x:h.evaluations,lap:h.bestLap,completion:h.completion}))}));
}

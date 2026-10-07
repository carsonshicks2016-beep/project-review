import {C,clamp,regime,coneAngle,boomTime,timeWindow,retardedTimes,doppler,nWave,shapedWave,AIRCRAFT_PRESETS,slantRange,carpetWidth,lateralAttenuation,lateralOverpressure,vaporConeIntensity} from './physics.mjs';
import {renderSchlieren} from './schlieren.mjs';
import {SpectrogramRenderer} from './spectrogram.mjs';
import {calculateCaustic,superboomWave,computeWavefrontEnvelope,TRAJECTORY_PRESETS,getTrajectoryPosition,getTrajectoryMach} from './caustics.mjs';
import {generateSubBassHarmonics,triggerHapticBoom,createImpulseResponse,flybyAudio} from './synth.mjs';

const $=id=>document.getElementById(id);
const scene=$('scene'),sc=scene.getContext('2d'),trace=$('trace'),tc=trace.getContext('2d');

let config={
  mach:2,
  altitude:500,
  lateralOffset:0,
  aircraft:'custom',
  trajectory:'level',
  environment:'desert',
  subbass:true,
  mode:'jet',
  pressure:60,
  duration:.18,
  rise:0.002
};

let bounds=timeWindow(2,500,0),t=.8,pristine=true,playing=false,building=true;
let ctx=null,gain=null,source=null,audioData=null,audioBuffer=null,audioStart=0,offset=0;
let job=0,worker=null,renderTimer,muted=false,traceMode='pass',viewMode='geo',lastUi=0,playIntent=false;
let hapticTriggered=false,convolverNode=null,reverbGain=null,dryGain=null;
let sceneSize={w:0,h:0},traceSize={w:0,h:0};

const spectrogramRenderer=new SpectrogramRenderer();
let spectrogramData=null;

// Trace buttons
const traceButtons=document.createElement('div');
traceButtons.className='trace-switch';
traceButtons.innerHTML='<button class="active" data-trace="pass">Full pass</button><button data-trace="nwave" id="trace-nwave-btn">N-wave</button><button data-trace="caustic" id="trace-caustic-btn">Superboom</button>';
document.querySelector('.readout-title').insertBefore(traceButtons,document.querySelector('.readout-state'));

for(const b of traceButtons.querySelectorAll('button')){
  b.onclick=()=>{
    traceMode=b.dataset.trace;
    for(const x of traceButtons.children)x.classList.toggle('active',x===b);
    $('trace-axis').textContent=traceMode==='pass'?'RELATIVE PRESSURE':(traceMode==='caustic'?'SUPERBOOM OVERPRESSURE · Pa':'MODEL OVERPRESSURE · Pa');
    $('trace-caption').textContent=traceMode==='pass'?'FULL PASS · PLAYBACK SIGNAL':(traceMode==='caustic'?'CAUSTIC FOCUS WAVEFORM · ASYMMETRIC U-WAVE':(config.aircraft==='x59'?'SHAPED LOW-BOOM · QUIET SIGNATURE':'BOOM DETAIL · 2 ms SHOCK EDGES'));
    drawTrace();
  };
}

// View buttons
const viewButtons=document.querySelectorAll('#view-switch button');
viewButtons.forEach(btn=>{
  btn.onclick=()=>{
    viewMode=btn.dataset.view;
    viewButtons.forEach(b=>b.classList.toggle('active',b===btn));
    draw();
  };
});

function resize(canvas,context){
  const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);
  canvas.width=Math.round(r.width*d);
  canvas.height=Math.round(r.height*d);
  context.setTransform(d,0,0,d,0,0);
  return {w:r.width,h:r.height};
}

new ResizeObserver(()=>{sceneSize=resize(scene,sc);traceSize=resize(trace,tc);draw();}).observe(document.querySelector('.simulation'));
new ResizeObserver(()=>{traceSize=resize(trace,tc);drawTrace();}).observe(document.querySelector('.trace-wrap'));

function stopSound(){
  if(source){
    source.onended=null;
    try{source.stop();}catch{}
    source.disconnect();
    source=null;
  }
}

function audioClock(){
  const stamp=ctx?.getOutputTimestamp?.();
  return stamp?.contextTime>0?stamp.contextTime+Math.max(0,performance.now()-stamp.performanceTime)/1000:ctx.currentTime;
}

function now(){
  return playing&&ctx?clamp(bounds.start+offset+audioClock()-audioStart,bounds.start,bounds.end):t;
}

function pause(){
  t=now();
  playing=false;
  playIntent=false;
  hapticTriggered=false;
  stopSound();
  updateUI();
}

function showAudioError(message){
  building=false;
  playIntent=false;
  $('play').disabled=false;
  $('play-label').textContent='Retry audio';
  $('event-explanation').textContent=message;
}

function updateReverb(){
  if(!ctx||!convolverNode)return;
  if(config.environment==='desert'){
    if(reverbGain)reverbGain.gain.setTargetAtTime(0,ctx.currentTime,0.02);
    return;
  }
  try{
    const ir=createImpulseResponse(config.environment,ctx.sampleRate,config.environment==='canyon'?1.8:1.2);
    const irBuf=ctx.createBuffer(2,ir.channelL.length,ctx.sampleRate);
    irBuf.copyToChannel(ir.channelL,0);
    irBuf.copyToChannel(ir.channelR,1);
    convolverNode.buffer=irBuf;
    if(reverbGain)reverbGain.gain.setTargetAtTime(config.environment==='canyon'?0.42:0.30,ctx.currentTime,0.02);
  }catch{}
}

function requestRender(){
  building=true;
  audioData=null;
  audioBuffer=null;
  const id=++job;
  clearTimeout(renderTimer);
  $('play-label').textContent='Preparing sound…';
  $('play').disabled=false;
  
  renderTimer=setTimeout(()=>{
    try{
      if(!worker){
        worker=new Worker(new URL('./audio-worker.mjs',import.meta.url),{type:'module'});
        worker.onmessage=({data})=>{
          if(data.id!==job)return;
          if(data.error){showAudioError('Audio generation failed. Use Listen to try again.');return;}
          audioData=data;
          
          // Apply sub-bass harmonic enhancement if enabled
          if(config.subbass&&audioData.samples){
            try{
              const sub=generateSubBassHarmonics(audioData.samples,audioData.sampleRate);
              if(audioData.isStereo&&audioData.samplesL&&audioData.samplesR){
                audioData.samplesL=generateSubBassHarmonics(audioData.samplesL,audioData.sampleRate);
                audioData.samplesR=generateSubBassHarmonics(audioData.samplesR,audioData.sampleRate);
              }
              audioData.samples=sub;
            }catch{}
          }
          
          // Generate spectrogram data
          try{
            spectrogramData=spectrogramRenderer.generateSpectrogramData(audioData.samples,audioData.sampleRate);
          }catch{}
          
          building=false;
          audioBuffer=null;
          updateUI();
          drawTrace();
          if(viewMode==='spectrogram')drawScene();
          if(playIntent)startPlayback();
        };
        worker.onerror=()=>{
          worker?.terminate();
          worker=null;
          showAudioError('The audio engine could not start. Use Listen to retry.');
        };
      }
      worker.postMessage({id,config});
    }catch{
      showAudioError('This browser could not load the audio engine. Open this site in a current browser.');
    }
  },100);
}

async function ensureContext(){
  if(!ctx){
    const Audio=window.AudioContext||window.webkitAudioContext;
    if(!Audio)throw Error('Web Audio is unavailable');
    ctx=new Audio();
    gain=ctx.createGain();
    gain.gain.value=muted?0:Number($('volume').value)/100;
    
    convolverNode=ctx.createConvolver();
    reverbGain=ctx.createGain();
    dryGain=ctx.createGain();
    
    reverbGain.gain.value=config.environment==='desert'?0:0.35;
    dryGain.gain.value=1.0;
    
    updateReverb();
    
    gain.connect(dryGain);
    gain.connect(convolverNode);
    convolverNode.connect(reverbGain);
    
    dryGain.connect(ctx.destination);
    reverbGain.connect(ctx.destination);
  }
  await ctx.resume();
  if(ctx.state!=='running')throw Error('Audio is paused by the browser');
}

async function startPlayback(){
  try{
    await ensureContext();
    if(building){playIntent=true;$('play-label').textContent='Preparing sound…';return;}
    if(!audioData){requestRender();playIntent=true;return;}
    
    stopSound();
    if(pristine||t>=bounds.end-.02)t=bounds.start;
    pristine=false;
    hapticTriggered=false;
    
    if(!audioBuffer){
      const isStereo=Boolean(audioData.isStereo&&audioData.samplesL&&audioData.samplesR);
      audioBuffer=ctx.createBuffer(isStereo?2:1,(audioData.samplesL||audioData.samples).length,audioData.sampleRate);
      if(isStereo){
        audioBuffer.copyToChannel(audioData.samplesL,0);
        audioBuffer.copyToChannel(audioData.samplesR,1);
      }else{
        audioBuffer.copyToChannel(audioData.samples,0);
      }
    }
    
    source=ctx.createBufferSource();
    source.buffer=audioBuffer;
    source.connect(gain);
    offset=clamp(t-bounds.start,0,audioBuffer.duration-.001);
    audioStart=ctx.currentTime+.03;
    playing=true;
    playIntent=false;
    source.start(audioStart,offset);
    source.onended=()=>{t=bounds.end;playing=false;source=null;updateUI();draw();};
    updateUI();
  }catch{
    showAudioError('Audio could not start. Press Listen again to enable browser audio.');
  }
}

$('play').onclick=()=>{if(playing||playIntent)pause();else startPlayback();};
$('restart').onclick=()=>{const resume=playing;pause();pristine=false;t=bounds.start;updateUI();draw();if(resume)startPlayback();};
$('scrub').oninput=()=>{pause();pristine=false;t=bounds.start+Number($('scrub').value)/1000*(bounds.end-bounds.start);updateUI();draw();};
$('volume').oninput=()=>{muted=false;if(gain)gain.gain.setTargetAtTime(Number($('volume').value)/100,ctx.currentTime,.015);updateVolume();};
$('mute').onclick=()=>{muted=!muted;if(gain)gain.gain.setTargetAtTime(muted?0:Number($('volume').value)/100,ctx.currentTime,.015);updateVolume();};

function updateVolume(){
  $('mute').textContent=muted?'⊘':'♫';
  $('mute').setAttribute('aria-label',muted?'Unmute sound':'Mute sound');
  $('mute').title=muted?'Unmute sound':'Mute sound';
}

function configure(change){
  pause();
  config={...config,...change};
  bounds=timeWindow(config.mach,config.altitude,config.lateralOffset);
  const arrival=boomTime(config.mach,config.altitude,config.lateralOffset);
  t=arrival===null?(config.mach===1?.3:-.7):Math.min(arrival*.65,1.5);
  pristine=true;
  syncControls();
  requestRender();
  draw();
}

function syncControls(){
  const {mach:m,altitude:h,lateralOffset:latY=0,aircraft:ac='custom',trajectory:tr='level',environment:env='desert'}=config;
  const angle=coneAngle(m),arrival=boomTime(m,h,latY);
  
  $('mach').value=m;
  $('altitude').value=h;
  $('lateral-offset').value=latY;
  $('aircraft-preset').value=ac;
  $('flight-trajectory').value=tr;
  $('environment-preset').value=env;
  $('subbass-toggle').checked=config.subbass;
  $('sound-mode').value=config.mode;
  $('pressure').value=config.pressure;
  $('duration').value=config.duration*1000;
  
  $('mach-value').textContent=m.toFixed(2);
  $('altitude-value').textContent=h.toLocaleString()+' m';
  $('lateral-offset-value').textContent=latY.toLocaleString()+' m';
  $('regime').textContent=regime(m);
  $('ground-speed').innerHTML=Math.round(m*C*3.6).toLocaleString()+' <small>km/h</small>';
  $('cone-angle').textContent=angle===null?'No finite cone':(angle*180/Math.PI).toFixed(1)+'°';
  $('arrival-label').textContent=arrival===null?'Overhead sound delay':'Boom after overhead';
  $('arrival-delay').innerHTML=(arrival??slantRange(h,latY)/C).toFixed(2)+' <small>s</small>';
  
  const rSlant=slantRange(h,latY);
  $('slant-range').innerHTML=Math.round(rSlant).toLocaleString()+' <small>m</small>';
  const cw=carpetWidth(m,h);
  $('carpet-width').innerHTML=cw>0?Math.round(cw).toLocaleString()+' <small>m</small>':'— <small>m</small>';
  
  const effP=Math.round(lateralOverpressure(config.pressure,h,latY));
  $('pressure-value').textContent=config.pressure+' Pa'+(latY>0?` (ground: ${effP} Pa)`:'');
  $('duration-value').textContent=Math.round(config.duration*1000)+' ms';
  
  const boomBtn=$('trace-nwave-btn');
  if(boomBtn)boomBtn.textContent=ac==='x59'?'Shaped wave':'N-wave';
  
  $('start-time').textContent=bounds.start.toFixed(1)+' s';
  $('end-time').textContent='+'+bounds.end.toFixed(1)+' s';
  
  document.querySelectorAll('[data-mach]').forEach(b=>b.classList.toggle('active',Math.abs(Number(b.dataset.mach)-m)<.001));
  
  $('insight-title').textContent=m>1?'Faster than its own sound.':m===1?'At the speed of sound.':'An audible shift in pitch.';
  $('insight-copy').textContent=m>1?(ac==='x59'?'NASA X-59 QueSST uses shaped airframe compression to prevent shock coalescence, replacing the loud double-boom with a soft thump.':'Above Mach 1, the rings pile up along a cone. You hear the boom when that cone sweeps over you.'):m===1?'The rings bunch up ahead of the source. A finite-angle Mach cone needs a speed strictly above Mach 1.':'Ahead of the jet, waves bunch together and the pitch rises. Behind it, waves spread apart and the pitch falls.';
  
  updateUI();
}

$('mach').oninput=()=>configure({mach:Number($('mach').value)});
$('altitude').oninput=()=>configure({altitude:Number($('altitude').value)});
$('lateral-offset').oninput=()=>configure({lateralOffset:Number($('lateral-offset').value)});
$('aircraft-preset').onchange=()=>{
  const id=$('aircraft-preset').value;
  const p=AIRCRAFT_PRESETS[id]||AIRCRAFT_PRESETS.custom;
  configure({aircraft:id,pressure:p.peak,duration:p.duration,rise:p.rise});
};
$('flight-trajectory').onchange=()=>{
  const traj=$('flight-trajectory').value;
  configure({trajectory:traj});
  const causticBtn=$('trace-caustic-btn');
  if(causticBtn&&(traj==='turn'||traj==='accel')){
    causticBtn.click();
  }
};
$('environment-preset').onchange=()=>{
  config.environment=$('environment-preset').value;
  updateReverb();
};
$('subbass-toggle').onchange=()=>{
  config.subbass=$('subbass-toggle').checked;
  requestRender();
};
$('sound-mode').onchange=()=>configure({mode:$('sound-mode').value});
$('pressure').oninput=()=>configure({pressure:Number($('pressure').value),aircraft:'custom'});
$('duration').oninput=()=>configure({duration:Number($('duration').value)/1000,aircraft:'custom'});
document.querySelectorAll('[data-mach]').forEach(b=>b.onclick=()=>configure({mach:Number(b.dataset.mach)}));

// Custom sound flyby-izer file handler
const customAudioBtn=$('custom-audio-btn');
const customAudioFile=$('custom-audio-file');
if(customAudioBtn&&customAudioFile){
  customAudioBtn.onclick=()=>customAudioFile.click();
  customAudioFile.onchange=async(e)=>{
    const file=e.target.files[0];
    if(!file)return;
    try{
      await ensureContext();
      const arrayBuf=await file.arrayBuffer();
      const decoded=await ctx.decodeAudioData(arrayBuf);
      const rawMono=decoded.getChannelData(0);
      const flyby=flybyAudio(rawMono,ctx.sampleRate,config.mach,config.altitude,config.lateralOffset);
      audioData={
        samples:flyby.samples,
        samplesL:flyby.samplesL,
        samplesR:flyby.samplesR,
        isStereo:true,
        sampleRate:flyby.sampleRate,
        peak:flyby.peak,
        rms:flyby.rms
      };
      if(spectrogramRenderer){
        spectrogramData=spectrogramRenderer.generateSpectrogramData(flyby.samples,flyby.sampleRate);
      }
      audioBuffer=null;
      bounds={start:flyby.start,end:flyby.end};
      t=bounds.start;
      $('event-explanation').textContent=`Custom sound "${file.name}" auralized through supersonic Doppler engine!`;
      updateUI();
      draw();
      startPlayback();
    }catch(err){
      showAudioError('Could not process custom audio: '+err.message);
    }
  };
}

const dialog=$('model-dialog');
for(const id of ['about-button','model-button'])$(id).onclick=()=>dialog.showModal();
$('close-dialog').onclick=()=>dialog.close();
dialog.onclick=e=>{
  if(e.target===dialog){
    const r=dialog.getBoundingClientRect();
    if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();
  }
};

document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)pause();});

function updateUI(){
  t=now();
  const {mach:m,altitude:h,lateralOffset:latY=0,aircraft:ac='custom',trajectory:tr='level'}=config;
  const arrival=boomTime(m,h,latY),roots=retardedTimes(t,m,h,latY),hasSound=roots.length>0;
  
  $('scene-clock').textContent='T '+(t<0?'−':'+')+Math.abs(t).toFixed(2)+' s';
  $('scrub').value=(t-bounds.start)/(bounds.end-bounds.start)*1000;
  
  $('play-icon').textContent=playing?'Ⅱ':'▶';
  $('play-label').textContent=building?'Preparing sound…':playing?'Pause flyby':t>=bounds.end-.02?'Replay flyby':pristine?'Listen to flyby':'Resume flyby';
  
  const inBoom=arrival!==null&&t>=arrival&&t<=arrival+config.duration;
  $('listener-state').textContent=config.mode==='boom'?(inBoom?'SHOCK AT LISTENER':arrival===null?'NO SONIC BOOM':t<arrival?'WAITING FOR SHOCK':'SHOCK HAS PASSED'):inBoom?'SHOCK AT LISTENER':hasSound?'SOUND ARRIVING':'WAITING FOR SOUND';
  
  const ratio=hasSound?doppler(roots.at(-1),m,h,latY):null;
  $('pitch-value').textContent=config.mode==='boom'?'N-WAVE ONLY':ratio&&Number.isFinite(ratio)?Math.round(200*ratio).toLocaleString()+' Hz'+(config.mode==='jet'?' ref.':''):'— Hz';
  
  const vIntensity=vaporConeIntensity(m);
  $('scene-kicker').textContent=tr==='accel'?'SUPERSONIC ACCELERATION (CAUSTIC)':tr==='turn'?'SUPERSONIC BANKING TURN':vIntensity>0.5?'TRANSONIC · PRANDTL-GLAUERT VAPOR CONE':regime(m)+' FLIGHT';
  
  $('scene-status').textContent=arrival!==null?(t<0?'The jet outruns its sound.':t<arrival?'Overhead. Still no sound.':inBoom?(ac==='x59'?'The shaped quiet thump reaches you.':'The shock front reaches you.'):'The boom passes. Sound follows.'):m===1?(t<=0?'Sound gathers ahead of the jet.':'Earlier sound is reaching you.'):(t<h/C?'Hear the approaching jet.':'Hear the pitch fall away.');
  
  $('event-explanation').textContent=arrival!==null?(t<arrival?'Shock arrival in '+(arrival-t).toFixed(2)+' s · '+arrival.toFixed(2)+' s after overhead.'+(latY>0?` (slant range ${Math.round(slantRange(h,latY))} m)`:'')+'.':`The shock passed at T +${arrival.toFixed(2)} s.`+(ac==='x59'?' NASA X-59 shaped low-boom profile.':` The two shocks are ${Math.round(config.duration*1000)} ms apart.`)+(latY>0?` Ground peak: ${Math.round(lateralOverpressure(config.pressure,h,latY))} Pa.`:'')):m===1?'Sonic limit: no added N-wave. Real transonic shocks require an aircraft flow model.':config.mode==='boom'?'No sonic boom below Mach 1. Select the jet or tone to hear this pass.':'The sound emitted directly overhead arrives '+(slantRange(h,latY)/C).toFixed(2)+' s later.';
}

function line(context,x1,y1,x2,y2,color,width=1,dash=[]){
  context.beginPath();
  context.strokeStyle=color;
  context.lineWidth=width;
  context.setLineDash(dash);
  context.moveTo(x1,y1);
  context.lineTo(x2,y2);
  context.stroke();
  context.setLineDash([]);
}

function label(context,text,x,y,color='#819aa6',align='left',size=11){
  context.font=`${size}px "DM Sans", sans-serif`;
  context.textAlign=align;
  context.fillStyle=color;
  context.fillText(text,x,y);
}

function drawScene(){
  const {w,h}=sceneSize;if(!w||!h)return;
  
  // Schlieren mode
  if(viewMode==='schlieren'){
    renderSchlieren(sc,w,h,{
      mach:config.mach,
      altitude:config.altitude,
      lateralOffset:config.lateralOffset,
      aircraft:config.aircraft,
      time:t,
      backgroundDistortion:true,
      palette:'cyber-cyan'
    });
    return;
  }
  
  // Spectrogram mode
  if(viewMode==='spectrogram'){
    if(spectrogramData){
      spectrogramRenderer.drawSpectrogram(scene,spectrogramData,t,bounds,{colormap:'cyberCyan'});
    }else{
      sc.clearRect(0,0,w,h);
      label(sc,'Calculating time-frequency spectrogram…',w/2,h/2,'#819aa6','center',14);
    }
    return;
  }
  
  // Standard geometry view
  const {mach:m,altitude:alt,lateralOffset:latY=0,aircraft:ac='custom',trajectory:tr='level'}=config;
  sc.clearRect(0,0,w,h);
  const ground=h*.77,scale=Math.min((ground-110)/alt,w/(2*alt*Math.max(2.1,m*1.25))),ox=w*.47,py=ground-alt*scale,px=ox+m*C*t*scale;
  
  const gridStep=[50,100,200,500,1000,2000,5000].find(s=>s*scale>=52)||5000,grid=gridStep*scale;
  for(let x=ox%grid;x<w;x+=grid)line(sc,x,0,x,ground,'#192c36');
  for(let y=ground%grid;y<ground;y+=grid)line(sc,0,y,w,y,'#192c36');
  
  sc.save();sc.beginPath();sc.rect(0,0,w,ground);sc.clip();
  const interval=alt/C/5,history=Math.max(14*alt/C,Math.abs(t)+5);let count=0;
  for(let tau=Math.floor(t/interval)*interval;tau>t-history&&count<110;tau-=interval,count++){
    const radius=C*(t-tau)*scale,cx=ox+m*C*tau*scale;if(radius<2)continue;
    sc.beginPath();sc.arc(cx,py,radius,0,Math.PI*2);sc.strokeStyle=`rgba(91,187,207,${.12+.24*Math.exp(-(t-tau)/(4*alt/C))})`;sc.lineWidth=1;sc.stroke();
  }
  
  if(m>1){
    const slope=1/Math.sqrt(m*m-1),left=-w*3;
    sc.beginPath();sc.moveTo(px,py);sc.lineTo(left,py+(px-left)*slope);sc.lineTo(left,py-(px-left)*slope);sc.closePath();sc.fillStyle='#c2f88109';sc.fill();
    line(sc,px,py,left,py+(px-left)*slope,'#c2f881aa',1.5);line(sc,px,py,left,py-(px-left)*slope,'#c2f88166',1.5);
    if(px>50&&px<w-70){
      sc.beginPath();sc.strokeStyle='#b0db7855';sc.arc(px,py,65,Math.PI-Math.asin(1/m),Math.PI);sc.stroke();
      label(sc,(coneAngle(m)*180/Math.PI).toFixed(1)+'°',px-80,py+18,'#c2f881');
    }
  }
  
  // Caustic focus visualization for maneuvers
  if(m>1&&(tr==='turn'||tr==='accel')){
    try{
      const causticInfo=calculateCaustic(tr,{mach:m,altitude:alt,acceleration:8.5,radius:3200});
      if(causticInfo.hasCaustic&&causticInfo.focusPoints&&causticInfo.focusPoints.length){
        const fp=causticInfo.focusPoints[0];
        const fpDist=fp.x-m*C*t;
        const fX=ox+fpDist*scale;
        if(fX>=-30&&fX<=w+30){
          sc.beginPath();sc.arc(fX,ground,10,0,Math.PI*2);sc.fillStyle='rgba(255,112,80,0.30)';sc.fill();
          sc.beginPath();sc.arc(fX,ground,5,0,Math.PI*2);sc.fillStyle='#ff7050';sc.fill();
          label(sc,`CAUSTIC FOCUS · ${causticInfo.focusFactor.toFixed(1)}× SUPERBOOM`,fX,ground-16,'#ff7050','center',10);
        }
      }
    }catch{}
  }
  
  const roots=retardedTimes(t,m,alt,latY);
  if(roots.length&&config.mode!=='boom'){
    const tau=roots.at(-1),ex=ox+m*C*tau*scale;line(sc,ex,py,ox,ground,'#65c8d580',1,[4,5]);
    sc.beginPath();sc.arc(ex,py,3,0,Math.PI*2);sc.fillStyle='#65c8d5';sc.fill();
  }
  line(sc,0,py,w,py,'#56727b66',1,[3,7]);sc.restore();
  
  sc.fillStyle='#0c171d';sc.fillRect(0,ground,w,h-ground);line(sc,0,ground,w,ground,'#51656c');
  for(let x=ox%grid;x<w;x+=grid){
    line(sc,x,ground,x,ground+5,'#51656c');
    if(Math.abs(x-ox)>grid*.3)label(sc,((x-ox)/scale/1000).toFixed(1)+' km',x,ground+21,'#657e8c','center',10);
  }
  
  const visibleX=clamp(px,24,w-24);
  
  // Render Prandtl-Glauert vapor cone when in transonic regime (M ≈ 0.92–1.06)
  const vIntensity=vaporConeIntensity(m);
  if(vIntensity>0.01){
    sc.save();sc.translate(visibleX,py);
    const grad=sc.createRadialGradient(-6,0,2,-18,0,32);
    grad.addColorStop(0,`rgba(240,252,255,${0.75*vIntensity})`);
    grad.addColorStop(0.35,`rgba(180,230,248,${0.50*vIntensity})`);
    grad.addColorStop(0.7,`rgba(100,200,225,${0.20*vIntensity})`);
    grad.addColorStop(1,'rgba(100,200,225,0)');
    sc.beginPath();
    sc.moveTo(6,0);
    sc.quadraticCurveTo(-6,-20,-32,-26);
    sc.lineTo(-26,0);
    sc.lineTo(-32,26);
    sc.quadraticCurveTo(-6,20,6,0);
    sc.closePath();
    sc.fillStyle=grad;
    sc.fill();
    sc.beginPath();
    sc.ellipse(-8,0,7,18,0,0,Math.PI*2);
    sc.strokeStyle=`rgba(255,255,255,${0.65*vIntensity})`;
    sc.lineWidth=1.5;
    sc.stroke();
    sc.restore();
  }
  
  // Aircraft marker
  sc.save();sc.translate(visibleX,py);sc.beginPath();sc.moveTo(20,0);sc.lineTo(1,-4);sc.lineTo(-10,-14);sc.lineTo(-7,-3);sc.lineTo(-18,-5);sc.lineTo(-15,0);sc.lineTo(-18,5);sc.lineTo(-7,3);sc.lineTo(-10,14);sc.lineTo(1,4);sc.closePath();sc.fillStyle='#edf5f1';sc.fill();sc.restore();
  
  if(px>20&&px<w-20){
    label(sc,'M '+m.toFixed(2)+(vIntensity>0.4?' · VAPOR CONE':''),visibleX,py-28,'#d7e6e9','center');
    line(sc,visibleX-35,py,visibleX-23,py,'#f3b786',2);
  }else{
    label(sc,(px<24?'← ':'')+'JET '+Math.abs(m*C*t/1000).toFixed(1)+' km'+(px>w-24?' →':''),visibleX,py-25,'#c6d9de',px<24?'left':'right');
  }
  
  const arrival=boomTime(m,alt,latY),flash=arrival===null?0:Math.max(0,1-Math.abs(t-arrival)/.35);
  if(flash>0){
    sc.beginPath();sc.arc(ox,ground-8,18+25*(1-flash),0,Math.PI*2);sc.strokeStyle=`rgba(194,248,129,${flash})`;sc.lineWidth=2;sc.stroke();
  }
  
  sc.fillStyle=flash>.1?'#c2f881':'#e6eeeb';sc.beginPath();sc.arc(ox,ground-18,4,0,Math.PI*2);sc.fill();
  line(sc,ox,ground-13,ox,ground-3,'#e6eeeb',2);line(sc,ox-4,ground,ox,ground-5,'#e6eeeb',2);line(sc,ox+4,ground,ox,ground-5,'#e6eeeb',2);line(sc,ox-5,ground-9,ox+5,ground-9,'#e6eeeb',1.5);
  
  sc.fillStyle='#0c171d';sc.fillRect(ox-65,ground+10,130,latY>0?36:18);
  label(sc,'YOU · GROUND LISTENER',ox,ground+23,'#c3d0d3','center',10);
  if(latY>0){
    label(sc,`OFFSET: ${latY.toLocaleString()} m · SLANT ${Math.round(slantRange(alt,latY)).toLocaleString()} m`,ox,ground+37,'#f3b786','center',9);
    const offLen=Math.min(32,(latY/5000)*22+10);
    line(sc,ox,ground,ox+offLen,ground+offLen*0.5,'#f3b786',1.5,[2,3]);
    sc.beginPath();sc.arc(ox+offLen,ground+offLen*0.5,2.5,0,Math.PI*2);sc.fillStyle='#f3b786';sc.fill();
  }
  
  const bx=w-24;line(sc,bx,py,bx,ground,'#7b969d66',1,[2,4]);line(sc,bx-4,py,bx+4,py,'#7b969d');line(sc,bx-4,ground,bx+4,ground,'#7b969d');sc.save();sc.translate(bx-8,(py+ground)/2);sc.rotate(-Math.PI/2);label(sc,alt.toLocaleString()+' m',0,0,'#96aeb8','center',10);sc.restore();
  
  if(m>1){
    const cw=carpetWidth(m,alt);
    label(sc,'CARPET WIDTH: '+Math.round(cw).toLocaleString()+' m',w-18,ground-12,'#708b97','right',9);
  }
  label(sc,'1 DIV = '+(gridStep>=1000?gridStep/1000+' km':gridStep+' m'),18,ground-12,'#708b97','left',9);
}

function drawTrace(){
  const {w,h}=traceSize;if(!w||!h)return;
  tc.clearRect(0,0,w,h);
  const top=23,bottom=h-23,mid=(top+bottom)/2;
  const arrival=boomTime(config.mach,config.altitude,config.lateralOffset);
  let start=bounds.start,end=bounds.end;
  const isX59=config.aircraft==='x59';
  const effPeak=lateralOverpressure(config.pressure,config.altitude,config.lateralOffset);
  
  if(traceMode==='caustic'){
    const causticInfo=calculateCaustic(config.trajectory,{mach:config.mach,altitude:config.altitude,acceleration:8.5,radius:3200});
    const focusFact=causticInfo.hasCaustic?causticInfo.focusFactor:2.5;
    const superPeak=effPeak*focusFact;
    if(arrival===null){
      line(tc,0,mid,w,mid,'#39505b');
      label(tc,'No superboom caustic below Mach 1.',w/2,mid-12,'#a5b7c0','center',14);
      return;
    }
    start=arrival-.045;end=arrival+config.duration+.045;
    const map=time=>(time-start)/(end-start)*w;
    for(let i=0;i<=8;i++)line(tc,w*i/8,top,w*i/8,bottom,'#21333d');
    line(tc,0,mid,w,mid,'#304750');
    tc.beginPath();tc.strokeStyle='#ff7050';tc.lineWidth=2.2;
    for(let x=0;x<=w;x++){
      const tVal=start+x/w*(end-start);
      const pVal=superboomWave(tVal,arrival,effPeak,focusFact,config.duration,config.rise);
      const y=mid-(pVal/superPeak)*(bottom-top)*.44;
      if(x===0)tc.moveTo(x,y);else tc.lineTo(x,y);
    }
    tc.stroke();
    label(tc,`+${Math.round(superPeak)} Pa (${focusFact.toFixed(1)}× SUPERBOOM)`,6,top+11,'#ff7050','left',10);
    label(tc,`−${Math.round(superPeak*0.7)} Pa`,6,bottom,'#ff7050','left',10);
    label(tc,'ASYMMETRIC U-WAVE CAUSTIC SIGNATURE',w/2,top+11,'#ff7050','center',10);
    label(tc,'FOCUSED PEAK',map(arrival)+6,h-3,'#ff9070','left',9);
    label(tc,'RECOMPRESSION',map(arrival+config.duration)-6,h-3,'#ff9070','right',9);
    const playX=map(t);
    if(playX>=0&&playX<=w){
      line(tc,playX,top-4,playX,bottom+3,'#f0f4f2',1.4);
      tc.beginPath();tc.moveTo(playX-3,top-8);tc.lineTo(playX+3,top-8);tc.lineTo(playX,top-3);tc.closePath();tc.fillStyle='#f0f4f2';tc.fill();
    }
    return;
  }
  
  if(traceMode==='nwave'){
    if(arrival===null){
      line(tc,0,mid,w,mid,'#39505b');
      label(tc,isX59?'No shaped wave at or below Mach 1.':'No N-wave at or below Mach 1.',w/2,mid-12,'#a5b7c0','center',14);
      return;
    }
    start=arrival-.045;end=arrival+config.duration+.045;
  }
  
  const map=time=>(time-start)/(end-start)*w;
  for(let i=0;i<=8;i++)line(tc,w*i/8,top,w*i/8,bottom,'#21333d');
  line(tc,0,mid,w,mid,'#304750');
  
  if(traceMode==='nwave'){
    tc.beginPath();tc.strokeStyle=isX59?'#65c8d5':'#c2f881';tc.lineWidth=2;
    const pNorm=Math.max(1,effPeak);
    for(let x=0;x<=w;x++){
      const tVal=start+x/w*(end-start);
      const pVal=shapedWave(tVal,arrival,config.aircraft,effPeak,config.duration,config.rise);
      const y=mid-(pVal/pNorm)*(bottom-top)*.42;
      if(x===0)tc.moveTo(x,y);else tc.lineTo(x,y);
    }
    tc.stroke();
    label(tc,'+'+Math.round(effPeak)+' Pa',6,top+11,isX59?'#65c8d5':'#9aad9b','left',10);
    label(tc,'−'+Math.round(effPeak)+' Pa',6,bottom,isX59?'#65c8d5':'#9aad9b','left',10);
    if(isX59){
      label(tc,'NASA X-59 SHAPED LOW-BOOM (QUIET THUMP)',w/2,top+11,'#65c8d5','center',10);
      label(tc,'NOSE RAMP (SOFT)',map(arrival)+6,h-3,'#65c8d5','left',9);
      label(tc,'TAIL RESTORATION',map(arrival+config.duration)-6,h-3,'#65c8d5','right',9);
    }else{
      label(tc,'BOW SHOCK',map(arrival)+6,h-3,'#b6ce9d','left',9);
      label(tc,'TAIL SHOCK',map(arrival+config.duration)-6,h-3,'#b6ce9d','right',9);
    }
  }else if(audioData){
    const samples=audioData.samples;tc.strokeStyle='#74c4cf';tc.lineWidth=1;
    const gainScale=(bottom-top)*.45/Math.max(.08,audioData.peak);
    tc.beginPath();
    for(let x=0;x<w;x++){
      const a=Math.floor(x/w*samples.length),b=Math.floor((x+1)/w*samples.length);
      let lo=0,hi=0;
      for(let i=a;i<b;i++){lo=Math.min(lo,samples[i]);hi=Math.max(hi,samples[i]);}
      tc.moveTo(x,mid-hi*gainScale);tc.lineTo(x,mid-lo*gainScale);
    }
    tc.stroke();
  }else{
    label(tc,'Calculating arriving sound…',w/2,mid-12,'#819aa6','center',12);
  }
  
  if(traceMode==='pass'){
    const over=map(0);line(tc,over,top,over,bottom,'#82929b',1,[3,4]);
    label(tc,'OVERHEAD',over+5,h-4,'#82929b','left',9);
    if(arrival!==null){
      const x=map(arrival);line(tc,x,top,x,bottom,'#c2f88166',1,[3,4]);
      label(tc,isX59?'SHAPED BOOM':'BOOM',x+6,top+9,isX59?'#65c8d5':'#c2f881','left',9);
    }
  }
  
  const playX=map(t);
  if(playX>=0&&playX<=w){
    line(tc,playX,top-4,playX,bottom+3,'#f0f4f2',1.4);
    tc.beginPath();tc.moveTo(playX-3,top-8);tc.lineTo(playX+3,top-8);tc.lineTo(playX,top-3);tc.closePath();tc.fillStyle='#f0f4f2';tc.fill();
  }
}

function draw(){drawScene();drawTrace();}

function frame(time){
  if(playing){
    t=now();
    const arr=boomTime(config.mach,config.altitude,config.lateralOffset);
    if(arr!==null&&Math.abs(t-arr)<0.035&&!hapticTriggered){
      triggerHapticBoom(Math.round(config.duration*1000));
      hapticTriggered=true;
    }
    drawScene();
    drawTrace();
    if(time-lastUi>70){updateUI();lastUi=time;}
  }
  requestAnimationFrame(frame);
}

// WebMCP agent tools
const modelContext=document.modelContext;
if(modelContext?.registerTool){
  const lifecycle=new AbortController();
  const register=tool=>{try{Promise.resolve(modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{}};
  register({
    name:'read_flyby',
    description:'Read the current flyby settings, simulation time, acoustic geometry, and caustics.',
    inputSchema:{type:'object',properties:{},additionalProperties:false},
    annotations:{readOnlyHint:true,untrustedContentHint:false},
    execute:()=>({...config,time:now(),playing,boomArrival:boomTime(config.mach,config.altitude,config.lateralOffset),coneHalfAngleDegrees:coneAngle(config.mach)===null?null:coneAngle(config.mach)*180/Math.PI})
  });
  register({
    name:'configure_flyby',
    description:'Set Mach number, altitude, lateral offset, aircraft preset, trajectory, environment, or sound source.',
    inputSchema:{
      type:'object',
      properties:{
        mach:{type:'number',minimum:.3,maximum:8},
        altitude:{type:'number',minimum:100,maximum:2000},
        lateralOffset:{type:'number',minimum:0,maximum:5000},
        aircraft:{type:'string',enum:['custom','f16','concorde','sr71','x59']},
        trajectory:{type:'string',enum:['level','dive','climb','turn','accel']},
        environment:{type:'string',enum:['desert','canyon','urban']},
        mode:{type:'string',enum:['jet','tone','boom']}
      },
      additionalProperties:false
    },
    annotations:{readOnlyHint:false,untrustedContentHint:false},
    execute:input=>{
      if(!input||typeof input!=='object'||Array.isArray(input))throw Error('Invalid flyby settings');
      configure(input);
      return {...config,audio:'preparing'};
    }
  });
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}

syncControls();
requestRender();
requestAnimationFrame(frame);

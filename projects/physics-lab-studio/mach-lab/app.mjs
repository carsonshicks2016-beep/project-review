import {C,clamp,regime,coneAngle,boomTime,timeWindow,retardedTimes,doppler,nWave} from './physics.mjs';
const $=id=>document.getElementById(id);
const scene=$('scene'),sc=scene.getContext('2d'),trace=$('trace'),tc=trace.getContext('2d');
let config={mach:2,altitude:500,mode:'jet',pressure:60,duration:.18};
let bounds=timeWindow(2,500),t=.8,pristine=true,playing=false,building=true,ctx=null,gain=null,source=null,audioData=null,audioBuffer=null,audioStart=0,offset=0,job=0,worker=null,renderTimer,muted=false,traceMode='pass',lastUi=0,playIntent=false;
let sceneSize={w:0,h:0},traceSize={w:0,h:0};
const traceButtons=document.createElement('div');traceButtons.className='trace-switch';traceButtons.innerHTML='<button class="active" data-trace="pass">Full pass</button><button data-trace="nwave">N-wave</button>';document.querySelector('.readout-title').insertBefore(traceButtons,document.querySelector('.readout-state'));
for(const b of traceButtons.querySelectorAll('button'))b.onclick=()=>{traceMode=b.dataset.trace;for(const x of traceButtons.children)x.classList.toggle('active',x===b);$('trace-axis').textContent=traceMode==='pass'?'RELATIVE PRESSURE':'MODEL OVERPRESSURE · Pa';$('trace-caption').textContent=traceMode==='pass'?'FULL PASS · PLAYBACK SIGNAL':'BOOM DETAIL · 2 ms SHOCK EDGES';drawTrace()};
function resize(canvas,context){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);context.setTransform(d,0,0,d,0,0);return {w:r.width,h:r.height}}
new ResizeObserver(()=>{sceneSize=resize(scene,sc);traceSize=resize(trace,tc);draw()}).observe(document.querySelector('.simulation'));
new ResizeObserver(()=>{traceSize=resize(trace,tc);drawTrace()}).observe(document.querySelector('.trace-wrap'));
function stopSound(){if(source){source.onended=null;try{source.stop()}catch{}source.disconnect();source=null}}
function audioClock(){const stamp=ctx?.getOutputTimestamp?.();return stamp?.contextTime>0?stamp.contextTime+Math.max(0,performance.now()-stamp.performanceTime)/1000:ctx.currentTime}
function now(){return playing&&ctx?clamp(bounds.start+offset+audioClock()-audioStart,bounds.start,bounds.end):t}
function pause(){t=now();playing=false;playIntent=false;stopSound();updateUI()}
function showAudioError(message){building=false;playIntent=false;$('play').disabled=false;$('play-label').textContent='Retry audio';$('event-explanation').textContent=message}
function requestRender(){
 building=true;audioData=null;audioBuffer=null;const id=++job;clearTimeout(renderTimer);$('play-label').textContent=playIntent?'Preparing sound…':'Preparing sound…';$('play').disabled=false;
 renderTimer=setTimeout(()=>{
  try{if(!worker){worker=new Worker(new URL('./audio-worker.mjs',import.meta.url),{type:'module'});worker.onmessage=({data})=>{if(data.id!==job)return;if(data.error){showAudioError('Audio generation failed. Use Listen to try again.');return}audioData=data;building=false;audioBuffer=null;updateUI();drawTrace();if(playIntent)startPlayback()};worker.onerror=()=>{worker?.terminate();worker=null;showAudioError('The audio engine could not start. Use Listen to retry.')}}worker.postMessage({id,config})}
  catch{showAudioError('This browser could not load the audio engine. Open this site in a current browser.')}
 },100);
}
async function ensureContext(){
 if(!ctx){const Audio=window.AudioContext||window.webkitAudioContext;if(!Audio)throw Error('Web Audio is unavailable');ctx=new Audio();gain=ctx.createGain();gain.gain.value=muted?0:Number($('volume').value)/100;gain.connect(ctx.destination)}
 await ctx.resume();if(ctx.state!=='running')throw Error('Audio is paused by the browser');
}
async function startPlayback(){
 try{await ensureContext();if(building){playIntent=true;$('play-label').textContent='Preparing sound…';return}if(!audioData){requestRender();playIntent=true;return}
 stopSound();if(pristine||t>=bounds.end-.02)t=bounds.start;pristine=false;
 if(!audioBuffer){audioBuffer=ctx.createBuffer(1,audioData.samples.length,audioData.sampleRate);audioBuffer.copyToChannel(audioData.samples,0)}
 source=ctx.createBufferSource();source.buffer=audioBuffer;source.connect(gain);offset=clamp(t-bounds.start,0,audioBuffer.duration-.001);audioStart=ctx.currentTime+.03;playing=true;playIntent=false;source.start(audioStart,offset);source.onended=()=>{t=bounds.end;playing=false;source=null;updateUI();draw()};updateUI();
 }catch{showAudioError('Audio could not start. Press Listen again to enable browser audio.')}
}
$('play').onclick=()=>{if(playing||playIntent)pause();else startPlayback()};
$('restart').onclick=()=>{const resume=playing;pause();pristine=false;t=bounds.start;updateUI();draw();if(resume)startPlayback()};
$('scrub').oninput=()=>{pause();pristine=false;t=bounds.start+Number($('scrub').value)/1000*(bounds.end-bounds.start);updateUI();draw()};
$('volume').oninput=()=>{muted=false;if(gain)gain.gain.setTargetAtTime(Number($('volume').value)/100,ctx.currentTime,.015);updateVolume()};
$('mute').onclick=()=>{muted=!muted;if(gain)gain.gain.setTargetAtTime(muted?0:Number($('volume').value)/100,ctx.currentTime,.015);updateVolume()};
function updateVolume(){$('mute').textContent=muted?'⊘':'♫';$('mute').setAttribute('aria-label',muted?'Unmute sound':'Mute sound');$('mute').title=muted?'Unmute sound':'Mute sound'}
function configure(change){pause();config={...config,...change};bounds=timeWindow(config.mach,config.altitude);const arrival=boomTime(config.mach,config.altitude);t=arrival===null?(config.mach===1?.3:-.7):Math.min(arrival*.65,1.5);pristine=true;syncControls();requestRender();draw()}
function syncControls(){
 const {mach:m,altitude:h}=config,angle=coneAngle(m),arrival=boomTime(m,h);
 $('mach').value=m;$('altitude').value=h;$('sound-mode').value=config.mode;$('pressure').value=config.pressure;$('duration').value=config.duration*1000;
 $('mach-value').textContent=m.toFixed(2);$('altitude-value').textContent=h.toLocaleString()+' m';$('regime').textContent=regime(m);$('ground-speed').innerHTML=Math.round(m*C*3.6).toLocaleString()+' <small>km/h</small>';
 $('cone-angle').textContent=angle===null?'No finite cone':(angle*180/Math.PI).toFixed(1)+'°';$('arrival-label').textContent=arrival===null?'Overhead sound delay':'Boom after overhead';$('arrival-delay').innerHTML=(arrival??h/C).toFixed(2)+' <small>s</small>';
 $('pressure-value').textContent=config.pressure+' Pa';$('duration-value').textContent=Math.round(config.duration*1000)+' ms';
 $('start-time').textContent=bounds.start.toFixed(1)+' s';$('end-time').textContent='+'+bounds.end.toFixed(1)+' s';
 document.querySelectorAll('[data-mach]').forEach(b=>b.classList.toggle('active',Math.abs(Number(b.dataset.mach)-m)<.001));
 $('insight-title').textContent=m>1?'Faster than its own sound.':m===1?'At the speed of sound.':'An audible shift in pitch.';
 $('insight-copy').textContent=m>1?'Above Mach 1, the rings pile up along a cone. You hear the boom when that cone sweeps over you.':m===1?'The rings bunch up ahead of the source. A finite-angle Mach cone needs a speed strictly above Mach 1.':'Ahead of the jet, waves bunch together and the pitch rises. Behind it, waves spread apart and the pitch falls.';
 updateUI();
}
$('mach').oninput=()=>configure({mach:Number($('mach').value)});$('altitude').oninput=()=>configure({altitude:Number($('altitude').value)});$('sound-mode').onchange=()=>configure({mode:$('sound-mode').value});$('pressure').oninput=()=>configure({pressure:Number($('pressure').value)});$('duration').oninput=()=>configure({duration:Number($('duration').value)/1000});document.querySelectorAll('[data-mach]').forEach(b=>b.onclick=()=>configure({mach:Number(b.dataset.mach)}));
const dialog=$('model-dialog');for(const id of ['about-button','model-button'])$(id).onclick=()=>dialog.showModal();$('close-dialog').onclick=()=>dialog.close();dialog.onclick=e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close()}};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)pause()});
function updateUI(){
 t=now();const {mach:m,altitude:h}=config,arrival=boomTime(m,h),roots=retardedTimes(t,m,h),hasSound=roots.length>0;
 $('scene-clock').textContent='T '+(t<0?'−':'+')+Math.abs(t).toFixed(2)+' s';$('scrub').value=(t-bounds.start)/(bounds.end-bounds.start)*1000;
 $('play-icon').textContent=playing?'Ⅱ':'▶';$('play-label').textContent=building?'Preparing sound…':playing?'Pause flyby':t>=bounds.end-.02?'Replay flyby':pristine?'Listen to flyby':'Resume flyby';
 const inBoom=arrival!==null&&t>=arrival&&t<=arrival+config.duration;
 $('listener-state').textContent=config.mode==='boom'?(inBoom?'SHOCK AT LISTENER':arrival===null?'NO SONIC BOOM':t<arrival?'WAITING FOR SHOCK':'SHOCK HAS PASSED'):inBoom?'SHOCK AT LISTENER':hasSound?'SOUND ARRIVING':'WAITING FOR SOUND';
 const ratio=hasSound?doppler(roots.at(-1),m,h):null;$('pitch-value').textContent=config.mode==='boom'?'N-WAVE ONLY':ratio&&Number.isFinite(ratio)?Math.round(200*ratio).toLocaleString()+' Hz'+(config.mode==='jet'?' ref.':''):'— Hz';
 $('scene-kicker').textContent=regime(m)+' FLIGHT';
 $('scene-status').textContent=arrival!==null?(t<0?'The jet outruns its sound.':t<arrival?'Overhead. Still no sound.':inBoom?'The shock front reaches you.':'The boom passes. Sound follows.'):m===1?(t<=0?'Sound gathers ahead of the jet.':'Earlier sound is reaching you.'):(t<h/C?'Hear the approaching jet.':'Hear the pitch fall away.');
 $('event-explanation').textContent=arrival!==null?(t<arrival?'Shock arrival in '+(arrival-t).toFixed(2)+' s · '+arrival.toFixed(2)+' s after overhead.':'The cone passed at T +'+arrival.toFixed(2)+' s. The two shocks are '+Math.round(config.duration*1000)+' ms apart.'):m===1?'Sonic limit: no added N-wave. Real transonic shocks require an aircraft flow model.':config.mode==='boom'?'No sonic boom below Mach 1. Select the jet or tone to hear this pass.':'The sound emitted directly overhead arrives '+(h/C).toFixed(2)+' s later.';
}
function line(context,x1,y1,x2,y2,color,width=1,dash=[]){context.beginPath();context.strokeStyle=color;context.lineWidth=width;context.setLineDash(dash);context.moveTo(x1,y1);context.lineTo(x2,y2);context.stroke();context.setLineDash([])}
function label(context,text,x,y,color='#819aa6',align='left',size=11){context.font=`${size}px "DM Sans", sans-serif`;context.textAlign=align;context.fillStyle=color;context.fillText(text,x,y)}
function drawScene(){
 const {w,h}=sceneSize;if(!w||!h)return;const {mach:m,altitude:alt}=config;
 sc.clearRect(0,0,w,h);const ground=h*.77,scale=Math.min((ground-110)/alt,w/(2*alt*Math.max(2.1,m*1.25))),ox=w*.47,py=ground-alt*scale,px=ox+m*C*t*scale;
 // Both horizontal and vertical distances use the same scale.
 const gridStep=[50,100,200,500,1000,2000,5000].find(s=>s*scale>=52)||5000,grid=gridStep*scale;
 for(let x=ox%grid;x<w;x+=grid)line(sc,x,0,x,ground,'#192c36');for(let y=ground%grid;y<ground;y+=grid)line(sc,0,y,w,y,'#192c36');
 sc.save();sc.beginPath();sc.rect(0,0,w,ground);sc.clip();
 const interval=alt/C/5,history=Math.max(14*alt/C,Math.abs(t)+5);let count=0;
 for(let tau=Math.floor(t/interval)*interval;tau>t-history&&count<110;tau-=interval,count++){
  const radius=C*(t-tau)*scale,cx=ox+m*C*tau*scale;if(radius<2)continue;
  sc.beginPath();sc.arc(cx,py,radius,0,Math.PI*2);sc.strokeStyle=`rgba(91,187,207,${.12+.24*Math.exp(-(t-tau)/(4*alt/C))})`;sc.lineWidth=1;sc.stroke();
 }
 if(m>1){const slope=1/Math.sqrt(m*m-1),left=-w*3;
  sc.beginPath();sc.moveTo(px,py);sc.lineTo(left,py+(px-left)*slope);sc.lineTo(left,py-(px-left)*slope);sc.closePath();sc.fillStyle='#c2f88109';sc.fill();
  line(sc,px,py,left,py+(px-left)*slope,'#c2f881aa',1.5);line(sc,px,py,left,py-(px-left)*slope,'#c2f88166',1.5);
  if(px>50&&px<w-70){sc.beginPath();sc.strokeStyle='#b0db7855';sc.arc(px,py,65,Math.PI-Math.asin(1/m),Math.PI);sc.stroke();label(sc,(coneAngle(m)*180/Math.PI).toFixed(1)+'°',px-80,py+18,'#c2f881')}
 }
 const roots=retardedTimes(t,m,alt);if(roots.length&&config.mode!=='boom'){
  const tau=roots.at(-1),ex=ox+m*C*tau*scale;line(sc,ex,py,ox,ground,'#65c8d580',1,[4,5]);sc.beginPath();sc.arc(ex,py,3,0,Math.PI*2);sc.fillStyle='#65c8d5';sc.fill();
 }
 line(sc,0,py,w,py,'#56727b66',1,[3,7]);sc.restore();
 sc.fillStyle='#0c171d';sc.fillRect(0,ground,w,h-ground);line(sc,0,ground,w,ground,'#51656c');
 for(let x=ox%grid;x<w;x+=grid){line(sc,x,ground,x,ground+5,'#51656c');if(Math.abs(x-ox)>grid*.3)label(sc,((x-ox)/scale/1000).toFixed(1)+' km',x,ground+21,'#657e8c','center',10)}
 // Functional aircraft marker, enlarged for legibility.
 const visibleX=clamp(px,24,w-24);sc.save();sc.translate(visibleX,py);sc.beginPath();sc.moveTo(20,0);sc.lineTo(1,-4);sc.lineTo(-10,-14);sc.lineTo(-7,-3);sc.lineTo(-18,-5);sc.lineTo(-15,0);sc.lineTo(-18,5);sc.lineTo(-7,3);sc.lineTo(-10,14);sc.lineTo(1,4);sc.closePath();sc.fillStyle='#edf5f1';sc.fill();sc.restore();
 if(px>20&&px<w-20){label(sc,'M '+m.toFixed(2),visibleX,py-28,'#d7e6e9','center');line(sc,visibleX-35,py,visibleX-23,py,'#f3b786',2)}else label(sc,(px<24?'← ':'')+'JET '+Math.abs(m*C*t/1000).toFixed(1)+' km'+(px>w-24?' →':''),visibleX,py-25,'#c6d9de',px<24?'left':'right');
 const arrival=boomTime(m,alt),flash=arrival===null?0:Math.max(0,1-Math.abs(t-arrival)/.35);
 if(flash>0){sc.beginPath();sc.arc(ox,ground-8,18+25*(1-flash),0,Math.PI*2);sc.strokeStyle=`rgba(194,248,129,${flash})`;sc.lineWidth=2;sc.stroke()}
 sc.fillStyle=flash>.1?'#c2f881':'#e6eeeb';sc.beginPath();sc.arc(ox,ground-18,4,0,Math.PI*2);sc.fill();line(sc,ox,ground-13,ox,ground-3,'#e6eeeb',2);line(sc,ox-4,ground,ox,ground-5,'#e6eeeb',2);line(sc,ox+4,ground,ox,ground-5,'#e6eeeb',2);line(sc,ox-5,ground-9,ox+5,ground-9,'#e6eeeb',1.5);
 sc.fillStyle='#0c171d';sc.fillRect(ox-60,ground+10,120,18);label(sc,'YOU · GROUND LISTENER',ox,ground+23,'#c3d0d3','center',10);
 const bx=w-24;line(sc,bx,py,bx,ground,'#7b969d66',1,[2,4]);line(sc,bx-4,py,bx+4,py,'#7b969d');line(sc,bx-4,ground,bx+4,ground,'#7b969d');sc.save();sc.translate(bx-8,(py+ground)/2);sc.rotate(-Math.PI/2);label(sc,alt.toLocaleString()+' m',0,0,'#96aeb8','center',10);sc.restore();
 label(sc,'1 DIV = '+(gridStep>=1000?gridStep/1000+' km':gridStep+' m'),18,ground-12,'#708b97','left',9);
}
function drawTrace(){
 const {w,h}=traceSize;if(!w||!h)return;tc.clearRect(0,0,w,h);const top=23,bottom=h-23,mid=(top+bottom)/2,arrival=boomTime(config.mach,config.altitude);let start=bounds.start,end=bounds.end;
 if(traceMode==='nwave'){if(arrival===null){line(tc,0,mid,w,mid,'#39505b');label(tc,'No N-wave at or below Mach 1.',w/2,mid-12,'#a5b7c0','center',14);return}start=arrival-.045;end=arrival+config.duration+.045}
 const map=time=>(time-start)/(end-start)*w;
 for(let i=0;i<=8;i++)line(tc,w*i/8,top,w*i/8,bottom,'#21333d');line(tc,0,mid,w,mid,'#304750');
 if(traceMode==='nwave'){
  tc.beginPath();tc.strokeStyle='#c2f881';tc.lineWidth=2;for(let x=0;x<=w;x++){const y=mid-nWave(start+x/w*(end-start),arrival,config.pressure,config.duration)/config.pressure*(bottom-top)*.42;if(x===0)tc.moveTo(x,y);else tc.lineTo(x,y)}tc.stroke();
  label(tc,'+'+config.pressure,6,top+11,'#9aad9b','left',10);label(tc,'−'+config.pressure,6,bottom,'#9aad9b','left',10);label(tc,'BOW SHOCK',map(arrival)+6,h-3,'#b6ce9d','left',9);label(tc,'TAIL SHOCK',map(arrival+config.duration)-6,h-3,'#b6ce9d','right',9);
 }else if(audioData){
  const samples=audioData.samples;tc.strokeStyle='#74c4cf';tc.lineWidth=1;const gainScale=(bottom-top)*.45/Math.max(.08,audioData.peak);
  tc.beginPath();for(let x=0;x<w;x++){const a=Math.floor(x/w*samples.length),b=Math.floor((x+1)/w*samples.length);let lo=0,hi=0;for(let i=a;i<b;i++){lo=Math.min(lo,samples[i]);hi=Math.max(hi,samples[i])}tc.moveTo(x,mid-hi*gainScale);tc.lineTo(x,mid-lo*gainScale)}tc.stroke();
 }else{label(tc,'Calculating arriving sound…',w/2,mid-12,'#819aa6','center',12)}
 if(traceMode==='pass'){
  const over=map(0);line(tc,over,top,over,bottom,'#82929b',1,[3,4]);label(tc,'OVERHEAD',over+5,h-4,'#82929b','left',9);
  if(arrival!==null){const x=map(arrival);line(tc,x,top,x,bottom,'#c2f88166',1,[3,4]);label(tc,'BOOM',x+6,top+9,'#c2f881','left',9)}
 }
 const playX=map(t);if(playX>=0&&playX<=w){line(tc,playX,top-4,playX,bottom+3,'#f0f4f2',1.4);tc.beginPath();tc.moveTo(playX-3,top-8);tc.lineTo(playX+3,top-8);tc.lineTo(playX,top-3);tc.closePath();tc.fillStyle='#f0f4f2';tc.fill()}
}
function draw(){drawScene();drawTrace()}
function frame(time){if(playing){t=now();drawScene();drawTrace();if(time-lastUi>70){updateUI();lastUi=time}}requestAnimationFrame(frame)}
// Optional agent integration uses the same validation and state as the visible controls.
const modelContext=document.modelContext;
if(modelContext?.registerTool){const lifecycle=new AbortController();const register=tool=>{try{Promise.resolve(modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{})}catch{}};
 register({name:'read_flyby',description:'Read the current flyby settings, simulation time, and acoustic geometry.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:false},execute:()=>({...config,time:now(),playing,boomArrival:boomTime(config.mach,config.altitude),coneHalfAngleDegrees:coneAngle(config.mach)===null?null:coneAngle(config.mach)*180/Math.PI})});
 register({name:'configure_flyby',description:'Set Mach number, altitude, or sound source. Stops playback and prepares the selected pass; does not start audio.',inputSchema:{type:'object',properties:{mach:{type:'number',minimum:.3,maximum:8},altitude:{type:'number',minimum:100,maximum:2000},mode:{type:'string',enum:['jet','tone','boom']}},additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute:input=>{if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).some(k=>!['mach','altitude','mode'].includes(k)))throw Error('Invalid flyby settings');if(input.mach!==undefined&&(!Number.isFinite(input.mach)||input.mach<.3||input.mach>8))throw Error('Mach must be 0.3–8');if(input.altitude!==undefined&&(!Number.isFinite(input.altitude)||input.altitude<100||input.altitude>2000))throw Error('Altitude must be 100–2000 m');if(input.mode!==undefined&&!['jet','tone','boom'].includes(input.mode))throw Error('Unknown sound source');configure(input);return {...config,audio:'preparing'}}});
 window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});}
syncControls();requestRender();requestAnimationFrame(frame);

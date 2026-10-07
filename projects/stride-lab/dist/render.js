import {G,clamp} from './physics.js';
const colors={lime:'#c3f460',dim:'#638039',blue:'#76bbd0',orange:'#ffac6e',grid:'#263541',text:'#a4b0ba'};
function fit(canvas){const r=canvas.getBoundingClientRect(),d=Math.min(window.devicePixelRatio||1,2);if(canvas.width!==Math.round(r.width*d)||canvas.height!==Math.round(r.height*d)){canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);}const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,r.width,r.height);return {c,w:r.width,h:r.height};}
function line(c,points,color,width=2,dash=[]){c.strokeStyle=color;c.lineWidth=width;c.lineCap='round';c.lineJoin='round';c.setLineDash(dash);c.beginPath();points.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.stroke();c.setLineDash([]);}
function dot(c,x,y,r,color){c.fillStyle=color;c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill();}
function arrow(c,a,b,color){line(c,[a,b],color,1.6);const ang=Math.atan2(b[1]-a[1],b[0]-a[0]);line(c,[[b[0]-7*Math.cos(ang-.45),b[1]-7*Math.sin(ang-.45)],b,[b[0]-7*Math.cos(ang+.45),b[1]-7*Math.sin(ang+.45)]],color,1.6);}
function knee(hip,foot,sign=1){const dx=foot[0]-hip[0],dy=foot[1]-hip[1],d=Math.hypot(dx,dy),len=Math.max(.555,d/2+.001),off=Math.sqrt(Math.max(0,len*len-d*d/4));return [(hip[0]+foot[0])/2-sign*dy/d*off,(hip[1]+foot[1])/2+sign*dx/d*off];}
export function drawTrack(canvas,trial,index=0,{forces=true}={}){
 const {c,w,h}=fit(canvas),f=trial?.frames?.[index];
 const scale=Math.min(h*.285,w*.15),ground=h*.79,anchor=w<520?w*.34:w*.38;
 const x=f?.x??0,origin=x-2;
 const X=v=>anchor+(v-x)*scale,Y=v=>ground-v*scale;
 c.fillStyle='#141c23';c.fillRect(0,0,w,h);
 for(let n=Math.floor(origin-5);n<origin+w/scale+6;n++){
  line(c,[[X(n),0],[X(n),ground]],colors.grid,.65);
  if(n>=0&&n%2===0){c.font='10px "IBM Plex Mono",monospace';c.fillStyle='#768995';c.textAlign='center';c.fillText(`${n} m`,X(n),ground+28);}
 }
 for(let yy=.5;yy<4;yy+=.5)line(c,[[0,Y(yy)],[w,Y(yy)]],colors.grid,.65);
 c.fillStyle='#1d2a32';c.fillRect(0,ground,w,h-ground);line(c,[[0,ground],[w,ground]],'#70848d',1.5);
 line(c,[[0,ground+14],[w,ground+14]],'#2f404b',.7);
 // Distance marks and obstacle geometry are in simulated world coordinates.
 if(trial){
  for(let i=0;i<trial.hurdles.length;i++){
   const hx=X(trial.hurdles[i]),hy=Y(trial.env.height),passed=f&&f.x>trial.hurdles[i]+.13;
   if(hx< -30||hx>w+30)continue;
   const color=passed?'#76875b':colors.orange;
   line(c,[[hx-2,ground],[hx-2,hy],[hx+8,hy]],color,3);line(c,[[hx-17,ground],[hx+13,ground]],color,3);
   c.font='10px "IBM Plex Mono",monospace';c.fillStyle=color;c.textAlign='center';c.fillText(`${i+1} / ${trial.hurdles.length}`,hx,hy-13);
  }
  const finishX=X(trial.event==='sprint'?100:60);if(finishX<w+20){line(c,[[finishX,ground],[finishX,ground-180]],'#c8d3d8',1,[4,4]);c.fillStyle='#d8e1e6';c.font='11px sans-serif';c.fillText('FINISH',finishX,ground-190);}
 }
 if(!f){c.textAlign='center';c.fillStyle=colors.text;c.font='14px sans-serif';c.fillText('Run a baseline trial to inspect the stride',w/2,h/2);return;}
 const start=Math.max(0,index-140);const path=[];for(let i=start;i<=index;i+=2){const q=trial.frames[i];path.push([X(q.x),Y(q.y)]);}line(c,path,'#798e9e',1,[3,4]);
 const hip=[f.x,f.y-.04],lean=clamp(f.vx/85,-.03,.11),shoulder=[f.x+lean,f.y+.47],head=[f.x+lean+.025,f.y+.67];
 const phase=f.t*13;let feet=[];
 if(f.stance){feet[f.leg]=[f.foot,0];feet[1-f.leg]=[f.x+f.swing,Math.max(.12,f.y-.70+.15*Math.cos(phase))];}
 else if(f.jump){const low=f.y-f.legLength;feet=[[f.x+f.swing,low],[f.x-.30,low+.05]];}
 else{const off=clamp(f.vx*.07,.08,.5),flight=Math.min(1,(f.t-f.lastTakeoff)/.14);feet[f.leg]=[f.x+f.swing,Math.max(.02,f.y-f.legLength+.12*(1-flight))];feet[1-f.leg]=[f.x-.3,Math.max(.08,f.y-.65)];}
 const P=p=>[X(p[0]),Y(p[1])];
 for(const i of [1,0]){
  const color=i===0?colors.lime:'#688244',k=knee(hip,feet[i],1);
  line(c,[P(hip),P(k),P(feet[i])],color,i===0?5:4);
  line(c,[P([feet[i][0]-.04,feet[i][1]]),P([feet[i][0]+.15,feet[i][1]])],color,4);
  dot(c,...P(k),3,'#142019');
 }
 const armSwing=f.jump?-.45:Math.sin(phase)*.35;
 for(const sign of [-1,1]){
  const elbow=[shoulder[0]+sign*armSwing,shoulder[1]-.24],hand=[elbow[0]+.23,elbow[1]+.08];
  line(c,[P(shoulder),P(elbow),P(hand)],sign===1?colors.lime:'#688244',4);
 }
 line(c,[P(hip),P(shoulder),P([head[0],head[1]-.09])],colors.lime,7);dot(c,...P(head),scale*.11,colors.lime);
 dot(c,X(f.x),Y(f.y),4,'#f0ffd3');dot(c,X(f.x),Y(f.y),2,'#19251b');
 if(forces){
  if(f.fy>0){const from=[X(f.foot),ground];arrow(c,from,[from[0]+f.fx/(trial.env.mass*G)*28,from[1]-f.fy/(trial.env.mass*G)*28],colors.orange);}
  arrow(c,[X(f.x),Y(f.y)],[X(f.x),Y(f.y)+35],colors.blue);
  if(f.vx>.1){arrow(c,[X(f.x)+20,Y(f.y)-12],[X(f.x)+20+f.vx*5,Y(f.y)-12],colors.lime);}
 }
 c.font='10px "IBM Plex Mono",monospace';c.fillStyle='#b7c7ce';c.textAlign='left';c.fillText(`${f.y.toFixed(2)} m`,X(f.x)+16,Y(f.y)+19);
 c.font='10px "IBM Plex Mono",monospace';c.fillStyle='#708590';c.textAlign='right';c.fillText('Limb pose is illustrative',w-17,ground+45);
}
export function drawTelemetry(canvas,trial,index=0,type='force'){
 const {c,w,h}=fit(canvas),left=37,right=10,top=8,bottom=21;
 if(!trial?.frames?.length){c.strokeStyle=colors.grid;for(let i=0;i<3;i++)line(c,[[left,top+i*28],[w,top+i*28]],colors.grid,1);return;}
 const frames=trial.frames,selected=frames[index],end=trial.duration;
 const defs=type==='force'?[{key:'fy',color:colors.orange,scale:1/(trial.env.mass*G)}]:type==='height'?[{key:'y',color:colors.lime,scale:1}]:[{key:'kinetic',color:colors.lime,scale:.001},{key:'potential',color:colors.blue,scale:.001},{key:'elastic',color:colors.orange,scale:.001}];
 let max=type==='force'?6:type==='height'?Math.ceil(trial.peakHeight*2)/2:Math.ceil(Math.max(...frames.map(f=>Math.max(f.kinetic,f.potential,f.elastic)))/500)*.5;
 max=Math.max(max,.5);const X=t=>left+t/end*(w-left-right),Y=v=>h-bottom-v/max*(h-top-bottom);
 c.font='9px "IBM Plex Mono",monospace';c.textAlign='right';
 for(let i=0;i<=3;i++){const v=max*i/3,y=Y(v);line(c,[[left,y],[w-right,y]],colors.grid,.8);c.fillStyle='#83949f';c.fillText(v.toFixed(type==='force'?0:1),left-8,y+3);}
 const stride=Math.max(1,Math.floor(frames.length/(w*1.4)));
 for(const d of defs){let pts=[];for(let i=0;i<frames.length;i+=stride)pts.push([X(frames[i].t),Y(frames[i][d.key]*d.scale)]);line(c,pts,d.color,1.3);}
 line(c,[[X(selected.t),top],[X(selected.t),h-bottom]],'#d4e0e6',1,[2,3]);
 for(const d of defs)dot(c,X(selected.t),Y(selected[d.key]*d.scale),3,d.color);
 c.fillStyle='#83949f';c.textAlign='center';for(let i=0;i<=4;i++)c.fillText(`${(end*i/4).toFixed(1)}s`,X(end*i/4),h-5);
}
export function drawLearning(canvas,history=[]){
 const {c,w,h}=fit(canvas);for(let i=1;i<=3;i++)line(c,[[0,i*h/4],[w,i*h/4]],colors.grid,.7);
 if(history.length<1)return;
 const lo=Math.min(...history.map(p=>p.mean))-10,hi=Math.max(...history.map(p=>p.score))+10;
 const xy=(v,i)=>[3+i/Math.max(1,history.length-1)*(w-6),h-6-(v-lo)/(hi-lo)*(h-12)];
 line(c,history.map((p,i)=>xy(p.mean,i)),'#586977',1);
 line(c,history.map((p,i)=>xy(p.score,i)),colors.lime,1.8);
 dot(c,...xy(history.at(-1).score,history.length-1),3,colors.lime);
}

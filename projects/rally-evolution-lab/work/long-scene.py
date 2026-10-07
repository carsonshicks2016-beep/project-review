from pathlib import Path
p=Path('dist/engine.mjs');s=p.read_text().replace(' const height=(x,z)=>elevation*(', ' const heightScale=lengthKm===null?1:Math.max(1,lengthKm/3);\n const height=(worldX,worldZ)=>{const x=worldX/heightScale,z=worldZ/heightScale;return elevation*(').replace('Math.sin((x+z)*.054));','Math.sin((x+z)*.054));};').replace('return {seed,elevation,lengthKm,height,','return {seed,elevation,lengthKm,heightScale,height,');p.write_text(s)
p=Path('dist/scene.mjs');s=p.read_text().replace(' let world=new THREE.Group(),carModels=[];', ' let world=new THREE.Group(),carModels=[];\n const terrainWidth=track.terrainWidth,terrainDepth=track.terrainDepth;')
s=s.replace('new THREE.PlaneGeometry(405,348,90,78)', 'new THREE.PlaneGeometry(terrainWidth,terrainDepth,track.lengthKm===null?90:Math.min(360,Math.ceil(terrainWidth/(3.5*track.heightScale))),track.lengthKm===null?78:Math.min(320,Math.ceil(terrainDepth/(3.5*track.heightScale))))')
a=s.index(' const rim=[];');b=s.index('let vertices=[];',a)
s=s[:a]+''' const rim=[];
 for(let i=0;i<=90;i++)rim.push([-terrainWidth/2+i/90*terrainWidth,-terrainDepth/2]);
 for(let i=0;i<=80;i++)rim.push([terrainWidth/2,-terrainDepth/2+i/80*terrainDepth]);
 for(let i=0;i<=90;i++)rim.push([terrainWidth/2-i/90*terrainWidth,terrainDepth/2]);
 for(let i=0;i<=80;i++)rim.push([-terrainWidth/2,terrainDepth/2-i/80*terrainDepth]);
 '''+s[b:]
s=s.replace('new THREE.PlaneGeometry(2000,2000)', 'new THREE.PlaneGeometry(terrainWidth*5,terrainDepth*5)')
s=s.replace('for(let i=0;i<205;i++){let x=(random()-.5)*385,z=(random()-.5)*327;', 'for(let i=0;i<(track.lengthKm===null?205:Math.min(2200,Math.round(track.lengthKm*220)));i++){let x=(random()-.5)*(terrainWidth-20),z=(random()-.5)*(terrainDepth-21);')
p.write_text(s)

import { VoxelGrid } from '../src/collision/voxelGrid.js';

let ok = true;
const check = (cond, label, extra='') => { if(!cond) ok=false; console.log(`${cond?'PASS':'FAIL'} ${label} ${extra}`); };

// Build a synthetic scene: a solid wall slab at x in [10,12], spanning y,z.
const pts = [];
for (let y=0;y<20;y+=0.4) for (let z=-20;z<20;z+=0.4) for (let x=10;x<12;x+=0.4) pts.push(x,y,z);
// Plus a lone stray point that must NOT become geometry.
pts.push(-30, 5, 0);
const grid = VoxelGrid.fromPoints(Float32Array.from(pts), { cellSize: 1.0, minPoints: 2 });
console.log(`grid dims=${grid.dims} cells=${grid.totalCells} solid=${grid.solidCount} (${(100*grid.solidCount/grid.totalCells).toFixed(1)}%)`);

check(grid.isSolid(11, 10, 0), 'wall interior is solid');
check(!grid.isSolid(0, 10, 0), 'open space is empty');
check(!grid.isSolid(-30, 5, 0), 'lone stray point rejected by minPoints');

// Ray straight at the wall from x=0 should hit near x=10.
const d1 = grid.raycast(0, 10, 0, 1, 0, 0, 100);
check(Math.abs(d1 - 10) < 1.5, 'ray hits wall at expected distance', `got ${d1.toFixed(2)} expect ~10`);

// Ray pointing away must miss.
const d2 = grid.raycast(0, 10, 0, -1, 0, 0, 100);
check(d2 === 100, 'ray away from wall returns maxDist', `got ${d2}`);

// Ray parallel to the wall must miss and still terminate.
const d3 = grid.raycast(0, 10, 0, 0, 0, 1, 100);
check(d3 === 100, 'parallel ray misses', `got ${d3}`);

// Starting inside geometry reports zero.
check(grid.raycast(11, 10, 0, 1, 0, 0, 100) === 0, 'ray starting inside solid returns 0');

// Diagonal ray: hits the wall plane at x=10, so distance ~ 10*sqrt(2) travelling (1,0,1)/|..|
const inv = 1/Math.SQRT2;
const d4 = grid.raycast(0, 10, 0, inv, 0, inv, 100);
check(Math.abs(d4 - 10*Math.SQRT2) < 2.0, 'diagonal ray distance', `got ${d4.toFixed(2)} expect ~${(10*Math.SQRT2).toFixed(2)}`);

// maxDist is respected.
check(grid.raycast(0, 10, 0, 1, 0, 0, 5) === 5, 'maxDist truncates before the wall');

// Serialize round-trip must preserve occupancy and raycasts.
const round = VoxelGrid.deserialize(JSON.parse(JSON.stringify({
  origin: grid.origin, dims: grid.dims, cellSize: grid.cellSize, bits: Array.from(grid.bits)
}, (k,v)=>v), (k,v)=> k==='bits' ? new Uint8Array(v).buffer : v));
check(round.isSolid(11,10,0) && !round.isSolid(0,10,0), 'serialize round-trip preserves occupancy');
check(Math.abs(round.raycast(0,10,0,1,0,0,100) - d1) < 1e-9, 'round-trip preserves raycast');

// Scale + offset must be applied (grid should sit where the rendered cloud sits).
const g2 = VoxelGrid.fromPoints(Float32Array.from([0,0,0, 0.1,0,0, 0,0.1,0]), { scale: 40, offsetY: 100, cellSize: 2, minPoints: 1 });
check(g2.isSolid(0, 100, 0), 'scale + offsetY applied to occupancy');
check(!g2.isSolid(0, 0, 0), 'un-offset location is empty');

// Rays originating OUTSIDE the grid must enter it correctly. Build a tight grid with no
// stray point so the origin genuinely falls outside.
{
  const wall = [];
  for (let y=0;y<40;y+=0.4) for (let z=-30;z<30;z+=0.4) for (let x=10;x<12;x+=0.4) wall.push(x,y,z);
  const g = VoxelGrid.fromPoints(Float32Array.from(wall), { cellSize: 1.0, minPoints: 2 });
  const spanLo = g.origin[0], spanHi = g.origin[0] + g.dims[0]*g.cellSize;
  check(spanLo > 0, 'grid is tight around the wall (origin outside ray start)', `x span ${spanLo.toFixed(1)}..${spanHi.toFixed(1)}`);
  const hit = g.raycast(0, 20, 0, 1, 0, 0, 30);
  check(Math.abs(hit - 10) < 1.5, 'ray from outside the grid enters and hits', `got ${hit.toFixed(2)} expect ~10`);
  check(g.raycast(0, 20, 0, -1, 0, 0, 30) === 30, 'ray from outside pointing away misses');
  check(g.raycast(0, 200, 0, 1, 0, 0, 30) === 30, 'ray from outside above the grid misses');
  const far = g.raycast(-100, 20, 0, 1, 0, 0, 30);
  check(far === 30, 'ray from outside beyond maxDist truncates', `got ${far}`);
}

console.log(ok ? 'VOXEL CHECK PASSED' : 'VOXEL CHECK FAILED');
process.exit(ok?0:1);

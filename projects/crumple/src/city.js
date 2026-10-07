// Procedural city: a road grid with buildings, curbs and scattered barriers.
// Buildings become static AABB colliders; sidewalks are visual only so cars
// don't trip over a 15cm curb at 40 m/s.

export const PITCH = 60;      // block-to-block spacing
export const BLOCK_HALF = 22; // half-width of a buildable block
export const ROAD_HALF = (PITCH - BLOCK_HALF * 2) / 2;
export const GRID = 4;        // blocks extend -GRID..GRID

// Small deterministic PRNG so the same city comes back every run.
function mulberry(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const PALETTE = [0x6e7681, 0x596170, 0x7d8794, 0x4d5563, 0x848d99, 0x656f7d];

export function buildCity(world, seed = 20260719) {
  const rnd = mulberry(seed);
  const buildings = [];
  const sidewalks = [];
  const props = [];

  for (let i = -GRID; i <= GRID; i++) {
    for (let j = -GRID; j <= GRID; j++) {
      const cx = i * PITCH, cz = j * PITCH;

      sidewalks.push({
        cx, cz, y: 0.07,
        sx: BLOCK_HALF * 2 + 2.5, sz: BLOCK_HALF * 2 + 2.5,
        color: 0x2f3339,
      });

      // Leave the middle block open as a plaza to spawn and turn around in.
      if (i === 0 && j === 0) continue;

      const cols = rnd() < 0.45 ? 1 : 2;
      const rows = rnd() < 0.45 ? 1 : 2;
      const cellX = (BLOCK_HALF * 2) / cols;
      const cellZ = (BLOCK_HALF * 2) / rows;

      for (let a = 0; a < cols; a++) {
        for (let b = 0; b < rows; b++) {
          if (rnd() < 0.12) continue; // vacant lot
          const inset = 1.5 + rnd() * 2.5;
          const bx0 = cx - BLOCK_HALF + a * cellX + inset;
          const bx1 = cx - BLOCK_HALF + (a + 1) * cellX - inset;
          const bz0 = cz - BLOCK_HALF + b * cellZ + inset;
          const bz1 = cz - BLOCK_HALF + (b + 1) * cellZ - inset;
          if (bx1 - bx0 < 4 || bz1 - bz0 < 4) continue;

          const dist = Math.hypot(cx, cz);
          const tall = Math.max(0, 1 - dist / (PITCH * GRID * 1.15));
          const h = 7 + rnd() * 16 + tall * tall * 46;

          const min = { x: bx0, y: 0, z: bz0 };
          const max = { x: bx1, y: h, z: bz1 };
          world.addBox(min, max, 0.95);
          buildings.push({
            min, max,
            color: PALETTE[(rnd() * PALETTE.length) | 0],
            windowSeed: rnd(),
          });
        }
      }
    }
  }

  // Concrete barriers on some intersections — things to smash into.
  for (let n = 0; n < 60; n++) {
    const i = ((rnd() * (GRID * 2 + 1)) | 0) - GRID;
    const j = ((rnd() * (GRID * 2 + 1)) | 0) - GRID;
    const onX = rnd() < 0.5;
    const cx = i * PITCH + (onX ? (rnd() - 0.5) * 20 : PITCH / 2);
    const cz = j * PITCH + (onX ? PITCH / 2 : (rnd() - 0.5) * 20);
    const sx = onX ? 3.2 : 0.7;
    const sz = onX ? 0.7 : 3.2;
    const min = { x: cx - sx, y: 0, z: cz - sz };
    const max = { x: cx + sx, y: 1.05, z: cz + sz };
    world.addBox(min, max, 0.9);
    props.push({ min, max, color: 0xb9a24a });
  }

  return { buildings, sidewalks, props, seed };
}

export function randomRoadPoint(rndFn = Math.random) {
  const k = Math.round((rndFn() * 2 - 1) * GRID);
  const along = (rndFn() * 2 - 1) * PITCH * GRID;
  if (rndFn() < 0.5) {
    return { x: PITCH / 2 + k * PITCH, z: along, heading: rndFn() < 0.5 ? 0 : Math.PI };
  }
  return { x: along, z: PITCH / 2 + k * PITCH, heading: rndFn() < 0.5 ? Math.PI / 2 : -Math.PI / 2 };
}

export const WORLD_EXTENT = PITCH * GRID + BLOCK_HALF + 40;

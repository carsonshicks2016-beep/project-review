import { rngFor } from './rng'

/**
 * Procedural terrain v0 — a seeded landscape for Ashvale to live on.
 *
 * Everything derives from the world seed through labeled rng streams, so the
 * same seed always produces the same coastline, rivers, forests, and town
 * site. Terrain is canonical sim state: pathfinding, construction materials,
 * and claims all read from it. The renderer only draws it.
 */

export const TERRAIN_W = 96
export const TERRAIN_H = 64
export const SEA_LEVEL = 0.34

export type BiomeId =
  | 'ocean'
  | 'lake'
  | 'river'
  | 'beach'
  | 'marsh'
  | 'clay_flats'
  | 'grass'
  | 'meadow'
  | 'forest'
  | 'dense_forest'
  | 'hills'
  | 'mountain'
  | 'snow'

export interface Landmark {
  kind: 'sea' | 'lake' | 'mountain' | 'forest'
  name: string
  x: number
  y: number
}

export interface Terrain {
  width: number
  height: number
  /** 0..1 per tile, row-major. */
  elevation: number[]
  /** 0..1 per tile. */
  moisture: number[]
  biome: BiomeId[]
  /** Gatherable construction materials per tile. */
  timber: number[]
  stone: number[]
  clay: number[]
  reeds: number[]
  landmarks: Landmark[]
  townCenter: { x: number; y: number }
}

export const idx = (x: number, y: number): number => y * TERRAIN_W + x
export const inBounds = (x: number, y: number): boolean =>
  x >= 0 && y >= 0 && x < TERRAIN_W && y < TERRAIN_H

const WATER: BiomeId[] = ['ocean', 'lake', 'river']
export const isWater = (b: BiomeId): boolean => WATER.includes(b)
export const isWalkable = (b: BiomeId): boolean => b !== 'ocean' && b !== 'lake'
const BUILDABLE: BiomeId[] = ['grass', 'meadow', 'clay_flats', 'beach', 'forest']
export const isBuildable = (b: BiomeId): boolean => BUILDABLE.includes(b)

/** Walking cost per tile kind; rivers are fordable but slow, peaks punishing. */
export function moveCost(b: BiomeId): number {
  switch (b) {
    case 'ocean':
    case 'lake':
      return Infinity
    case 'river':
      return 6
    case 'marsh':
      return 3
    case 'forest':
      return 2.2
    case 'dense_forest':
      return 4
    case 'hills':
      return 2.5
    case 'mountain':
      return 8
    case 'snow':
      return 10
    case 'beach':
      return 1.3
    default:
      return 1
  }
}

// ---------------------------------------------------------------------------
// Seeded value noise

const smoothstep = (t: number): number => t * t * (3 - 2 * t)

class Noise {
  private cache = new Map<string, number>()
  constructor(
    private seed: string,
    private channel: string,
  ) {}

  private lattice(gx: number, gy: number): number {
    const key = `${gx}:${gy}`
    let v = this.cache.get(key)
    if (v === undefined) {
      v = rngFor(this.seed, 'terrain', this.channel, gx, gy).next()
      this.cache.set(key, v)
    }
    return v
  }

  /** Bilinear value noise at one spatial scale. */
  at(x: number, y: number, scale: number): number {
    const fx = x / scale
    const fy = y / scale
    const x0 = Math.floor(fx)
    const y0 = Math.floor(fy)
    const tx = smoothstep(fx - x0)
    const ty = smoothstep(fy - y0)
    const a = this.lattice(x0, y0)
    const b = this.lattice(x0 + 1, y0)
    const c = this.lattice(x0, y0 + 1)
    const d = this.lattice(x0 + 1, y0 + 1)
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
  }

  /** Four-octave fractal noise, 0..1. */
  fbm(x: number, y: number): number {
    return (
      this.at(x, y, 28) * 0.5 +
      this.at(x, y, 14) * 0.25 +
      this.at(x, y, 7) * 0.15 +
      this.at(x, y, 3.5) * 0.1
    )
  }
}

// ---------------------------------------------------------------------------
// Generation

export function generateTerrain(seed: string): Terrain {
  const size = TERRAIN_W * TERRAIN_H
  const elevNoise = new Noise(seed, 'elev')
  const moistNoise = new Noise(seed, 'moist')

  // Elevation: noise + a western coastal shelf + a gentle eastward rise.
  const elevation = new Array<number>(size)
  for (let y = 0; y < TERRAIN_H; y++) {
    for (let x = 0; x < TERRAIN_W; x++) {
      const base = elevNoise.fbm(x, y)
      const coast = Math.min(1, x / (TERRAIN_W * 0.3))
      const eastRise = x / TERRAIN_W
      elevation[idx(x, y)] = base * 0.58 + smoothstep(coast) * 0.27 + eastRise * 0.15
    }
  }

  // Water bodies: below sea level. Edge-connected water is ocean, the rest lakes.
  const water = elevation.map((e) => e < SEA_LEVEL)
  const ocean = new Array<boolean>(size).fill(false)
  const queue: number[] = []
  for (let y = 0; y < TERRAIN_H; y++) {
    for (const x of [0, TERRAIN_W - 1]) {
      const i = idx(x, y)
      if (water[i] && !ocean[i]) {
        ocean[i] = true
        queue.push(i)
      }
    }
  }
  for (let x = 0; x < TERRAIN_W; x++) {
    for (const y of [0, TERRAIN_H - 1]) {
      const i = idx(x, y)
      if (water[i] && !ocean[i]) {
        ocean[i] = true
        queue.push(i)
      }
    }
  }
  while (queue.length > 0) {
    const i = queue.pop()!
    const x = i % TERRAIN_W
    const y = Math.floor(i / TERRAIN_W)
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = x + dx
      const ny = y + dy
      if (!inBounds(nx, ny)) continue
      const ni = idx(nx, ny)
      if (water[ni] && !ocean[ni]) {
        ocean[ni] = true
        queue.push(ni)
      }
    }
  }

  const biome = new Array<BiomeId>(size).fill('grass')
  for (let i = 0; i < size; i++) {
    if (water[i]) biome[i] = ocean[i] ? 'ocean' : 'lake'
  }

  // Guarantee at least one inland lake: carve a small one at the wettest
  // far-from-ocean spot if noise produced none.
  if (!biome.some((b) => b === 'lake')) {
    let best = -1
    let bestScore = -Infinity
    for (let y = 6; y < TERRAIN_H - 6; y++) {
      for (let x = Math.floor(TERRAIN_W * 0.45); x < TERRAIN_W - 8; x++) {
        const i = idx(x, y)
        if (water[i]) continue
        const score = moistNoise.fbm(x, y) - elevation[i] * 0.5
        if (score > bestScore) {
          bestScore = score
          best = i
        }
      }
    }
    if (best >= 0) {
      const bx = best % TERRAIN_W
      const by = Math.floor(best / TERRAIN_W)
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          if (Math.abs(dx) === 2 && dy !== 0) continue
          const i = idx(bx + dx, by + dy)
          if (inBounds(bx + dx, by + dy)) {
            biome[i] = 'lake'
            elevation[i] = Math.min(elevation[i], SEA_LEVEL - 0.02)
          }
        }
      }
    }
  }

  // Rivers: springs in the high east descend to water, carving as they go.
  const riverRng = rngFor(seed, 'terrain', 'rivers')
  const springs: number[] = []
  const highTiles: number[] = []
  for (let y = 4; y < TERRAIN_H - 4; y++) {
    for (let x = Math.floor(TERRAIN_W * 0.55); x < TERRAIN_W - 3; x++) {
      const i = idx(x, y)
      if (elevation[i] > 0.7 && !water[i]) highTiles.push(i)
    }
  }
  highTiles.sort((a, b) => elevation[b] - elevation[a])
  const springCount = Math.min(3, highTiles.length)
  for (let s = 0; s < springCount; s++) {
    // Spread springs apart: take from the top of the sorted list in chunks.
    const pool = highTiles.slice(s * 40, s * 40 + 40)
    if (pool.length > 0) springs.push(pool[riverRng.int(0, pool.length - 1)])
  }
  for (const spring of springs) {
    let cur = spring
    const visited = new Set<number>([cur])
    for (let step = 0; step < 300; step++) {
      const x = cur % TERRAIN_W
      const y = Math.floor(cur / TERRAIN_W)
      if (biome[cur] === 'ocean' || biome[cur] === 'lake') break
      if (biome[cur] !== 'river') biome[cur] = 'river'
      let next = -1
      let lowest = Infinity
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const nx = x + dx
        const ny = y + dy
        if (!inBounds(nx, ny)) continue
        const ni = idx(nx, ny)
        if (visited.has(ni)) continue
        if (elevation[ni] < lowest) {
          lowest = elevation[ni]
          next = ni
        }
      }
      if (next === -1) break // boxed in: stop as a pool
      // Carve so the river never flows uphill on replay.
      elevation[next] = Math.min(elevation[next], elevation[cur])
      visited.add(next)
      cur = next
    }
  }

  // Moisture: noise plus proximity to fresh water.
  const freshDist = distanceTo(biome, (b) => b === 'lake' || b === 'river')
  const moisture = new Array<number>(size)
  for (let y = 0; y < TERRAIN_H; y++) {
    for (let x = 0; x < TERRAIN_W; x++) {
      const i = idx(x, y)
      const near = Math.max(0, 1 - freshDist[i] / 6)
      moisture[i] = Math.min(1, moistNoise.fbm(x, y) * 0.75 + near * 0.35)
    }
  }

  // Land biomes.
  const oceanDist = distanceTo(biome, (b) => b === 'ocean')
  const clayRng = rngFor(seed, 'terrain', 'clay')
  for (let y = 0; y < TERRAIN_H; y++) {
    for (let x = 0; x < TERRAIN_W; x++) {
      const i = idx(x, y)
      if (isWater(biome[i])) continue
      const e = elevation[i]
      const m = moisture[i]
      if (e < SEA_LEVEL + 0.03 && oceanDist[i] <= 2) biome[i] = 'beach'
      else if (e > 0.86) biome[i] = 'snow'
      else if (e > 0.74) biome[i] = 'mountain'
      else if (e > 0.62) biome[i] = 'hills'
      else if (m > 0.74 && e < 0.48 && freshDist[i] <= 2) biome[i] = 'marsh'
      else if (freshDist[i] === 1 && clayRng.chance(0.45)) biome[i] = 'clay_flats'
      else if (m > 0.7) biome[i] = 'dense_forest'
      else if (m > 0.55) biome[i] = 'forest'
      else if (m > 0.42) biome[i] = 'meadow'
      else biome[i] = 'grass'
    }
  }

  // Materials.
  const timber = new Array<number>(size).fill(0)
  const stone = new Array<number>(size).fill(0)
  const clay = new Array<number>(size).fill(0)
  const reeds = new Array<number>(size).fill(0)
  for (let i = 0; i < size; i++) {
    switch (biome[i]) {
      case 'dense_forest':
        timber[i] = 8
        break
      case 'forest':
        timber[i] = 5
        break
      case 'mountain':
        stone[i] = 7
        break
      case 'snow':
        stone[i] = 4
        break
      case 'hills':
        stone[i] = 3
        break
      case 'clay_flats':
        clay[i] = 5
        break
      case 'marsh':
        reeds[i] = 5
        break
    }
  }

  const townCenter = pickTownSite(biome, freshDist)
  const landmarks = nameLandmarks(seed, biome, elevation)

  return {
    width: TERRAIN_W,
    height: TERRAIN_H,
    elevation: elevation.map((e) => Math.round(e * 1000) / 1000),
    moisture: moisture.map((m) => Math.round(m * 1000) / 1000),
    biome,
    timber,
    stone,
    clay,
    reeds,
    landmarks,
    townCenter,
  }
}

/** BFS distance (4-neighbor) to the nearest tile matching `match`. */
function distanceTo(biome: BiomeId[], match: (b: BiomeId) => boolean): number[] {
  const size = TERRAIN_W * TERRAIN_H
  const dist = new Array<number>(size).fill(Infinity)
  const queue: number[] = []
  for (let i = 0; i < size; i++) {
    if (match(biome[i])) {
      dist[i] = 0
      queue.push(i)
    }
  }
  let head = 0
  while (head < queue.length) {
    const i = queue[head++]
    const x = i % TERRAIN_W
    const y = Math.floor(i / TERRAIN_W)
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = x + dx
      const ny = y + dy
      if (!inBounds(nx, ny)) continue
      const ni = idx(nx, ny)
      if (dist[ni] > dist[i] + 1) {
        dist[ni] = dist[i] + 1
        queue.push(ni)
      }
    }
  }
  return dist
}

/**
 * Deterministic town site: the most buildable neighborhood with fresh water
 * close by and timber within reach. Ties break toward the lowest index.
 */
function pickTownSite(biome: BiomeId[], freshDist: number[]): { x: number; y: number } {
  const timberDist = distanceTo(biome, (b) => b === 'forest' || b === 'dense_forest')
  let best = { x: Math.floor(TERRAIN_W / 2), y: Math.floor(TERRAIN_H / 2) }
  let bestScore = -Infinity
  for (let y = 10; y < TERRAIN_H - 10; y++) {
    for (let x = 14; x < TERRAIN_W - 10; x++) {
      const i = idx(x, y)
      if (!isBuildable(biome[i]) || biome[i] === 'forest') continue
      let open = 0
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const ni = idx(x + dx, y + dy)
          if (inBounds(x + dx, y + dy) && isBuildable(biome[ni]) && biome[ni] !== 'forest') open++
        }
      }
      if (open < 26) continue
      if (freshDist[i] > 8) continue
      const score = open + (8 - freshDist[i]) * 5 + Math.max(0, 10 - timberDist[i]) * 1.5
      if (score > bestScore) {
        bestScore = score
        best = { x, y }
      }
    }
  }
  return best
}

// ---------------------------------------------------------------------------
// Landmark naming — geography enters culture by having names to gossip about.

const LAKE_NAMES = ['Cold', 'Mirror', 'Reed', 'Hollow', 'Grey', 'Pike', 'Ember']
const LAKE_KINDS = ['Lake', 'Mere', 'Tarn']
const PEAK_NAMES = ['Bare', 'Iron', 'Raven', 'Storm', 'Old Grumble', 'Widow’s']
const PEAK_KINDS = ['Peak', 'Tor', 'Fell']
const FOREST_NAMES = ['Whisper', 'Tangle', 'Elder', 'Black', 'Hush', 'Crooked']
const FOREST_KINDS = ['Pines', 'Wood', 'Thicket']
const SEA_NAMES = ['the Wide Water', 'the Ash Sea', 'the Western Deep', 'the Pale Gulf']

function nameLandmarks(seed: string, biome: BiomeId[], elevation: number[]): Landmark[] {
  const landmarks: Landmark[] = []
  const rng = rngFor(seed, 'terrain', 'names')

  // The sea, labeled mid-height near the west edge if there is one.
  const seaTile = biome.findIndex((b) => b === 'ocean')
  if (seaTile >= 0) {
    landmarks.push({
      kind: 'sea',
      name: rng.pick(SEA_NAMES),
      x: 6,
      y: Math.floor(TERRAIN_H / 2),
    })
  }

  // Largest lake.
  const lakeBlob = largestBlob(biome, (b) => b === 'lake')
  if (lakeBlob) {
    landmarks.push({
      kind: 'lake',
      name: `${rng.pick(LAKE_NAMES)} ${rng.pick(LAKE_KINDS)}`,
      x: lakeBlob.x,
      y: lakeBlob.y,
    })
  }

  // Highest peak.
  let peak = -1
  for (let i = 0; i < biome.length; i++) {
    if ((biome[i] === 'mountain' || biome[i] === 'snow') && (peak < 0 || elevation[i] > elevation[peak])) {
      peak = i
    }
  }
  if (peak >= 0) {
    landmarks.push({
      kind: 'mountain',
      name: `${rng.pick(PEAK_NAMES)} ${rng.pick(PEAK_KINDS)}`,
      x: peak % TERRAIN_W,
      y: Math.floor(peak / TERRAIN_W),
    })
  }

  // Largest forest.
  const forestBlob = largestBlob(biome, (b) => b === 'forest' || b === 'dense_forest')
  if (forestBlob) {
    landmarks.push({
      kind: 'forest',
      name: `${rng.pick(FOREST_NAMES)} ${rng.pick(FOREST_KINDS)}`,
      x: forestBlob.x,
      y: forestBlob.y,
    })
  }

  return landmarks
}

/** Centroid of the largest connected blob matching `match`, if any. */
function largestBlob(
  biome: BiomeId[],
  match: (b: BiomeId) => boolean,
): { x: number; y: number; size: number } | null {
  const seen = new Array<boolean>(biome.length).fill(false)
  let best: { x: number; y: number; size: number } | null = null
  for (let start = 0; start < biome.length; start++) {
    if (seen[start] || !match(biome[start])) continue
    let sumX = 0
    let sumY = 0
    let count = 0
    const queue = [start]
    seen[start] = true
    while (queue.length > 0) {
      const i = queue.pop()!
      const x = i % TERRAIN_W
      const y = Math.floor(i / TERRAIN_W)
      sumX += x
      sumY += y
      count++
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const nx = x + dx
        const ny = y + dy
        if (!inBounds(nx, ny)) continue
        const ni = idx(nx, ny)
        if (!seen[ni] && match(biome[ni])) {
          seen[ni] = true
          queue.push(ni)
        }
      }
    }
    if (count >= 8 && (!best || count > best.size)) {
      best = { x: Math.round(sumX / count), y: Math.round(sumY / count), size: count }
    }
  }
  return best
}

// ---------------------------------------------------------------------------
// Town placement onto terrain

/**
 * Find the nearest buildable, unoccupied tile to a desired offset from the
 * town center, searching outward ring by ring (deterministic order).
 */
export function findBuildSite(
  terrain: Terrain,
  occupied: Set<number>,
  desiredX: number,
  desiredY: number,
): { x: number; y: number } {
  for (let r = 0; r < 20; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
        const x = desiredX + dx
        const y = desiredY + dy
        if (!inBounds(x, y)) continue
        const i = idx(x, y)
        if (occupied.has(i)) continue
        if (!isBuildable(terrain.biome[i])) continue
        return { x, y }
      }
    }
  }
  return { x: terrain.townCenter.x, y: terrain.townCenter.y }
}

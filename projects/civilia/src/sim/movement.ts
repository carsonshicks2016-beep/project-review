import type { World } from './types'
import { TERRAIN_H, TERRAIN_W, idx, inBounds, moveCost } from './terrain'

/**
 * Movement v0: agents walk tile routes between places, found with A* over
 * the terrain. Walking a tile wears it; worn tiles are cheaper to walk, so
 * popular routes self-reinforce from trail to path to road. Wear decays
 * slowly when a route falls out of use.
 *
 * No route caching: wear changes as agents walk within a phase, and cached
 * module state would break save/load replay identity.
 */

/** Wear thresholds used by sim and renderer alike. */
export const WEAR_TRAIL = 3
export const WEAR_PATH = 8
export const WEAR_ROAD = 18

const wearKey = (x: number, y: number): string => `${x},${y}`

function tileCost(world: World, x: number, y: number): number {
  const base = moveCost(world.terrain.biome[idx(x, y)])
  if (!isFinite(base)) return Infinity
  const wear = world.pathWear[wearKey(x, y)] ?? 0
  // A well-trodden road is up to ~1.8x faster than raw ground.
  return base / (1 + Math.min(wear, 25) * 0.032)
}

/**
 * A* route between two places. Returns the tile path including both
 * endpoints, or null if no route exists. Deterministic: fixed neighbor
 * order, ties broken by tile index.
 */
export function findRoute(
  world: World,
  from: { x: number; y: number },
  to: { x: number; y: number },
): { x: number; y: number }[] | null {
  const start = idx(from.x, from.y)
  const goal = idx(to.x, to.y)
  if (start === goal) return [from]

  const size = TERRAIN_W * TERRAIN_H
  const gScore = new Float64Array(size).fill(Infinity)
  const cameFrom = new Int32Array(size).fill(-1)
  const closed = new Uint8Array(size)
  gScore[start] = 0

  // Binary heap of [fScore, tileIndex].
  const heap: [number, number][] = [[heuristic(start, goal), start]]
  const push = (f: number, i: number) => {
    heap.push([f, i])
    let c = heap.length - 1
    while (c > 0) {
      const p = (c - 1) >> 1
      if (heap[p][0] < heap[c][0] || (heap[p][0] === heap[c][0] && heap[p][1] <= heap[c][1])) break
      ;[heap[p], heap[c]] = [heap[c], heap[p]]
      c = p
    }
  }
  const pop = (): [number, number] => {
    const top = heap[0]
    const last = heap.pop()!
    if (heap.length > 0) {
      heap[0] = last
      let p = 0
      for (;;) {
        const l = p * 2 + 1
        const r = l + 1
        let m = p
        if (l < heap.length && (heap[l][0] < heap[m][0] || (heap[l][0] === heap[m][0] && heap[l][1] < heap[m][1]))) m = l
        if (r < heap.length && (heap[r][0] < heap[m][0] || (heap[r][0] === heap[m][0] && heap[r][1] < heap[m][1]))) m = r
        if (m === p) break
        ;[heap[p], heap[m]] = [heap[m], heap[p]]
        p = m
      }
    }
    return top
  }

  while (heap.length > 0) {
    const [, current] = pop()
    if (current === goal) {
      const path: { x: number; y: number }[] = []
      let i = current
      while (i !== -1) {
        path.push({ x: i % TERRAIN_W, y: Math.floor(i / TERRAIN_W) })
        i = cameFrom[i]
      }
      return path.reverse()
    }
    if (closed[current]) continue
    closed[current] = 1
    const cx = current % TERRAIN_W
    const cy = Math.floor(current / TERRAIN_W)
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = cx + dx
      const ny = cy + dy
      if (!inBounds(nx, ny)) continue
      const ni = idx(nx, ny)
      if (closed[ni]) continue
      const cost = tileCost(world, nx, ny)
      if (!isFinite(cost)) continue
      const tentative = gScore[current] + cost
      if (tentative < gScore[ni]) {
        gScore[ni] = tentative
        cameFrom[ni] = current
        push(tentative + heuristic(ni, goal), ni)
      }
    }
  }
  return null
}

/** Manhattan distance scaled to the cheapest tile cost (admissible: roads floor at ~0.55). */
function heuristic(a: number, b: number): number {
  const ax = a % TERRAIN_W
  const ay = Math.floor(a / TERRAIN_W)
  const bx = b % TERRAIN_W
  const by = Math.floor(b / TERRAIN_W)
  return (Math.abs(ax - bx) + Math.abs(ay - by)) * 0.5
}

/** Walking a route leaves wear on every tile it crosses. */
export function recordTravel(world: World, path: { x: number; y: number }[]): void {
  for (const { x, y } of path) {
    const key = wearKey(x, y)
    world.pathWear[key] = (world.pathWear[key] ?? 0) + 1
  }
}

/** Nightly regrowth: unused routes fade over weeks, roads take months. */
export function decayPathWear(world: World): void {
  for (const key of Object.keys(world.pathWear)) {
    const next = world.pathWear[key] * 0.97
    if (next < 0.15) delete world.pathWear[key]
    else world.pathWear[key] = Math.round(next * 1000) / 1000
  }
}

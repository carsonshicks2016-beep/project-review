import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import {
  TERRAIN_H,
  TERRAIN_W,
  generateTerrain,
  idx,
  isBuildable,
  isWalkable,
} from '../../src/sim/terrain'

describe('procedural terrain', () => {
  it('is deterministic per seed and differs across seeds', () => {
    const a = generateTerrain('terra-1')
    const b = generateTerrain('terra-1')
    const c = generateTerrain('terra-2')
    expect(a).toEqual(b)
    expect(a.biome.join('')).not.toEqual(c.biome.join(''))
  })

  it('has an ocean, fresh water, forests, and high ground', () => {
    const t = generateTerrain('terra-1')
    const counts = new Map<string, number>()
    for (const b of t.biome) counts.set(b, (counts.get(b) ?? 0) + 1)
    expect(counts.get('ocean') ?? 0).toBeGreaterThan(50)
    const fresh = (counts.get('lake') ?? 0) + (counts.get('river') ?? 0)
    expect(fresh).toBeGreaterThan(0)
    expect((counts.get('forest') ?? 0) + (counts.get('dense_forest') ?? 0)).toBeGreaterThan(30)
    expect((counts.get('mountain') ?? 0) + (counts.get('hills') ?? 0)).toBeGreaterThan(10)
  })

  it('provides gatherable materials', () => {
    const t = generateTerrain('terra-1')
    expect(t.timber.some((v) => v > 0)).toBe(true)
    expect(t.stone.some((v) => v > 0)).toBe(true)
  })

  it('names its landmarks', () => {
    const t = generateTerrain('terra-1')
    expect(t.landmarks.length).toBeGreaterThan(0)
    for (const lm of t.landmarks) {
      expect(lm.name.length).toBeGreaterThan(2)
      expect(lm.x).toBeGreaterThanOrEqual(0)
      expect(lm.y).toBeLessThan(TERRAIN_H)
    }
  })

  it('picks a buildable town site away from map edges', () => {
    const t = generateTerrain('terra-1')
    const { x, y } = t.townCenter
    expect(x).toBeGreaterThan(8)
    expect(x).toBeLessThan(TERRAIN_W - 6)
    expect(y).toBeGreaterThan(4)
    expect(y).toBeLessThan(TERRAIN_H - 4)
    expect(isWalkable(t.biome[idx(x, y)])).toBe(true)
  })

  it('settles every town place on its own walkable tile', () => {
    const world = createWorld('terra-town')
    const seen = new Set<string>()
    for (const place of Object.values(world.places)) {
      const key = `${place.tile.x},${place.tile.y}`
      expect(seen.has(key)).toBe(false)
      seen.add(key)
      const biome = world.terrain.biome[idx(place.tile.x, place.tile.y)]
      expect(isWalkable(biome)).toBe(true)
      expect(isBuildable(biome)).toBe(true)
    }
  })
})

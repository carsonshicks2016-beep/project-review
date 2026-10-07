import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import { decayPathWear, findRoute, recordTravel } from '../../src/sim/movement'
import { idx, isWalkable } from '../../src/sim/terrain'

const SEED = 'move-test'

describe('movement and path wear', () => {
  it('routes between places over walkable tiles only', () => {
    const world = createWorld(SEED)
    const farm = world.places['place_farm']
    const tavern = world.places['place_tavern']
    const route = findRoute(world, farm.tile, tavern.tile)
    expect(route).not.toBeNull()
    expect(route![0]).toEqual(farm.tile)
    expect(route![route!.length - 1]).toEqual(tavern.tile)
    for (const { x, y } of route!) {
      expect(isWalkable(world.terrain.biome[idx(x, y)])).toBe(true)
    }
    // Adjacent steps only.
    for (let i = 1; i < route!.length; i++) {
      const d = Math.abs(route![i].x - route![i - 1].x) + Math.abs(route![i].y - route![i - 1].y)
      expect(d).toBe(1)
    }
  })

  it('wears routes with repeated walking and decays them overnight', () => {
    const world = createWorld(SEED)
    const a = world.places['place_well'].tile
    const b = world.places['place_farm'].tile
    const route = findRoute(world, a, b)!
    for (let i = 0; i < 5; i++) recordTravel(world, route)
    const key = `${route[1].x},${route[1].y}`
    expect(world.pathWear[key]).toBe(5)
    decayPathWear(world)
    expect(world.pathWear[key]).toBeLessThan(5)
    expect(world.pathWear[key]).toBeGreaterThan(4)
  })

  it('worn routes become preferred routes (roads attract walkers)', () => {
    const world = createWorld(SEED)
    const a = world.places['place_well'].tile
    const b = world.places['place_tavern'].tile
    const before = findRoute(world, a, b)!
    // Heavily wear the existing route, then re-route: cost must not increase,
    // and the worn route should be at least as attractive as before.
    for (let i = 0; i < 30; i++) recordTravel(world, before)
    const after = findRoute(world, a, b)!
    const overlap = after.filter((t) => before.some((u) => u.x === t.x && u.y === t.y)).length
    expect(overlap / after.length).toBeGreaterThan(0.6)
  })

  it('accumulates wear deterministically across full sim runs', () => {
    const w1 = createWorld(SEED)
    const w2 = createWorld(SEED)
    for (let d = 0; d < 8; d++) {
      stepDay(w1)
      stepDay(w2)
    }
    expect(w1.pathWear).toEqual(w2.pathWear)
    // Agents actually walked somewhere.
    expect(Object.keys(w1.pathWear).length).toBeGreaterThan(5)
  })
})

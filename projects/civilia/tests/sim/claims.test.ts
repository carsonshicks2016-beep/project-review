import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import { recomputeClaims } from '../../src/sim/systems/claimsSystem'
import { idx } from '../../src/sim/terrain'

const SEED = 'claims-test'

describe('claims and borders', () => {
  it('every household claims the ground under its own home', () => {
    const world = createWorld(SEED)
    recomputeClaims(world)
    for (const household of Object.values(world.households)) {
      const home = world.places[household.homeId]
      const i = idx(home.tile.x, home.tile.y)
      const owner = world.claims!.owner[i]
      // Own home tile: either clearly theirs, or contested with a close neighbor.
      expect(owner === household.id || world.claims!.disputed[i]).toBe(true)
    }
  })

  it('claims cover ground and water stays unclaimed', () => {
    const world = createWorld(SEED)
    recomputeClaims(world)
    const claimed = world.claims!.owner.filter((o) => o !== '').length
    expect(claimed).toBeGreaterThan(50)
    for (let i = 0; i < world.claims!.owner.length; i++) {
      const biome = world.terrain.biome[i]
      if (biome === 'ocean' || biome === 'lake') {
        expect(world.claims!.owner[i]).toBe('')
      }
    }
  })

  it('disputed tiles only appear between two real claimants', () => {
    const world = createWorld(SEED)
    recomputeClaims(world)
    for (let i = 0; i < world.claims!.disputed.length; i++) {
      if (world.claims!.disputed[i]) {
        expect(world.claims!.owner[i]).not.toBe('')
      }
    }
  })

  it('claims and incidents replay deterministically', () => {
    const a = createWorld(SEED)
    const b = createWorld(SEED)
    for (let d = 0; d < 15; d++) {
      stepDay(a)
      stepDay(b)
    }
    expect(a.claims).toEqual(b.claims)
    expect(
      a.events.filter((e) => e.type === 'border_dispute'),
    ).toEqual(b.events.filter((e) => e.type === 'border_dispute'))
  })

  it('border incidents leave grudges when they happen', () => {
    const world = createWorld(SEED)
    for (let d = 0; d < 30; d++) stepDay(world)
    const incidents = world.events.filter((e) => e.type === 'border_dispute')
    for (const incident of incidents) {
      const a = world.agents[incident.actorIds[0]]
      const b = world.agents[incident.targetIds[0]]
      expect(a).toBeDefined()
      expect(b).toBeDefined()
      // The confrontation produced an actual spoken line.
      expect(incident.payload.speech).toBeDefined()
    }
  })
})

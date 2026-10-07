import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import type { World } from '../../src/sim/types'

/** Fingerprint of everything that should replay identically for a seed. */
function fingerprint(world: World): string {
  const events = world.events.map(
    (e) => `${e.id}|${e.type}|${e.day}|${e.phase}|${e.actorIds.join(',')}|${e.targetIds.join(',')}|${e.witnessIds.join(',')}`,
  )
  const agents = Object.values(world.agents).map(
    (a) => `${a.id}|${a.locationId}|${a.money}|${a.needs.hunger.toFixed(3)}|${a.breadInventory}|${a.status}`,
  )
  return [...events, ...agents].join('\n')
}

describe('determinism', () => {
  it('same seed produces the same event log and end state', () => {
    const a = createWorld('test-seed-1')
    const b = createWorld('test-seed-1')
    for (let i = 0; i < 10; i++) {
      stepDay(a)
      stepDay(b)
    }
    expect(fingerprint(a)).toBe(fingerprint(b))
  })

  it('different seeds produce different histories', () => {
    const a = createWorld('test-seed-1')
    const b = createWorld('test-seed-2')
    for (let i = 0; i < 5; i++) {
      stepDay(a)
      stepDay(b)
    }
    expect(fingerprint(a)).not.toBe(fingerprint(b))
  })

  it('world generation is deterministic before any steps', () => {
    const a = createWorld('gen-check')
    const b = createWorld('gen-check')
    expect(JSON.stringify(a)).toBe(JSON.stringify(b))
  })
})

import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay, stepPhase } from '../../src/sim/stepWorld'
import { SAVE_VERSION, deserializeWorld, serializeWorld } from '../../src/sim/persistence'

const SEED = 'persist-test'

describe('persistence v0', () => {
  it('round-trips a world losslessly', () => {
    const world = createWorld(SEED)
    for (let i = 0; i < 3; i++) stepDay(world)
    const restored = deserializeWorld(serializeWorld(world))
    expect(restored).toEqual(world)
  })

  it('save/load mid-run continues exactly like an uninterrupted run', () => {
    const uninterrupted = createWorld(SEED)
    for (let d = 0; d < 10; d++) stepDay(uninterrupted)

    const interrupted = createWorld(SEED)
    for (let d = 0; d < 4; d++) stepDay(interrupted)
    stepPhase(interrupted) // save mid-day, not just on day boundaries
    const restored = deserializeWorld(serializeWorld(interrupted))
    while (restored.day < 11) stepPhase(restored)

    expect(restored.events).toEqual(uninterrupted.events)
    expect(restored.agents).toEqual(uninterrupted.agents)
    expect(restored.rumors).toEqual(uninterrupted.rumors)
  })

  it('rejects snapshots from other versions or junk input', () => {
    expect(() => deserializeWorld('not json')).toThrow(/not valid JSON/)
    expect(() => deserializeWorld('{"saveVersion": 999, "world": {}}')).toThrow(/version/)
    expect(() =>
      deserializeWorld(`{"saveVersion": ${SAVE_VERSION}, "world": {"seed": "x"}}`),
    ).toThrow(/core world state/)
  })
})

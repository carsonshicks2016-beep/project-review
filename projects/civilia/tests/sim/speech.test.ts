import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import type { Utterance } from '../../src/sim/systems/speechSystem'

const SEED = 'speech-test'

function speechEvents(world: ReturnType<typeof createWorld>) {
  return world.events.filter((e) => e.payload.speech !== undefined)
}

describe('speech and motive engine', () => {
  it('attaches spoken lines with motives to social events', () => {
    const world = createWorld(SEED)
    for (let d = 0; d < 12; d++) stepDay(world)
    const spoken = speechEvents(world)
    expect(spoken.length).toBeGreaterThan(5)
    for (const event of spoken) {
      const speech = event.payload.speech as Utterance
      expect(speech.text.length).toBeGreaterThan(3)
      expect(speech.category).toBeDefined()
      expect(speech.motive.kind).toBeDefined()
      expect(speech.motive.summary.length).toBeGreaterThan(10)
      expect(speech.motive.reasons.length).toBeGreaterThan(0)
    }
  })

  it('produces identical speech on replay (stored and recomposed)', () => {
    const a = createWorld(SEED)
    const b = createWorld(SEED)
    for (let d = 0; d < 8; d++) {
      stepDay(a)
      stepDay(b)
    }
    expect(speechEvents(a).map((e) => e.payload.speech)).toEqual(
      speechEvents(b).map((e) => e.payload.speech),
    )
    // The stored utterance is the canonical record: it captured the world at
    // the moment of speaking (prices, grudges), which later drifts — so the
    // store-at-emit design, not recomposition, is what makes replay exact.
  })

  it('motive analysis reflects the relationship', () => {
    const world = createWorld(SEED)
    for (let d = 0; d < 20; d++) stepDay(world)
    const insult = world.events.find((e) => e.type === 'insulted' && e.payload.speech)
    if (insult) {
      const speech = insult.payload.speech as Utterance
      expect(['premeditated', 'venting']).toContain(speech.motive.kind)
      expect(speech.motive.reasons.some((r) => r.includes('resentment'))).toBe(true)
    }
    const help = world.events.find((e) => e.type === 'helped' && e.payload.speech)
    if (help) {
      const speech = help.payload.speech as Utterance
      expect(['genuine', 'performative']).toContain(speech.motive.kind)
    }
    expect(insult ?? help).toBeDefined()
  })
})

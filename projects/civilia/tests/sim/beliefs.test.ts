import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import { relationshipWith } from '../../src/sim/relationships'
import {
  beliefAbout,
  believeFromWitness,
  damagingBeliefsAbout,
  receiveRumorClaim,
} from '../../src/sim/systems/beliefSystem'
import { createRumor } from '../../src/sim/systems/rumorSystem'
import type { Utterance } from '../../src/sim/systems/speechSystem'
import type { World } from '../../src/sim/types'

const SEED = 'belief-test'

function testRumor(world: World, aboutId: string, knowerId: string) {
  return createRumor(world, {
    claim: `${world.agents[aboutId].name} stole bread from the Bray bakery`,
    aboutAgentId: aboutId,
    emotionalCharge: 0.5,
    firstKnowerIds: [knowerId],
    tags: ['crime', 'bread'],
  })
}

describe('stage 15 — belief system', () => {
  it('seeds the founding suspicion as a graded inference', () => {
    const world = createWorld(SEED)
    const holder = Object.values(world.agents).find((a) => a.beliefIds.length > 0)!
    const belief = world.beliefs[holder.beliefIds[0]]
    expect(belief.source).toBe('inference')
    expect(belief.confidence).toBeCloseTo(0.55)
    expect(belief.truthStatus).toBe('true') // the reserve is scenario-true
    expect(belief.claim).toContain('grain reserve')
  })

  it('an eyewitness knows; a listener merely weighs the teller', () => {
    const world = createWorld(SEED)
    const witness = world.agents['agent_0001']
    const rumor = testRumor(world, 'agent_0009', 'agent_0001')
    believeFromWitness(world, witness, {
      claim: rumor.originalClaim,
      subjectId: 'agent_0009',
      rumorId: rumor.id,
      emotionalCharge: 0.6,
      tags: ['crime'],
    })
    const known = beliefAbout(world, witness, { rumorId: rumor.id })!
    expect(known.confidence).toBe(0.95)
    expect(known.source).toBe('witnessed')

    // Trusted teller vs distrusted teller: same claim, different conviction.
    const trusting = world.agents['agent_0002']
    const wary = world.agents['agent_0003']
    relationshipWith(trusting, witness.id).trust = 0.9
    relationshipWith(wary, witness.id).trust = 0.05
    const heard1 = receiveRumorClaim(world, trusting, witness, rumor, rumor.currentClaim)
    const heard2 = receiveRumorClaim(world, wary, witness, rumor, rumor.currentClaim)
    expect(heard1.belief.confidence).toBeGreaterThan(heard2.belief.confidence)
    expect(heard1.factors.some((f) => f.includes('trust'))).toBe(true)
  })

  it('disposition bends belief: grudges make accusations easy to believe', () => {
    const world = createWorld(SEED)
    const teller = world.agents['agent_0001']
    const subject = 'agent_0009'
    const hater = world.agents['agent_0004']
    const friend = world.agents['agent_0005']
    relationshipWith(hater, teller.id).trust = 0.4
    relationshipWith(friend, teller.id).trust = 0.4
    relationshipWith(hater, subject).resentment = 0.8
    relationshipWith(friend, subject).affection = 0.8
    const rumor = testRumor(world, subject, teller.id)
    const heardByHater = receiveRumorClaim(world, hater, teller, rumor, rumor.currentClaim)
    const heardByFriend = receiveRumorClaim(world, friend, teller, rumor, rumor.currentClaim)
    expect(heardByHater.belief.confidence).toBeGreaterThan(heardByFriend.belief.confidence)
  })

  it('repetition moves belief with inertia, and eyewitnesses barely move', () => {
    const world = createWorld(SEED)
    const teller = world.agents['agent_0001']
    const listener = world.agents['agent_0002']
    relationshipWith(listener, teller.id).trust = 0.8
    const rumor = testRumor(world, 'agent_0009', teller.id)

    const firstHearing = receiveRumorClaim(world, listener, teller, rumor, rumor.currentClaim)
    const secondHearing = receiveRumorClaim(world, listener, teller, rumor, rumor.currentClaim)
    // Same evidence twice: the second telling moves confidence less.
    expect(Math.abs(secondHearing.delta)).toBeLessThan(Math.abs(firstHearing.delta))

    // An eyewitness hears a watered-down version: conviction barely shifts.
    const witness = world.agents['agent_0003']
    believeFromWitness(world, witness, {
      claim: rumor.originalClaim,
      subjectId: 'agent_0009',
      rumorId: rumor.id,
      emotionalCharge: 0.6,
      tags: ['crime'],
    })
    relationshipWith(witness, teller.id).trust = 0.1
    const witnessHears = receiveRumorClaim(world, witness, teller, rumor, rumor.currentClaim)
    expect(witnessHears.belief.confidence).toBeGreaterThan(0.85)
    expect(witnessHears.belief.source).toBe('witnessed')
  })

  it('keeps rumor believer/skeptic rosters in sync with graded conviction', () => {
    const world = createWorld(SEED)
    const teller = world.agents['agent_0001']
    const devotee = world.agents['agent_0002']
    const doubter = world.agents['agent_0003']
    relationshipWith(devotee, teller.id).trust = 1
    relationshipWith(devotee, 'agent_0009').resentment = 0.9
    relationshipWith(doubter, teller.id).trust = 0
    relationshipWith(doubter, 'agent_0009').affection = 1
    const rumor = testRumor(world, 'agent_0009', teller.id)
    receiveRumorClaim(world, devotee, teller, rumor, rumor.currentClaim)
    receiveRumorClaim(world, doubter, teller, rumor, rumor.currentClaim)
    expect(rumor.believerIds).toContain(devotee.id)
    expect(rumor.believerIds).not.toContain(doubter.id)
    expect(rumor.skepticIds).toContain(doubter.id)
  })

  it('beliefs surface in dialogue and the reasoning payload cites them', () => {
    const world = createWorld(SEED)
    for (let d = 0; d < 25; d++) stepDay(world)
    const spoken = world.events.filter((e) => e.payload.speech)
    const cited = spoken.filter((e) => {
      const speech = e.payload.speech as Utterance
      return speech.citedMemory || speech.citedBelief
    })
    expect(cited.length).toBeGreaterThan(0)
    for (const event of cited) {
      const speech = event.payload.speech as Utterance
      // Citations are snapshots: they outlive the memory/belief they cite.
      if (speech.citedMemory) {
        expect(speech.citedMemory.summary.length).toBeGreaterThan(5)
        expect(speech.citedMemory.day).toBeLessThanOrEqual(event.day)
      }
      if (speech.citedBelief) {
        expect(speech.citedBelief.confidence).toBeGreaterThan(0.5)
      }
    }
    // Damaging convictions also bias insult scoring with a visible reason.
    const insults = world.events.filter((e) => e.type === 'insulted')
    expect(insults.length).toBeGreaterThan(0)
  })

  it('full runs replay deterministically with beliefs', () => {
    const a = createWorld(SEED)
    const b = createWorld(SEED)
    for (let d = 0; d < 15; d++) {
      stepDay(a)
      stepDay(b)
    }
    expect(a.beliefs).toEqual(b.beliefs)
    expect(damagingBeliefsAbout(a, a.agents['agent_0001'], 'agent_0004')).toEqual(
      damagingBeliefsAbout(b, b.agents['agent_0001'], 'agent_0004'),
    )
  })
})

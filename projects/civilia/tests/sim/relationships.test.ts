import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import { rngFor } from '../../src/sim/rng'
import { applyDebtPressure, relationshipWith } from '../../src/sim/relationships'
import { apologize, repayFavor } from '../../src/sim/actions/repair'
import { helpAgent, insultAgent } from '../../src/sim/actions/social'
import { decorateWithSpeech } from '../../src/sim/systems/speechSystem'
import type { ActionContext, Agent, World } from '../../src/sim/types'

const SEED = 'rel-test'

function contextFor(world: World, agent: Agent, placeId: string): ActionContext {
  agent.locationId = placeId
  return {
    world,
    agent,
    phase: 'midday',
    othersHere: Object.values(world.agents).filter(
      (a) => a.id !== agent.id && a.locationId === placeId,
    ),
    place: world.places[placeId],
    rng: rngFor(world.seed, 'test', world.day, agent.id),
  }
}

function meet(a: Agent, b: Agent): void {
  a.locationId = 'place_well'
  b.locationId = 'place_well'
}

describe('stage 13 — relationship repair and the favor economy', () => {
  it('cross-household help creates a favor debt; family help does not', () => {
    const world = createWorld(SEED)
    const giver = world.agents['agent_0001'] // Mara Calder
    const strangerInNeed = world.agents['agent_0004'] // Odet Bray
    const kinInNeed = world.agents['agent_0003'] // Tessa Calder
    giver.money = 20
    giver.breadInventory = 0

    strangerInNeed.needs.hunger = 80
    meet(giver, strangerInNeed)
    helpAgent.execute(contextFor(world, giver, 'place_well'), {
      actionId: 'help_agent',
      label: '',
      targetAgentId: strangerInNeed.id,
      score: 1,
      reasons: [],
    })
    expect(relationshipWith(strangerInNeed, giver.id).debt).toBe(2)

    kinInNeed.needs.hunger = 80
    meet(giver, kinInNeed)
    helpAgent.execute(contextFor(world, giver, 'place_well'), {
      actionId: 'help_agent',
      label: '',
      targetAgentId: kinInNeed.id,
      score: 1,
      reasons: [],
    })
    expect(relationshipWith(kinInNeed, giver.id).debt).toBe(0)
  })

  it('repaying a favor clears the debt and warms the creditor', () => {
    const world = createWorld(SEED)
    const debtor = world.agents['agent_0004']
    const creditor = world.agents['agent_0001']
    relationshipWith(debtor, creditor.id).debt = 4
    debtor.money = 10
    const trustBefore = relationshipWith(creditor, debtor.id).trust

    meet(debtor, creditor)
    const ctx = contextFor(world, debtor, 'place_well')
    const candidates = repayFavor.candidates.call(repayFavor, ctx)
    expect(candidates.length).toBe(1)
    repayFavor.execute(ctx, candidates[0])

    expect(relationshipWith(debtor, creditor.id).debt).toBe(0)
    expect(debtor.money).toBe(6)
    expect(creditor.money).toBeGreaterThan(0)
    expect(relationshipWith(creditor, debtor.id).trust).toBeGreaterThan(trustBefore)
    expect(world.events.at(-1)!.type).toBe('debt_repaid')
  })

  it('unpaid debts corrode the creditor patience overnight', () => {
    const world = createWorld(SEED)
    const debtor = world.agents['agent_0004']
    const creditor = world.agents['agent_0001']
    relationshipWith(debtor, creditor.id).debt = 5
    const before = relationshipWith(creditor, debtor.id).resentment
    applyDebtPressure(world)
    expect(relationshipWith(creditor, debtor.id).resentment).toBeGreaterThan(before)
  })

  it('insults leave grievances that apologies can repair', () => {
    const world = createWorld(SEED)
    const offender = world.agents['agent_0004']
    const offended = world.agents['agent_0001']
    relationshipWith(offender, offended.id).resentment = 0.5
    meet(offender, offended)
    insultAgent.execute(contextFor(world, offender, 'place_well'), {
      actionId: 'insult_agent',
      label: '',
      targetAgentId: offended.id,
      score: 1,
      reasons: [],
    })
    const view = relationshipWith(offended, offender.id)
    expect(view.grievances).toBe(1)
    expect(view.lastHarmDay).toBe(world.day)
  })

  it('a sincere, fresh, restitution-backed apology to a humble listener lands', () => {
    const world = createWorld(SEED)
    world.day = 2 // fixes the decision stream this test rolls against
    const offender = world.agents['agent_0004']
    const offended = world.agents['agent_0001']
    // Engineer the spec's thresholds: sincerity, timing, restitution, trust, low pride.
    offender.traits.empathy = 0.9
    offender.traits.pride = 0.1
    offender.money = 10
    offended.traits.pride = 0.1
    const view = relationshipWith(offended, offender.id)
    view.grievances = 1
    view.lastHarmDay = world.day
    view.trust = 0.6
    view.resentment = 0.5

    meet(offender, offended)
    const ctx = contextFor(world, offender, 'place_well')
    const candidates = apologize.candidates.call(apologize, ctx)
    expect(candidates.length).toBe(1)
    decorateWithSpeech(world, apologize.execute(ctx, candidates[0]))

    const event = world.events.at(-1)!
    expect(event.type).toBe('apologized')
    expect(event.payload.acceptance as number).toBeGreaterThan(0.8)
    expect(event.payload.accepted).toBe(true)
    expect(view.grievances).toBe(0)
    expect(view.resentment).toBeLessThan(0.5)
    expect(event.payload.speech).toBeDefined()
  })

  it('a stale apology from a proud offender to a prouder victim mostly fails', () => {
    const world = createWorld(SEED)
    const offender = world.agents['agent_0004']
    const offended = world.agents['agent_0001']
    offender.traits.empathy = 0.1
    offender.money = 0
    offended.traits.pride = 0.95
    const view = relationshipWith(offended, offender.id)
    view.grievances = 4
    view.lastHarmDay = 1
    view.trust = 0.05
    world.day = 20
    const statusBefore = offender.status

    meet(offender, offended)
    const ctx = contextFor(world, offender, 'place_well')
    apologize.execute(ctx, {
      actionId: 'apologize',
      label: '',
      targetAgentId: offended.id,
      score: 1,
      reasons: [],
    })
    const event = world.events.at(-1)!
    expect(event.payload.acceptance as number).toBeLessThanOrEqual(0.1)
    expect(event.payload.accepted).toBe(false)
    expect(view.grievances).toBe(4)
    expect(offender.status).toBeLessThan(statusBefore)
  })

  it('repairs occur in real runs and replay deterministically', () => {
    const a = createWorld(SEED)
    const b = createWorld(SEED)
    for (let d = 0; d < 40; d++) {
      stepDay(a)
      stepDay(b)
    }
    const repairs = a.events.filter((e) => e.type === 'apologized' || e.type === 'debt_repaid')
    expect(repairs.length).toBeGreaterThan(0)
    expect(repairs).toEqual(b.events.filter((e) => e.type === 'apologized' || e.type === 'debt_repaid'))
  })
})

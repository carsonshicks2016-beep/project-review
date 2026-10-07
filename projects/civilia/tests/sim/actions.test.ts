import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay, stepPhase } from '../../src/sim/stepWorld'
import { decideAndAct } from '../../src/sim/systems/decisionSystem'
import { rngFor } from '../../src/sim/rng'
import type { ActionContext, World } from '../../src/sim/types'
import { currentPhase } from '../../src/sim/types'
import { buyFood } from '../../src/sim/actions/food'
import { stealItem } from '../../src/sim/actions/crime'
import { gossip, insultAgent } from '../../src/sim/actions/social'

function contextFor(world: World, agentId: string): ActionContext {
  const agent = world.agents[agentId]
  return {
    world,
    agent,
    phase: currentPhase(world),
    othersHere: Object.values(world.agents).filter((a) => a.id !== agentId && a.locationId === agent.locationId),
    place: world.places[agent.locationId],
    rng: rngFor(world.seed, 'test', agentId),
  }
}

describe('needs', () => {
  it('hunger rises if an agent does not eat', () => {
    const world = createWorld('needs-test')
    const agent = world.agents['agent_0001']
    agent.breadInventory = 0
    agent.money = 0 // cannot buy, cannot eat
    const before = agent.needs.hunger
    stepPhase(world)
    expect(agent.needs.hunger).toBeGreaterThan(before)
  })
})

describe('buy_food', () => {
  it('requires money, stock, and a place that sells food', () => {
    const world = createWorld('buy-test')
    const agent = world.agents['agent_0001']
    const bakery = world.places['place_bakery']

    agent.locationId = 'place_bakery'
    agent.breadInventory = 0
    agent.money = 0
    expect(buyFood.candidates(contextFor(world, agent.id))).toHaveLength(0)

    agent.money = 10
    expect(buyFood.candidates(contextFor(world, agent.id)).length).toBeGreaterThan(0)

    bakery.foodStock = 0
    expect(buyFood.candidates(contextFor(world, agent.id))).toHaveLength(0)

    agent.money = 10
    agent.locationId = 'place_well' // the well sells nothing
    expect(buyFood.candidates(contextFor(world, agent.id))).toHaveLength(0)
  })

  it('transfers money and stock and emits paired events', () => {
    const world = createWorld('buy-test-2')
    const agent = world.agents['agent_0001']
    const bakery = world.places['place_bakery']
    agent.locationId = 'place_bakery'
    agent.breadInventory = 0
    agent.money = 10

    const stockBefore = bakery.foodStock
    const price = bakery.foodPrice
    const ctx = contextFor(world, agent.id)
    const eventIds = buyFood.execute(ctx, buyFood.candidates(ctx)[0])

    expect(agent.money).toBe(10 - price)
    expect(agent.breadInventory).toBe(1)
    expect(bakery.foodStock).toBe(stockBefore - 1)
    const types = eventIds.map((id) => world.events.find((e) => e.id === id)!.type)
    expect(types).toContain('bought_good')
    expect(types).toContain('sold_good')
  })
})

describe('steal_item', () => {
  it('emits stole_item, and witnesses react when they notice', () => {
    const world = createWorld('steal-test')
    const thief = world.agents['agent_0003'] // Tessa, no job
    const guard = world.agents['agent_0006'] // Bram, guard
    thief.locationId = 'place_bakery'
    thief.needs.hunger = 90
    thief.money = 0
    guard.locationId = 'place_bakery'
    guard.traits.paranoia = 1 // guarantees noticing (0.45 + 0.25 + 0.2 > 1)

    const ctx = contextFor(world, thief.id)
    const candidates = stealItem.candidates(ctx)
    expect(candidates.length).toBeGreaterThan(0)

    const trustBefore = guard.relationships[thief.id].trust
    const eventIds = stealItem.execute(ctx, candidates[0])
    const events = eventIds.map((id) => world.events.find((e) => e.id === id)!)

    expect(events[0].type).toBe('stole_item')
    expect(thief.breadInventory).toBeGreaterThan(0)
    expect(events.some((e) => e.type === 'witnessed_crime')).toBe(true)
    expect(guard.relationships[thief.id].trust).toBeLessThan(trustBefore)
    // A witnessed theft seeds a rumor known to the witness.
    expect(guard.knownRumorIds.length).toBeGreaterThan(0)
  })

  it('is not considered without desperation or nerve', () => {
    const world = createWorld('steal-test-2')
    const agent = world.agents['agent_0001']
    agent.locationId = 'place_bakery'
    agent.needs.hunger = 10
    agent.money = 20
    agent.traits.riskTolerance = 0.1
    expect(stealItem.candidates(contextFor(world, agent.id))).toHaveLength(0)
  })
})

describe('gossip', () => {
  it('creates a shared_rumor event and the listener learns the rumor', () => {
    const world = createWorld('gossip-test')
    const speaker = Object.values(world.agents).find((a) => a.knownRumorIds.length > 0)!
    const listener = Object.values(world.agents).find(
      (a) => a.id !== speaker.id && a.knownRumorIds.length === 0 && world.rumors[speaker.knownRumorIds[0]].aboutAgentId !== a.id,
    )!
    listener.locationId = speaker.locationId

    const ctx = contextFor(world, speaker.id)
    const candidates = gossip.candidates(ctx)
    const toListener = candidates.find((c) => c.targetAgentId === listener.id)
    expect(toListener).toBeDefined()

    const eventIds = gossip.execute(ctx, toListener!)
    const shared = world.events.find((e) => e.id === eventIds[0])!
    expect(shared.type).toBe('shared_rumor')
    expect(listener.knownRumorIds.length).toBe(1)
  })
})

describe('insult', () => {
  it('raises the target\'s resentment toward the insulter', () => {
    const world = createWorld('insult-test')
    const a = world.agents['agent_0001']
    const b = world.agents['agent_0004']
    b.locationId = a.locationId
    a.relationships[b.id].resentment = 0.6

    const ctx = contextFor(world, a.id)
    const candidates = insultAgent.candidates(ctx)
    const atB = candidates.find((c) => c.targetAgentId === b.id)
    expect(atB).toBeDefined()

    const resentmentBefore = b.relationships[a.id].resentment
    insultAgent.execute(ctx, atB!)
    expect(b.relationships[a.id].resentment).toBeGreaterThan(resentmentBefore)
  })
})

describe('long run', () => {
  it('runs 30 days without crashing and produces varied behavior', () => {
    const world = createWorld('ashvale-1')
    for (let i = 0; i < 30; i++) stepDay(world)

    expect(world.day).toBe(31)
    expect(world.events.length).toBeGreaterThan(500)

    // Agents do not all do the same thing every phase.
    const actionTypes = new Set(world.events.map((e) => e.type))
    expect(actionTypes.size).toBeGreaterThanOrEqual(8)

    // Every agent has decision traces with scored candidates.
    for (const agent of Object.values(world.agents)) {
      const traces = world.decisionTraces[agent.id]
      expect(traces.length).toBeGreaterThan(0)
      const last = traces[traces.length - 1]
      expect(last.candidates.length).toBeGreaterThan(0)
      expect(last.candidates.every((c) => c.reasons.length > 0)).toBe(true)
    }

    // Social consequences exist: some relationship moved away from defaults.
    const someResentment = Object.values(world.agents).some((a) =>
      Object.values(a.relationships).some((r) => r.resentment > 0.6),
    )
    expect(someResentment).toBe(true)
  })

  it('decideAndAct records the chosen candidate in the trace', () => {
    const world = createWorld('trace-test')
    const agent = world.agents['agent_0001']
    const trace = decideAndAct(world, agent)
    expect(trace.candidates.length).toBeGreaterThan(0)
    if (trace.chosenIndex >= 0) {
      expect(trace.candidates[trace.chosenIndex]).toBeDefined()
    }
  })
})

import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import { rngFor } from '../../src/sim/rng'
import { relationshipWith } from '../../src/sim/relationships'
import { insultAgent, talk } from '../../src/sim/actions/social'
import {
  decayMemories,
  peekMemories,
  processEventsIntoMemories,
  recallMemories,
} from '../../src/sim/systems/memorySystem'
import type { ActionContext, Agent, World } from '../../src/sim/types'

const SEED = 'mem-test'

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

function memoriesOf(world: World, agent: Agent) {
  return agent.memoryIds.map((id) => world.memories[id]).filter(Boolean)
}

describe('stage 14 — memory system', () => {
  it('an insult marks the target hard, the insulter lightly, and small talk not at all', () => {
    const world = createWorld(SEED)
    const offender = world.agents['agent_0004']
    const offended = world.agents['agent_0001']
    relationshipWith(offender, offended.id).resentment = 0.5
    offended.locationId = 'place_well'
    const ctx = contextFor(world, offender, 'place_well')
    processEventsIntoMemories(
      world,
      insultAgent.execute(ctx, {
        actionId: 'insult_agent',
        label: '',
        targetAgentId: offended.id,
        score: 1,
        reasons: [],
      }),
    )

    const offendedMemory = memoriesOf(world, offended)[0]
    expect(offendedMemory).toBeDefined()
    expect(offendedMemory.summary).toContain('insulted me')
    expect(['angry', 'ashamed']).toContain(offendedMemory.tone)
    const offenderMemory = memoriesOf(world, offender)[0]
    expect(offenderMemory.importance).toBeLessThan(offendedMemory.importance)

    // Small talk evaporates: a plain conversation forms no memory.
    const talker = world.agents['agent_0002']
    talker.locationId = 'place_well'
    const talkCtx = contextFor(world, talker, 'place_well')
    const candidates = talk.candidates.call(talk, talkCtx)
    processEventsIntoMemories(world, talk.execute(talkCtx, candidates[0]))
    expect(memoriesOf(world, talker).length).toBe(0)
  })

  it('memories fade nightly and the trivial are forgotten before the searing', () => {
    const world = createWorld(SEED)
    const agent = world.agents['agent_0001']
    world.memories['memory_x'] = {
      id: 'memory_x',
      agentId: agent.id,
      sourceEventIds: [],
      summary: 'a forgettable afternoon',
      kind: 'witnessed',
      importance: 0.3,
      emotionalCharge: 0.1,
      tone: 'worried',
      strength: 0.4,
      aboutAgentIds: [],
      tags: [],
      createdDay: 1,
      timesRecalled: 0,
    }
    world.memories['memory_y'] = {
      id: 'memory_y',
      agentId: agent.id,
      sourceEventIds: [],
      summary: 'the night everything burned',
      kind: 'experience',
      importance: 0.9,
      emotionalCharge: 0.9,
      tone: 'afraid',
      strength: 0.4,
      aboutAgentIds: [],
      tags: [],
      createdDay: 1,
      timesRecalled: 0,
    }
    agent.memoryIds.push('memory_x', 'memory_y')

    for (let night = 0; night < 16; night++) decayMemories(world)
    expect(world.memories['memory_x']).toBeUndefined()
    expect(world.memories['memory_y']).toBeDefined()
    expect(agent.memoryIds).toEqual(['memory_y'])
  })

  it('recalling strengthens; peeking does not', () => {
    const world = createWorld(SEED)
    const agent = world.agents['agent_0001']
    world.memories['memory_z'] = {
      id: 'memory_z',
      agentId: agent.id,
      sourceEventIds: [],
      summary: 'they shamed me at the market',
      kind: 'experience',
      importance: 0.7,
      emotionalCharge: 0.7,
      tone: 'ashamed',
      strength: 0.5,
      aboutAgentIds: ['agent_0004'],
      tags: ['social'],
      createdDay: 1,
      timesRecalled: 0,
    }
    agent.memoryIds.push('memory_z')

    const peeked = peekMemories(world, agent, { aboutAgentId: 'agent_0004' })
    expect(peeked.length).toBe(1)
    expect(world.memories['memory_z'].strength).toBe(0.5)
    expect(world.memories['memory_z'].timesRecalled).toBe(0)

    recallMemories(world, agent, { aboutAgentId: 'agent_0004' })
    expect(world.memories['memory_z'].strength).toBeCloseTo(0.58)
    expect(world.memories['memory_z'].timesRecalled).toBe(1)
    expect(world.memories['memory_z'].lastRecalledDay).toBe(world.day)
  })

  it('old wounds bias decisions: a remembered insult raises the urge to strike back', () => {
    const world = createWorld(SEED)
    const avenger = world.agents['agent_0001']
    const offender = world.agents['agent_0004']
    relationshipWith(avenger, offender.id).resentment = 0.4
    offender.locationId = 'place_well'

    const ctxBefore = contextFor(world, avenger, 'place_well')
    const before = insultAgent.candidates.call(insultAgent, ctxBefore)[0]

    world.memories['memory_w'] = {
      id: 'memory_w',
      agentId: avenger.id,
      sourceEventIds: [],
      summary: 'Odet insulted me in front of everyone at the market square',
      kind: 'experience',
      importance: 0.8,
      emotionalCharge: 0.8,
      tone: 'ashamed',
      strength: 0.9,
      aboutAgentIds: [offender.id],
      tags: ['social', 'conflict'],
      createdDay: 1,
      timesRecalled: 0,
    }
    avenger.memoryIds.push('memory_w')

    const ctxAfter = contextFor(world, avenger, 'place_well')
    const after = insultAgent.candidates.call(insultAgent, ctxAfter)[0]
    expect(after.score).toBeGreaterThan(before.score)
    expect(after.reasons.some((r) => r.includes('still remembers'))).toBe(true)
  })

  it('agents accumulate bounded, deterministic memories over real runs', () => {
    const a = createWorld(SEED)
    const b = createWorld(SEED)
    for (let d = 0; d < 25; d++) {
      stepDay(a)
      stepDay(b)
    }
    expect(a.memories).toEqual(b.memories)
    const counts = Object.values(a.agents).map((agent) => agent.memoryIds.length)
    expect(Math.max(...counts)).toBeLessThanOrEqual(36)
    expect(counts.reduce((x, y) => x + y, 0)).toBeGreaterThan(10)
    // Every memory traces back to a real logged event.
    for (const memory of Object.values(a.memories)) {
      for (const eventId of memory.sourceEventIds) {
        expect(a.events.find((e) => e.id === eventId)).toBeDefined()
      }
    }
  })
})

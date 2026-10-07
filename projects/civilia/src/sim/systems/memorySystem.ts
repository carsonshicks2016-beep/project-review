import type { Agent, Memory, MemoryKind, MemoryTone, SimEvent, World } from '../types'

/**
 * Memory v0 — not every event becomes a memory, and memories are not the
 * event log. Each one is a single agent's first-person, emotionally toned,
 * decaying record of something that mattered to them. Humiliations stick,
 * small talk evaporates, dwelling on a wound keeps it fresh, and what an
 * agent acts on is their memory of the town — not the truth of it.
 *
 * Formation and decay are deterministic; summaries are template-built
 * (qwen3:4b compression replaces the wording, not the data, at stage 30).
 */

const KEEP_THRESHOLD = 0.3
const FORGET_BELOW = 0.1
const MAX_MEMORIES_PER_AGENT = 36
const RECALL_BOOST = 0.08

interface Impression {
  importance: number
  tone: MemoryTone
  summary: string
  kind: MemoryKind
  emotionalCharge: number
}

const first = (name: string): string => name.split(' ')[0]

/**
 * How one participant experiences one event. Returns null when the moment
 * isn't memorable from that seat.
 */
function impressionOf(
  world: World,
  event: SimEvent,
  agent: Agent,
  role: 'actor' | 'target' | 'witness',
): Impression | null {
  const p = event.payload as Record<string, unknown>
  const actor = world.agents[event.actorIds[0]]
  const target = event.targetIds[0] ? world.agents[event.targetIds[0]] : undefined
  const place = event.locationId ? world.places[event.locationId] : undefined
  const placeName = place?.name ?? 'town'
  const publicly = event.witnessIds.length > 0

  switch (event.type) {
    case 'insulted': {
      if (role === 'target') {
        // Humiliations stick, and pride makes them stick harder.
        return {
          importance: 0.6 + (publicly ? 0.15 : 0) + agent.traits.pride * 0.15,
          tone: publicly ? 'ashamed' : 'angry',
          summary: `${first(actor.name)} insulted me${publicly ? ' in front of everyone' : ''} at ${placeName}`,
          kind: 'experience',
          emotionalCharge: 0.7 + (publicly ? 0.15 : 0),
        }
      }
      if (role === 'actor') {
        return {
          importance: 0.35,
          tone: 'bitter',
          summary: `I told ${first(target?.name ?? 'them')} exactly what I thought of them`,
          kind: 'experience',
          emotionalCharge: 0.45,
        }
      }
      return {
        importance: 0.35 + agent.traits.curiosity * 0.1,
        tone: 'worried',
        summary: `I watched ${first(actor.name)} tear into ${first(target?.name ?? 'someone')} at ${placeName}`,
        kind: 'witnessed',
        emotionalCharge: 0.4,
      }
    }
    case 'helped': {
      const gift = String(p.gift ?? 'helped me')
      if (role === 'target') {
        return {
          importance: 0.55 + agent.traits.empathy * 0.1,
          tone: 'warm',
          summary: `${first(actor.name)} ${gift} when I was going hungry`,
          kind: 'experience',
          emotionalCharge: 0.55,
        }
      }
      if (role === 'actor') {
        return {
          importance: 0.4,
          tone: 'proud',
          summary: `I helped ${first(target?.name ?? 'someone')} when they were going hungry`,
          kind: 'experience',
          emotionalCharge: 0.4,
        }
      }
      return {
        importance: 0.3,
        tone: 'warm',
        summary: `I saw ${first(actor.name)} help ${first(target?.name ?? 'someone')} at ${placeName}`,
        kind: 'witnessed',
        emotionalCharge: 0.3,
      }
    }
    case 'stole_item': {
      const caught = p.caught === true
      if (role === 'actor') {
        return {
          importance: caught ? 0.8 : 0.55,
          tone: caught ? 'ashamed' : 'afraid',
          summary: caught
            ? `I was caught stealing bread from ${placeName}`
            : `I took bread from ${placeName} and no one saw`,
          kind: 'experience',
          emotionalCharge: caught ? 0.85 : 0.6,
        }
      }
      // Witnesses: paranoid people never forget a thief.
      return {
        importance: 0.6 + agent.traits.paranoia * 0.15,
        tone: 'angry',
        summary: `I saw ${first(actor.name)} steal bread from ${placeName}`,
        kind: 'witnessed',
        emotionalCharge: 0.6,
      }
    }
    case 'apologized': {
      const accepted = p.accepted === true
      const restitution = Number(p.restitution ?? 0)
      if (role === 'actor') {
        return {
          importance: accepted ? 0.5 : 0.65 + agent.traits.pride * 0.15,
          tone: accepted ? 'warm' : 'ashamed',
          summary: accepted
            ? `I apologized to ${first(target?.name ?? 'them')} and they forgave me`
            : `I apologized to ${first(target?.name ?? 'them')} and they threw it back at me`,
          kind: 'experience',
          emotionalCharge: accepted ? 0.5 : 0.75,
        }
      }
      if (role === 'target') {
        return {
          importance: 0.5,
          tone: accepted ? 'warm' : 'proud',
          summary: accepted
            ? `${first(actor.name)} apologized to me${restitution ? ' with coins in hand' : ''}, and I let it go`
            : `${first(actor.name)} came to apologize and I sent them off`,
          kind: 'experience',
          emotionalCharge: 0.5,
        }
      }
      return null
    }
    case 'shared_rumor': {
      if (role !== 'target') return null
      const claim = String(p.claim ?? 'something')
      const about = p.rumorId ? world.rumors[String(p.rumorId)]?.aboutAgentId : undefined
      return {
        importance: 0.4 + (about ? 0.1 : 0),
        tone: 'worried',
        summary: `${first(actor.name)} told me: "${claim}"`,
        kind: 'hearsay',
        emotionalCharge: 0.45,
      }
    }
    case 'border_dispute': {
      if (role !== 'actor' && role !== 'target') return null
      const otherName = role === 'actor' ? target?.name : actor.name
      return {
        importance: 0.55 + agent.traits.pride * 0.1,
        tone: 'angry',
        summary: `I had words with ${first(otherName ?? 'a neighbor')} over the boundary near ${placeName}`,
        kind: 'experience',
        emotionalCharge: 0.55,
      }
    }
    case 'building_completed': {
      if (role !== 'actor') return null
      return {
        importance: 0.6,
        tone: 'proud',
        summary: `I drove the last nail into ${placeName}`,
        kind: 'experience',
        emotionalCharge: 0.55,
      }
    }
    case 'household_formed': {
      if (role !== 'actor') return null
      return {
        importance: 0.9,
        tone: 'proud',
        summary: `I moved out and made a home of my own`,
        kind: 'experience',
        emotionalCharge: 0.8,
      }
    }
    default:
      return null // work, meals, sleep, travel, small talk: lived, not remembered
  }
}

function rememberFor(world: World, agent: Agent, event: SimEvent, role: 'actor' | 'target' | 'witness'): void {
  const impression = impressionOf(world, event, agent, role)
  if (!impression || impression.importance < KEEP_THRESHOLD) return

  const about = new Set<string>()
  for (const id of [...event.actorIds, ...event.targetIds]) {
    if (id !== agent.id && world.agents[id]) about.add(id)
  }

  const memory: Memory = {
    id: `memory_${String(world.nextMemoryId).padStart(6, '0')}`,
    agentId: agent.id,
    sourceEventIds: [event.id],
    summary: impression.summary,
    kind: impression.kind,
    importance: Math.min(1, Math.round(impression.importance * 100) / 100),
    emotionalCharge: Math.min(1, impression.emotionalCharge),
    tone: impression.tone,
    strength: Math.min(1, 0.6 + impression.importance * 0.4),
    aboutAgentIds: [...about].sort(),
    locationId: event.locationId,
    tags: [...event.tags],
    createdDay: world.day,
    timesRecalled: 0,
  }
  world.nextMemoryId++
  world.memories[memory.id] = memory
  agent.memoryIds.push(memory.id)

  // A full head keeps what burns brightest and lets the rest go.
  if (agent.memoryIds.length > MAX_MEMORIES_PER_AGENT) {
    const weakest = [...agent.memoryIds].sort((a, b) => {
      const ma = world.memories[a]
      const mb = world.memories[b]
      return ma.strength * ma.importance - mb.strength * mb.importance
    })[0]
    forget(world, agent, weakest)
  }
}

function forget(world: World, agent: Agent, memoryId: string): void {
  agent.memoryIds = agent.memoryIds.filter((id) => id !== memoryId)
  delete world.memories[memoryId]
}

/** Form memories for everyone an event touched. Called after actions execute. */
export function processEventsIntoMemories(world: World, eventIds: string[]): void {
  for (const eventId of eventIds) {
    const event = world.events.find((e) => e.id === eventId)
    if (!event) continue
    const seen = new Set<string>()
    for (const id of event.actorIds) {
      const agent = world.agents[id]
      if (agent && !seen.has(id)) {
        rememberFor(world, agent, event, 'actor')
        seen.add(id)
      }
    }
    for (const id of event.targetIds) {
      const agent = world.agents[id]
      if (agent && !seen.has(id)) {
        rememberFor(world, agent, event, 'target')
        seen.add(id)
      }
    }
    for (const id of event.witnessIds) {
      const agent = world.agents[id]
      if (agent && !seen.has(id)) {
        rememberFor(world, agent, event, 'witness')
        seen.add(id)
      }
    }
  }
}

/**
 * Nightly fade. Important and emotionally hot memories resist; everything
 * else slides toward forgetting. Below the floor, the memory is gone.
 */
export function decayMemories(world: World): void {
  for (const agent of Object.values(world.agents)) {
    for (const memoryId of [...agent.memoryIds]) {
      const memory = world.memories[memoryId]
      if (!memory) continue
      const resistance = (1 - memory.importance * 0.5) * (1 - memory.emotionalCharge * 0.5)
      memory.strength = Math.round((memory.strength - 0.025 * resistance) * 1000) / 1000
      if (memory.strength < FORGET_BELOW) forget(world, agent, memoryId)
    }
  }
}

export interface MemoryQuery {
  aboutAgentId?: string
  tones?: MemoryTone[]
  tags?: string[]
  limit?: number
}

/** Read-only retrieval for decision scoring — never strengthens. */
export function peekMemories(world: World, agent: Agent, query: MemoryQuery): Memory[] {
  const results: { memory: Memory; score: number }[] = []
  for (const memoryId of agent.memoryIds) {
    const memory = world.memories[memoryId]
    if (!memory) continue
    if (query.aboutAgentId && !memory.aboutAgentIds.includes(query.aboutAgentId)) continue
    if (query.tones && !query.tones.includes(memory.tone)) continue
    if (query.tags && !query.tags.some((t) => memory.tags.includes(t))) continue
    const score = memory.strength * (memory.importance + memory.emotionalCharge) / 2
    results.push({ memory, score })
  }
  results.sort((a, b) => b.score - a.score || a.memory.id.localeCompare(b.memory.id))
  return results.slice(0, query.limit ?? 5).map((r) => r.memory)
}

/** Active recall: the agent dwells on it, which keeps the memory fresh. */
export function recallMemories(world: World, agent: Agent, query: MemoryQuery): Memory[] {
  const recalled = peekMemories(world, agent, query)
  for (const memory of recalled) {
    memory.strength = Math.min(1, Math.round((memory.strength + RECALL_BOOST) * 1000) / 1000)
    memory.lastRecalledDay = world.day
    memory.timesRecalled++
  }
  return recalled
}

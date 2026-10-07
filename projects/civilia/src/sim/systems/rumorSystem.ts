import type { Rng } from '../rng'
import type { Agent, Rumor, SimEvent, World } from '../types'
import { adjustRelationship } from '../relationships'
import { receiveRumorClaim } from './beliefSystem'

/**
 * Shallow Rumor River for Milestone 0: rumors are structured objects that
 * spread agent-to-agent and can mutate through fixed distortion templates.
 * Canonical truth (the event log) is never modified by a rumor.
 */

export function createRumor(
  world: World,
  opts: {
    claim: string
    aboutAgentId?: string
    sourceEventIds?: string[]
    emotionalCharge: number
    firstKnowerIds: string[]
    tags?: string[]
  },
): Rumor {
  const rumor: Rumor = {
    id: `rumor_${String(world.nextRumorId).padStart(4, '0')}`,
    sourceEventIds: opts.sourceEventIds ?? [],
    originalClaim: opts.claim,
    currentClaim: opts.claim,
    aboutAgentId: opts.aboutAgentId,
    variants: [],
    believerIds: [...opts.firstKnowerIds],
    skepticIds: [],
    truthDistance: 0,
    emotionalCharge: opts.emotionalCharge,
    createdDay: world.day,
    tags: opts.tags ?? [],
  }
  world.nextRumorId++
  world.rumors[rumor.id] = rumor
  for (const id of opts.firstKnowerIds) {
    const knower = world.agents[id]
    if (knower && !knower.knownRumorIds.includes(rumor.id)) knower.knownRumorIds.push(rumor.id)
  }
  return rumor
}

/** A theft was witnessed: the witnesses now carry a story about it. */
export function createCrimeRumor(
  world: World,
  thief: Agent,
  stealEvent: SimEvent,
  witnessIds: string[],
): Rumor {
  const place = stealEvent.locationId ? world.places[stealEvent.locationId] : undefined
  return createRumor(world, {
    claim: `${thief.name} stole bread from ${place?.name ?? 'somewhere in town'}`,
    aboutAgentId: thief.id,
    sourceEventIds: [stealEvent.id],
    emotionalCharge: 0.5,
    firstKnowerIds: witnessIds,
    tags: ['crime', 'bread'],
  })
}

interface MutationTemplate {
  render: (subjectName: string) => string
  truthDistanceDelta: number
  chargeDelta: number
  kind: string
}

const MUTATIONS: MutationTemplate[] = [
  {
    kind: 'exaggeration',
    render: (s) => `${s} has been stealing bread for weeks`,
    truthDistanceDelta: 0.15,
    chargeDelta: 0.1,
  },
  {
    kind: 'blame_shift',
    render: (s) => `${s} only steals because the bakery's prices are robbery`,
    truthDistanceDelta: 0.2,
    chargeDelta: 0.15,
  },
  {
    kind: 'conspiracy',
    render: (s) => `${s} is hoarding bread to sell it dear when the shelves run empty`,
    truthDistanceDelta: 0.25,
    chargeDelta: 0.2,
  },
  {
    kind: 'moral_panic',
    render: (s) => `${s} would steal from their own kin if the lock were loose`,
    truthDistanceDelta: 0.2,
    chargeDelta: 0.15,
  },
]

export interface SpreadResult {
  claim: string
  mutated: boolean
  mutationKind?: string
  listenerBelieves: boolean
  /** Listener's graded conviction after this telling, 0..1. */
  confidence: number
  /** Why the claim landed (or didn't) — for inspectors. */
  beliefFactors: string[]
}

/**
 * Speaker retells a rumor to a listener. May mutate the claim; updates the
 * listener's knowledge and their feelings about the rumor's subject.
 */
export function spreadRumor(
  world: World,
  rng: Rng,
  speaker: Agent,
  listener: Agent,
  rumor: Rumor,
): SpreadResult {
  let claim = rumor.currentClaim
  let mutated = false
  let mutationKind: string | undefined

  const subject = rumor.aboutAgentId ? world.agents[rumor.aboutAgentId] : undefined
  // Distortion is more likely from proud, paranoid, or unempathetic tellers.
  const mutationChance =
    0.2 + speaker.traits.pride * 0.15 + speaker.traits.paranoia * 0.15 + (1 - speaker.traits.empathy) * 0.1
  if (subject && rng.chance(mutationChance)) {
    const template = rng.pick(MUTATIONS)
    claim = template.render(subject.name)
    mutated = true
    mutationKind = template.kind
    rumor.currentClaim = claim
    rumor.truthDistance = Math.min(1, rumor.truthDistance + template.truthDistanceDelta)
    rumor.emotionalCharge = Math.min(1, rumor.emotionalCharge + template.chargeDelta)
    rumor.variants.push({ text: claim, byAgentId: speaker.id, day: world.day })
  }

  if (!listener.knownRumorIds.includes(rumor.id)) listener.knownRumorIds.push(rumor.id)

  // Graded belief update: source trust, disposition, corroboration, hunger.
  const reception = receiveRumorClaim(world, listener, speaker, rumor, claim)

  // Conviction sours the listener on the subject, in proportion to it.
  if (subject && subject.id !== listener.id && reception.belief.confidence >= 0.5) {
    adjustRelationship(listener, subject.id, {
      resentment: (0.05 + rumor.emotionalCharge * 0.08) * reception.belief.confidence,
      trust: -0.06 * reception.belief.confidence,
    })
  }

  return {
    claim,
    mutated,
    mutationKind,
    listenerBelieves: reception.believes,
    confidence: reception.belief.confidence,
    beliefFactors: reception.factors,
  }
}

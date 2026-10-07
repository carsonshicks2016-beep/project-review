import type { Agent, Belief, BeliefSource, Rumor, World } from '../types'
import { relationshipWith } from '../relationships'

/**
 * Belief v0 — conviction is graded, not binary. Seeing something makes a
 * near-certain belief; hearing it makes a belief whose confidence depends
 * on trust in the teller, what the listener already feels about the
 * subject, whether the world corroborates it (empty shelves make hoarding
 * stories ring true), and what they've witnessed themselves. Existing
 * beliefs have inertia: new tellings move them, but only partway.
 *
 * Canonical truth never depends on belief — `truthStatus` exists so the
 * inspector can show the gap between what's true and what's believed.
 */

const BELIEVER_THRESHOLD = 0.6
const SKEPTIC_THRESHOLD = 0.35
/** How far a new telling moves an existing belief toward the new evidence. */
const RECEPTIVITY = 0.45
/** Eyewitness conviction barely moves for hearsay. */
const WITNESSED_RECEPTIVITY = 0.12

interface HoldOptions {
  claim: string
  subjectId?: string
  rumorId?: string
  confidence: number
  source: BeliefSource
  truthStatus: Belief['truthStatus']
  emotionalCharge: number
  sourceEventIds?: string[]
  heardFromId?: string
  tags?: string[]
}

/** Find the holder's belief tracking a rumor (or exact claim). */
export function beliefAbout(world: World, holder: Agent, key: { rumorId?: string; claim?: string }): Belief | undefined {
  for (const id of holder.beliefIds) {
    const belief = world.beliefs[id]
    if (!belief) continue
    if (key.rumorId && belief.rumorId === key.rumorId) return belief
    if (key.claim && belief.claim === key.claim) return belief
  }
  return undefined
}

function createBelief(world: World, holder: Agent, opts: HoldOptions): Belief {
  const belief: Belief = {
    id: `belief_${String(world.nextBeliefId).padStart(6, '0')}`,
    holderId: holder.id,
    subjectId: opts.subjectId,
    claim: opts.claim,
    rumorId: opts.rumorId,
    confidence: clamp01(opts.confidence),
    source: opts.source,
    truthStatus: opts.truthStatus,
    emotionalCharge: clamp01(opts.emotionalCharge),
    sourceEventIds: opts.sourceEventIds ?? [],
    heardFromIds: opts.heardFromId ? [opts.heardFromId] : [],
    createdDay: world.day,
    lastUpdatedDay: world.day,
    tags: opts.tags ?? [],
  }
  world.nextBeliefId++
  world.beliefs[belief.id] = belief
  holder.beliefIds.push(belief.id)
  return belief
}

const clamp01 = (x: number): number => Math.max(0, Math.min(1, x))
const round3 = (x: number): number => Math.round(x * 1000) / 1000

/**
 * An eyewitness knows what they saw. Creates (or hardens) a near-certain
 * belief that later gossip can barely shake.
 */
export function believeFromWitness(world: World, witness: Agent, opts: Omit<HoldOptions, 'source' | 'confidence' | 'truthStatus'>): Belief {
  const existing = beliefAbout(world, witness, { rumorId: opts.rumorId, claim: opts.claim })
  if (existing) {
    existing.confidence = Math.max(existing.confidence, 0.95)
    existing.source = 'witnessed'
    existing.truthStatus = 'true'
    existing.lastUpdatedDay = world.day
    if (opts.sourceEventIds) {
      for (const id of opts.sourceEventIds) {
        if (!existing.sourceEventIds.includes(id)) existing.sourceEventIds.push(id)
      }
    }
    return existing
  }
  return createBelief(world, witness, {
    ...opts,
    source: 'witnessed',
    confidence: 0.95,
    truthStatus: 'true',
  })
}

/** A private hunch with no teller — the seed of every suspicion. */
export function suspect(world: World, holder: Agent, opts: Omit<HoldOptions, 'source'>): Belief {
  const existing = beliefAbout(world, holder, { rumorId: opts.rumorId, claim: opts.claim })
  if (existing) return existing
  return createBelief(world, holder, { ...opts, source: 'inference' })
}

export interface ReceptionResult {
  belief: Belief
  /** Evidence strength this telling carried, 0..1. */
  evidence: number
  /** Signed confidence movement this telling caused. */
  delta: number
  believes: boolean
  /** Human-readable factors for inspectors. */
  factors: string[]
}

function truthStatusFor(rumor: Rumor): Belief['truthStatus'] {
  if (rumor.truthDistance < 0.3) return 'true'
  if (rumor.truthDistance < 0.6) return 'unknown'
  return 'false'
}

/**
 * The spec's belief-update rule: a told claim lands with strength built
 * from source trust, prior feeling about the subject, hunger and shortage
 * when the story is about bread, and the listener's own eyewitness
 * corroboration — then moves the existing belief only partway (inertia).
 */
export function receiveRumorClaim(
  world: World,
  listener: Agent,
  speaker: Agent,
  rumor: Rumor,
  claim: string,
): ReceptionResult {
  const factors: string[] = []
  let evidence = 0.3
  factors.push('+30 a story half the town would tell')

  const trust = relationshipWith(listener, speaker.id).trust
  evidence += trust * 0.35
  factors.push(`+${Math.round(trust * 35)} trust in ${speaker.name.split(' ')[0]}`)

  const subject = rumor.aboutAgentId ? world.agents[rumor.aboutAgentId] : undefined
  if (subject) {
    const feeling = relationshipWith(listener, subject.id)
    if (feeling.resentment > 0) {
      evidence += feeling.resentment * 0.2
      factors.push(`+${Math.round(feeling.resentment * 20)} wants it true of ${subject.name.split(' ')[0]}`)
    }
    if (feeling.affection > 0) {
      evidence -= feeling.affection * 0.25
      factors.push(`-${Math.round(feeling.affection * 25)} can't picture ${subject.name.split(' ')[0]} doing it`)
    }
    // Their own eyes corroborate: an eyewitness belief about the same person.
    const corroborating = listener.beliefIds.some((id) => {
      const other = world.beliefs[id]
      return other && other.source === 'witnessed' && other.subjectId === subject.id && other.rumorId !== rumor.id
    })
    if (corroborating) {
      evidence += 0.25
      factors.push('+25 saw something like it with their own eyes')
    }
  }

  const breadStory = rumor.tags.some((t) => ['bread', 'grain', 'hoarding'].includes(t))
  if (breadStory) {
    if (listener.needs.hunger > 60) {
      evidence += 0.1
      factors.push('+10 hungry enough to believe it')
    }
    const bakery = world.places['place_bakery']
    if (bakery && bakery.foodStock <= 3) {
      evidence += 0.1
      factors.push('+10 the shelves really are bare')
    }
  }

  evidence = clamp01(evidence)

  let belief = beliefAbout(world, listener, { rumorId: rumor.id })
  let delta: number
  if (!belief) {
    belief = createBelief(world, listener, {
      claim,
      subjectId: rumor.aboutAgentId,
      rumorId: rumor.id,
      confidence: evidence,
      source: 'rumor',
      truthStatus: truthStatusFor(rumor),
      emotionalCharge: rumor.emotionalCharge,
      sourceEventIds: [...rumor.sourceEventIds],
      heardFromId: speaker.id,
      tags: [...rumor.tags],
    })
    delta = evidence
  } else {
    const receptivity = belief.source === 'witnessed' ? WITNESSED_RECEPTIVITY : RECEPTIVITY
    const before = belief.confidence
    belief.confidence = round3(clamp01(before + (evidence - before) * receptivity))
    delta = round3(belief.confidence - before)
    belief.claim = claim // they carry the latest version they were told
    belief.emotionalCharge = Math.max(belief.emotionalCharge, rumor.emotionalCharge)
    belief.truthStatus = belief.source === 'witnessed' ? belief.truthStatus : truthStatusFor(rumor)
    belief.lastUpdatedDay = world.day
    if (!belief.heardFromIds.includes(speaker.id)) belief.heardFromIds.push(speaker.id)
  }

  syncRumorConviction(rumor, listener.id, belief.confidence)
  return { belief, evidence: round3(evidence), delta, believes: belief.confidence >= BELIEVER_THRESHOLD, factors }
}

/** Keep the rumor's believer/skeptic rosters consistent with graded belief. */
export function syncRumorConviction(rumor: Rumor, agentId: string, confidence: number): void {
  const inBelievers = rumor.believerIds.includes(agentId)
  const inSkeptics = rumor.skepticIds.includes(agentId)
  if (confidence >= BELIEVER_THRESHOLD) {
    if (!inBelievers) rumor.believerIds.push(agentId)
    if (inSkeptics) rumor.skepticIds = rumor.skepticIds.filter((id) => id !== agentId)
  } else if (confidence <= SKEPTIC_THRESHOLD) {
    if (!inSkeptics) rumor.skepticIds.push(agentId)
    if (inBelievers) rumor.believerIds = rumor.believerIds.filter((id) => id !== agentId)
  } else {
    // Unsure: on neither roster.
    if (inBelievers) rumor.believerIds = rumor.believerIds.filter((id) => id !== agentId)
    if (inSkeptics) rumor.skepticIds = rumor.skepticIds.filter((id) => id !== agentId)
  }
}

/** Damaging things the holder believes about a subject, strongest first. */
export function damagingBeliefsAbout(world: World, holder: Agent, subjectId: string): Belief[] {
  return holder.beliefIds
    .map((id) => world.beliefs[id])
    .filter(
      (b): b is Belief =>
        !!b && b.subjectId === subjectId && b.confidence >= 0.55 && b.emotionalCharge >= 0.35,
    )
    .sort((a, b) => b.confidence * b.emotionalCharge - a.confidence * a.emotionalCharge)
}

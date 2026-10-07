import type { ActionCandidate, ActionDefinition } from '../types'
import {
  adjustRelationship,
  clampNeed,
  clampStatus,
  recordGrievance,
  relationshipWith,
} from '../relationships'
import { emitEvent } from '../systems/eventSystem'
import { spreadRumor } from '../systems/rumorSystem'
import { peekMemories, recallMemories } from '../systems/memorySystem'
import { damagingBeliefsAbout } from '../systems/beliefSystem'

const TALK_TOPICS = [
  'the harvest',
  'bread prices',
  'the weather turning',
  'old stories',
  'the state of the well',
  'tavern songs',
  'family matters',
  'who was seen with whom',
]

export const talk: ActionDefinition = {
  id: 'talk',

  candidates(ctx) {
    const { agent, othersHere, phase } = ctx
    const candidates: ActionCandidate[] = []
    for (const other of othersHere) {
      const rel = relationshipWith(agent, other.id)
      const reasons: string[] = []
      let score = agent.traits.sociability * 22
      reasons.push(`+${score.toFixed(0)} sociability`)

      const lonelinessBonus = (100 - agent.needs.belonging) * 0.25
      score += lonelinessBonus
      reasons.push(`+${lonelinessBonus.toFixed(0)} wants company`)

      const affectionBonus = rel.affection * 18
      score += affectionBonus
      reasons.push(`+${affectionBonus.toFixed(0)} likes ${other.name}`)

      const warmMemory = peekMemories(ctx.world, agent, {
        aboutAgentId: other.id,
        tones: ['warm'],
        limit: 1,
      })[0]
      if (warmMemory) {
        const memoryBonus = warmMemory.strength * 5
        score += memoryBonus
        reasons.push(`+${memoryBonus.toFixed(0)} good memories of them`)
      }

      const grudgePenalty = rel.resentment * 14
      score -= grudgePenalty
      if (grudgePenalty > 1) reasons.push(`-${grudgePenalty.toFixed(0)} resents ${other.name}`)

      if (phase === 'morning') {
        score -= 8
        reasons.push('-8 busy morning')
      }

      candidates.push({
        actionId: this.id,
        label: `talk with ${other.name}`,
        targetAgentId: other.id,
        score,
        reasons,
      })
    }
    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent, place, rng } = ctx
    const other = world.agents[candidate.targetAgentId!]
    const topic = rng.pick(TALK_TOPICS)

    agent.needs.belonging = clampNeed(agent.needs.belonging + 10)
    other.needs.belonging = clampNeed(other.needs.belonging + 8)
    adjustRelationship(agent, other.id, { trust: 0.02, affection: 0.03 })
    adjustRelationship(other, agent.id, { trust: 0.02, affection: 0.03 })

    const talked = emitEvent(world, {
      type: 'talked',
      locationId: place.id,
      actorIds: [agent.id],
      targetIds: [other.id],
      witnessIds: ctx.othersHere.filter((a) => a.id !== other.id).map((a) => a.id),
      payload: { topic },
      visibility: 'witnessed',
      tags: ['social'],
    })
    return [talked.id]
  },
}

export const gossip: ActionDefinition = {
  id: 'gossip',

  candidates(ctx) {
    const { agent, world, othersHere } = ctx
    if (agent.knownRumorIds.length === 0) return []
    const candidates: ActionCandidate[] = []

    for (const other of othersHere) {
      // Prefer telling someone the freshest rumor they haven't heard.
      const newRumorId = [...agent.knownRumorIds].reverse().find((id) => !other.knownRumorIds.includes(id))
      const rumorId = newRumorId ?? agent.knownRumorIds[agent.knownRumorIds.length - 1]
      const rumor = world.rumors[rumorId]
      if (!rumor) continue
      // No fun retelling someone a story about themselves.
      if (rumor.aboutAgentId === other.id) continue

      const reasons: string[] = []
      let score = agent.traits.sociability * 14 + agent.traits.curiosity * 10
      reasons.push(`+${score.toFixed(0)} loves to talk`)

      const chargeBonus = rumor.emotionalCharge * 25
      score += chargeBonus
      reasons.push(`+${chargeBonus.toFixed(0)} juicy claim`)

      if (newRumorId) {
        score += 8
        reasons.push(`+8 ${other.name} hasn't heard it`)
      } else {
        score -= 10
        reasons.push(`-10 ${other.name} already knows`)
      }

      candidates.push({
        actionId: this.id,
        label: `gossip with ${other.name}`,
        targetAgentId: other.id,
        score,
        reasons,
      })
    }
    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent, place, rng } = ctx
    const listener = world.agents[candidate.targetAgentId!]
    const rumorId =
      [...agent.knownRumorIds].reverse().find((id) => !listener.knownRumorIds.includes(id)) ??
      agent.knownRumorIds[agent.knownRumorIds.length - 1]
    const rumor = world.rumors[rumorId]

    const result = spreadRumor(world, rng, agent, listener, rumor)
    agent.needs.belonging = clampNeed(agent.needs.belonging + 8)
    listener.needs.belonging = clampNeed(listener.needs.belonging + 5)

    const shared = emitEvent(world, {
      type: 'shared_rumor',
      locationId: place.id,
      actorIds: [agent.id],
      targetIds: [listener.id],
      payload: {
        rumorId: rumor.id,
        claim: result.claim,
        mutated: result.mutated,
        mutationKind: result.mutationKind,
        listenerBelieves: result.listenerBelieves,
        confidence: result.confidence,
        beliefFactors: result.beliefFactors,
      },
      visibility: 'private',
      consequenceLevel: result.mutated ? 1 : 0,
      tags: ['social', 'rumor', ...rumor.tags],
    })
    return [shared.id]
  },
}

export const helpAgent: ActionDefinition = {
  id: 'help_agent',

  candidates(ctx) {
    const { agent, othersHere } = ctx
    const candidates: ActionCandidate[] = []
    const canGiveBread = agent.breadInventory >= 1 && agent.needs.hunger < 60
    const canGiveCoins = agent.money >= 4

    for (const other of othersHere) {
      if (other.needs.hunger < 60) continue
      if (!canGiveBread && !canGiveCoins) continue
      const rel = relationshipWith(agent, other.id)

      const reasons: string[] = []
      let score = agent.traits.empathy * 38
      reasons.push(`+${score.toFixed(0)} empathy`)

      const bondBonus = rel.affection * 20
      score += bondBonus
      reasons.push(`+${bondBonus.toFixed(0)} cares for ${other.name}`)

      const owedKindness = peekMemories(ctx.world, agent, {
        aboutAgentId: other.id,
        tones: ['warm'],
        limit: 1,
      })[0]
      if (owedKindness) {
        const memoryBonus = owedKindness.strength * 8
        score += memoryBonus
        reasons.push(`+${memoryBonus.toFixed(0)} remembers: "${owedKindness.summary}"`)
      }

      if (rel.kinship === 'household') {
        score += 10
        reasons.push('+10 family')
      }

      const pridePenalty = agent.traits.pride * 6
      score -= pridePenalty
      reasons.push(`-${pridePenalty.toFixed(0)} pride`)

      candidates.push({
        actionId: this.id,
        label: `help ${other.name}, who is going hungry`,
        targetAgentId: other.id,
        score,
        reasons,
      })
    }
    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent, place } = ctx
    const other = world.agents[candidate.targetAgentId!]
    let gift: string
    let favorValue: number
    if (agent.breadInventory >= 1 && agent.needs.hunger < 60) {
      agent.breadInventory--
      other.breadInventory++
      gift = 'gave them bread'
      favorValue = 2
    } else {
      agent.money -= 2
      other.money += 2
      gift = 'gave them 2 coins'
      favorValue = 2
    }
    // Family shares freely; between households, a gift is a favor owed.
    if (relationshipWith(agent, other.id).kinship !== 'household') {
      adjustRelationship(other, agent.id, { debt: favorValue })
    }

    adjustRelationship(other, agent.id, { affection: 0.1, trust: 0.08 })
    adjustRelationship(agent, other.id, { affection: 0.04 })
    agent.status = clampStatus(agent.status + 2)
    other.needs.belonging = clampNeed(other.needs.belonging + 6)

    const helped = emitEvent(world, {
      type: 'helped',
      locationId: place.id,
      actorIds: [agent.id],
      targetIds: [other.id],
      witnessIds: ctx.othersHere.filter((a) => a.id !== other.id).map((a) => a.id),
      payload: { gift },
      visibility: 'witnessed',
      consequenceLevel: 1,
      tags: ['social', 'kindness'],
    })
    return [helped.id]
  },
}

export const insultAgent: ActionDefinition = {
  id: 'insult_agent',

  candidates(ctx) {
    const { agent, othersHere } = ctx
    const candidates: ActionCandidate[] = []
    for (const other of othersHere) {
      const rel = relationshipWith(agent, other.id)
      if (rel.resentment < 0.3) continue

      const reasons: string[] = []
      let score = rel.resentment * 55
      reasons.push(`+${score.toFixed(0)} old grudge against ${other.name}`)

      const wound = peekMemories(ctx.world, agent, {
        aboutAgentId: other.id,
        tones: ['angry', 'ashamed', 'bitter'],
        limit: 1,
      })[0]
      if (wound) {
        const memoryBonus = wound.strength * 10
        score += memoryBonus
        reasons.push(`+${memoryBonus.toFixed(0)} still remembers: "${wound.summary}"`)
      }

      const conviction = damagingBeliefsAbout(ctx.world, agent, other.id)[0]
      if (conviction) {
        const beliefBonus = conviction.confidence * conviction.emotionalCharge * 14
        score += beliefBonus
        reasons.push(`+${beliefBonus.toFixed(0)} believes: "${conviction.claim}"`)
      }

      const prideBonus = agent.traits.pride * 12
      score += prideBonus
      reasons.push(`+${prideBonus.toFixed(0)} pride`)

      const empathyPenalty = agent.traits.empathy * 30
      score -= empathyPenalty
      reasons.push(`-${empathyPenalty.toFixed(0)} conscience`)

      const trustPenalty = rel.trust * 10
      score -= trustPenalty
      if (trustPenalty > 1) reasons.push(`-${trustPenalty.toFixed(0)} some remaining respect`)

      candidates.push({
        actionId: this.id,
        label: `insult ${other.name}`,
        targetAgentId: other.id,
        score,
        reasons,
      })
    }
    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent, place } = ctx
    const other = world.agents[candidate.targetAgentId!]
    const witnesses = ctx.othersHere.filter((a) => a.id !== other.id).map((a) => a.id)

    // Lashing out means dwelling on the wound, which keeps it fresh.
    recallMemories(world, agent, {
      aboutAgentId: other.id,
      tones: ['angry', 'ashamed', 'bitter'],
      limit: 1,
    })

    adjustRelationship(other, agent.id, { resentment: 0.15, affection: -0.08, trust: -0.06 })
    recordGrievance(other, agent.id, world.day)
    // Venting takes a little heat out of the grudge.
    adjustRelationship(agent, other.id, { resentment: -0.05 })
    if (witnesses.length > 0) {
      other.status = clampStatus(other.status - 2)
      agent.status = clampStatus(agent.status - 1)
    }

    const insulted = emitEvent(world, {
      type: 'insulted',
      locationId: place.id,
      actorIds: [agent.id],
      targetIds: [other.id],
      witnessIds: witnesses,
      payload: { publicly: witnesses.length > 0 },
      visibility: 'witnessed',
      consequenceLevel: witnesses.length > 0 ? 1 : 0,
      tags: ['social', 'conflict'],
    })
    return [insulted.id]
  },
}

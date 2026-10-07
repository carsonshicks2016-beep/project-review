import type { ActionDefinition } from '../types'
import { adjustRelationship, clampStatus, recordGrievance } from '../relationships'
import { emitEvent } from '../systems/eventSystem'
import { createCrimeRumor } from '../systems/rumorSystem'
import { believeFromWitness } from '../systems/beliefSystem'

export const stealItem: ActionDefinition = {
  id: 'steal_item',

  candidates(ctx) {
    const { agent, place } = ctx
    if (place.foodPrice <= 0 || place.foodStock <= 0) return []

    const desperate = agent.needs.hunger > 55 && agent.money < place.foodPrice
    const opportunist = agent.traits.riskTolerance > 0.7 && agent.needs.hunger > 40
    if (!desperate && !opportunist) return []

    const reasons: string[] = []
    let score = agent.needs.hunger * 0.7
    reasons.push(`+${score.toFixed(0)} hunger`)

    if (agent.money < place.foodPrice) {
      score += 18
      reasons.push('+18 cannot afford bread')
    }

    const nerveBonus = agent.traits.riskTolerance * 12
    score += nerveBonus
    reasons.push(`+${nerveBonus.toFixed(0)} nerve`)

    const conscience = agent.traits.empathy * 28
    score -= conscience
    reasons.push(`-${conscience.toFixed(0)} conscience`)

    const fearOfBeingSeen = agent.traits.paranoia * 12 + ctx.othersHere.length * 4
    score -= fearOfBeingSeen
    reasons.push(`-${fearOfBeingSeen.toFixed(0)} fear of watchers`)

    score -= 12
    reasons.push('-12 it is wrong and they know it')

    return [
      {
        actionId: this.id,
        label: `steal bread from ${place.name}`,
        targetPlaceId: place.id,
        score,
        reasons,
      },
    ]
  },

  execute(ctx) {
    const { world, agent, place, rng } = ctx
    place.foodStock--
    agent.breadInventory++

    // Who notices? Guards are trained to; the wary are watchful.
    const noticedBy = ctx.othersHere.filter((witness) => {
      let chance = 0.45 + witness.traits.paranoia * 0.25
      if (witness.jobId === 'guard') chance += 0.2
      return rng.chance(chance)
    })

    const stole = emitEvent(world, {
      type: 'stole_item',
      locationId: place.id,
      actorIds: [agent.id],
      witnessIds: noticedBy.map((a) => a.id),
      payload: { good: 'bread', caught: noticedBy.length > 0 },
      visibility: noticedBy.length > 0 ? 'witnessed' : 'private',
      consequenceLevel: noticedBy.length > 0 ? 2 : 1,
      tags: ['crime', 'bread'],
    })
    const eventIds = [stole.id]

    if (noticedBy.length > 0) {
      agent.status = clampStatus(agent.status - 4)
      for (const witness of noticedBy) {
        adjustRelationship(witness, agent.id, { trust: -0.2, resentment: 0.18 })
        recordGrievance(witness, agent.id, world.day)
        const saw = emitEvent(world, {
          type: 'witnessed_crime',
          locationId: place.id,
          actorIds: [witness.id],
          targetIds: [agent.id],
          payload: { crime: 'theft', good: 'bread' },
          visibility: 'private',
          consequenceLevel: 1,
          tags: ['crime'],
          parentEventIds: [stole.id],
        })
        eventIds.push(saw.id)
      }
      const rumor = createCrimeRumor(world, agent, stole, noticedBy.map((a) => a.id))
      // Eyewitnesses don't "believe" — they know. Gossip can barely move them.
      for (const witness of noticedBy) {
        believeFromWitness(world, witness, {
          claim: rumor.originalClaim,
          subjectId: agent.id,
          rumorId: rumor.id,
          emotionalCharge: 0.6,
          sourceEventIds: [stole.id],
          tags: ['crime', 'bread'],
        })
      }
    }

    return eventIds
  },
}

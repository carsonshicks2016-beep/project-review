import type { ActionDefinition } from '../types'
import { clampNeed } from '../relationships'
import { emitEvent } from '../systems/eventSystem'

const FATIGUE_RELIEF = 60

export const sleep: ActionDefinition = {
  id: 'sleep',

  candidates(ctx) {
    const { agent, phase, world } = ctx
    const home = world.households[agent.householdId]?.homeId
    const atHome = agent.locationId === home
    const exhausted = agent.needs.fatigue > 85

    // Normal sleep happens at home in the evening or at night; collapsing
    // from exhaustion can happen anywhere.
    if (!exhausted && !(atHome && (phase === 'evening' || phase === 'night'))) return []

    const reasons: string[] = []
    let score = agent.needs.fatigue * 0.7
    reasons.push(`+${score.toFixed(0)} fatigue`)
    if (phase === 'night') {
      score += 25
      reasons.push('+25 nighttime')
    }
    if (!atHome) {
      score -= 10
      reasons.push('-10 not at home')
    }

    return [
      {
        actionId: this.id,
        label: atHome ? 'sleep at home' : 'collapse and sleep where they stand',
        score,
        reasons,
      },
    ]
  },

  execute(ctx) {
    const { world, agent, place } = ctx
    agent.needs.fatigue = clampNeed(agent.needs.fatigue - FATIGUE_RELIEF)
    agent.sleptTonight = true
    const slept = emitEvent(world, {
      type: 'slept',
      locationId: place.id,
      actorIds: [agent.id],
      payload: {},
      visibility: 'private',
      tags: ['rest'],
    })
    return [slept.id]
  },
}

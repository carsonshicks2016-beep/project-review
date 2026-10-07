import type { ActionDefinition, JobId } from '../types'
import { clampNeed } from '../relationships'
import { emitEvent } from '../systems/eventSystem'

const WAGES: Record<JobId, number> = {
  farmer: 3,
  baker: 4,
  trader: 5,
  laborer: 2,
  guard: 3,
}

const WORK_PHASES = ['morning', 'midday', 'afternoon']
const MAX_SHIFTS_PER_DAY = 2
const BAKERY_BATCH = 6
const BAKERY_STOCK_CAP = 24

export const workShift: ActionDefinition = {
  id: 'work_shift',

  candidates(ctx) {
    const { agent, phase, place } = ctx
    if (!agent.jobId || !agent.workplaceId) return []
    if (!WORK_PHASES.includes(phase)) return []
    if (agent.locationId !== agent.workplaceId) return []
    if (agent.workedShiftsToday >= MAX_SHIFTS_PER_DAY) return []
    if (agent.needs.fatigue > 88) return []

    const reasons: string[] = []
    let score = 30
    reasons.push('+30 a shift is available')

    const ambitionBonus = agent.traits.ambition * 20
    score += ambitionBonus
    reasons.push(`+${ambitionBonus.toFixed(0)} ambition`)

    if (agent.money < 8) {
      score += 18
      reasons.push('+18 short on coin')
    }

    const fatiguePenalty = agent.needs.fatigue * 0.25
    score -= fatiguePenalty
    reasons.push(`-${fatiguePenalty.toFixed(0)} fatigue`)

    return [
      {
        actionId: this.id,
        label: `work a shift at ${place.name}`,
        score,
        reasons,
      },
    ]
  },

  execute(ctx) {
    const { world, agent, place } = ctx
    const wage = WAGES[agent.jobId as JobId]
    agent.workedShiftsToday++
    agent.needs.fatigue = clampNeed(agent.needs.fatigue + 14)
    agent.money += wage

    const worked = emitEvent(world, {
      type: 'worked',
      locationId: place.id,
      actorIds: [agent.id],
      payload: { job: agent.jobId },
      visibility: 'public',
      tags: ['work'],
    })
    const paid = emitEvent(world, {
      type: 'wage_paid',
      locationId: place.id,
      actorIds: [agent.id],
      payload: { amount: wage },
      visibility: 'private',
      tags: ['work', 'money'],
      parentEventIds: [worked.id],
    })
    const eventIds = [worked.id, paid.id]

    if (agent.jobId === 'baker' && place.kind === 'bakery') {
      // A stocked granary keeps the ovens running longer.
      let batchSize = BAKERY_BATCH
      const granaryPlace = Object.values(world.places).find(
        (p) => p.kind === 'granary' && p.buildingId && world.buildings[p.buildingId]?.state === 'complete',
      )
      if (granaryPlace && granaryPlace.grainStore >= 2) {
        granaryPlace.grainStore -= 2
        batchSize += 2
      }
      const batch = Math.min(batchSize, BAKERY_STOCK_CAP - place.foodStock)
      if (batch > 0) {
        place.foodStock += batch
        const produced = emitEvent(world, {
          type: 'produced_good',
          locationId: place.id,
          actorIds: [agent.id],
          payload: { good: 'bread', quantity: batch },
          visibility: 'public',
          tags: ['work', 'bread'],
          parentEventIds: [worked.id],
        })
        eventIds.push(produced.id)
      }
    } else if ((agent.jobId === 'farmer' || agent.jobId === 'laborer') && place.kind === 'farm') {
      place.grainStore += 2
      const produced = emitEvent(world, {
        type: 'produced_good',
        locationId: place.id,
        actorIds: [agent.id],
        payload: { good: 'grain', quantity: 2 },
        visibility: 'public',
        tags: ['work', 'grain'],
        parentEventIds: [worked.id],
      })
      eventIds.push(produced.id)
    }

    return eventIds
  },
}

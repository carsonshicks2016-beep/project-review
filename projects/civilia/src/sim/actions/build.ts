import type { ActionDefinition } from '../types'
import { clampNeed } from '../relationships'
import { emitEvent } from '../systems/eventSystem'
import {
  completeBuilding,
  gatherMaterials,
  materialsComplete,
} from '../systems/constructionSystem'

const WORK_PHASES = ['morning', 'midday', 'afternoon']
const PROGRESS_PER_SHIFT = 0.18
const BUILD_WAGE = 2

/**
 * Raise the building at the agent's location. Builders without a trade (or
 * laborers between farm shifts) gather materials first, then build. The
 * town pays a small levy wage per shift.
 */
export const build: ActionDefinition = {
  id: 'build',

  candidates(ctx) {
    const { agent, world, phase, place } = ctx
    if (!WORK_PHASES.includes(phase)) return []
    if (agent.jobId && agent.jobId !== 'laborer') return []
    if (agent.workedShiftsToday >= 2) return []
    if (agent.needs.fatigue > 85) return []
    const building = place.buildingId ? world.buildings[place.buildingId] : undefined
    if (!building || building.state === 'complete') return []

    const reasons: string[] = []
    let score = 26
    reasons.push('+26 the site needs hands')
    const ambition = agent.traits.ambition * 12
    score += ambition
    reasons.push(`+${ambition.toFixed(0)} ambition`)
    if (agent.money < 6) {
      score += 10
      reasons.push('+10 the levy wage helps')
    }
    const fatiguePenalty = agent.needs.fatigue * 0.2
    score -= fatiguePenalty
    reasons.push(`-${fatiguePenalty.toFixed(0)} fatigue`)

    const task = materialsComplete(building) ? 'raise' : 'gather materials for'
    return [
      {
        actionId: this.id,
        label: `${task} ${place.name}`,
        targetPlaceId: place.id,
        score,
        reasons,
      },
    ]
  },

  execute(ctx) {
    const { world, agent, place } = ctx
    const building = world.buildings[place.buildingId!]
    agent.workedShiftsToday++
    agent.needs.fatigue = clampNeed(agent.needs.fatigue + 12)
    agent.money += BUILD_WAGE

    if (!materialsComplete(building)) {
      if (building.state === 'planned') building.state = 'under_construction'
      const gathered = gatherMaterials(world, building)
      const total = Object.values(gathered).reduce((a, b) => a + b, 0)
      const event = emitEvent(world, {
        type: 'gathered_materials',
        locationId: place.id,
        actorIds: [agent.id],
        payload: { gathered, total, kind: building.kind },
        visibility: 'public',
        tags: ['construction', building.kind],
      })
      return [event.id]
    }

    building.progress = Math.min(1, building.progress + PROGRESS_PER_SHIFT)
    const progressed = emitEvent(world, {
      type: 'construction_progress',
      locationId: place.id,
      actorIds: [agent.id],
      payload: { progress: building.progress, kind: building.kind },
      visibility: 'public',
      tags: ['construction', building.kind],
    })
    const eventIds = [progressed.id]
    if (building.progress >= 1) {
      eventIds.push(...completeBuilding(world, building, agent.id))
    }
    return eventIds
  },
}

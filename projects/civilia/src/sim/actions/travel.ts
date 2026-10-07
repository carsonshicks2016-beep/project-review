import type { ActionCandidate, ActionDefinition } from '../types'
import { clampNeed } from '../relationships'
import { emitEvent } from '../systems/eventSystem'
import { findRoute, recordTravel } from '../movement'

const TRAVEL_COST = 8
const WORK_PHASES = ['morning', 'midday', 'afternoon']

/**
 * Travel candidates are generated per worthwhile destination; the utility of
 * being there (minus the cost of walking) competes with acting where you are.
 */
export const travelToPlace: ActionDefinition = {
  id: 'travel_to_place',

  candidates(ctx) {
    const { agent, world, phase } = ctx
    const candidates: ActionCandidate[] = []
    const home = world.households[agent.householdId]?.homeId

    const consider = (placeId: string, utility: number, why: string) => {
      if (placeId === agent.locationId) return
      const place = world.places[placeId]
      if (!place) return
      const score = utility - TRAVEL_COST
      if (score <= 0) return
      candidates.push({
        actionId: this.id,
        label: `travel to ${place.name}`,
        targetPlaceId: placeId,
        score,
        reasons: [`+${utility.toFixed(0)} ${why}`, `-${TRAVEL_COST} travel time`],
      })
    }

    // Work calls during work phases.
    if (agent.jobId && agent.workplaceId && WORK_PHASES.includes(phase) && agent.workedShiftsToday < 2) {
      let utility = 28 + agent.traits.ambition * 18
      if (agent.money < 8) utility += 15
      consider(agent.workplaceId, utility, 'a shift is waiting')
    }

    // Building sites call to anyone with free hands.
    if ((!agent.jobId || agent.jobId === 'laborer') && WORK_PHASES.includes(phase) && agent.workedShiftsToday < 2) {
      for (const building of Object.values(world.buildings)) {
        if (building.state === 'complete') continue
        let utility = 24 + agent.traits.ambition * 10
        if (agent.money < 6) utility += 10
        consider(building.placeId, utility, 'a building site needs hands')
      }
    }

    // Food calls when hungry with coins in pocket and nothing to eat.
    if (agent.needs.hunger > 45 && agent.breadInventory === 0) {
      for (const place of Object.values(world.places)) {
        if (place.foodPrice > 0 && place.foodStock > 0 && agent.money >= place.foodPrice) {
          consider(place.id, agent.needs.hunger * 0.55, 'food for sale there')
        }
      }
    }

    // The tavern calls in the evening.
    if (phase === 'evening') {
      let utility = agent.traits.sociability * 28
      if (agent.needs.belonging < 45) utility += 14
      consider('place_tavern', utility, 'company at the tavern')
    }

    // Home calls when tired or late.
    if ((phase === 'evening' || phase === 'night') && home) {
      consider(home, agent.needs.fatigue * 0.5 + 12, 'time to head home')
    }

    // Idle wandering keeps the town mixing.
    if (phase === 'midday' || phase === 'afternoon') {
      consider('place_well', 8 + agent.traits.curiosity * 8, 'errands at the well')
      consider('place_market', 6 + agent.traits.sociability * 10, 'people at the market')
    }

    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent } = ctx
    const fromId = agent.locationId
    const toId = candidate.targetPlaceId!
    const route = findRoute(world, world.places[fromId].tile, world.places[toId].tile)
    const distance = route ? route.length - 1 : 0
    if (route) recordTravel(world, route)
    agent.locationId = toId
    agent.needs.fatigue = clampNeed(agent.needs.fatigue + 2 + distance * 0.05)
    const traveled = emitEvent(world, {
      type: 'traveled',
      locationId: toId,
      actorIds: [agent.id],
      payload: { fromId, toId, distance },
      visibility: 'witnessed',
      tags: ['movement'],
    })
    return [traveled.id]
  },
}

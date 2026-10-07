import type { Agent, DayPhase, Place } from '../types'

/**
 * Indoor/outdoor occupancy v0. Whether an agent is inside follows from what
 * they just did and where: sleeping and indoor work happen inside, market
 * stalls and field work happen in the open, and people drift indoors at
 * home in the evening. The map renders outdoor agents directly and shows
 * indoor agents through their building.
 */

const INTERIOR_KINDS: Place['kind'][] = ['home', 'bakery', 'tavern', 'granary']

export function updateIndoors(
  agent: Agent,
  place: Place,
  actionId: string | null,
  phase: DayPhase,
): void {
  if (!INTERIOR_KINDS.includes(place.kind)) {
    agent.indoors = false
    return
  }
  if (actionId === 'sleep') {
    agent.indoors = true
    return
  }
  if (actionId === 'work_shift' || actionId === 'eat_food' || actionId === 'buy_food') {
    agent.indoors = true
    return
  }
  if (
    (actionId === null || actionId === 'idle') &&
    (phase === 'evening' || phase === 'night') &&
    place.kind === 'home'
  ) {
    agent.indoors = true
    return
  }
  agent.indoors = false
}

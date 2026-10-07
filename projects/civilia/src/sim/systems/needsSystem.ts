import type { World } from '../types'
import { clampNeed } from '../relationships'

/** Per-phase pressure: over a full day this is +36 hunger, +24 fatigue, -15 belonging. */
const HUNGER_PER_PHASE = 6
const FATIGUE_PER_PHASE = 4
const BELONGING_DECAY_PER_PHASE = 2.5

export function applyPhaseDrift(world: World): void {
  for (const agent of Object.values(world.agents)) {
    agent.needs.hunger = clampNeed(agent.needs.hunger + HUNGER_PER_PHASE)
    agent.needs.fatigue = clampNeed(agent.needs.fatigue + FATIGUE_PER_PHASE)
    agent.needs.belonging = clampNeed(agent.needs.belonging - BELONGING_DECAY_PER_PHASE)
  }
}

/** Dawn housekeeping: yesterday's shift counters and sleep flags reset. */
export function resetDaily(world: World): void {
  for (const agent of Object.values(world.agents)) {
    agent.workedShiftsToday = 0
    agent.sleptTonight = false
  }
}

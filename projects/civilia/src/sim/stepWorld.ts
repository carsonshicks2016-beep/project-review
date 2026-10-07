import type { World } from './types'
import { PHASES, currentPhase } from './types'
import { applyPhaseDrift, resetDaily } from './systems/needsSystem'
import { decideAndAct } from './systems/decisionSystem'
import { adjustFoodPrice } from './actions/food'
import { decayPathWear } from './movement'
import { applyDebtPressure } from './relationships'
import { decayMemories } from './systems/memorySystem'
import { granaryIntake, maybeProposeBuildings } from './systems/constructionSystem'
import { maybeBorderIncident, recomputeClaims } from './systems/claimsSystem'
import { ChronicleSystem } from './chronicle/chronicleSystem'
import { DEFAULT_CHRONICLE_CONFIG } from './chronicle/types'

const TAVERN_DAILY_STOCK = 8

/**
 * Advance the world by one phase. Agents act in id order; all randomness is
 * derived from the world seed, so the same seed replays the same history.
 */
export function stepPhase(world: World): void {
  const phase = currentPhase(world)
  if (phase === 'dawn') {
    resetDaily(world)
    recomputeClaims(world)
    maybeBorderIncident(world)
  }
  applyPhaseDrift(world)

  const agentIds = Object.keys(world.agents).sort()
  for (const id of agentIds) {
    decideAndAct(world, world.agents[id])
  }

  world.phaseIndex++
  if (world.phaseIndex >= PHASES.length) {
    world.phaseIndex = 0
    world.day++
    endOfDay(world)
  }
}

/** Advance to the next dawn (runs the remaining phases of the current day). */
export function stepDay(world: World): void {
  do {
    stepPhase(world)
  } while (world.phaseIndex !== 0)
}

function endOfDay(world: World): void {
  // Grass regrows a little on unused routes overnight.
  decayPathWear(world)

  // The town notices its needs and starts projects; the farm carts grain.
  maybeProposeBuildings(world)
  granaryIntake(world)

  // Unpaid favors quietly corrode creditors' patience overnight.
  applyDebtPressure(world)

  // Memories fade in the night; the unimportant ones are gone by morning.
  decayMemories(world)

  // The tavern lays in fresh meals overnight.
  const tavern = world.places['place_tavern']
  if (tavern && tavern.foodStock < TAVERN_DAILY_STOCK) {
    tavern.foodStock = TAVERN_DAILY_STOCK
    adjustFoodPrice(world, tavern, 3)
  }

  // Chronicle: check if the period boundary was crossed
  const chronicleReq = ChronicleSystem.maybeGenerate(
    world,
    world.chronicleConfig ?? DEFAULT_CHRONICLE_CONFIG,
  )
  if (chronicleReq) {
    // Store the request on the world for the UI/LLM layer to pick up
    world.pendingChronicle = chronicleReq
  }
}

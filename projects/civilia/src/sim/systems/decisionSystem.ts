import { rngFor } from '../rng'
import type { ActionContext, Agent, DecisionTrace, ScoredCandidate, World } from '../types'
import { currentPhase } from '../types'
import { ALL_ACTIONS } from '../actions'
import { updateIndoors } from './occupancySystem'
import { decorateWithSpeech } from './speechSystem'
import { processEventsIntoMemories } from './memorySystem'

const MAX_TRACES_PER_AGENT = 30
/** Agents pick among their best few options, not strictly the best one. */
const SELECTION_POOL = 5

/**
 * One agent takes one action: gather candidates from every action definition,
 * add seeded noise, pick by squared-score weighted randomness, execute, and
 * record a full decision trace.
 */
export function decideAndAct(world: World, agent: Agent): DecisionTrace {
  const phase = currentPhase(world)
  const rng = rngFor(world.seed, 'decide', world.day, phase, agent.id)
  const ctx: ActionContext = {
    world,
    agent,
    phase,
    othersHere: Object.values(world.agents).filter(
      (a) => a.id !== agent.id && a.locationId === agent.locationId,
    ),
    place: world.places[agent.locationId],
    rng,
  }

  const scored: ScoredCandidate[] = []
  for (const action of ALL_ACTIONS) {
    for (const candidate of action.candidates(ctx)) {
      const noise = rng.range(-7, 7)
      scored.push({ ...candidate, noise, finalScore: candidate.score + noise })
    }
  }
  // Doing nothing is always on the table, and wins when nothing presses.
  scored.push({
    actionId: 'idle',
    label: 'idle and watch the day go by',
    score: 4,
    noise: rng.range(-2, 2),
    finalScore: 4,
    reasons: ['+4 nothing urgent'],
  })

  scored.sort((a, b) => b.finalScore - a.finalScore)
  const pool = scored.filter((c) => c.finalScore > 0).slice(0, SELECTION_POOL)

  let chosenIndex = -1
  let eventIds: string[] = []
  if (pool.length > 0) {
    const weights = pool.map((c) => c.finalScore * c.finalScore)
    const chosen = rng.weightedPick(pool, weights)
    chosenIndex = scored.indexOf(chosen)
    if (chosen.actionId !== 'idle') {
      const action = ALL_ACTIONS.find((a) => a.id === chosen.actionId)!
      eventIds = action.execute(ctx, chosen)
      decorateWithSpeech(world, eventIds)
      processEventsIntoMemories(world, eventIds)
    }
  }

  const chosenActionId = chosenIndex >= 0 ? scored[chosenIndex].actionId : null
  updateIndoors(agent, world.places[agent.locationId], chosenActionId, phase)

  const trace: DecisionTrace = {
    id: `trace_${String(world.nextTraceId).padStart(6, '0')}`,
    agentId: agent.id,
    day: world.day,
    phase,
    locationId: ctx.place.id,
    candidates: scored,
    chosenIndex,
    eventIds,
  }
  world.nextTraceId++
  const traces = (world.decisionTraces[agent.id] ??= [])
  traces.push(trace)
  if (traces.length > MAX_TRACES_PER_AGENT) traces.shift()
  return trace
}

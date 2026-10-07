/**
 * Civilia core data model — Milestone 0.
 *
 * Canonical world truth lives in these structures. The event log is the spine:
 * every meaningful action appends a typed Event, and decision traces record
 * why each agent chose what it chose.
 */

import type { Rng } from './rng'
import type { ChronicleArticle, ChronicleConfig } from './chronicle/types'
import type { ChronicleGenerationRequest } from './chronicle/chronicleSystem'
import type { Terrain } from './terrain'

export const PHASES = ['dawn', 'morning', 'midday', 'afternoon', 'evening', 'night'] as const
export type DayPhase = (typeof PHASES)[number]

// ---------------------------------------------------------------------------
// Agents

export interface Needs {
  /** 0 = full, 100 = starving */
  hunger: number
  /** 0 = rested, 100 = exhausted */
  fatigue: number
  /** 0 = isolated, 100 = socially satisfied */
  belonging: number
}

/** All traits are 0..1. */
export interface PersonalityTraits {
  ambition: number
  empathy: number
  pride: number
  paranoia: number
  curiosity: number
  riskTolerance: number
  sociability: number
}

/** Directed relationship from one agent toward another. All dims 0..1. */
export interface Relationship {
  trust: number
  affection: number
  resentment: number
  /** Coins this agent owes the other (favor economy: gifts create debts). */
  debt: number
  /** Unresolved harms the other has done to this agent; apologies repair them. */
  grievances: number
  /** Day of the most recent harm — apologies land best while the wound is fresh. */
  lastHarmDay?: number
  kinship?: 'household'
}

export type JobId = 'farmer' | 'baker' | 'trader' | 'laborer' | 'guard'

export interface Agent {
  id: string
  name: string
  age: number
  householdId: string
  locationId: string
  /** True when the agent is inside the building at their location. */
  indoors: boolean
  jobId?: JobId
  /** Where this agent works, if employed. */
  workplaceId?: string
  traits: PersonalityTraits
  needs: Needs
  money: number
  /** 0..100 public standing in town. */
  status: number
  /** Units of bread carried. */
  breadInventory: number
  relationships: Record<string, Relationship>
  /** This agent's memories (ids into world.memories), oldest first. */
  memoryIds: string[]
  /** This agent's beliefs (ids into world.beliefs). */
  beliefIds: string[]
  /** Rumor IDs this agent has heard. */
  knownRumorIds: string[]
  /** Shifts worked today (capped per day). */
  workedShiftsToday: number
  /** True if the agent slept since last dawn. */
  sleptTonight: boolean
}

export interface Household {
  id: string
  name: string
  homeId: string
  memberIds: string[]
}

// ---------------------------------------------------------------------------
// Places

export type PlaceKind = 'home' | 'farm' | 'bakery' | 'market' | 'well' | 'tavern' | 'granary'

export interface Place {
  id: string
  name: string
  kind: PlaceKind
  /** Tile coordinates on the terrain grid. */
  tile: { x: number; y: number }
  /** The building standing (or rising) here, if any. */
  buildingId?: string
  /** Units of food for sale here (bread at the bakery, meals at the tavern). */
  foodStock: number
  /** Base price per unit of food; 0 means food is not sold here. */
  foodPrice: number
  /** Grain produced by farm work. Flavor counter for now. */
  grainStore: number
}

// ---------------------------------------------------------------------------
// Buildings — colonists raise these from local materials (stages 11/19/20)

export type MaterialId = 'timber' | 'stone' | 'clay' | 'reeds'

export type BuildingKind =
  | 'house'
  | 'cottage'
  | 'farmhouse'
  | 'bakery'
  | 'market_stalls'
  | 'well'
  | 'tavern'
  | 'granary'

export type BuildingState = 'planned' | 'under_construction' | 'complete'

export interface Building {
  id: string
  kind: BuildingKind
  placeId: string
  state: BuildingState
  /** 0..1, advances once materials are delivered. */
  progress: number
  materialsNeeded: Record<MaterialId, number>
  materialsDelivered: Record<MaterialId, number>
  ownerHouseholdId?: string
  proposedDay: number
  completedDay?: number
  /** Why the town wanted this building — construction is always motivated. */
  reason: string
}

// ---------------------------------------------------------------------------
// Events

export type EventType =
  | 'worked'
  | 'wage_paid'
  | 'produced_good'
  | 'bought_good'
  | 'sold_good'
  | 'ate_food'
  | 'slept'
  | 'traveled'
  | 'talked'
  | 'shared_rumor'
  | 'helped'
  | 'insulted'
  | 'apologized'
  | 'debt_repaid'
  | 'stole_item'
  | 'witnessed_crime'
  | 'price_changed'
  | 'chronicle_generated'
  | 'building_proposed'
  | 'gathered_materials'
  | 'construction_progress'
  | 'building_completed'
  | 'household_formed'
  | 'border_dispute'

export type EventVisibility = 'private' | 'witnessed' | 'public'

export interface SimEvent {
  id: string
  type: EventType
  day: number
  phase: DayPhase
  locationId?: string
  actorIds: string[]
  targetIds: string[]
  witnessIds: string[]
  payload: Record<string, unknown>
  /** 0 mundane .. 3 town-shaking. Used later for Chronicle selection. */
  consequenceLevel: number
  visibility: EventVisibility
  tags: string[]
  parentEventIds: string[]
}

// ---------------------------------------------------------------------------
// Memories — each agent's personal, fallible record of what happened.
// The event log stays canonical; memories are what agents act on.

export type MemoryKind = 'experience' | 'witnessed' | 'hearsay'

export type MemoryTone =
  | 'warm'
  | 'proud'
  | 'angry'
  | 'ashamed'
  | 'afraid'
  | 'worried'
  | 'bitter'

export interface Memory {
  id: string
  agentId: string
  sourceEventIds: string[]
  /** First-person, template-built summary (qwen3:4b compression at stage 30). */
  summary: string
  kind: MemoryKind
  /** 0..1 — how much this mattered to the agent; gates retention and recall. */
  importance: number
  /** 0..1 — feeling intensity at formation; hot memories resist decay. */
  emotionalCharge: number
  tone: MemoryTone
  /** 0..1 — fades nightly, strengthens when recalled; forgotten below ~0.1. */
  strength: number
  aboutAgentIds: string[]
  locationId?: string
  tags: string[]
  createdDay: number
  lastRecalledDay?: number
  timesRecalled: number
}

// ---------------------------------------------------------------------------
// Beliefs — graded conviction about claims. What an agent saw, what they
// were told, and what they merely suspect carry different weight, and
// none of it has to match canonical truth.

export type BeliefSource = 'witnessed' | 'rumor' | 'inference'

export interface Belief {
  id: string
  holderId: string
  /** Who the claim is about, when it concerns a person. */
  subjectId?: string
  /** The version of the claim the holder currently carries. */
  claim: string
  /** The rumor this belief tracks, when it arrived as gossip. */
  rumorId?: string
  /** 0..1 — how sure the holder is. Not binary, per the spec. */
  confidence: number
  source: BeliefSource
  /** Canonical check for the debug inspector: did this actually happen? */
  truthStatus: 'true' | 'false' | 'unknown'
  emotionalCharge: number
  sourceEventIds: string[]
  /** Everyone who has told them some version of this claim. */
  heardFromIds: string[]
  createdDay: number
  lastUpdatedDay: number
  tags: string[]
}

// ---------------------------------------------------------------------------
// Rumors (shallow placeholder for the Rumor River)

export interface RumorVariant {
  text: string
  byAgentId: string
  day: number
}

export interface Rumor {
  id: string
  sourceEventIds: string[]
  /** The claim as first formed. */
  originalClaim: string
  /** The claim as most recently retold. */
  currentClaim: string
  aboutAgentId?: string
  variants: RumorVariant[]
  believerIds: string[]
  skepticIds: string[]
  /** 0..1, how far retellings have drifted from the source event. */
  truthDistance: number
  /** 0..1, how juicy/alarming the claim is. */
  emotionalCharge: number
  createdDay: number
  tags: string[]
}

// ---------------------------------------------------------------------------
// Decisions

export interface ScoredCandidate {
  actionId: string
  /** Human-readable summary, e.g. "travel to the bakery". */
  label: string
  targetAgentId?: string
  targetPlaceId?: string
  /** Utility before noise. */
  score: number
  /** Seeded noise added for this decision. */
  noise: number
  /** score + noise, used for selection. */
  finalScore: number
  /** Why the score is what it is, e.g. "+38 hunger pressure". */
  reasons: string[]
}

export interface DecisionTrace {
  id: string
  agentId: string
  day: number
  phase: DayPhase
  locationId: string
  candidates: ScoredCandidate[]
  /** Index into candidates of the chosen action. -1 if the agent idled. */
  chosenIndex: number
  /** IDs of events emitted by executing the chosen action. */
  eventIds: string[]
}

// ---------------------------------------------------------------------------
// World

export interface World {
  id: string
  seed: string
  name: string
  day: number
  /** Index into PHASES for the phase about to execute. */
  phaseIndex: number
  /** Seeded landscape: biomes, water, materials, landmarks, town site. */
  terrain: Terrain
  /**
   * Accumulated foot traffic per tile (key "x,y"). Worn routes become
   * trails, paths, and roads; wear decays slowly when unused.
   */
  pathWear: Record<string, number>
  agents: Record<string, Agent>
  places: Record<string, Place>
  households: Record<string, Household>
  buildings: Record<string, Building>
  /** Append-only event log. */
  events: SimEvent[]
  /** All living memories, keyed by id; agents reference them via memoryIds. */
  memories: Record<string, Memory>
  /** All beliefs, keyed by id; agents reference them via beliefIds. */
  beliefs: Record<string, Belief>
  rumors: Record<string, Rumor>
  /** Recent decision traces per agent (newest last, capped). */
  decisionTraces: Record<string, DecisionTrace[]>
  nextEventId: number
  nextRumorId: number
  nextTraceId: number
  nextBuildingId: number
  nextMemoryId: number
  nextBeliefId: number
  /**
   * Territorial claims, recomputed each dawn from building ownership.
   * Derived state — but border incidents it triggers are real events.
   */
  claims?: import('./systems/claimsSystem').ClaimsState
  /** Chronicle configuration for newspaper generation cadence. */
  chronicleConfig?: ChronicleConfig
  /** Pending chronicle request for the UI/LLM layer to pick up. */
  pendingChronicle?: ChronicleGenerationRequest
  /** Published chronicle articles. */
  chronicleArticles: ChronicleArticle[]
}

export function currentPhase(world: World): DayPhase {
  return PHASES[world.phaseIndex]
}

// ---------------------------------------------------------------------------
// Action plumbing

/** Everything an action needs to evaluate and execute for one agent. */
export interface ActionContext {
  world: World
  agent: Agent
  phase: DayPhase
  /** Other agents at the same location. */
  othersHere: Agent[]
  place: Place
  rng: Rng
}

export interface ActionCandidate {
  actionId: string
  label: string
  targetAgentId?: string
  targetPlaceId?: string
  score: number
  reasons: string[]
}

export interface ActionDefinition {
  id: string
  /** Produce zero or more scored candidates for this agent this phase. */
  candidates(ctx: ActionContext): ActionCandidate[]
  /** Apply effects and emit events. Returns emitted event IDs. */
  execute(ctx: ActionContext, candidate: ActionCandidate): string[]
}

import type { DayPhase, EventType, EventVisibility, SimEvent, World } from '../types'
import { currentPhase } from '../types'

export interface EmitOptions {
  type: EventType
  locationId?: string
  actorIds?: string[]
  targetIds?: string[]
  witnessIds?: string[]
  payload?: Record<string, unknown>
  consequenceLevel?: number
  visibility?: EventVisibility
  tags?: string[]
  parentEventIds?: string[]
}

/** Append a typed event to the world's append-only log and return it. */
export function emitEvent(world: World, opts: EmitOptions): SimEvent {
  const event: SimEvent = {
    id: `event_${String(world.nextEventId).padStart(6, '0')}`,
    type: opts.type,
    day: world.day,
    phase: currentPhase(world),
    locationId: opts.locationId,
    actorIds: opts.actorIds ?? [],
    targetIds: opts.targetIds ?? [],
    witnessIds: opts.witnessIds ?? [],
    payload: opts.payload ?? {},
    consequenceLevel: opts.consequenceLevel ?? 0,
    visibility: opts.visibility ?? 'witnessed',
    tags: opts.tags ?? [],
    parentEventIds: opts.parentEventIds ?? [],
  }
  world.nextEventId++
  world.events.push(event)
  return event
}

/** Human-readable one-liner for the timeline UI and headless logs. */
export function describeEvent(world: World, event: SimEvent): string {
  const name = (id?: string) => (id && world.agents[id] ? world.agents[id].name : id ?? '?')
  const placeName = (id?: string) => (id && world.places[id] ? world.places[id].name : id ?? '?')
  const actor = name(event.actorIds[0])
  const target = name(event.targetIds[0])
  const p = event.payload as Record<string, any>

  switch (event.type) {
    case 'worked':
      return `${actor} worked a shift at ${placeName(event.locationId)}`
    case 'wage_paid':
      return `${actor} was paid ${p.amount} coins`
    case 'produced_good':
      return `${actor} produced ${p.quantity} ${p.good} at ${placeName(event.locationId)}`
    case 'bought_good':
      return `${actor} bought ${p.good} for ${p.price} coins at ${placeName(event.locationId)}`
    case 'sold_good':
      return `${placeName(event.locationId)} sold ${p.good} to ${actor}`
    case 'ate_food':
      return `${actor} ate ${p.good ?? 'bread'}${event.locationId ? ` at ${placeName(event.locationId)}` : ''}`
    case 'slept':
      return `${actor} slept at ${placeName(event.locationId)}`
    case 'traveled':
      return `${actor} went from ${placeName(p.fromId)} to ${placeName(p.toId)}`
    case 'talked':
      return `${actor} talked with ${target} about ${p.topic}`
    case 'shared_rumor':
      return `${actor} whispered to ${target}: "${p.claim}"`
    case 'helped':
      return `${actor} helped ${target} (${p.gift})`
    case 'insulted':
      return `${actor} insulted ${target}${event.witnessIds.length > 0 ? ' in front of others' : ''}`
    case 'apologized':
      return `${actor} apologized to ${target}${p.restitution ? ` with ${p.restitution} coins in hand` : ''} — ${p.accepted ? 'and was forgiven' : 'and was rebuffed'}`
    case 'debt_repaid':
      return `${actor} repaid ${p.amount} coins to ${target}`
    case 'stole_item':
      return `${actor} stole ${p.good} from ${placeName(event.locationId)}`
    case 'witnessed_crime':
      return `${actor} saw ${target} stealing at ${placeName(event.locationId)}`
    case 'price_changed':
      return `${placeName(event.locationId)} now charges ${p.newPrice} coins for ${p.good}`
    case 'building_proposed':
      return `the town planned a ${p.kind} at ${placeName(event.locationId)} — ${p.reason}`
    case 'gathered_materials':
      return `${actor} hauled ${p.total} loads of material to ${placeName(event.locationId)}`
    case 'construction_progress':
      return `${actor} worked on ${placeName(event.locationId)} (${Math.round((p.progress ?? 0) * 100)}% raised)`
    case 'building_completed':
      return `${placeName(event.locationId)} is finished — ${p.reason}`
    case 'household_formed':
      return `${actor} moved out and founded ${p.householdName}`
    case 'border_dispute':
      return `${actor} confronted ${target} over the boundary near ${placeName(event.locationId)}`
    default: {
      const t: string = event.type
      return `${actor}: ${t}`
    }
  }
}

export function phaseLabel(day: number, phase: DayPhase): string {
  return `Day ${day}, ${phase}`
}

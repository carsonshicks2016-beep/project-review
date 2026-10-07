import type {
  Building,
  BuildingKind,
  Household,
  MaterialId,
  Place,
  World,
} from '../types'
import { emitEvent } from './eventSystem'
import { findBuildSite, idx } from '../terrain'

/**
 * Construction v0 — buildings are always motivated. The town proposes a
 * project when a need shows up in the event log (shortage, crowding), sites
 * it on buildable ground, and colonists with free hands gather local
 * materials and raise it shift by shift. Completion changes the town:
 * a granary buffers grain, a cottage spins off a new household.
 */

export const MATERIALS: MaterialId[] = ['timber', 'stone', 'clay', 'reeds']

const RECIPES: Partial<Record<BuildingKind, Record<MaterialId, number>>> = {
  cottage: { timber: 10, stone: 0, clay: 4, reeds: 4 },
  granary: { timber: 8, stone: 5, clay: 3, reeds: 0 },
}

const MAX_ACTIVE_PROJECTS = 2

const zeroMaterials = (): Record<MaterialId, number> => ({
  timber: 0,
  stone: 0,
  clay: 0,
  reeds: 0,
})

/** Found an already-standing building (world generation only). */
export function foundBuilding(
  world: World,
  kind: BuildingKind,
  placeId: string,
  ownerHouseholdId?: string,
): Building {
  const building: Building = {
    id: `building_${String(world.nextBuildingId).padStart(4, '0')}`,
    kind,
    placeId,
    state: 'complete',
    progress: 1,
    materialsNeeded: zeroMaterials(),
    materialsDelivered: zeroMaterials(),
    ownerHouseholdId,
    proposedDay: 0,
    completedDay: 0,
    reason: 'standing since the founding',
  }
  world.nextBuildingId++
  world.buildings[building.id] = building
  world.places[placeId].buildingId = building.id
  return building
}

function occupiedTiles(world: World): Set<number> {
  const occupied = new Set<number>()
  for (const place of Object.values(world.places)) {
    occupied.add(idx(place.tile.x, place.tile.y))
  }
  return occupied
}

/**
 * Builders use what the land offers: any material that can't be found near
 * the site is substituted with timber (and timber itself is clamped to what
 * actually stands within hauling distance).
 */
function adaptRecipe(
  world: World,
  site: { x: number; y: number },
  recipe: Record<MaterialId, number>,
): Record<MaterialId, number> {
  const available = zeroMaterials()
  for (let dy = -18; dy <= 18; dy++) {
    for (let dx = -18; dx <= 18; dx++) {
      const x = site.x + dx
      const y = site.y + dy
      if (x < 0 || y < 0 || x >= world.terrain.width || y >= world.terrain.height) continue
      const i = idx(x, y)
      for (const m of MATERIALS) available[m] += world.terrain[m][i]
    }
  }
  const adapted = { ...recipe }
  for (const m of MATERIALS) {
    if (m === 'timber') continue
    if (available[m] < adapted[m]) {
      adapted.timber += adapted[m] - Math.max(0, available[m])
      adapted[m] = Math.max(0, Math.min(adapted[m], available[m]))
    }
  }
  adapted.timber = Math.min(adapted.timber, Math.max(4, available.timber))
  return adapted
}

function proposeBuilding(
  world: World,
  kind: BuildingKind,
  place: Place,
  reason: string,
  ownerHouseholdId?: string,
): Building {
  const recipe = { ...zeroMaterials(), ...RECIPES[kind] }
  const building: Building = {
    id: `building_${String(world.nextBuildingId).padStart(4, '0')}`,
    kind,
    placeId: place.id,
    state: 'planned',
    progress: 0,
    materialsNeeded: adaptRecipe(world, place.tile, recipe),
    materialsDelivered: zeroMaterials(),
    ownerHouseholdId,
    proposedDay: world.day,
    reason,
  }
  world.nextBuildingId++
  world.buildings[building.id] = building
  world.places[place.id] = place
  place.buildingId = building.id

  emitEvent(world, {
    type: 'building_proposed',
    locationId: place.id,
    payload: { kind, reason, ownerHouseholdId },
    visibility: 'public',
    consequenceLevel: 1,
    tags: ['construction', kind],
  })
  return building
}

/** Persistent bread shortages convince the town it needs a granary. */
function wantsGranary(world: World): boolean {
  if (Object.values(world.buildings).some((b) => b.kind === 'granary')) return false
  const recentShortages = world.events.filter(
    (e) =>
      e.type === 'price_changed' &&
      e.day > world.day - 7 &&
      (e.payload as { reason?: string }).reason === 'shortage',
  )
  return recentShortages.length >= 2
}

/** A crowded, solvent household wants a cottage for its grown child. */
function crowdedHousehold(world: World): Household | null {
  if (world.day < 5) return null
  for (const household of Object.values(world.households)) {
    if (household.memberIds.length < 3) continue
    const alreadyBuilding = Object.values(world.buildings).some(
      (b) => b.kind === 'cottage' && b.ownerHouseholdId === household.id,
    )
    if (alreadyBuilding) continue
    const wealth = household.memberIds.reduce((sum, id) => sum + world.agents[id].money, 0)
    if (wealth >= 20) return household
  }
  return null
}

/** End-of-day check: does the town need to start a project? */
export function maybeProposeBuildings(world: World): void {
  const active = Object.values(world.buildings).filter((b) => b.state !== 'complete').length
  if (active >= MAX_ACTIVE_PROJECTS) return

  if (wantsGranary(world)) {
    const bakery = world.places['place_bakery']
    const site = findBuildSite(world.terrain, occupiedTiles(world), bakery.tile.x + 1, bakery.tile.y + 1)
    const place: Place = {
      id: 'place_granary',
      name: 'the town granary',
      kind: 'granary',
      tile: site,
      foodStock: 0,
      foodPrice: 0,
      grainStore: 0,
    }
    world.places[place.id] = place
    proposeBuilding(world, 'granary', place, 'bread keeps running short; the town wants a grain store')
    return
  }

  const crowded = crowdedHousehold(world)
  if (crowded) {
    const home = world.places[crowded.homeId]
    const site = findBuildSite(world.terrain, occupiedTiles(world), home.tile.x + 2, home.tile.y + 2)
    const mover = youngestAdult(world, crowded)
    const place: Place = {
      id: `place_cottage_${crowded.name.toLowerCase()}`,
      name: `${firstName(mover?.name ?? crowded.name)}'s cottage`,
      kind: 'home',
      tile: site,
      foodStock: 0,
      foodPrice: 0,
      grainStore: 0,
    }
    world.places[place.id] = place
    proposeBuilding(
      world,
      'cottage',
      place,
      `the ${crowded.name} home is crowded; ${firstName(mover?.name ?? '')} wants a place of their own`,
      crowded.id,
    )
  }
}

function youngestAdult(world: World, household: Household) {
  const adults = household.memberIds
    .map((id) => world.agents[id])
    .filter((a) => a.age >= 15)
    .sort((a, b) => a.age - b.age || a.id.localeCompare(b.id))
  return adults[0]
}

const firstName = (full: string): string => full.split(' ')[0]

/**
 * Gather up to `capacity` units of still-needed materials from the tiles
 * nearest the site (deterministic outward ring scan). Depletes the terrain.
 */
export function gatherMaterials(world: World, building: Building, capacity = 4): Record<MaterialId, number> {
  const site = world.places[building.placeId].tile
  const gathered = zeroMaterials()
  let remaining = capacity

  for (const material of MATERIALS) {
    if (remaining <= 0) break
    let needed = building.materialsNeeded[material] - building.materialsDelivered[material]
    if (needed <= 0) continue
    const supply = world.terrain[material]
    for (let r = 0; r <= 18 && needed > 0 && remaining > 0; r++) {
      for (let dy = -r; dy <= r && needed > 0 && remaining > 0; dy++) {
        for (let dx = -r; dx <= r && needed > 0 && remaining > 0; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
          const x = site.x + dx
          const y = site.y + dy
          if (x < 0 || y < 0 || x >= world.terrain.width || y >= world.terrain.height) continue
          const i = idx(x, y)
          if (supply[i] <= 0) continue
          const take = Math.min(supply[i], needed, remaining)
          supply[i] -= take
          building.materialsDelivered[material] += take
          gathered[material] += take
          needed -= take
          remaining -= take
        }
      }
    }
  }
  return gathered
}

export function materialsComplete(building: Building): boolean {
  return MATERIALS.every(
    (m) => building.materialsDelivered[m] >= building.materialsNeeded[m],
  )
}

/** Mark a building finished and apply what it changes about the town. */
export function completeBuilding(world: World, building: Building, builderId: string): string[] {
  building.state = 'complete'
  building.progress = 1
  building.completedDay = world.day

  const completed = emitEvent(world, {
    type: 'building_completed',
    locationId: building.placeId,
    actorIds: [builderId],
    payload: { kind: building.kind, reason: building.reason },
    visibility: 'public',
    consequenceLevel: 2,
    tags: ['construction', building.kind],
  })
  const eventIds = [completed.id]

  // A finished cottage spins off a new household: the youngest adult moves out.
  if (building.kind === 'cottage' && building.ownerHouseholdId) {
    const parent = world.households[building.ownerHouseholdId]
    const mover = youngestAdult(world, parent)
    if (mover && parent.memberIds.length > 1) {
      parent.memberIds = parent.memberIds.filter((id) => id !== mover.id)
      const household: Household = {
        id: `household_${building.id}`,
        name: `${firstName(mover.name)}'s cottage`,
        homeId: building.placeId,
        memberIds: [mover.id],
      }
      world.households[household.id] = household
      mover.householdId = household.id
      building.ownerHouseholdId = household.id
      const formed = emitEvent(world, {
        type: 'household_formed',
        locationId: building.placeId,
        actorIds: [mover.id],
        payload: { householdName: household.name, parentHousehold: parent.name },
        visibility: 'public',
        consequenceLevel: 2,
        tags: ['family', 'construction'],
        parentEventIds: [completed.id],
      })
      eventIds.push(formed.id)
    }
  }
  return eventIds
}

/** Nightly grain flow: the farm carts surplus into a finished granary. */
export function granaryIntake(world: World): void {
  const granary = Object.values(world.buildings).find(
    (b) => b.kind === 'granary' && b.state === 'complete',
  )
  if (!granary) return
  const farm = world.places['place_farm']
  const store = world.places[granary.placeId]
  const transfer = Math.min(3, farm.grainStore)
  if (transfer > 0) {
    farm.grainStore -= transfer
    store.grainStore += transfer
  }
}

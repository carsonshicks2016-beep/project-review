import { rngFor } from './rng'
import { createRumor } from './systems/rumorSystem'
import { suspect, syncRumorConviction } from './systems/beliefSystem'
import { foundBuilding } from './systems/constructionSystem'
import { findBuildSite, generateTerrain, idx } from './terrain'
import type { Terrain } from './terrain'
import type {
  Agent,
  BuildingKind,
  Household,
  JobId,
  PersonalityTraits,
  Place,
  Relationship,
  World,
} from './types'

interface AgentSpec {
  name: string
  age: number
  jobId?: JobId
  workplaceId?: string
  startingMoney: number
}

interface HouseholdSpec {
  name: string
  members: AgentSpec[]
}

/**
 * Ashvale's founding roster. Fixed by design: the cast is part of the scenario,
 * while traits, relationships, and everything that happens come from the seed.
 */
const HOUSEHOLD_SPECS: HouseholdSpec[] = [
  {
    name: 'Calder',
    members: [
      { name: 'Mara Calder', age: 34, jobId: 'farmer', workplaceId: 'place_farm', startingMoney: 8 },
      { name: 'Joren Calder', age: 37, jobId: 'laborer', workplaceId: 'place_farm', startingMoney: 5 },
      { name: 'Tessa Calder', age: 16, startingMoney: 2 },
    ],
  },
  {
    name: 'Bray',
    members: [
      { name: 'Odet Bray', age: 51, jobId: 'baker', workplaceId: 'place_bakery', startingMoney: 22 },
      { name: 'Calla Bray', age: 48, jobId: 'trader', workplaceId: 'place_market', startingMoney: 18 },
    ],
  },
  {
    name: 'Fenn',
    members: [
      { name: 'Bram Fenn', age: 42, jobId: 'guard', workplaceId: 'place_market', startingMoney: 10 },
      { name: 'Petra Fenn', age: 39, jobId: 'farmer', workplaceId: 'place_farm', startingMoney: 9 },
      { name: 'Ren Fenn', age: 19, startingMoney: 3 },
    ],
  },
  {
    name: 'Hale',
    members: [
      { name: 'Sylas Hale', age: 28, jobId: 'laborer', workplaceId: 'place_farm', startingMoney: 4 },
      { name: 'Ivy Hale', age: 26, startingMoney: 6 },
    ],
  },
]

const TRAIT_KEYS: (keyof PersonalityTraits)[] = [
  'ambition',
  'empathy',
  'pride',
  'paranoia',
  'curiosity',
  'riskTolerance',
  'sociability',
]

const NO_TILE = { x: 0, y: 0 }

function makePlaces(): Record<string, Place> {
  const publicPlaces: Place[] = [
    { id: 'place_farm', name: 'the Westfield farm', kind: 'farm', tile: NO_TILE, foodStock: 0, foodPrice: 0, grainStore: 6 },
    // The baker starts with a quiet surplus — the scenario's first secret.
    { id: 'place_bakery', name: 'the Bray bakery', kind: 'bakery', tile: NO_TILE, foodStock: 12, foodPrice: 2, grainStore: 0 },
    { id: 'place_market', name: 'the market square', kind: 'market', tile: NO_TILE, foodStock: 0, foodPrice: 0, grainStore: 0 },
    { id: 'place_well', name: 'the old well', kind: 'well', tile: NO_TILE, foodStock: 0, foodPrice: 0, grainStore: 0 },
    { id: 'place_tavern', name: 'the Ash & Ember tavern', kind: 'tavern', tile: NO_TILE, foodStock: 8, foodPrice: 3, grainStore: 0 },
  ]
  const places: Record<string, Place> = {}
  for (const p of publicPlaces) places[p.id] = p
  for (const spec of HOUSEHOLD_SPECS) {
    const id = homeIdFor(spec.name)
    places[id] = {
      id,
      name: `the ${spec.name} home`,
      kind: 'home',
      tile: NO_TILE,
      foodStock: 0,
      foodPrice: 0,
      grainStore: 0,
    }
  }
  return places
}

/** Village layout: desired tile offsets from the town center per place. */
const TOWN_LAYOUT: Record<string, { dx: number; dy: number }> = {
  place_well: { dx: 0, dy: 0 },
  place_market: { dx: 2, dy: 0 },
  place_bakery: { dx: -2, dy: -1 },
  place_tavern: { dx: 3, dy: 2 },
  place_farm: { dx: -5, dy: 2 },
  place_home_calder: { dx: -3, dy: 4 },
  place_home_bray: { dx: -1, dy: -3 },
  place_home_fenn: { dx: 2, dy: 4 },
  place_home_hale: { dx: 5, dy: -2 },
}

/**
 * Settle each place onto the nearest buildable tile around the town center
 * and clear the lot (the founders felled whatever stood there).
 */
function assignTownTiles(terrain: Terrain, places: Record<string, Place>): void {
  const occupied = new Set<number>()
  const ids = Object.keys(TOWN_LAYOUT).filter((id) => places[id])
  for (const id of ids) {
    const { dx, dy } = TOWN_LAYOUT[id]
    const site = findBuildSite(terrain, occupied, terrain.townCenter.x + dx, terrain.townCenter.y + dy)
    places[id].tile = site
    const i = idx(site.x, site.y)
    occupied.add(i)
    terrain.biome[i] = 'grass'
    terrain.timber[i] = 0
    // The farm clears a small field beside the farmhouse.
    if (places[id].kind === 'farm') {
      for (const [fx, fy] of [
        [site.x + 1, site.y],
        [site.x + 1, site.y + 1],
        [site.x, site.y + 1],
      ]) {
        const fi = idx(fx, fy)
        if (terrain.biome[fi] !== 'ocean' && terrain.biome[fi] !== 'lake' && terrain.biome[fi] !== 'river') {
          terrain.biome[fi] = 'grass'
          terrain.timber[fi] = 0
          occupied.add(fi)
        }
      }
    }
  }
}

function homeIdFor(householdName: string): string {
  return `place_home_${householdName.toLowerCase()}`
}

function rollTraits(seed: string, agentId: string): PersonalityTraits {
  const traits = {} as PersonalityTraits
  for (const key of TRAIT_KEYS) {
    // One stream per trait so adding traits later never reshuffles old worlds.
    traits[key] = Math.round(rngFor(seed, 'trait', agentId, key).range(0.05, 0.95) * 100) / 100
  }
  return traits
}

function defaultRelationship(): Relationship {
  return { trust: 0, affection: 0, resentment: 0, debt: 0, grievances: 0 }
}

export function createWorld(seed: string): World {
  const terrain = generateTerrain(seed)
  const places = makePlaces()
  assignTownTiles(terrain, places)
  const agents: Record<string, Agent> = {}
  const households: Record<string, Household> = {}

  let agentIndex = 1
  for (const spec of HOUSEHOLD_SPECS) {
    const householdId = `household_${spec.name.toLowerCase()}`
    const homeId = homeIdFor(spec.name)
    const memberIds: string[] = []

    for (const member of spec.members) {
      const id = `agent_${String(agentIndex).padStart(4, '0')}`
      agentIndex++
      memberIds.push(id)
      agents[id] = {
        id,
        name: member.name,
        age: member.age,
        householdId,
        locationId: homeId,
        indoors: true,
        jobId: member.jobId,
        workplaceId: member.workplaceId,
        traits: rollTraits(seed, id),
        needs: {
          hunger: rngFor(seed, 'hunger0', id).int(20, 45),
          fatigue: rngFor(seed, 'fatigue0', id).int(5, 25),
          belonging: rngFor(seed, 'belonging0', id).int(40, 70),
        },
        money: member.startingMoney,
        status: member.jobId ? 50 : 40,
        breadInventory: rngFor(seed, 'bread0', id).chance(0.4) ? 1 : 0,
        relationships: {},
        memoryIds: [],
        beliefIds: [],
        knownRumorIds: [],
        workedShiftsToday: 0,
        sleptTonight: false,
      }
    }

    households[householdId] = { id: householdId, name: spec.name, homeId, memberIds }
  }

  // Baseline relationships: warm inside a household, lukewarm across town.
  const ids = Object.keys(agents)
  for (const a of ids) {
    for (const b of ids) {
      if (a === b) continue
      const sameHouse = agents[a].householdId === agents[b].householdId
      const rel = defaultRelationship()
      const r = rngFor(seed, 'rel0', a, b)
      if (sameHouse) {
        rel.kinship = 'household'
        rel.trust = r.range(0.55, 0.8)
        rel.affection = r.range(0.5, 0.8)
        rel.resentment = r.range(0, 0.1)
      } else {
        rel.trust = r.range(0.2, 0.45)
        rel.affection = r.range(0.1, 0.35)
        rel.resentment = r.range(0, 0.08)
      }
      agents[a].relationships[b] = rel
    }
  }

  // Seed two cross-household grudges so the town starts with social fuel.
  const grudgeRng = rngFor(seed, 'grudges')
  const crossPairs: [string, string][] = []
  for (const a of ids) {
    for (const b of ids) {
      if (a < b && agents[a].householdId !== agents[b].householdId) crossPairs.push([a, b])
    }
  }
  for (let i = 0; i < 2 && crossPairs.length > 0; i++) {
    const [a, b] = crossPairs[grudgeRng.int(0, crossPairs.length - 1)]
    agents[a].relationships[b].resentment = grudgeRng.range(0.45, 0.6)
    agents[b].relationships[a].resentment = grudgeRng.range(0.3, 0.55)
    agents[a].relationships[b].trust *= 0.5
    agents[b].relationships[a].trust *= 0.5
  }

  const world: World = {
    id: `world_${seed}`,
    seed,
    name: 'Ashvale',
    day: 1,
    phaseIndex: 0,
    terrain,
    pathWear: {},
    agents,
    places,
    households,
    buildings: {},
    events: [],
    memories: {},
    beliefs: {},
    rumors: {},
    decisionTraces: {},
    nextEventId: 1,
    nextRumorId: 1,
    nextTraceId: 1,
    nextBuildingId: 1,
    nextMemoryId: 1,
    nextBeliefId: 1,
    chronicleArticles: [],
  }

  // The founding town stands already built. Workplaces belong to the
  // households that run them; the market square and well are civic ground.
  const ownerOf: Record<string, string | undefined> = {
    place_farm: 'household_calder',
    place_bakery: 'household_bray',
    place_tavern: 'household_hale',
  }
  const buildingKindFor: Record<string, BuildingKind> = {
    farm: 'farmhouse',
    bakery: 'bakery',
    market: 'market_stalls',
    well: 'well',
    tavern: 'tavern',
    home: 'house',
    granary: 'granary',
  }
  for (const place of Object.values(places)) {
    const owner =
      place.kind === 'home'
        ? Object.values(households).find((h) => h.homeId === place.id)?.id
        : ownerOf[place.id]
    foundBuilding(world, buildingKindFor[place.kind], place.id, owner)
  }

  // Starting secret: someone outside the Bray household suspects the baker's
  // reserve. True, as it happens — the bakery starts well stocked.
  const baker = ids.find((id) => agents[id].jobId === 'baker')
  if (baker) {
    const outsiders = ids.filter((id) => agents[id].householdId !== agents[baker].householdId)
    const firstKnower = rngFor(seed, 'secret0').pick(outsiders)
    const rumor = createRumor(world, {
      claim: `${agents[baker].name} keeps a private grain reserve under the bakery`,
      aboutAgentId: baker,
      emotionalCharge: 0.55,
      firstKnowerIds: [firstKnower],
      tags: ['grain', 'hoarding', 'bakery'],
    })
    // Their hunch is a private inference, not testimony — graded accordingly.
    suspect(world, world.agents[firstKnower], {
      claim: rumor.originalClaim,
      subjectId: baker,
      rumorId: rumor.id,
      confidence: 0.55,
      truthStatus: 'true',
      emotionalCharge: 0.55,
      tags: [...rumor.tags],
    })
    syncRumorConviction(rumor, firstKnower, 0.55)
  }

  return world
}

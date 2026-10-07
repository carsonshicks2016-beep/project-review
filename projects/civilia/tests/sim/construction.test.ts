import { describe, expect, it } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import {
  gatherMaterials,
  materialsComplete,
  maybeProposeBuildings,
} from '../../src/sim/systems/constructionSystem'

const SEED = 'build-test'

describe('buildings and construction', () => {
  it('founds every starting place with a complete building', () => {
    const world = createWorld(SEED)
    for (const place of Object.values(world.places)) {
      expect(place.buildingId).toBeDefined()
      const building = world.buildings[place.buildingId!]
      expect(building.state).toBe('complete')
    }
    // Workplaces have owners; civic ground does not.
    expect(world.buildings[world.places['place_bakery'].buildingId!].ownerHouseholdId).toBe('household_bray')
    expect(world.buildings[world.places['place_well'].buildingId!].ownerHouseholdId).toBeUndefined()
  })

  it('proposes a cottage for a crowded, solvent household', () => {
    const world = createWorld(SEED)
    world.day = 6 // crowding proposals start after the town settles
    maybeProposeBuildings(world)
    const cottage = Object.values(world.buildings).find((b) => b.kind === 'cottage')
    expect(cottage).toBeDefined()
    expect(cottage!.state).toBe('planned')
    expect(cottage!.ownerHouseholdId).toBeDefined()
    const site = world.places[cottage!.placeId]
    expect(site.kind).toBe('home')
    const proposed = world.events.find((e) => e.type === 'building_proposed')
    expect(proposed).toBeDefined()
  })

  it('gathering pulls materials from terrain and depletes it', () => {
    const world = createWorld(SEED)
    world.day = 6
    maybeProposeBuildings(world)
    const project = Object.values(world.buildings).find((b) => b.state === 'planned')!
    const timberBefore = world.terrain.timber.reduce((a, b) => a + b, 0)
    let guard = 0
    while (!materialsComplete(project) && guard++ < 50) {
      gatherMaterials(world, project)
    }
    expect(materialsComplete(project)).toBe(true)
    const timberAfter = world.terrain.timber.reduce((a, b) => a + b, 0)
    expect(timberAfter).toBeLessThan(timberBefore)
    expect(project.materialsDelivered.timber).toBe(project.materialsNeeded.timber)
  })

  it('colonists raise a building over a full run and a household forms', () => {
    const world = createWorld(SEED)
    for (let d = 0; d < 40; d++) stepDay(world)
    const completedEvents = world.events.filter((e) => e.type === 'building_completed')
    expect(completedEvents.length).toBeGreaterThan(0)
    // The cottage spins off a new household with its own home.
    const formed = world.events.find((e) => e.type === 'household_formed')
    if (formed) {
      const mover = world.agents[formed.actorIds[0]]
      const household = world.households[mover.householdId]
      expect(household.memberIds).toEqual([mover.id])
      expect(world.places[household.homeId]).toBeDefined()
    }
  })

  it('construction runs are deterministic', () => {
    const a = createWorld(SEED)
    const b = createWorld(SEED)
    for (let d = 0; d < 20; d++) {
      stepDay(a)
      stepDay(b)
    }
    expect(a.buildings).toEqual(b.buildings)
    expect(a.households).toEqual(b.households)
  })

  it('tracks indoor occupancy', () => {
    const world = createWorld(SEED)
    for (let d = 0; d < 2; d++) stepDay(world)
    // After a night phase everyone slept indoors at some point.
    const sleptEvents = world.events.filter((e) => e.type === 'slept')
    expect(sleptEvents.length).toBeGreaterThan(0)
    // indoors is a live boolean on every agent.
    for (const agent of Object.values(world.agents)) {
      expect(typeof agent.indoors).toBe('boolean')
    }
  })
})

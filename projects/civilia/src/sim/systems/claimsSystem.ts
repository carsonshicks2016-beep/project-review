import { rngFor } from '../rng'
import type { Agent, Household, World } from '../types'
import { emitEvent } from './eventSystem'
import { adjustRelationship, recordGrievance } from '../relationships'
import { createRumor } from './rumorSystem'
import { decorateWithSpeech } from './speechSystem'
import { processEventsIntoMemories } from './memorySystem'
import { TERRAIN_H, TERRAIN_W, idx, isWater } from '../terrain'

/**
 * Claims v0 — every faction-like group (households today; cults, clubs, and
 * governments when they exist) projects territory around the buildings it
 * owns. Where two projections land with near-equal strength the ground is
 * disputed, and disputed ground breeds incidents: confrontations, grudges,
 * and rumors about boundary stones that walk at night.
 *
 * Claims are derived state, recomputed each dawn from canonical buildings —
 * but incidents are real events that feed back into relationships.
 */

export interface ClaimsState {
  /** Per-tile owning household id, '' when unclaimed. */
  owner: string[]
  disputed: boolean[]
}

interface ClaimSource {
  x: number
  y: number
  weight: number
}

function claimSources(world: World, household: Household): ClaimSource[] {
  const sources: ClaimSource[] = []
  for (const building of Object.values(world.buildings)) {
    if (building.ownerHouseholdId !== household.id) continue
    const tile = world.places[building.placeId].tile
    // Rising buildings already project a (weaker) claim — construction is politics.
    const weight = building.state === 'complete' ? 1 : 0.6
    sources.push({ x: tile.x, y: tile.y, weight })
  }
  return sources
}

function householdRadius(world: World, household: Household, buildingCount: number): number {
  const members = household.memberIds.map((id) => world.agents[id])
  if (members.length === 0) return 0
  const statusAvg = members.reduce((s, a) => s + a.status, 0) / members.length
  return Math.min(9, 3 + members.length * 0.8 + statusAvg / 40 + buildingCount * 0.6)
}

/** Recompute the claim map from current buildings and standing. */
export function recomputeClaims(world: World): void {
  const size = TERRAIN_W * TERRAIN_H
  const owner = new Array<string>(size).fill('')
  const disputed = new Array<boolean>(size).fill(false)
  const best = new Float64Array(size)
  const second = new Float64Array(size)
  const secondOwner = new Array<string>(size).fill('')

  const households = Object.values(world.households).sort((a, b) => a.id.localeCompare(b.id))
  for (const household of households) {
    const sources = claimSources(world, household)
    if (sources.length === 0) continue
    const radius = householdRadius(world, household, sources.length)
    const r = Math.ceil(radius)
    for (const source of sources) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const x = source.x + dx
          const y = source.y + dy
          if (x < 0 || y < 0 || x >= TERRAIN_W || y >= TERRAIN_H) continue
          const i = idx(x, y)
          if (isWater(world.terrain.biome[i])) continue
          const dist = Math.max(Math.abs(dx), Math.abs(dy))
          const strength = (radius - dist) * source.weight
          if (strength <= 0) continue
          if (owner[i] === household.id) {
            best[i] = Math.max(best[i], strength)
          } else if (strength > best[i]) {
            second[i] = best[i]
            secondOwner[i] = owner[i]
            best[i] = strength
            owner[i] = household.id
          } else if (strength > second[i]) {
            second[i] = strength
            secondOwner[i] = household.id
          }
        }
      }
    }
  }

  for (let i = 0; i < size; i++) {
    if (owner[i] && secondOwner[i] && best[i] - second[i] < 1) disputed[i] = true
  }
  world.claims = { owner, disputed }
}

function householdHead(world: World, householdId: string): Agent | undefined {
  const household = world.households[householdId]
  if (!household) return undefined
  return household.memberIds
    .map((id) => world.agents[id])
    .sort((a, b) => b.age - a.age || a.id.localeCompare(b.id))[0]
}

/**
 * Dawn check: disputed ground occasionally flares into a confrontation
 * between household heads — resentment on both sides, sometimes a rumor.
 */
export function maybeBorderIncident(world: World): void {
  const claims = world.claims
  if (!claims) return
  const disputedTiles: number[] = []
  for (let i = 0; i < claims.disputed.length; i++) {
    if (claims.disputed[i]) disputedTiles.push(i)
  }
  if (disputedTiles.length === 0) return

  const rng = rngFor(world.seed, 'border', world.day)
  if (!rng.chance(0.25)) return

  const tile = disputedTiles[rng.int(0, disputedTiles.length - 1)]
  const ownerId = claims.owner[tile]
  // Find the strongest rival at this tile by checking adjacent ownership.
  const x = tile % TERRAIN_W
  const y = Math.floor(tile / TERRAIN_W)
  let rivalId = ''
  for (const [dx, dy] of [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ]) {
    const nx = x + dx
    const ny = y + dy
    if (nx < 0 || ny < 0 || nx >= TERRAIN_W || ny >= TERRAIN_H) continue
    const neighborOwner = claims.owner[idx(nx, ny)]
    if (neighborOwner && neighborOwner !== ownerId) {
      rivalId = neighborOwner
      break
    }
  }
  if (!rivalId) return

  const headA = householdHead(world, ownerId)
  const headB = householdHead(world, rivalId)
  if (!headA || !headB) return

  // Nearest place gives the confrontation an address.
  let nearestPlaceId: string | undefined
  let nearestDist = Infinity
  for (const place of Object.values(world.places)) {
    const d = Math.abs(place.tile.x - x) + Math.abs(place.tile.y - y)
    if (d < nearestDist) {
      nearestDist = d
      nearestPlaceId = place.id
    }
  }

  adjustRelationship(headA, headB.id, { resentment: 0.06, trust: -0.03 })
  adjustRelationship(headB, headA.id, { resentment: 0.06, trust: -0.03 })
  recordGrievance(headA, headB.id, world.day)
  recordGrievance(headB, headA.id, world.day)

  const event = emitEvent(world, {
    type: 'border_dispute',
    locationId: nearestPlaceId,
    actorIds: [headA.id],
    targetIds: [headB.id],
    payload: {
      tile: { x, y },
      households: [ownerId, rivalId],
    },
    visibility: 'public',
    consequenceLevel: 1,
    tags: ['conflict', 'border'],
  })
  decorateWithSpeech(world, [event.id])
  processEventsIntoMemories(world, [event.id])

  if (rng.chance(0.5)) {
    const aName = world.households[ownerId]?.name ?? 'someone'
    const bName = world.households[rivalId]?.name ?? 'someone'
    createRumor(world, {
      claim: `the ${bName}s moved the boundary stones on the ${aName} side by night`,
      sourceEventIds: [event.id],
      emotionalCharge: 0.45,
      firstKnowerIds: [headA.id],
      tags: ['border', 'conflict'],
    })
  }
}

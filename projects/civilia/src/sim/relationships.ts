import type { Agent, Relationship, World } from './types'

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x))
}

/** Get (or lazily create) the directed relationship from `agent` toward `otherId`. */
export function relationshipWith(agent: Agent, otherId: string): Relationship {
  let rel = agent.relationships[otherId]
  if (!rel) {
    rel = { trust: 0.25, affection: 0.15, resentment: 0, debt: 0, grievances: 0 }
    agent.relationships[otherId] = rel
  }
  return rel
}

/** The offender wronged this agent: remember it until it is repaired. */
export function recordGrievance(offended: Agent, offenderId: string, day: number): void {
  const rel = relationshipWith(offended, offenderId)
  rel.grievances = Math.min(8, rel.grievances + 1)
  rel.lastHarmDay = day
}

/**
 * Nightly favor-economy pressure: owing money quietly corrodes how the
 * creditor sees the debtor. Small, but it compounds until someone repays.
 */
export function applyDebtPressure(world: World): void {
  for (const debtor of Object.values(world.agents)) {
    for (const [creditorId, rel] of Object.entries(debtor.relationships)) {
      if (rel.debt < 2) continue
      const creditor = world.agents[creditorId]
      if (!creditor) continue
      adjustRelationship(creditor, debtor.id, {
        resentment: Math.min(0.015, 0.004 * rel.debt),
      })
    }
  }
}

export interface RelationshipDeltas {
  trust?: number
  affection?: number
  resentment?: number
  debt?: number
}

/** Apply clamped deltas to agent's relationship toward otherId. */
export function adjustRelationship(agent: Agent, otherId: string, deltas: RelationshipDeltas): void {
  const rel = relationshipWith(agent, otherId)
  if (deltas.trust) rel.trust = clamp01(rel.trust + deltas.trust)
  if (deltas.affection) rel.affection = clamp01(rel.affection + deltas.affection)
  if (deltas.resentment) rel.resentment = clamp01(rel.resentment + deltas.resentment)
  if (deltas.debt) rel.debt = Math.max(0, rel.debt + deltas.debt)
}

export function clampNeed(x: number): number {
  return Math.max(0, Math.min(100, x))
}

export function clampStatus(x: number): number {
  return Math.max(0, Math.min(100, x))
}

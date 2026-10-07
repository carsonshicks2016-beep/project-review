import type { ActionCandidate, ActionDefinition } from '../types'
import { adjustRelationship, clampNeed, clampStatus, relationshipWith } from '../relationships'
import { emitEvent } from '../systems/eventSystem'

/**
 * Relationship repair — the other half of the social ledger.
 *
 * Apologies follow the spec's forgiveness thresholds: they only land when
 * timing, sincerity, restitution, relationship, status, and witnesses align.
 * A rebuffed apology is a real social cost, so proud agents rarely risk it.
 *
 * Repaying favors closes the debts that gifts open; unpaid debts quietly
 * corrode the creditor's patience (see applyDebtPressure).
 */

export const apologize: ActionDefinition = {
  id: 'apologize',

  candidates(ctx) {
    const { agent, othersHere } = ctx
    const candidates: ActionCandidate[] = []
    for (const other of othersHere) {
      // The offender knows what they did: the other side holds the grievance.
      const theirView = relationshipWith(other, agent.id)
      if (theirView.grievances <= 0) continue
      const myView = relationshipWith(agent, other.id)

      const reasons: string[] = []
      let score = agent.traits.empathy * 30
      reasons.push(`+${score.toFixed(0)} conscience`)

      const bond = myView.affection * 15
      score += bond
      reasons.push(`+${bond.toFixed(0)} misses ${other.name}`)

      const fresh =
        theirView.lastHarmDay !== undefined && ctx.world.day - theirView.lastHarmDay <= 2
      if (fresh) {
        score += 12
        reasons.push('+12 the wound is fresh')
      }

      if (agent.status + 8 < other.status) {
        score += 8
        reasons.push(`+8 needs ${other.name}'s goodwill`)
      }

      const pridePenalty = agent.traits.pride * 28
      score -= pridePenalty
      reasons.push(`-${pridePenalty.toFixed(0)} pride`)

      const audiencePenalty = (ctx.othersHere.length - 1) * 3
      if (audiencePenalty > 0) {
        score -= audiencePenalty
        reasons.push(`-${audiencePenalty} eating crow in public`)
      }

      candidates.push({
        actionId: this.id,
        label: `apologize to ${other.name}`,
        targetAgentId: other.id,
        score,
        reasons,
      })
    }
    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent, place, rng } = ctx
    const other = world.agents[candidate.targetAgentId!]
    const theirView = relationshipWith(other, agent.id)
    const witnesses = ctx.othersHere.filter((a) => a.id !== other.id).map((a) => a.id)
    const publicly = witnesses.length > 0

    // Restitution: coins offered when the offender can spare them and cares.
    const restitution =
      agent.money >= 2 && (agent.traits.empathy > 0.4 || relationshipWith(agent, other.id).debt > 0)
        ? 2
        : 0

    // Forgiveness thresholds: timing, sincerity, restitution, relationship,
    // the offended party's pride, the audience, and the depth of the wound.
    let acceptance = 0.25
    acceptance += agent.traits.empathy * 0.25
    const age = theirView.lastHarmDay !== undefined ? world.day - theirView.lastHarmDay : 99
    if (age <= 2) acceptance += 0.15
    else if (age > 6) acceptance -= 0.1
    if (restitution > 0) acceptance += 0.2
    acceptance += theirView.trust * 0.15
    acceptance -= other.traits.pride * 0.25
    acceptance += Math.min(0.12, witnesses.length * 0.06)
    acceptance -= Math.max(0, theirView.grievances - 1) * 0.05
    acceptance = Math.max(0.05, Math.min(0.95, acceptance))

    const accepted = rng.chance(acceptance)
    const grievancesBefore = theirView.grievances

    if (restitution > 0) {
      agent.money -= restitution
      other.money += restitution
    }

    if (accepted) {
      theirView.grievances = 0
      adjustRelationship(other, agent.id, { resentment: -0.22, trust: 0.1, affection: 0.06 })
      adjustRelationship(agent, other.id, { resentment: -0.05 })
      agent.needs.belonging = clampNeed(agent.needs.belonging + 8)
      other.needs.belonging = clampNeed(other.needs.belonging + 6)
      if (publicly) {
        agent.status = clampStatus(agent.status + 1)
        other.status = clampStatus(other.status + 1)
      }
    } else {
      agent.status = clampStatus(agent.status - (publicly ? 2 : 1))
      adjustRelationship(agent, other.id, { resentment: 0.06 })
      adjustRelationship(other, agent.id, { resentment: -0.04 })
    }

    const event = emitEvent(world, {
      type: 'apologized',
      locationId: place.id,
      actorIds: [agent.id],
      targetIds: [other.id],
      witnessIds: witnesses,
      payload: { accepted, restitution, publicly, grievancesBefore, acceptance: Math.round(acceptance * 100) / 100 },
      visibility: 'witnessed',
      consequenceLevel: publicly ? 1 : 0,
      tags: ['social', 'repair'],
    })
    return [event.id]
  },
}

export const repayFavor: ActionDefinition = {
  id: 'repay_favor',

  candidates(ctx) {
    const { agent, othersHere } = ctx
    const candidates: ActionCandidate[] = []
    for (const other of othersHere) {
      const myView = relationshipWith(agent, other.id)
      if (myView.debt < 2) continue
      if (agent.money - 1 < 2) continue

      const reasons: string[] = []
      let score = agent.traits.pride * 22
      reasons.push(`+${score.toFixed(0)} hates owing anyone`)

      const trustBonus = myView.trust * 8
      score += trustBonus
      reasons.push(`+${trustBonus.toFixed(0)} values ${other.name}`)

      const weight = myView.debt * 2.5
      score += weight
      reasons.push(`+${weight.toFixed(0)} the debt weighs ${myView.debt} coins`)

      if (agent.money < 5) {
        score -= 14
        reasons.push('-14 can barely spare it')
      }

      candidates.push({
        actionId: this.id,
        label: `repay ${other.name} what they're owed`,
        targetAgentId: other.id,
        score,
        reasons,
      })
    }
    return candidates
  },

  execute(ctx, candidate) {
    const { world, agent, place } = ctx
    const other = world.agents[candidate.targetAgentId!]
    const myView = relationshipWith(agent, other.id)
    const amount = Math.min(myView.debt, agent.money - 1)

    agent.money -= amount
    other.money += amount
    adjustRelationship(agent, other.id, { debt: -amount })
    adjustRelationship(other, agent.id, { trust: 0.06, affection: 0.03, resentment: -0.05 })
    agent.needs.belonging = clampNeed(agent.needs.belonging + 4)

    const event = emitEvent(world, {
      type: 'debt_repaid',
      locationId: place.id,
      actorIds: [agent.id],
      targetIds: [other.id],
      witnessIds: ctx.othersHere.filter((a) => a.id !== other.id).map((a) => a.id),
      payload: { amount, remaining: myView.debt },
      visibility: 'witnessed',
      tags: ['social', 'money', 'repair'],
    })
    return [event.id]
  },
}

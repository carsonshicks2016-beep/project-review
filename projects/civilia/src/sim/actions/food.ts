import type { ActionDefinition, Place, World } from '../types'
import { clampNeed } from '../relationships'
import { emitEvent } from '../systems/eventSystem'

const MAX_CARRIED_BREAD = 2
const HUNGER_RELIEF = 50

/** Shortage pricing: +1 coin when stock is nearly gone, back to base when it recovers. */
export function adjustFoodPrice(world: World, place: Place, basePrice: number): void {
  const shortagePrice = basePrice + 1
  const target = place.foodStock <= 3 ? shortagePrice : basePrice
  if (place.foodPrice !== target) {
    place.foodPrice = target
    emitEvent(world, {
      type: 'price_changed',
      locationId: place.id,
      payload: { good: 'bread', newPrice: target, reason: place.foodStock <= 3 ? 'shortage' : 'restock' },
      visibility: 'public',
      consequenceLevel: 1,
      tags: ['economy', 'bread'],
    })
  }
}

export const buyFood: ActionDefinition = {
  id: 'buy_food',

  candidates(ctx) {
    const { agent, place } = ctx
    if (place.foodPrice <= 0 || place.foodStock <= 0) return []
    if (agent.money < place.foodPrice) return []
    if (agent.breadInventory >= MAX_CARRIED_BREAD) return []

    const reasons: string[] = []
    let score = agent.needs.hunger * 0.65
    reasons.push(`+${score.toFixed(0)} hunger pressure`)

    if (agent.breadInventory === 0) {
      score += 12
      reasons.push('+12 nothing to eat later')
    }

    const pricePenalty = place.foodPrice * 1.5
    score -= pricePenalty
    reasons.push(`-${pricePenalty.toFixed(0)} price of ${place.foodPrice} coins`)

    return [
      {
        actionId: this.id,
        label: `buy bread at ${place.name}`,
        targetPlaceId: place.id,
        score,
        reasons,
      },
    ]
  },

  execute(ctx) {
    const { world, agent, place } = ctx
    const price = place.foodPrice
    const basePrice = place.kind === 'tavern' ? 3 : 2
    agent.money -= price
    place.foodStock--
    agent.breadInventory++

    const bought = emitEvent(world, {
      type: 'bought_good',
      locationId: place.id,
      actorIds: [agent.id],
      payload: { good: 'bread', price },
      visibility: 'witnessed',
      witnessIds: ctx.othersHere.map((a) => a.id),
      tags: ['economy', 'bread'],
    })
    const sold = emitEvent(world, {
      type: 'sold_good',
      locationId: place.id,
      actorIds: [agent.id],
      payload: { good: 'bread', price },
      visibility: 'witnessed',
      tags: ['economy', 'bread'],
      parentEventIds: [bought.id],
    })
    adjustFoodPrice(world, place, basePrice)
    return [bought.id, sold.id]
  },
}

export const eatFood: ActionDefinition = {
  id: 'eat_food',

  candidates(ctx) {
    const { agent } = ctx
    if (agent.breadInventory <= 0) return []
    if (agent.needs.hunger < 25) return []

    const score = agent.needs.hunger * 0.85 - 5
    return [
      {
        actionId: this.id,
        label: 'eat some bread',
        score,
        reasons: [`+${(agent.needs.hunger * 0.85).toFixed(0)} hunger pressure`, '-5 takes time'],
      },
    ]
  },

  execute(ctx) {
    const { world, agent, place } = ctx
    agent.breadInventory--
    agent.needs.hunger = clampNeed(agent.needs.hunger - HUNGER_RELIEF)
    const ate = emitEvent(world, {
      type: 'ate_food',
      locationId: place.id,
      actorIds: [agent.id],
      payload: { good: 'bread' },
      visibility: 'witnessed',
      tags: ['food'],
    })
    return [ate.id]
  },
}

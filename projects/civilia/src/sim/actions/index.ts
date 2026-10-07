import type { ActionDefinition } from '../types'
import { workShift } from './work'
import { buyFood, eatFood } from './food'
import { sleep } from './rest'
import { travelToPlace } from './travel'
import { gossip, helpAgent, insultAgent, talk } from './social'
import { apologize, repayFavor } from './repair'
import { stealItem } from './crime'
import { build } from './build'

/** Evaluation order is fixed so decisions replay identically for a given seed. */
export const ALL_ACTIONS: ActionDefinition[] = [
  workShift,
  buyFood,
  eatFood,
  sleep,
  travelToPlace,
  talk,
  gossip,
  helpAgent,
  insultAgent,
  apologize,
  repayFavor,
  stealItem,
  build,
]

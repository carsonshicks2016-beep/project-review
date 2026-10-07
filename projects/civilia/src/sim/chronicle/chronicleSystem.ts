/**
 * Chronicle system — collects notable events, clusters them, and builds
 * LLM-ready prompts for newspaper article generation.
 *
 * The system is deliberately pure: it never calls an LLM. It returns a
 * ChronicleGenerationRequest that the UI/LLM layer can pick up and fulfill.
 */

import type { SimEvent, World } from '../types'
import type { ChronicleConfig } from './types'
import { DEFAULT_CHRONICLE_CONFIG } from './types'
import { FACTION_TEMPLATES, GENERIC_TEMPLATE } from './promptTemplates'
import { describeEvent } from '../systems/eventSystem'

// ---------------------------------------------------------------------------
// Public types

export interface ChronicleCluster {
  events: SimEvent[]
  factionId: string
  prompt: string
}

export interface ChronicleGenerationRequest {
  period: string
  clusters: ChronicleCluster[]
}

// ---------------------------------------------------------------------------
// Cadence helpers

/** Number of sim-days per period for a given cadence. */
function cadenceDays(cadence: ChronicleConfig['cadence']): number {
  switch (cadence) {
    case 'weekly': return 7
    case 'monthly': return 30
    case 'seasonal': return 90
  }
}

/** Compute the human-readable period label for a given day. */
function periodLabel(day: number, cadence: ChronicleConfig['cadence']): string {
  const span = cadenceDays(cadence)
  const index = Math.ceil(day / span)
  switch (cadence) {
    case 'weekly': return `Week ${index}`
    case 'monthly': return `Month ${index}`
    case 'seasonal': return `Season ${index}`
  }
}

/** The first day of the period containing `day`. */
function periodStart(day: number, cadence: ChronicleConfig['cadence']): number {
  const span = cadenceDays(cadence)
  return Math.floor((day - 1) / span) * span + 1
}

/** Return true when `day` is the last day of its period. */
function isPeriodBoundary(day: number, cadence: ChronicleConfig['cadence']): boolean {
  const span = cadenceDays(cadence)
  return day % span === 0
}

// ---------------------------------------------------------------------------
// Core logic

/** Filter events by consequenceLevel and visibility. */
export function collectNotableEvents(
  world: World,
  fromDay: number,
  toDay: number,
  threshold: number,
): SimEvent[] {
  return world.events.filter(
    (e) =>
      e.day >= fromDay &&
      e.day <= toDay &&
      e.consequenceLevel >= threshold &&
      e.visibility !== 'private',
  )
}

/**
 * Group events into clusters by location + type proximity.
 *
 * Events at the same location with the same type go into one cluster.
 * Events with no location are grouped by type alone.
 */
export function clusterEvents(events: SimEvent[]): SimEvent[][] {
  const map = new Map<string, SimEvent[]>()
  for (const e of events) {
    const key = `${e.locationId ?? '_nowhere_'}::${e.type}`
    const bucket = map.get(key)
    if (bucket) {
      bucket.push(e)
    } else {
      map.set(key, [e])
    }
  }
  return Array.from(map.values())
}

/** Fill in a prompt template with concrete values. */
export function buildPrompt(
  world: World,
  cluster: SimEvent[],
  factionId: string,
  period: string,
  townName: string,
): string {
  const template = FACTION_TEMPLATES[factionId] ?? GENERIC_TEMPLATE

  const eventLines = cluster
    .map((e) => `- ${describeEvent(world, e)}`)
    .join('\n')

  return template
    .replace(/\{\{EVENTS_SUMMARY\}\}/g, eventLines)
    .replace(/\{\{PERIOD\}\}/g, period)
    .replace(/\{\{TOWN_NAME\}\}/g, townName)
}

// ---------------------------------------------------------------------------
// Faction list (derived from the world's households)

function factionIds(world: World): string[] {
  return Object.values(world.households).map((h) => h.name)
}

// ---------------------------------------------------------------------------
// Entry point

/**
 * Check whether the just-completed day crosses a period boundary.
 * If so, collect notable events, cluster them, build prompts for
 * every faction, and return a generation request.
 *
 * Returns null if:
 * - chronicle is disabled
 * - we haven't reached a period boundary
 * - there are no notable events in the period
 */
export function maybeGenerate(
  world: World,
  config?: ChronicleConfig,
): ChronicleGenerationRequest | null {
  const cfg = config ?? DEFAULT_CHRONICLE_CONFIG
  if (!cfg.enabled) return null

  // world.day is already incremented to the *next* day when endOfDay runs,
  // so the day that just finished is world.day - 1.
  const justFinished = world.day - 1
  if (justFinished < 1) return null
  if (!isPeriodBoundary(justFinished, cfg.cadence)) return null

  const fromDay = periodStart(justFinished, cfg.cadence)
  const toDay = justFinished
  const period = periodLabel(justFinished, cfg.cadence)
  const notable = collectNotableEvents(world, fromDay, toDay, cfg.consequenceThreshold)

  if (notable.length === 0) return null

  const eventClusters = clusterEvents(notable)
  const factions = factionIds(world)
  const clusters: ChronicleCluster[] = []

  // For each faction, build prompts for each event cluster (capped).
  for (const faction of factions) {
    let count = 0
    for (const cluster of eventClusters) {
      if (count >= cfg.maxArticlesPerPeriod) break
      clusters.push({
        events: cluster,
        factionId: faction,
        prompt: buildPrompt(world, cluster, faction, period, world.name),
      })
      count++
    }
  }

  return { period, clusters }
}

/** Convenience namespace for static-style access. */
export const ChronicleSystem = {
  collectNotableEvents,
  clusterEvents,
  buildPrompt,
  maybeGenerate,
} as const

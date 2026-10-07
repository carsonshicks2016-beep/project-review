import { describe, it, expect } from 'vitest'
import { createWorld } from '../../src/sim/createWorld'
import { stepDay } from '../../src/sim/stepWorld'
import {
  collectNotableEvents,
  clusterEvents,
  maybeGenerate,
} from '../../src/sim/chronicle/chronicleSystem'
import { validateChronicleArticle } from '../../src/sim/chronicle/schema'
import { DEFAULT_CHRONICLE_CONFIG } from '../../src/sim/chronicle/types'
import type { ChronicleConfig } from '../../src/sim/chronicle/types'
import type { SimEvent } from '../../src/sim/types'

// ---------------------------------------------------------------------------
// Helpers

function worldAtDay(targetDay: number) {
  const world = createWorld('test-chronicle')
  while (world.day < targetDay) {
    stepDay(world)
  }
  return world
}

function fakeEvent(overrides: Partial<SimEvent> = {}): SimEvent {
  return {
    id: 'event_test_001',
    type: 'worked',
    day: 1,
    phase: 'morning',
    locationId: 'place_farm',
    actorIds: ['agent_0001'],
    targetIds: [],
    witnessIds: [],
    payload: {},
    consequenceLevel: 0,
    visibility: 'witnessed',
    tags: [],
    parentEventIds: [],
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// collectNotableEvents

describe('collectNotableEvents', () => {
  it('filters by consequenceLevel', () => {
    const world = createWorld('filter-test')
    world.events = [
      fakeEvent({ id: 'e1', consequenceLevel: 0, day: 1, visibility: 'public' }),
      fakeEvent({ id: 'e2', consequenceLevel: 1, day: 1, visibility: 'public' }),
      fakeEvent({ id: 'e3', consequenceLevel: 2, day: 1, visibility: 'public' }),
    ]
    const result = collectNotableEvents(world, 1, 1, 1)
    expect(result.map((e) => e.id)).toEqual(['e2', 'e3'])
  })

  it('excludes private events', () => {
    const world = createWorld('private-test')
    world.events = [
      fakeEvent({ id: 'e1', consequenceLevel: 2, day: 1, visibility: 'private' }),
      fakeEvent({ id: 'e2', consequenceLevel: 2, day: 1, visibility: 'witnessed' }),
    ]
    const result = collectNotableEvents(world, 1, 1, 1)
    expect(result.map((e) => e.id)).toEqual(['e2'])
  })

  it('filters by day range', () => {
    const world = createWorld('range-test')
    world.events = [
      fakeEvent({ id: 'e1', consequenceLevel: 2, day: 1, visibility: 'public' }),
      fakeEvent({ id: 'e2', consequenceLevel: 2, day: 3, visibility: 'public' }),
      fakeEvent({ id: 'e3', consequenceLevel: 2, day: 5, visibility: 'public' }),
    ]
    const result = collectNotableEvents(world, 2, 4, 1)
    expect(result.map((e) => e.id)).toEqual(['e2'])
  })
})

// ---------------------------------------------------------------------------
// clusterEvents

describe('clusterEvents', () => {
  it('groups events by location + type', () => {
    const events = [
      fakeEvent({ id: 'e1', locationId: 'place_farm', type: 'worked' }),
      fakeEvent({ id: 'e2', locationId: 'place_farm', type: 'worked' }),
      fakeEvent({ id: 'e3', locationId: 'place_bakery', type: 'worked' }),
      fakeEvent({ id: 'e4', locationId: 'place_farm', type: 'stole_item' }),
    ]
    const clusters = clusterEvents(events)
    expect(clusters.length).toBe(3)
    const farmWork = clusters.find((c) => c[0].locationId === 'place_farm' && c[0].type === 'worked')
    expect(farmWork?.length).toBe(2)
  })

  it('returns empty array for no events', () => {
    expect(clusterEvents([])).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// maybeGenerate

describe('maybeGenerate', () => {
  it('returns null when chronicle is disabled', () => {
    const world = worldAtDay(8)
    const config: ChronicleConfig = { ...DEFAULT_CHRONICLE_CONFIG, enabled: false }
    expect(maybeGenerate(world, config)).toBeNull()
  })

  it('returns null before period boundary (day 3 of weekly)', () => {
    const world = worldAtDay(4) // day is 4, just finished day 3
    const result = maybeGenerate(world, DEFAULT_CHRONICLE_CONFIG)
    expect(result).toBeNull()
  })

  it('returns a request after 7 days (weekly cadence) when notable events exist', () => {
    const world = createWorld('weekly-test')
    // Inject some notable events across days 1-7
    for (let d = 1; d <= 7; d++) {
      world.events.push(
        fakeEvent({
          id: `notable_${d}`,
          day: d,
          consequenceLevel: 2,
          visibility: 'public',
          locationId: 'place_market',
          type: 'stole_item',
        }),
      )
    }
    // Simulate being at the end of day 7 (world.day is already incremented to 8)
    world.day = 8
    const result = maybeGenerate(world, DEFAULT_CHRONICLE_CONFIG)
    expect(result).not.toBeNull()
    expect(result!.period).toBe('Week 1')
    expect(result!.clusters.length).toBeGreaterThan(0)
  })

  it('returns null when no notable events in the period', () => {
    const world = createWorld('empty-period')
    // Only mundane events
    for (let d = 1; d <= 7; d++) {
      world.events.push(
        fakeEvent({
          id: `mundane_${d}`,
          day: d,
          consequenceLevel: 0,
          visibility: 'public',
        }),
      )
    }
    world.day = 8
    const result = maybeGenerate(world, DEFAULT_CHRONICLE_CONFIG)
    expect(result).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// validateChronicleArticle

describe('validateChronicleArticle', () => {
  const validArticle = {
    id: 'article-001',
    title: 'Bread Prices Soar',
    authorFaction: 'Bray',
    bias: 'pro' as const,
    period: 'Week 1',
    sourceEventIds: ['event_000001', 'event_000002'],
    content: 'The Bray bakery reported record sales...',
    generatedAt: Date.now(),
  }

  it('accepts a valid article', () => {
    const result = validateChronicleArticle(validArticle)
    expect(result.ok).toBe(true)
    if (result.ok) {
      expect(result.article.title).toBe('Bread Prices Soar')
    }
  })

  it('rejects null', () => {
    const result = validateChronicleArticle(null)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.errors).toContain('expected an object')
  })

  it('rejects missing fields', () => {
    const result = validateChronicleArticle({ id: 'x' })
    expect(result.ok).toBe(false)
  })

  it('rejects invalid bias value', () => {
    const result = validateChronicleArticle({ ...validArticle, bias: 'extreme' })
    expect(result.ok).toBe(false)
    if (!result.ok) {
      expect(result.errors.some((e) => e.includes('bias'))).toBe(true)
    }
  })

  it('rejects non-array sourceEventIds', () => {
    const result = validateChronicleArticle({ ...validArticle, sourceEventIds: 'not-an-array' })
    expect(result.ok).toBe(false)
  })

  it('rejects non-finite generatedAt', () => {
    const result = validateChronicleArticle({ ...validArticle, generatedAt: Infinity })
    expect(result.ok).toBe(false)
  })
})

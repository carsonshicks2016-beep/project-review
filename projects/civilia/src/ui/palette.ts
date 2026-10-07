import type { BiomeId, DayPhase, World } from '../sim'

/** Tile size in world pixels; the whole map is 960×640 world px. */
export const TILE = 10

export const PHASE_EMOJI: Record<DayPhase, string> = {
  dawn: '🌅',
  morning: '☀️',
  midday: '🌞',
  afternoon: '🌤',
  evening: '🌆',
  night: '🌙',
}

/** Night falls over the whole map; dawn glows. */
export const PHASE_TINT: Record<DayPhase, string> = {
  dawn: 'rgba(255, 170, 110, 0.12)',
  morning: 'rgba(0, 0, 0, 0)',
  midday: 'rgba(255, 255, 220, 0.04)',
  afternoon: 'rgba(255, 200, 120, 0.07)',
  evening: 'rgba(235, 120, 60, 0.16)',
  night: 'rgba(15, 25, 70, 0.38)',
}

/**
 * 16-bit-flavored biome palette: each biome gets a base and one or two
 * jitter shades so the ground reads as pixel-art texture, not flat fill.
 */
export const BIOME_SHADES: Record<BiomeId, string[]> = {
  ocean: ['#1d4e6e', '#1a4763', '#216080'],
  lake: ['#2d6f8c', '#28647e', '#347a99'],
  river: ['#3f87a5', '#3a7d99'],
  beach: ['#d8c08a', '#cdb47e', '#e0ca96'],
  marsh: ['#5d7d52', '#54724a', '#66875c'],
  clay_flats: ['#b07850', '#a56f4a', '#b98158'],
  grass: ['#6f9e4f', '#679648', '#77a657'],
  meadow: ['#8cb96b', '#82af62', '#95c274'],
  forest: ['#4e8347', '#468040', '#57894f'],
  dense_forest: ['#39673a', '#325e34', '#406f41'],
  hills: ['#8a8f6a', '#818661', '#939873'],
  mountain: ['#8d8d93', '#84848a', '#97979d'],
  snow: ['#e8edf2', '#dfe5ec', '#f2f6fa'],
}

/** Canopy colors stamped over forest tiles. */
export const TREE_DARK = '#2c5a33'
export const TREE_DARKER = '#1f4427'

/** Path-wear rendering, faint trail to packed road. */
export const WEAR_COLORS = {
  trail: 'rgba(141, 110, 70, 0.45)',
  path: '#8b6d47',
  road: '#9d7c50',
}

const HOUSEHOLD_PALETTE = [
  '#e4564a',
  '#4a90d9',
  '#58b368',
  '#d9a44a',
  '#9a6dd7',
  '#50c8c2',
  '#d96aa8',
  '#8a8f4a',
]

/** Stable color per household (sorted id order, so new households extend). */
export function householdColor(world: World, householdId: string): string {
  const ids = Object.keys(world.households).sort()
  const index = ids.indexOf(householdId)
  return HOUSEHOLD_PALETTE[(index + HOUSEHOLD_PALETTE.length) % HOUSEHOLD_PALETTE.length]
}

/** Bubble fill per speech category. */
export const SPEECH_COLORS: Record<string, string> = {
  chat: '#f6f1e3',
  rumor: '#e6d7f5',
  hostile: '#f5c1b8',
  kind: '#cfe8c6',
  status: '#d9e4f0',
  alarm: '#ffd28a',
  thought: '#e8e8ee',
}

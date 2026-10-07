export type MetricDirection = 'higher' | 'lower' | 'neutral'

export type MetricCategoryKey =
  | 'recovery'
  | 'sleep'
  | 'vitals'
  | 'load'
  | 'body'

export interface MetricDefinition {
  key: string
  label: string
  shortLabel: string
  unit: string
  category: MetricCategoryKey
  direction: MetricDirection
  description: string
  aliases: string[]
  decimals?: number
  accent: string
}

export interface MetricSectionDefinition {
  key: MetricCategoryKey
  title: string
  description: string
}

export interface CustomMetricValue {
  key: string
  label: string
  unit: string
  value: number
}

export interface DailyEntry {
  id: string
  date: string
  source: 'manual' | 'csv' | 'seed' | 'backup'
  notes: string
  metrics: Record<string, number | null>
  customMetrics: CustomMetricValue[]
  createdAt: string
  updatedAt: string
}

export type WorkoutMarkerType =
  | 'threshold'
  | 'recovery'
  | 'drift'
  | 'surge'
  | 'fatigue'
  | 'steady'
  | 'custom'

export interface WorkoutMarker {
  id: string
  x: number
  y: number
  type: WorkoutMarkerType
  label: string
  value: number | null
  unit: string
  note: string
}

export interface WorkoutEntry {
  id: string
  date: string
  title: string
  sport: string
  durationMin: number | null
  strain: number | null
  avgHr: number | null
  maxHr: number | null
  hrRecovery1m: number | null
  zone2Minutes: number | null
  zone3Minutes: number | null
  zone4Minutes: number | null
  zone5Minutes: number | null
  screenshot: string
  notes: string
  tags: string[]
  markers: WorkoutMarker[]
  createdAt: string
  updatedAt: string
}

export interface BackupPayload {
  version: number
  exportedAt: string
  dailyEntries: DailyEntry[]
  workouts: WorkoutEntry[]
}

export interface MetaRecord {
  key: string
  value: string
}

export interface MetricChange {
  key: string
  label: string
  currentAverage: number
  previousAverage: number
  delta: number
  percentDelta: number
  direction: MetricDirection
}

export interface CorrelationInsight {
  key: string
  label: string
  correlation: number
  sampleSize: number
}

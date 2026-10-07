import { format } from 'date-fns'
import { mean, sampleCorrelation } from 'simple-statistics'
import {
  cardioSignalMetricKeys,
  getMetricDefinition,
  metricCatalog,
} from './metricCatalog'
import type {
  CorrelationInsight,
  DailyEntry,
  MetricChange,
  MetricDefinition,
  WorkoutEntry,
} from './types'

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

export function getMetricValue(entry: DailyEntry, metricKey: string) {
  const standardValue = entry.metrics[metricKey]
  if (typeof standardValue === 'number' && Number.isFinite(standardValue)) {
    return standardValue
  }

  const customValue = entry.customMetrics.find((metric) => metric.key === metricKey)?.value
  return typeof customValue === 'number' && Number.isFinite(customValue)
    ? customValue
    : null
}

export function sortByDate<T extends { date: string }>(items: T[]) {
  return [...items].sort((left, right) => left.date.localeCompare(right.date))
}

export function buildTrendSeries(entries: DailyEntry[], metricKeys: string[]) {
  return sortByDate(entries).map((entry) => {
    const row: Record<string, number | string | null> = {
      date: entry.date,
      dateLabel: format(new Date(`${entry.date}T12:00:00`), 'MMM d'),
    }

    for (const key of metricKeys) {
      row[key] = getMetricValue(entry, key)
    }

    return row
  })
}

export function getAvailableMetricDefinitions(entries: DailyEntry[]) {
  const customMap = new Map<string, MetricDefinition>()

  for (const entry of entries) {
    for (const metric of entry.customMetrics) {
      if (!customMap.has(metric.key)) {
        customMap.set(metric.key, {
          key: metric.key,
          label: metric.label,
          shortLabel: metric.label,
          unit: metric.unit,
          category: 'body',
          direction: 'neutral',
          description: 'Custom metric imported or entered manually.',
          aliases: [metric.label, metric.key],
          accent: '#f4a261',
        })
      }
    }
  }

  return [...metricCatalog, ...customMap.values()]
}

export function computeMetricChange(
  entries: DailyEntry[],
  metricKey: string,
  windowSize = 14,
) {
  const definition = getMetricDefinition(metricKey)
  if (!definition) {
    return null
  }

  const values = sortByDate(entries)
    .map((entry) => ({ date: entry.date, value: getMetricValue(entry, metricKey) }))
    .filter((point) => typeof point.value === 'number') as Array<{
    date: string
    value: number
  }>

  if (values.length < windowSize * 2) {
    return null
  }

  const currentWindow = values.slice(-windowSize).map((point) => point.value)
  const previousWindow = values.slice(-windowSize * 2, -windowSize).map((point) => point.value)

  const currentAverage = mean(currentWindow)
  const previousAverage = mean(previousWindow)
  const delta = currentAverage - previousAverage
  const denominator = Math.abs(previousAverage) > 0.0001 ? Math.abs(previousAverage) : 1
  const percentDelta = (delta / denominator) * 100

  return {
    key: metricKey,
    label: definition.shortLabel,
    currentAverage,
    previousAverage,
    delta,
    percentDelta,
    direction: definition.direction,
  } satisfies MetricChange
}

export function computeCardioSignal(entries: DailyEntry[], workouts: WorkoutEntry[]) {
  const changes = cardioSignalMetricKeys
    .map((metricKey) => computeMetricChange(entries, metricKey))
    .filter((change): change is MetricChange => Boolean(change))

  const weightedSignals = changes
    .filter((change) => change.direction !== 'neutral')
    .map((change) => {
      const directionMultiplier = change.direction === 'higher' ? 1 : -1
      const weight =
        change.key === 'hrv' || change.key === 'rhr'
          ? 1.4
          : change.key === 'recoveryScore'
            ? 1.2
            : 1

      return clamp(change.percentDelta * directionMultiplier, -40, 40) * weight
    })

  const workoutRecoveryTrend = computeWorkoutRecoveryTrend(workouts)
  if (workoutRecoveryTrend) {
    weightedSignals.push(clamp(workoutRecoveryTrend.delta, -15, 15) * 1.4)
  }

  const score = weightedSignals.length
    ? clamp(mean(weightedSignals) * 2.5, -100, 100)
    : 0

  const label =
    score >= 35
      ? 'Building'
      : score >= 10
        ? 'Improving'
        : score <= -35
          ? 'Flagged'
          : score <= -10
            ? 'Mixed'
            : 'Stable'

  const reasons = changes
    .sort((left, right) => Math.abs(right.percentDelta) - Math.abs(left.percentDelta))
    .slice(0, 3)
    .map((change) => {
      const trendWord =
        change.direction === 'lower'
          ? change.delta < 0
            ? 'down'
            : 'up'
          : change.delta > 0
            ? 'up'
            : 'down'

      return `${change.label} ${trendWord} ${Math.abs(change.percentDelta).toFixed(1)}%`
    })

  const summary =
    reasons.length > 0
      ? `${label} cardio signal based on your last 2 x 14-day windows: ${reasons.join(', ')}.`
      : 'Not enough history yet for a reliable cardio signal.'

  return {
    score,
    label,
    summary,
    changes,
    workoutRecoveryTrend,
  }
}

export function computeWorkoutRecoveryTrend(workouts: WorkoutEntry[]) {
  const values = sortByDate(workouts)
    .map((workout) => workout.hrRecovery1m)
    .filter((value): value is number => typeof value === 'number')

  if (values.length < 8) {
    return null
  }

  const currentAverage = mean(values.slice(-4))
  const previousAverage = mean(values.slice(-8, -4))
  const delta = currentAverage - previousAverage

  return {
    currentAverage,
    previousAverage,
    delta,
  }
}

export function buildWorkoutSeries(workouts: WorkoutEntry[]) {
  return sortByDate(workouts).map((workout) => ({
    date: workout.date,
    dateLabel: format(new Date(`${workout.date}T12:00:00`), 'MMM d'),
    avgHr: workout.avgHr,
    maxHr: workout.maxHr,
    durationMin: workout.durationMin,
    hrRecovery1m: workout.hrRecovery1m,
    strain: workout.strain,
  }))
}

export function buildCorrelationInsights(
  entries: DailyEntry[],
  targetMetricKey: string,
) {
  const availableMetrics = getAvailableMetricDefinitions(entries)
  const insights: CorrelationInsight[] = []

  for (const metric of availableMetrics) {
    if (metric.key === targetMetricKey) {
      continue
    }

    const left: number[] = []
    const right: number[] = []

    for (const entry of entries) {
      const targetValue = getMetricValue(entry, targetMetricKey)
      const candidateValue = getMetricValue(entry, metric.key)

      if (
        typeof targetValue === 'number' &&
        Number.isFinite(targetValue) &&
        typeof candidateValue === 'number' &&
        Number.isFinite(candidateValue)
      ) {
        left.push(targetValue)
        right.push(candidateValue)
      }
    }

    if (left.length < 4) {
      continue
    }

    try {
      const correlation = sampleCorrelation(left, right)
      if (Number.isFinite(correlation)) {
        insights.push({
          key: metric.key,
          label: metric.shortLabel,
          correlation,
          sampleSize: left.length,
        })
      }
    } catch {
      continue
    }
  }

  return insights.sort(
    (left, right) => Math.abs(right.correlation) - Math.abs(left.correlation),
  )
}

export function formatMetricValue(value: number | null | undefined, metricKey: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'No data'
  }

  const definition = getMetricDefinition(metricKey)
  const decimals =
    typeof definition?.decimals === 'number'
      ? definition.decimals
      : Number.isInteger(value)
        ? 0
        : 1

  return `${value.toFixed(decimals)}${definition?.unit ? ` ${definition.unit}` : ''}`
}

export function getLatestMetric(entries: DailyEntry[], metricKey: string) {
  const latest = sortByDate(entries)
    .reverse()
    .find((entry) => typeof getMetricValue(entry, metricKey) === 'number')

  return latest ? getMetricValue(latest, metricKey) : null
}

export function getWorkoutHighlights(workouts: WorkoutEntry[]) {
  const sorted = sortByDate(workouts)
  const latest = sorted.at(-1)
  const avgHr = mean(
    sorted
      .map((workout) => workout.avgHr)
      .filter((value): value is number => typeof value === 'number'),
  )
  const recovery = mean(
    sorted
      .map((workout) => workout.hrRecovery1m)
      .filter((value): value is number => typeof value === 'number'),
  )

  return {
    latestWorkout: latest,
    avgHr: Number.isFinite(avgHr) ? avgHr : null,
    recovery: Number.isFinite(recovery) ? recovery : null,
  }
}

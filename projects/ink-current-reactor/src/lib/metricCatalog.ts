import type {
  MetricDefinition,
  MetricSectionDefinition,
} from './types'

export const metricSections: MetricSectionDefinition[] = [
  {
    key: 'recovery',
    title: 'Recovery signals',
    description: 'Autonomic and recovery markers that typically move with fitness and fatigue.',
  },
  {
    key: 'sleep',
    title: 'Sleep quality',
    description: 'Sleep quantity, consistency, and debt markers pulled from nightly recovery.',
  },
  {
    key: 'vitals',
    title: 'Vitals',
    description: 'Respiratory and temperature trends that help add context to recovery changes.',
  },
  {
    key: 'load',
    title: 'Training load',
    description: 'Daily load markers that make workout trends easier to interpret.',
  },
  {
    key: 'body',
    title: 'Body context',
    description: 'Optional body markers and manually tracked measures that influence performance.',
  },
]

export const metricCatalog: MetricDefinition[] = [
  {
    key: 'recoveryScore',
    label: 'Recovery Score',
    shortLabel: 'Recovery',
    unit: '%',
    category: 'recovery',
    direction: 'higher',
    description: 'Whoop recovery percentage.',
    aliases: ['recovery', 'recovery score', 'recovery %', 'recoveryscore'],
    accent: '#7ce2ae',
  },
  {
    key: 'hrv',
    label: 'Heart Rate Variability',
    shortLabel: 'HRV',
    unit: 'ms',
    category: 'recovery',
    direction: 'higher',
    description: 'Nightly HRV in milliseconds.',
    aliases: ['hrv', 'heart rate variability'],
    accent: '#8fd2ff',
  },
  {
    key: 'rhr',
    label: 'Resting Heart Rate',
    shortLabel: 'RHR',
    unit: 'bpm',
    category: 'recovery',
    direction: 'lower',
    description: 'Resting heart rate overnight or on wake.',
    aliases: ['rhr', 'resting heart rate', 'resting hr'],
    accent: '#ff9d74',
  },
  {
    key: 'sleepPerformance',
    label: 'Sleep Performance',
    shortLabel: 'Sleep %',
    unit: '%',
    category: 'sleep',
    direction: 'higher',
    description: 'Sleep performance percentage.',
    aliases: ['sleep performance', 'sleep %', 'sleep score', 'sleepperformance'],
    accent: '#b4abff',
  },
  {
    key: 'sleepDuration',
    label: 'Sleep Duration',
    shortLabel: 'Sleep Hrs',
    unit: 'hr',
    category: 'sleep',
    direction: 'higher',
    description: 'Total sleep duration in hours.',
    aliases: ['sleep duration', 'sleep hours', 'hours asleep', 'time asleep'],
    decimals: 1,
    accent: '#5bc0eb',
  },
  {
    key: 'sleepNeed',
    label: 'Sleep Need',
    shortLabel: 'Need',
    unit: 'hr',
    category: 'sleep',
    direction: 'neutral',
    description: 'Calculated need for sleep in hours.',
    aliases: ['sleep need', 'need'],
    decimals: 1,
    accent: '#77d4d9',
  },
  {
    key: 'sleepDebt',
    label: 'Sleep Debt',
    shortLabel: 'Debt',
    unit: 'hr',
    category: 'sleep',
    direction: 'lower',
    description: 'Estimated sleep debt in hours.',
    aliases: ['sleep debt', 'debt'],
    decimals: 1,
    accent: '#ffd166',
  },
  {
    key: 'sleepConsistency',
    label: 'Sleep Consistency',
    shortLabel: 'Consistency',
    unit: '%',
    category: 'sleep',
    direction: 'higher',
    description: 'How consistent your sleep timing has been.',
    aliases: ['sleep consistency', 'consistency'],
    accent: '#60d394',
  },
  {
    key: 'respiratoryRate',
    label: 'Respiratory Rate',
    shortLabel: 'Resp Rate',
    unit: 'rpm',
    category: 'vitals',
    direction: 'lower',
    description: 'Average overnight breathing rate.',
    aliases: ['respiratory rate', 'breathing rate', 'resp rate', 'respiratoryrate'],
    decimals: 1,
    accent: '#ffb4a2',
  },
  {
    key: 'skinTemp',
    label: 'Skin Temperature Delta',
    shortLabel: 'Skin Temp',
    unit: 'deg',
    category: 'vitals',
    direction: 'neutral',
    description: 'Skin temperature delta versus baseline.',
    aliases: ['skin temp', 'skin temperature', 'skin temperature delta', 'temperature'],
    decimals: 1,
    accent: '#ff8fab',
  },
  {
    key: 'bloodOxygen',
    label: 'Blood Oxygen',
    shortLabel: 'SpO2',
    unit: '%',
    category: 'vitals',
    direction: 'higher',
    description: 'Nightly oxygen saturation.',
    aliases: ['spo2', 'blood oxygen', 'oxygen saturation', 'bloodoxygen'],
    accent: '#9bf6ff',
  },
  {
    key: 'stress',
    label: 'Stress',
    shortLabel: 'Stress',
    unit: 'score',
    category: 'vitals',
    direction: 'lower',
    description: 'Stress score or perceived load.',
    aliases: ['stress', 'stress score'],
    decimals: 1,
    accent: '#ffadad',
  },
  {
    key: 'dayStrain',
    label: 'Day Strain',
    shortLabel: 'Strain',
    unit: 'strain',
    category: 'load',
    direction: 'neutral',
    description: 'Whole-day strain or training load.',
    aliases: ['day strain', 'strain', 'daily strain', 'load'],
    decimals: 1,
    accent: '#ffcc66',
  },
  {
    key: 'steps',
    label: 'Steps',
    shortLabel: 'Steps',
    unit: 'steps',
    category: 'load',
    direction: 'higher',
    description: 'Daily step count.',
    aliases: ['steps', 'step count'],
    accent: '#bde0fe',
  },
  {
    key: 'calories',
    label: 'Calories Burned',
    shortLabel: 'Calories',
    unit: 'kcal',
    category: 'load',
    direction: 'neutral',
    description: 'Daily calories burned.',
    aliases: ['calories', 'calories burned', 'burned calories', 'kcal'],
    accent: '#ffd6a5',
  },
  {
    key: 'distance',
    label: 'Distance',
    shortLabel: 'Distance',
    unit: 'mi',
    category: 'load',
    direction: 'higher',
    description: 'Distance covered on the day.',
    aliases: ['distance', 'miles', 'mi', 'kilometers', 'km'],
    decimals: 1,
    accent: '#a0c4ff',
  },
  {
    key: 'weight',
    label: 'Weight',
    shortLabel: 'Weight',
    unit: 'lb',
    category: 'body',
    direction: 'neutral',
    description: 'Optional body weight.',
    aliases: ['weight', 'body weight'],
    decimals: 1,
    accent: '#d8e2dc',
  },
]

export const defaultTrendMetricKeys = [
  'hrv',
  'rhr',
  'sleepPerformance',
  'dayStrain',
]

export const cardioSignalMetricKeys = [
  'hrv',
  'rhr',
  'recoveryScore',
  'sleepPerformance',
  'sleepDuration',
  'respiratoryRate',
]

const normalizedAliasMap = new Map<string, string>()

for (const metric of metricCatalog) {
  for (const alias of [metric.key, metric.label, metric.shortLabel, ...metric.aliases]) {
    normalizedAliasMap.set(normalizeHeader(alias), metric.key)
  }
}

export function normalizeHeader(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
}

export function findMetricKeyFromHeader(value: string) {
  return normalizedAliasMap.get(normalizeHeader(value))
}

export function getMetricDefinition(metricKey: string) {
  return metricCatalog.find((metric) => metric.key === metricKey)
}

export function toCustomMetricKey(label: string) {
  return `custom_${normalizeHeader(label)}`
}

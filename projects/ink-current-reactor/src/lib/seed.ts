import { subDays } from 'date-fns'
import { db } from './db'
import type { DailyEntry, WorkoutEntry } from './types'

function pseudoRandom(seed: number) {
  const value = Math.sin(seed * 999) * 10000
  return value - Math.floor(value)
}

function createDemoWorkoutImage(date: string, avgHr: number, maxHr: number) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#081421"/>
          <stop offset="100%" stop-color="#1d3557"/>
        </linearGradient>
        <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#86efac"/>
          <stop offset="100%" stop-color="#f97316"/>
        </linearGradient>
      </defs>
      <rect width="900" height="520" rx="32" fill="url(#bg)"/>
      <g fill="#d8f3ff" font-family="Avenir Next, Arial, sans-serif">
        <text x="52" y="74" font-size="36" font-weight="700">Demo Whoop Heart Rate Capture</text>
        <text x="52" y="112" font-size="20" opacity="0.8">${date}</text>
        <text x="52" y="164" font-size="18" opacity="0.8">Avg HR</text>
        <text x="52" y="204" font-size="44" font-weight="700">${avgHr} bpm</text>
        <text x="250" y="164" font-size="18" opacity="0.8">Peak HR</text>
        <text x="250" y="204" font-size="44" font-weight="700">${maxHr} bpm</text>
      </g>
      <path d="M 55 395 C 120 365, 160 225, 215 245 S 310 350, 365 300 S 468 205, 520 252 S 610 352, 662 320 S 740 232, 835 278"
        fill="none" stroke="url(#line)" stroke-width="16" stroke-linecap="round"/>
      <g stroke="#5fa8d3" stroke-opacity="0.35">
        <line x1="55" x2="845" y1="430" y2="430"/>
        <line x1="55" x2="845" y1="360" y2="360"/>
        <line x1="55" x2="845" y1="290" y2="290"/>
        <line x1="55" x2="845" y1="220" y2="220"/>
      </g>
    </svg>
  `

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`
}

function createDailyEntry(index: number, today: Date): DailyEntry {
  const date = subDays(today, 69 - index).toISOString().slice(0, 10)
  const noise = (amount: number) => (pseudoRandom(index + amount) - 0.5) * amount
  const createdAt = new Date(`${date}T12:00:00`).toISOString()

  return {
    id: date,
    date,
    source: 'seed',
    notes:
      index % 7 === 0
        ? 'Demo note: harder week with travel and shorter sleep.'
        : index % 9 === 0
          ? 'Demo note: smooth training block with better consistency.'
          : '',
    metrics: {
      recoveryScore: Math.round(64 + index * 0.28 + noise(10)),
      hrv: Math.round(67 + index * 0.18 + noise(8)),
      rhr: Math.round(58 - index * 0.05 + noise(3)),
      sleepPerformance: Math.round(82 + index * 0.08 + noise(8)),
      sleepDuration: Number((7.1 + index * 0.01 + noise(0.8)).toFixed(1)),
      sleepNeed: Number((7.9 + noise(0.4)).toFixed(1)),
      sleepDebt: Number(Math.max(0, 1.9 - index * 0.02 + noise(0.5)).toFixed(1)),
      sleepConsistency: Math.round(73 + index * 0.15 + noise(8)),
      respiratoryRate: Number((14.9 - index * 0.01 + noise(0.5)).toFixed(1)),
      skinTemp: Number((noise(0.6)).toFixed(1)),
      bloodOxygen: Math.round(96 + noise(1.5)),
      stress: Number((2.5 - index * 0.01 + noise(0.6)).toFixed(1)),
      dayStrain: Number((11.5 + noise(2.2)).toFixed(1)),
      steps: Math.round(8300 + index * 40 + noise(1800)),
      calories: Math.round(2150 + index * 10 + noise(250)),
      distance: Number((3.3 + index * 0.02 + noise(0.7)).toFixed(1)),
      weight: Number((176 - index * 0.03 + noise(0.7)).toFixed(1)),
    },
    customMetrics: [
      {
        key: 'custom_zone2focus',
        label: 'Zone 2 Focus',
        unit: 'min',
        value: Math.round(18 + index * 0.3 + noise(9)),
      },
    ],
    createdAt,
    updatedAt: createdAt,
  }
}

function createWorkoutEntry(index: number, today: Date): WorkoutEntry {
  const date = subDays(today, 60 - index * 6).toISOString().slice(0, 10)
  const avgHr = Math.round(151 - index * 1.1)
  const maxHr = Math.round(176 - index * 0.6)
  const durationMin = Math.round(38 + index * 2.5)
  const createdAt = new Date(`${date}T15:00:00`).toISOString()

  return {
    id: crypto.randomUUID(),
    date,
    title: index % 2 === 0 ? 'Tempo Run' : 'Bike Endurance Block',
    sport: index % 2 === 0 ? 'Run' : 'Ride',
    durationMin,
    strain: Number((11.6 + index * 0.5).toFixed(1)),
    avgHr,
    maxHr,
    hrRecovery1m: 21 + index * 2,
    zone2Minutes: Math.round(durationMin * 0.48),
    zone3Minutes: Math.round(durationMin * 0.24),
    zone4Minutes: Math.round(durationMin * 0.16),
    zone5Minutes: Math.round(durationMin * 0.08),
    screenshot: createDemoWorkoutImage(date, avgHr, maxHr),
    notes:
      index >= 6
        ? 'Demo note: lower heart rate at similar effort, strong finish.'
        : 'Demo note: aerobic build session.',
    tags: index % 2 === 0 ? ['tempo', 'run'] : ['bike', 'zone2'],
    markers: [
      {
        id: crypto.randomUUID(),
        x: 0.26,
        y: 0.43,
        type: 'threshold',
        label: 'Threshold rise',
        value: 162,
        unit: 'bpm',
        note: 'Intensity climbed here.',
      },
      {
        id: crypto.randomUUID(),
        x: 0.68,
        y: 0.58,
        type: 'recovery',
        label: 'Fast settle',
        value: 29 + index,
        unit: '1m drop',
        note: 'Recovery improved after the hard segment.',
      },
    ],
    createdAt,
    updatedAt: createdAt,
  }
}

export async function ensureSeedData() {
  return db.transaction('rw', db.dailyEntries, db.workouts, db.meta, async () => {
    const seeded = await db.meta.get('demo-seeded')
    if (seeded) {
      return false
    }

    const today = new Date()
    const dailyEntries = Array.from({ length: 70 }, (_, index) =>
      createDailyEntry(index, today),
    )
    const workouts = Array.from({ length: 9 }, (_, index) =>
      createWorkoutEntry(index, today),
    )

    await db.dailyEntries.bulkPut(dailyEntries)
    await db.workouts.bulkPut(workouts)
    await db.meta.put({
      key: 'demo-seeded',
      value: new Date().toISOString(),
    })

    return true
  })
}

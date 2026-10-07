import Papa from 'papaparse'
import { findMetricKeyFromHeader, toCustomMetricKey } from './metricCatalog'
import type { BackupPayload, DailyEntry } from './types'

function parseNumber(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }

  if (typeof value !== 'string') {
    return null
  }

  const trimmed = value.trim().replace(/,/g, '')
  if (!trimmed) {
    return null
  }

  const numeric = Number.parseFloat(trimmed)
  return Number.isFinite(numeric) ? numeric : null
}

function normalizeDateValue(value: unknown) {
  if (typeof value !== 'string') {
    return null
  }

  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  const direct = /^\d{4}-\d{2}-\d{2}$/.test(trimmed)
    ? new Date(`${trimmed}T12:00:00`)
    : new Date(trimmed)

  if (Number.isNaN(direct.getTime())) {
    return null
  }

  return direct.toISOString().slice(0, 10)
}

export async function parseWhoopCsv(file: File) {
  const result = await new Promise<Papa.ParseResult<Record<string, string>>>((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: resolve,
      error: reject,
    })
  })

  const rows = result.data
  if (rows.length === 0) {
    return []
  }

  const headers = Object.keys(rows[0] ?? {})
  const dateHeader =
    headers.find((header) =>
      ['date', 'day', 'recovery date', 'sleep date'].includes(header.toLowerCase()),
    ) ?? headers.find((header) => header.toLowerCase().includes('date'))

  if (!dateHeader) {
    throw new Error('No date column was found in the CSV.')
  }

  const importedEntries: DailyEntry[] = []
  const now = new Date().toISOString()

  for (const row of rows) {
    const date = normalizeDateValue(row[dateHeader])
    if (!date) {
      continue
    }

    const metrics: Record<string, number | null> = {}
    const customMetrics: DailyEntry['customMetrics'] = []

    for (const header of headers) {
      if (header === dateHeader) {
        continue
      }

      const value = parseNumber(row[header])
      if (value === null) {
        continue
      }

      const knownKey = findMetricKeyFromHeader(header)
      if (knownKey) {
        metrics[knownKey] = value
        continue
      }

      customMetrics.push({
        key: toCustomMetricKey(header),
        label: header,
        unit: '',
        value,
      })
    }

    importedEntries.push({
      id: date,
      date,
      source: 'csv',
      notes: `Imported from ${file.name}`,
      metrics,
      customMetrics,
      createdAt: now,
      updatedAt: now,
    })
  }

  return importedEntries
}

export async function readBackupFile(file: File) {
  const raw = await file.text()
  const parsed = JSON.parse(raw) as BackupPayload

  if (!Array.isArray(parsed.dailyEntries) || !Array.isArray(parsed.workouts)) {
    throw new Error('Backup file is missing daily entries or workouts.')
  }

  return parsed
}

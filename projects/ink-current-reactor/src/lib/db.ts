import Dexie, { type Table } from 'dexie'
import type { DailyEntry, MetaRecord, WorkoutEntry } from './types'

export class WhoopPerformanceDB extends Dexie {
  dailyEntries!: Table<DailyEntry, string>
  workouts!: Table<WorkoutEntry, string>
  meta!: Table<MetaRecord, string>

  constructor() {
    super('whoop-performance-lab')

    this.version(1).stores({
      dailyEntries: 'id, date, updatedAt',
      workouts: 'id, date, updatedAt',
      meta: 'key',
    })
  }
}

export const db = new WhoopPerformanceDB()

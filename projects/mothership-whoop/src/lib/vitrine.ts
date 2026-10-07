/**
 * Data shaping for the Vitrine dashboard.
 *
 * Pure functions over plain rows — no React, no Prisma import — so this can run
 * on the server, in a route handler, or in a test.
 *
 * Two things WHOOP does not give us, handled honestly rather than faked:
 *   1. No beat-to-beat RR intervals. `hrv` is one scalar per recovery, so the
 *      scatter is a lag-1 recurrence plot over days, not a Poincare plot.
 *   2. No hypnogram. Sleep stores stage totals, so the core column is
 *      proportional composition, not a chronological sequence.
 *
 * Day strain comes from Cycle.strain. A cycle and its recovery share the same
 * WHOOP id, so they join exactly rather than by date proximity. Where a cycle
 * has not been scored yet we fall back to the day's peak workout strain and say
 * so via `strainSource`, because the two are not the same measure.
 */

const MIN = 60_000;

export type RecoveryRow = {
  /** WHOOP's cycle id — the join key to Cycle. */
  whoopId: string;
  createdAt: Date | string;
  score: number | null;
  restingHr: number | null;
  hrv: number | null;
};

export type SleepRow = {
  createdAt: Date | string;
  start?: Date | string | null;
  end?: Date | string | null;
  totalInBedTimeMilli: number | null;
  totalAwakeTimeMilli: number | null;
  totalLightSleepTimeMilli: number | null;
  totalSlowWaveSleepTimeMilli: number | null;
  totalRemSleepTimeMilli: number | null;
  sleepNeededMilli: number | null;
  respiratoryRate: number | null;
  sleepPerformancePercentage: number | null;
  sleepEfficiencyPercentage: number | null;
};

export type WorkoutRow = {
  start: Date | string;
  strain: number | null;
  end?: Date | string | null;
  sportId?: number | null;
  averageHeartRate?: number | null;
  maxHeartRate?: number | null;
  kilojoules?: number | null;
  distanceMeter?: number | null;
  zoneDurationMilli?: string | null;
};

export type CycleRow = {
  whoopId: string;
  start: Date | string;
  strain: number | null;
};

/** Where a strain figure came from — the two measures are not interchangeable. */
export type StrainSource = 'cycle' | 'workout' | 'none';

export type Vitals = {
  dayKey: string;
  label: string;
  recovery: number;
  hrv: number;
  restingHr: number;
  strain: number;
  strainSource: StrainSource;
};

export type VitrineData = {
  /** Most recent day, with everything the hero needs. Null when there is no data. */
  latest: {
    dayKey: string;
    label: string;
    recovery: number;
    hrv: number;
    restingHr: number;
    /** Breaths per minute — drives the membrane's breathing period. */
    respiratoryRate: number;
    sleepPerformance: number;
    efficiency: number;
    inBedMin: number;
    asleepMin: number;
    awakeMin: number;
    lightMin: number;
    remMin: number;
    deepMin: number;
    neededMin: number;
    /** WHOOP day strain when the cycle is scored; peak workout strain otherwise. */
    strain: number;
    strainSource: StrainSource;
  } | null;
  /** HRV(n) vs HRV(n+1) across the window. */
  lag: { x: number; y: number }[];
  /** HRV series, normalised 0..1, for the density band. */
  band: number[];
  /** Last seven days, oldest first. */
  week: Vitals[];
};

const toDate = (v: Date | string) => (v instanceof Date ? v : new Date(v));

/** Local calendar day, so a 2am sync doesn't land on the wrong date. */
function dayKey(v: Date | string): string {
  const d = toDate(v);
  const m = `${d.getMonth() + 1}`.padStart(2, '0');
  const day = `${d.getDate()}`.padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

const dayLabel = (v: Date | string) =>
  toDate(v).toLocaleDateString('en-US', { weekday: 'short' });

const num = (v: number | null | undefined, fallback = 0) =>
  typeof v === 'number' && Number.isFinite(v) ? v : fallback;

const toMin = (milli: number | null | undefined) => Math.round(num(milli) / MIN);

export function buildVitrineData({
  recoveries,
  sleeps,
  workouts,
  cycles = [],
}: {
  recoveries: RecoveryRow[];
  sleeps: SleepRow[];
  workouts: WorkoutRow[];
  cycles?: CycleRow[];
}): VitrineData {
  // Rows arrive ascending from the page query; sort defensively anyway.
  const recs = [...recoveries].sort(
    (a, b) => +toDate(a.createdAt) - +toDate(b.createdAt),
  );
  const scored = recs.filter((r) => typeof r.score === 'number');

  const hrvSeries = scored.map((r) => num(r.hrv));
  const lag = hrvSeries
    .slice(0, -1)
    .map((x, i) => ({ x, y: hrvSeries[i + 1] }))
    .filter((p) => p.x > 0 && p.y > 0);

  const lo = Math.min(...hrvSeries, 0);
  const hi = Math.max(...hrvSeries, 1);
  const band = hrvSeries.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5));

  const strainByCycle = new Map(
    cycles
      .filter((c) => typeof c.strain === 'number')
      .map((c) => [c.whoopId, num(c.strain)]),
  );

  // Workout strain does not sum into a day figure, so the fallback is the
  // single hardest effort of that day, not a total.
  const peakWorkoutStrain = (key: string) =>
    workouts
      .filter((w) => dayKey(w.start) === key)
      .reduce((max, w) => Math.max(max, num(w.strain)), 0);

  function strainFor(r: RecoveryRow): { strain: number; strainSource: StrainSource } {
    const cycleStrain = strainByCycle.get(r.whoopId);
    if (typeof cycleStrain === 'number') {
      return { strain: cycleStrain, strainSource: 'cycle' };
    }
    const peak = peakWorkoutStrain(dayKey(r.createdAt));
    return peak > 0
      ? { strain: peak, strainSource: 'workout' }
      : { strain: 0, strainSource: 'none' };
  }

  const week: Vitals[] = scored.slice(-7).map((r) => ({
    dayKey: dayKey(r.createdAt),
    label: dayLabel(r.createdAt),
    recovery: Math.round(num(r.score)),
    hrv: Math.round(num(r.hrv)),
    restingHr: Math.round(num(r.restingHr)),
    ...strainFor(r),
  }));

  const lastRec = scored.at(-1);
  if (!lastRec) return { latest: null, lag, band, week };

  const key = dayKey(lastRec.createdAt);

  // Pair the recovery with the sleep that produced it, falling back to the most
  // recent sleep so the hero still renders when the days don't line up.
  const sleepsAsc = [...sleeps].sort(
    (a, b) => +toDate(a.createdAt) - +toDate(b.createdAt),
  );
  const sleep =
    sleepsAsc.filter((s) => dayKey(s.createdAt) === key).at(-1) ??
    sleepsAsc.at(-1) ??
    null;

  const inBedMin = toMin(sleep?.totalInBedTimeMilli);
  const awakeMin = toMin(sleep?.totalAwakeTimeMilli);

  return {
    latest: {
      dayKey: key,
      label: toDate(lastRec.createdAt).toLocaleDateString('en-US', {
        weekday: 'long',
        day: 'numeric',
        month: 'long',
      }),
      recovery: Math.round(num(lastRec.score)),
      hrv: Math.round(num(lastRec.hrv)),
      restingHr: Math.round(num(lastRec.restingHr)),
      respiratoryRate: num(sleep?.respiratoryRate, 0),
      sleepPerformance: Math.round(num(sleep?.sleepPerformancePercentage)),
      efficiency: num(sleep?.sleepEfficiencyPercentage),
      inBedMin,
      asleepMin: Math.max(0, inBedMin - awakeMin),
      awakeMin,
      lightMin: toMin(sleep?.totalLightSleepTimeMilli),
      remMin: toMin(sleep?.totalRemSleepTimeMilli),
      deepMin: toMin(sleep?.totalSlowWaveSleepTimeMilli),
      neededMin: toMin(sleep?.sleepNeededMilli),
      ...strainFor(lastRec),
    },
    lag,
    band,
    week,
  };
}

/** 466 -> "7:46". */
export function hhmm(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return `${h}:${`${m}`.padStart(2, '0')}`;
}

/**
 * The one place recovery becomes colour. Cyan is parasympathetic (recovered),
 * magenta is sympathetic (depleted), violet is the middle.
 */
export function recoveryColor(recovery: number): string {
  if (recovery >= 67) return '#8FFFE0';
  if (recovery >= 34) return '#5B3FFF';
  return '#FF4FD8';
}

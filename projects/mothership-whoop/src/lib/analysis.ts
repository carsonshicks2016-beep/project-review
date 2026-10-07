/**
 * Derived physiology for the Vitrine console.
 *
 * Pure functions over plain rows — no React, no Prisma — so every claim the UI
 * makes can be tested without a browser or a database.
 *
 * Standing rule in here: never present a computed number as more certain than
 * the data supports. Small samples carry their `n`, absent inputs return
 * `available: false` rather than a zero that reads like a measurement.
 */

import type { RecoveryRow, SleepRow, WorkoutRow, CycleRow } from './vitrine';

const MIN = 60_000;
export const WINDOWS = [7, 14, 30] as const;
export type Window = (typeof WINDOWS)[number];

const toDate = (v: Date | string) => (v instanceof Date ? v : new Date(v));
const num = (v: number | null | undefined, f = 0) =>
  typeof v === 'number' && Number.isFinite(v) ? v : f;
const toMin = (milli: number | null | undefined) => Math.round(num(milli) / MIN);

export function dayKey(v: Date | string): string {
  const d = toDate(v);
  return `${d.getFullYear()}-${`${d.getMonth() + 1}`.padStart(2, '0')}-${`${d.getDate()}`.padStart(2, '0')}`;
}
const shortDay = (v: Date | string) =>
  toDate(v).toLocaleDateString('en-US', { weekday: 'short' });

/* ------------------------------------------------------------------ stats */

export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Sample standard deviation (n-1). Returns 0 for fewer than two points. */
export function stdev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
}

export type Baseline = {
  mean: number;
  sd: number;
  /** Standard deviations from baseline. 0 when the baseline is degenerate. */
  z: number;
  /** Raw difference from the mean, in the metric's own units. */
  delta: number;
  n: number;
  available: boolean;
};

/**
 * Where today sits against its own recent history. `history` must EXCLUDE the
 * current value, or the point being measured drags its own baseline.
 */
export function baseline(history: number[], current: number): Baseline {
  const xs = history.filter((x) => Number.isFinite(x));
  const m = mean(xs);
  const sd = stdev(xs);
  return {
    mean: m,
    sd,
    z: sd > 0 ? (current - m) / sd : 0,
    delta: current - m,
    n: xs.length,
    available: xs.length >= 3,
  };
}

/** Pearson correlation. Null when n < 3 or either series is constant. */
export function pearson(xs: number[], ys: number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 3) return null;
  const mx = mean(xs.slice(0, n));
  const my = mean(ys.slice(0, n));
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx === 0 || syy === 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}

/* ------------------------------------------------------------- sleep debt */

export type DebtPoint = {
  dayKey: string;
  label: string;
  neededMin: number;
  asleepMin: number;
  /** Positive means short of need. */
  deficitMin: number;
  cumulativeMin: number;
};

/**
 * Running sleep debt. Requires sleepNeededMilli, which WHOOP supplies but an
 * older sync may not have stored — hence `available`.
 */
export function sleepDebt(sleeps: SleepRow[]): {
  points: DebtPoint[];
  totalMin: number;
  available: boolean;
} {
  const rows = [...sleeps].sort((a, b) => +toDate(a.createdAt) - +toDate(b.createdAt));
  const usable = rows.filter((s) => num(s.sleepNeededMilli) > 0);

  let cumulative = 0;
  const points = usable.map((s) => {
    const neededMin = toMin(s.sleepNeededMilli);
    const asleepMin = toMin(s.totalInBedTimeMilli) - toMin(s.totalAwakeTimeMilli);
    const deficitMin = neededMin - asleepMin;
    cumulative += deficitMin;
    return {
      dayKey: dayKey(s.createdAt),
      label: shortDay(s.createdAt),
      neededMin,
      asleepMin,
      deficitMin,
      cumulativeMin: cumulative,
    };
  });

  return { points, totalMin: cumulative, available: points.length > 0 };
}

/* --------------------------------------------------------------- circadian */

export type CircadianPoint = {
  dayKey: string;
  label: string;
  /** Minutes from midnight; negative for an onset before midnight. */
  onsetMin: number;
  /** Minutes from midnight. */
  wakeMin: number;
  durationMin: number;
};

/**
 * Bed and wake clock times per night. Onset is folded around midnight so a
 * 23:40 start reads as -20 and a 00:30 start as +30, keeping the axis
 * continuous instead of jumping a full day.
 */
export function circadian(sleeps: SleepRow[]): {
  points: CircadianPoint[];
  onsetSd: number;
  wakeSd: number;
  available: boolean;
} {
  const rows = [...sleeps]
    .filter((s) => s.start && s.end)
    .sort((a, b) => +toDate(a.start!) - +toDate(b.start!));

  const points = rows.map((s) => {
    const start = toDate(s.start!);
    const end = toDate(s.end!);
    const raw = start.getHours() * 60 + start.getMinutes();
    return {
      dayKey: dayKey(s.createdAt),
      label: shortDay(s.createdAt),
      onsetMin: raw >= 720 ? raw - 1440 : raw,
      wakeMin: end.getHours() * 60 + end.getMinutes(),
      durationMin: Math.round((+end - +start) / MIN),
    };
  });

  return {
    points,
    onsetSd: stdev(points.map((p) => p.onsetMin)),
    wakeSd: stdev(points.map((p) => p.wakeMin)),
    available: points.length >= 2,
  };
}

/** -0:20 -> "23:40", 450 -> "07:30". */
export function clockLabel(minFromMidnight: number): string {
  const m = ((Math.round(minFromMidnight) % 1440) + 1440) % 1440;
  return `${`${Math.floor(m / 60)}`.padStart(2, '0')}:${`${m % 60}`.padStart(2, '0')}`;
}

/* ------------------------------------------------------------ correlations */

export type Correlation = {
  key: string;
  label: string;
  detail: string;
  r: number | null;
  n: number;
};

/**
 * Relationships worth watching. All are correlational over a handful of days —
 * the UI must show `n` and must not phrase any of them as causal.
 */
export function correlations({
  recoveries,
  sleeps,
  cycles,
}: {
  recoveries: RecoveryRow[];
  sleeps: SleepRow[];
  cycles: CycleRow[];
}): Correlation[] {
  const recs = [...recoveries]
    .filter((r) => typeof r.score === 'number')
    .sort((a, b) => +toDate(a.createdAt) - +toDate(b.createdAt));

  const strainByCycle = new Map(
    cycles.filter((c) => typeof c.strain === 'number').map((c) => [c.whoopId, num(c.strain)]),
  );
  const sleepByDay = new Map(sleeps.map((s) => [dayKey(s.createdAt), s]));

  // Strain today against recovery tomorrow.
  const sx: number[] = [];
  const sy: number[] = [];
  for (let i = 0; i < recs.length - 1; i++) {
    const strain = strainByCycle.get(recs[i].whoopId);
    if (typeof strain === 'number') {
      sx.push(strain);
      sy.push(num(recs[i + 1].score));
    }
  }

  // Sleep duration on the night against that morning's recovery.
  const dx: number[] = [];
  const dy: number[] = [];
  for (const r of recs) {
    const s = sleepByDay.get(dayKey(r.createdAt));
    if (s && num(s.totalInBedTimeMilli) > 0) {
      dx.push(toMin(s.totalInBedTimeMilli) - toMin(s.totalAwakeTimeMilli));
      dy.push(num(r.score));
    }
  }

  // HRV against recovery — expected to be strong; a weak result means something
  // is off in the data rather than in the physiology.
  const hx = recs.filter((r) => num(r.hrv) > 0).map((r) => num(r.hrv));
  const hy = recs.filter((r) => num(r.hrv) > 0).map((r) => num(r.score));

  // Resting heart rate against recovery — expected negative.
  const rx = recs.filter((r) => num(r.restingHr) > 0).map((r) => num(r.restingHr));
  const ry = recs.filter((r) => num(r.restingHr) > 0).map((r) => num(r.score));

  return [
    {
      key: 'strain-next',
      label: 'Strain → next-day recovery',
      detail: "Yesterday's day strain against this morning's score",
      r: pearson(sx, sy),
      n: sx.length,
    },
    {
      key: 'sleep-recovery',
      label: 'Sleep duration → recovery',
      detail: 'Hours actually asleep against the same morning',
      r: pearson(dx, dy),
      n: dx.length,
    },
    {
      key: 'hrv-recovery',
      label: 'HRV → recovery',
      detail: 'Should be strongly positive; if not, suspect the data',
      r: pearson(hx, hy),
      n: hx.length,
    },
    {
      key: 'rhr-recovery',
      label: 'Resting HR → recovery',
      detail: 'Should be negative — a higher resting rate costs you',
      r: pearson(rx, ry),
      n: rx.length,
    },
  ];
}

/* ---------------------------------------------------------------- rhythm */

export type WeekdayCell = { weekday: string; index: number; mean: number; n: number };

/** Average recovery by day of the week — the pattern a line chart hides. */
export function weekdayRhythm(recoveries: RecoveryRow[]): WeekdayCell[] {
  const names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const buckets: number[][] = names.map(() => []);

  for (const r of recoveries) {
    if (typeof r.score !== 'number') continue;
    buckets[toDate(r.createdAt).getDay()].push(r.score);
  }

  // Weeks start Monday here; Sunday moves to the end.
  return [1, 2, 3, 4, 5, 6, 0].map((i) => ({
    weekday: names[i],
    index: i,
    mean: buckets[i].length ? mean(buckets[i]) : 0,
    n: buckets[i].length,
  }));
}

/* ----------------------------------------------------------------- zones */

export type ZoneTotals = { zone: number; label: string; minutes: number }[];

/**
 * Heart-rate zone time, summed across workouts. The sync stores WHOOP's
 * zone_duration object as a JSON string, so it has to be parsed back.
 */
export function hrZones(workouts: (WorkoutRow & { zoneDurationMilli?: string | null })[]): {
  totals: ZoneTotals;
  totalMin: number;
  available: boolean;
} {
  const keys = [
    'zone_zero_milli',
    'zone_one_milli',
    'zone_two_milli',
    'zone_three_milli',
    'zone_four_milli',
    'zone_five_milli',
  ];
  const labels = ['0 – 50%', '50 – 60%', '60 – 70%', '70 – 80%', '80 – 90%', '90 – 100%'];
  const sums = new Array(6).fill(0);
  let parsed = 0;

  for (const w of workouts) {
    if (!w.zoneDurationMilli) continue;
    try {
      const z = JSON.parse(w.zoneDurationMilli);
      keys.forEach((k, i) => (sums[i] += num(z?.[k])));
      parsed++;
    } catch {
      // A malformed row should not take the panel down.
    }
  }

  const totals = sums.map((ms, i) => ({
    zone: i,
    label: labels[i],
    minutes: Math.round(ms / MIN),
  }));

  return {
    totals,
    totalMin: totals.reduce((a, z) => a + z.minutes, 0),
    available: parsed > 0 && totals.some((z) => z.minutes > 0),
  };
}

/* ---------------------------------------------------------------- sports */

/** WHOOP sport ids. Unlisted ids fall back to their number. */
const SPORTS: Record<number, string> = {
  [-1]: 'Activity', 0: 'Running', 1: 'Cycling', 16: 'Baseball', 17: 'Basketball',
  18: 'Rowing', 19: 'Fencing', 20: 'Field Hockey', 21: 'Football', 22: 'Golf',
  24: 'Ice Hockey', 25: 'Lacrosse', 27: 'Rugby', 28: 'Sailing', 29: 'Skiing',
  30: 'Soccer', 31: 'Softball', 32: 'Squash', 33: 'Swimming', 34: 'Tennis',
  35: 'Track & Field', 36: 'Volleyball', 37: 'Water Polo', 38: 'Wrestling',
  39: 'Boxing', 42: 'Dance', 43: 'Pilates', 44: 'Yoga', 45: 'Weightlifting',
  47: 'Cross Country Skiing', 48: 'Functional Fitness', 49: 'Duathlon',
  51: 'Gymnastics', 52: 'Hiking/Rucking', 53: 'Horseback Riding',
  55: 'Kayaking', 56: 'Martial Arts', 57: 'Meditation', 59: 'Mountain Biking',
  60: 'Powerlifting', 61: 'Rock Climbing', 62: 'Paddleboarding', 63: 'Triathlon',
  64: 'Walking', 65: 'Surfing', 66: 'Elliptical', 70: 'Spin', 71: 'Stairmaster',
  73: 'Pit Practice', 74: 'Diving', 75: 'Operations - Tactical',
  82: 'Barre', 83: 'Stretching', 84: 'Table Tennis', 85: 'Badminton',
  86: 'Netball', 87: 'Sauna', 88: 'Disc Golf', 89: 'Climber',
  91: 'Jiu Jitsu', 92: 'Manual Labour', 93: 'Cricket', 94: 'Pickleball',
  95: 'Inline Skating', 96: 'Box Fitness', 97: 'Spikeball', 98: 'Wheelchair Pushing',
  101: 'Ultimate', 102: 'Climbing', 104: 'Ice Bath', 105: 'Cold Water Swim',
  125: 'Motocross', 126: 'Caddying', 131: 'Obstacle Course Racing',
  230: 'Cheerleading', 231: 'Wrestling (Folkstyle)',
};
export const sportName = (id: number | null | undefined): string =>
  typeof id === 'number' ? (SPORTS[id] ?? `Sport ${id}`) : 'Activity';

/* --------------------------------------------------------- records/streak */

export type Extreme = { label: string; value: string; when: string };

export function records({
  recoveries,
  sleeps,
  cycles,
}: {
  recoveries: RecoveryRow[];
  sleeps: SleepRow[];
  cycles: CycleRow[];
}): Extreme[] {
  const fmt = (v: Date | string) =>
    toDate(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  const hm = (m: number) => `${Math.floor(m / 60)}:${`${m % 60}`.padStart(2, '0')}`;
  const out: Extreme[] = [];

  const scored = recoveries.filter((r) => typeof r.score === 'number');
  if (scored.length) {
    const best = scored.reduce((a, b) => (num(b.score) > num(a.score) ? b : a));
    const worst = scored.reduce((a, b) => (num(b.score) < num(a.score) ? b : a));
    out.push({ label: 'Best recovery', value: `${Math.round(num(best.score))}%`, when: fmt(best.createdAt) });
    out.push({ label: 'Worst recovery', value: `${Math.round(num(worst.score))}%`, when: fmt(worst.createdAt) });
  }

  const withHrv = recoveries.filter((r) => num(r.hrv) > 0);
  if (withHrv.length) {
    const peak = withHrv.reduce((a, b) => (num(b.hrv) > num(a.hrv) ? b : a));
    out.push({ label: 'Peak HRV', value: `${Math.round(num(peak.hrv))} ms`, when: fmt(peak.createdAt) });
  }

  const withSleep = sleeps.filter((s) => num(s.totalInBedTimeMilli) > 0);
  if (withSleep.length) {
    const longest = withSleep.reduce((a, b) =>
      num(b.totalInBedTimeMilli) > num(a.totalInBedTimeMilli) ? b : a,
    );
    out.push({
      label: 'Longest sleep',
      value: hm(toMin(longest.totalInBedTimeMilli) - toMin(longest.totalAwakeTimeMilli)),
      when: fmt(longest.createdAt),
    });
  }

  const strained = cycles.filter((c) => typeof c.strain === 'number');
  if (strained.length) {
    const hardest = strained.reduce((a, b) => (num(b.strain) > num(a.strain) ? b : a));
    out.push({ label: 'Hardest day', value: num(hardest.strain).toFixed(1), when: fmt(hardest.start) });
  }

  return out;
}

export type Streak = { current: number; best: number; threshold: number };

/** Consecutive days at or above `threshold` recovery, counted back from today. */
export function streaks(recoveries: RecoveryRow[], threshold = 67): Streak {
  const scored = [...recoveries]
    .filter((r) => typeof r.score === 'number')
    .sort((a, b) => +toDate(a.createdAt) - +toDate(b.createdAt));

  let best = 0;
  let run = 0;
  for (const r of scored) {
    run = num(r.score) >= threshold ? run + 1 : 0;
    best = Math.max(best, run);
  }
  return { current: run, best, threshold };
}

/* ------------------------------------------------------------------- csv */

const csvCell = (v: unknown) => {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  return [
    cols.join(','),
    ...rows.map((r) => cols.map((c) => csvCell(r[c])).join(',')),
  ].join('\n');
}

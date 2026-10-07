export type FlatValue = { path: string; value: string | number | boolean | null };

export function flattenValues(value: unknown, prefix = ''): FlatValue[] {
  if (value === null || ['string', 'number', 'boolean'].includes(typeof value)) {
    return [{ path: prefix || 'value', value: value as string | number | boolean | null }];
  }
  if (Array.isArray(value)) {
    if (!value.length) return [{ path: prefix || 'value', value: '[]' }];
    return value.flatMap((item, index) => flattenValues(item, `${prefix}[${index}]`));
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return [{ path: prefix || 'value', value: '{}' }];
    return entries.flatMap(([key, item]) => flattenValues(item, prefix ? `${prefix}.${key}` : key));
  }
  return [{ path: prefix || 'value', value: String(value) }];
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export type DailyMetric = { date: string; recovery: number | null; hrv: number | null; restingHr: number | null; sleepPerformance: number | null; strain: number | null; sleepMinutes: number | null };

export function buildDailySeries(records: Array<{ category: string; record_id?: string; start_at: string | null; timezone_offset?: string | null; raw: Record<string, any> }>): DailyMetric[] {
  const daily = new Map<string, DailyMetric>();
  const localDay = (instant: string | null, offset?: string | null) => {
    if (!instant) return '';
    const match = offset?.match(/^([+-])(\d{2}):(\d{2})$/);
    const minutes = match ? (Number(match[2]) * 60 + Number(match[3])) * (match[1] === '+' ? 1 : -1) : 0;
    const time = Date.parse(instant);
    return Number.isFinite(time) ? new Date(time + minutes * 60_000).toISOString().slice(0, 10) : '';
  };
  const cycleDay = new Map<string, string>();
  for (const record of records) if (record.category === 'cycles' && record.record_id) cycleDay.set(record.record_id, localDay(record.start_at, record.timezone_offset));
  const get = (date: string) => {
    if (!daily.has(date)) daily.set(date, { date, recovery: null, hrv: null, restingHr: null, sleepPerformance: null, strain: null, sleepMinutes: null });
    return daily.get(date)!;
  };
  for (const record of records) {
    const relatedCycle = record.raw.cycle_id === undefined ? '' : String(record.raw.cycle_id);
    const date = (record.category === 'recovery' || record.category === 'sleep') && cycleDay.has(relatedCycle)
      ? cycleDay.get(relatedCycle)!
      : localDay(record.start_at, record.timezone_offset);
    if (!date) continue;
    const target = get(date);
    const score = record.raw.score ?? {};
    if (record.category === 'recovery') {
      target.recovery = numberOrNull(score.recovery_score);
      target.hrv = numberOrNull(score.hrv_rmssd_milli);
      target.restingHr = numberOrNull(score.resting_heart_rate);
    } else if (record.category === 'cycles') target.strain = numberOrNull(score.strain);
    else if (record.category === 'sleep') {
      target.sleepPerformance = numberOrNull(score.sleep_performance_percentage);
      const stages = score.stage_summary ?? {};
      const inBed = numberOrNull(stages.total_in_bed_time_milli);
      if (inBed !== null) target.sleepMinutes = Math.round(inBed / 60_000);
    }
  }
  return [...daily.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function recoveryInsights(series: DailyMetric[]) {
  const scored = series.filter(point => point.recovery !== null);
  const latest = scored.at(-1) ?? null;
  const prior = scored.slice(-22, -1).map(point => point.recovery!).filter(Number.isFinite);
  const center = median(prior);
  const mad = center === null ? null : median(prior.map(value => Math.abs(value - center)));
  const threshold = center === null || mad === null ? null : Math.max(12, mad * 2.5);
  const shift = latest && center !== null && threshold !== null && Math.abs(latest.recovery! - center) >= threshold
    ? { date: latest.date, value: latest.recovery, baseline: Math.round(center), delta: Math.round(latest.recovery! - center), confidence: prior.length >= 14 ? 'moderate' : 'low' }
    : null;
  const baselineWindow = scored.slice(-22, -1).map(point => point.recovery!).filter(Number.isFinite);
  return {
    latest, baseline: baselineWindow.length >= 5 ? Math.round(median(baselineWindow)!) : null,
    baselineSamples: baselineWindow.length, shift,
    note: baselineWindow.length < 5 ? 'More scored recovery days are needed to establish your personal baseline.' : 'Compared with your own recent history; this is a wearable signal, not an explanation or diagnosis.',
  };
}

export function eventRecoveryResponse(series: DailyMetric[], eventDate: string, eventEndDate = eventDate) {
  const eventTime = Date.parse(`${eventDate}T00:00:00Z`);
  const endTime = Date.parse(`${eventEndDate}T00:00:00Z`);
  const preceding = series.filter(point => point.recovery !== null && Date.parse(`${point.date}T00:00:00Z`) < eventTime && Date.parse(`${point.date}T00:00:00Z`) >= eventTime - 28 * 86_400_000).map(point => point.recovery!);
  const after = series.filter(point => point.recovery !== null && Date.parse(`${point.date}T00:00:00Z`) > endTime && Date.parse(`${point.date}T00:00:00Z`) <= endTime + 21 * 86_400_000);
  if (preceding.length < 5) return { status: 'insufficient_baseline', days: null, baseline: null, baselineSamples: preceding.length, returnDate: null };
  const center = median(preceding)!;
  const mad = median(preceding.map(value => Math.abs(value - center))) ?? 0;
  const lowerBound = center - Math.max(8, mad * 2);
  for (let index = 0; index + 1 < after.length; index += 1) {
    if (after[index].recovery! >= lowerBound && after[index + 1].recovery! >= lowerBound) {
      const returnDate = after[index].date;
      return { status: 'observed', days: Math.round((Date.parse(`${returnDate}T00:00:00Z`) - endTime) / 86_400_000), baseline: Math.round(center), baselineSamples: preceding.length, returnDate };
    }
  }
  const latestTime = Date.parse(`${series.at(-1)?.date ?? eventDate}T00:00:00Z`);
  return { status: latestTime < endTime + 14 * 86_400_000 ? 'waiting' : 'not_observed', days: null, baseline: Math.round(center), baselineSamples: preceding.length, returnDate: null };
}

export function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? '' : typeof value === 'string' ? value : JSON.stringify(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function fieldUnit(path: string): string {
  const key = path.toLowerCase();
  if (key.includes('percentage') || key.includes('recovery_score') || key.includes('spo2')) return '%';
  if (key.includes('heart_rate') || key.includes('resting_hr')) return 'bpm';
  if (key.includes('hrv') || key.endsWith('_milli')) return 'ms';
  if (key.includes('temperature') || key.includes('skin_temp')) return '°C';
  if (key.includes('respiratory_rate')) return 'breaths/min';
  if (key.includes('kilojoule')) return 'kJ';
  if (key.includes('kilogram') || key.includes('weight')) return 'kg';
  if (key.includes('meter')) return 'm';
  if (key.includes('step_count')) return 'steps';
  if (key.endsWith('.strain') || key === 'strain') return 'strain';
  return '—';
}

export function fieldDefinition(path: string): string {
  const known: Record<string, string> = {
    'score.recovery_score': 'WHOOP Recovery Score for the physiological cycle.',
    'score.resting_heart_rate': 'Resting heart rate recorded for this recovery.',
    'score.hrv_rmssd_milli': 'Heart rate variability measured as RMSSD.',
    'score.spo2_percentage': 'Blood oxygen percentage when available for the device.',
    'score.skin_temp_celsius': 'Skin temperature in Celsius relative to WHOOP measurement.',
    'score.strain': 'WHOOP cardiovascular Strain for this cycle or workout.',
    'score.average_heart_rate': 'Average heart rate for the cycle or workout.',
    'score.max_heart_rate': 'Maximum heart rate for the cycle or workout.',
    'score.kilojoule': 'Estimated energy expenditure in kilojoules.',
    'score.respiratory_rate': 'Breaths per minute during sleep.',
    'score.sleep_performance_percentage': 'Sleep achieved as a percentage of sleep need.',
    'score.sleep_consistency_percentage': 'Consistency of sleep timing as a percentage.',
    'score.sleep_efficiency_percentage': 'Sleep time as a percentage of time in bed.',
    'score.stage_summary.total_in_bed_time_milli': 'Total time in bed, in milliseconds.',
    'score.stage_summary.total_awake_time_milli': 'Time awake during the sleep, in milliseconds.',
    'score.stage_summary.total_light_sleep_time_milli': 'Light sleep duration, in milliseconds.',
    'score.stage_summary.total_slow_wave_sleep_time_milli': 'Slow-wave sleep duration, in milliseconds.',
    'score.stage_summary.total_rem_sleep_time_milli': 'REM sleep duration, in milliseconds.',
    'step_count': 'Steps during the physiological cycle, when available.',
    'height_meter': 'Body height reported in meters.',
    'weight_kilogram': 'Body weight reported in kilograms.',
    'max_heart_rate': 'Maximum heart rate used by WHOOP.',
  };
  return known[path] ?? 'WHOOP provider field. Open the original record to inspect its complete context.';
}

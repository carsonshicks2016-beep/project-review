import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildDailySeries, csvCell, eventRecoveryResponse, fieldUnit, flattenValues, recoveryInsights } from './analytics.js';

test('flattenValues preserves nested paths, arrays, booleans, and nulls', () => {
  assert.deepEqual(flattenValues({ score: { recovery: 88 }, stages: [1, null], ready: false }), [
    { path: 'score.recovery', value: 88 }, { path: 'stages[0]', value: 1 },
    { path: 'stages[1]', value: null }, { path: 'ready', value: false },
  ]);
});

test('buildDailySeries derives only values present in provider records', () => {
  const result = buildDailySeries([
    { category: 'recovery', start_at: '2026-01-01T08:00:00Z', raw: { score: { recovery_score: 70, hrv_rmssd_milli: 42 } } },
    { category: 'cycles', start_at: '2026-01-01T08:00:00Z', raw: { score: { strain: 8 } } },
    { category: 'sleep', start_at: '2026-01-01T08:00:00Z', raw: { score: { stage_summary: { total_in_bed_time_milli: 28_800_000 } } } },
  ]);
  assert.deepEqual(result, [{ date: '2026-01-01', recovery: 70, hrv: 42, restingHr: null, sleepPerformance: null, strain: 8, sleepMinutes: 480 }]);
});

test('buildDailySeries assigns recovery to its physiological cycle day and timezone', () => {
  const result = buildDailySeries([
    { category: 'cycles', record_id: '99', start_at: '2026-01-01T03:30:00Z', timezone_offset: '-05:00', raw: { id: 99 } },
    { category: 'recovery', start_at: '2026-01-02T11:00:00Z', raw: { cycle_id: 99, score: { recovery_score: 88 } } },
  ]);
  assert.equal(result[0].date, '2025-12-31');
  assert.equal(result[0].recovery, 88);
});

test('recoveryInsights waits for enough data and labels deviations as shifts', () => {
  const points = Array.from({ length: 8 }, (_, index) => ({ date: `2026-01-${String(index + 1).padStart(2, '0')}`, recovery: index === 7 ? 40 : 80, hrv: null, restingHr: null, sleepPerformance: null, strain: null, sleepMinutes: null }));
  const result = recoveryInsights(points);
  assert.equal(result.shift?.delta, -40);
  assert.equal(result.shift?.confidence, 'low');
  assert.match(result.note, /not an explanation/);
  assert.equal(recoveryInsights(points.slice(0, 4)).baseline, null);
});

test('eventRecoveryResponse finds two consecutive days returning within a personal band', () => {
  const series = Array.from({ length: 10 }, (_, index) => ({ date: `2026-02-${String(index + 1).padStart(2, '0')}`, recovery: index < 5 ? 80 : index === 5 ? 50 : index === 6 ? 75 : index >= 7 ? 78 : null, hrv: null, restingHr: null, sleepPerformance: null, strain: null, sleepMinutes: null }));
  const response = eventRecoveryResponse(series, '2026-02-06');
  assert.equal(response.status, 'observed');
  assert.equal(response.returnDate, '2026-02-07');
  assert.equal(response.days, 1);
});

test('csvCell safely quotes values containing delimiters', () => {
  assert.equal(csvCell('one,"two"'), '"one,""two"""');
  assert.equal(csvCell(12), '12');
});

test('fieldUnit labels known WHOOP fields and leaves unknown units explicit', () => {
  assert.equal(fieldUnit('score.hrv_rmssd_milli'), 'ms');
  assert.equal(fieldUnit('score.average_heart_rate'), 'bpm');
  assert.equal(fieldUnit('score.recovery_score'), '%');
  assert.equal(fieldUnit('unmapped_value'), '—');
});

'use client';

import { useCallback, useMemo, useState } from 'react';
import {
  baseline,
  circadian,
  correlations,
  hrZones,
  records,
  sleepDebt,
  streaks,
  toCsv,
  weekdayRhythm,
  WINDOWS,
  type Window,
} from '@/lib/analysis';
import {
  buildVitrineData,
  type CycleRow,
  type RecoveryRow,
  type SleepRow,
  type WorkoutRow,
} from '@/lib/vitrine';
import { CommandPalette, type Command } from './CommandPalette';
import { BaselinePanel, CorrelationPanel, WeekdayPanel, type BaselineRow } from './PanelsSignal';
import { CircadianPanel, SleepDebtPanel } from './PanelsNight';
import { RecordsPanel, WorkoutPanel, ZonePanel } from './PanelsLoad';
import { VitrineStage } from './VitrineStage';
import { Label } from './primitives';

export type ConsoleRows = {
  recoveries: RecoveryRow[];
  sleeps: SleepRow[];
  workouts: WorkoutRow[];
  cycles: CycleRow[];
};

const SECTIONS = [
  { id: 'status', label: 'Status' },
  { id: 'signal', label: 'Signal' },
  { id: 'night', label: 'Night' },
  { id: 'load', label: 'Load' },
] as const;

const DAY = 86_400_000;
const at = (v: Date | string) => +new Date(v);

export function VitrineConsole({ rows }: { rows: ConsoleRows }) {
  const [days, setDays] = useState<Window>(30);

  /**
   * The window is anchored to the most recent record, not to now. Anchoring to
   * the clock would empty every panel the moment a sync goes a few days stale,
   * which reads as a broken dashboard rather than as old data.
   */
  const { view, range } = useMemo(() => {
    const anchor = rows.recoveries.reduce((max, r) => Math.max(max, at(r.createdAt)), 0);
    if (!anchor) {
      return { view: rows, range: null as null | { from: Date; to: Date } };
    }
    const cutoff = anchor - (days - 1) * DAY;
    const keep = (v: Date | string) => at(v) >= cutoff;

    return {
      view: {
        recoveries: rows.recoveries.filter((r) => keep(r.createdAt)),
        sleeps: rows.sleeps.filter((s) => keep(s.createdAt)),
        workouts: rows.workouts.filter((w) => keep(w.start)),
        cycles: rows.cycles.filter((c) => keep(c.start)),
      },
      range: { from: new Date(cutoff), to: new Date(anchor) },
    };
  }, [rows, days]);

  const data = useMemo(() => buildVitrineData(view), [view]);

  const analysis = useMemo(() => {
    const scored = [...view.recoveries]
      .filter((r) => typeof r.score === 'number')
      .sort((a, b) => at(a.createdAt) - at(b.createdAt));

    // The baseline must exclude today, or the point being judged drags its own
    // reference. `slice(0, -1)` is doing that work.
    const history = scored.slice(0, -1);
    const latest = scored.at(-1);

    const baselines: BaselineRow[] = latest
      ? [
          {
            key: 'hrv',
            label: 'HRV',
            unit: ' ms',
            current: Math.round(latest.hrv ?? 0),
            stat: baseline(history.map((r) => r.hrv ?? 0), latest.hrv ?? 0),
            higherIsBetter: true,
          },
          {
            key: 'rhr',
            label: 'Resting HR',
            unit: ' bpm',
            current: Math.round(latest.restingHr ?? 0),
            stat: baseline(history.map((r) => r.restingHr ?? 0), latest.restingHr ?? 0),
            higherIsBetter: false,
          },
          {
            key: 'recovery',
            label: 'Recovery',
            unit: '%',
            current: Math.round(latest.score ?? 0),
            stat: baseline(history.map((r) => r.score ?? 0), latest.score ?? 0),
            higherIsBetter: true,
          },
        ]
      : [];

    return {
      baselines,
      correlations: correlations(view),
      weekday: weekdayRhythm(view.recoveries),
      debt: sleepDebt(view.sleeps),
      circadian: circadian(view.sleeps),
      zones: hrZones(view.workouts),
      extremes: records(view),
      streak: streaks(view.recoveries),
    };
  }, [view]);

  const jump = useCallback((id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  const exportCsv = useCallback(() => {
    const byDay = new Map(view.sleeps.map((s) => [new Date(s.createdAt).toDateString(), s]));
    const strain = new Map(view.cycles.map((c) => [c.whoopId, c.strain]));

    const csv = toCsv(
      [...view.recoveries]
        .sort((a, b) => at(a.createdAt) - at(b.createdAt))
        .map((r) => {
          const s = byDay.get(new Date(r.createdAt).toDateString());
          const ms = (v: number | null | undefined) => (v ? Math.round(v / 60000) : '');
          return {
            date: new Date(r.createdAt).toISOString().slice(0, 10),
            recovery: r.score ?? '',
            hrv_ms: r.hrv ?? '',
            resting_hr: r.restingHr ?? '',
            day_strain: strain.get(r.whoopId) ?? '',
            sleep_performance: s?.sleepPerformancePercentage ?? '',
            respiratory_rate: s?.respiratoryRate ?? '',
            in_bed_min: ms(s?.totalInBedTimeMilli),
            awake_min: ms(s?.totalAwakeTimeMilli),
            light_min: ms(s?.totalLightSleepTimeMilli),
            rem_min: ms(s?.totalRemSleepTimeMilli),
            deep_min: ms(s?.totalSlowWaveSleepTimeMilli),
            sleep_needed_min: ms(s?.sleepNeededMilli),
          };
        }),
    );

    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `mothership-${days}d-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [view, days]);

  const commands: Command[] = useMemo(
    () => [
      ...SECTIONS.map((s) => ({
        id: `jump-${s.id}`,
        group: 'Go to',
        label: s.label,
        run: () => jump(s.id),
      })),
      ...WINDOWS.map((w) => ({
        id: `window-${w}`,
        group: 'Window',
        label: `Last ${w} days`,
        hint: days === w ? 'current' : undefined,
        run: () => setDays(w),
      })),
      { id: 'export', group: 'Data', label: 'Export window as CSV', run: exportCsv },
      {
        id: 'top',
        group: 'Go to',
        label: 'Back to top',
        run: () => window.scrollTo({ top: 0, behavior: 'smooth' }),
      },
    ],
    [days, exportCsv, jump],
  );

  const fmt = (d: Date) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4">
      <CommandPalette commands={commands} />

      {/* console chrome: window selector + section nav */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border border-vit-rule bg-vit-bg px-4 py-3">
        {/* Each group can take a full line on narrow screens. */}
        <div className="flex items-center gap-1">
          {WINDOWS.map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => setDays(w)}
              aria-pressed={days === w}
              className={`border px-2.5 py-1.5 font-vit-mono text-[10px] tracking-[0.14em] uppercase transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-vit-cy ${
                days === w
                  ? 'border-vit-cy text-vit-cy'
                  : 'border-vit-rule text-vit-dim hover:text-vit-bone'
              }`}
            >
              {w}d
            </button>
          ))}
        </div>

        {range && (
          <Label className="order-3 min-w-0 basis-full sm:order-none sm:basis-auto">
            {fmt(range.from)} — {fmt(range.to)} ·{' '}
            <b className="font-normal text-vit-bone">{view.recoveries.length}</b> days of data
          </Label>
        )}

        <nav className="flex min-w-0 flex-wrap items-center gap-1 sm:ml-auto">
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => jump(s.id)}
              className="px-2 py-1 font-vit-mono text-[10px] tracking-[0.14em] text-vit-dim uppercase hover:text-vit-bone focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-vit-cy"
            >
              {s.label}
            </button>
          ))}
          <button
            type="button"
            onClick={exportCsv}
            className="border border-vit-rule px-2.5 py-1.5 font-vit-mono text-[10px] tracking-[0.14em] text-vit-dim uppercase hover:text-vit-bone focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-vit-cy"
          >
            CSV
          </button>
          <span className="ml-2 hidden font-vit-mono text-[9px] tracking-[0.14em] text-vit-dim uppercase sm:inline">
            ⌘K
          </span>
        </nav>
      </div>

      <div id="status" className="min-w-0 scroll-mt-4">
        <VitrineStage data={data} />
      </div>

      <div id="signal" className="grid grid-cols-[minmax(0,1fr)] gap-4 scroll-mt-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <BaselinePanel rows={analysis.baselines} />
        <CorrelationPanel rows={analysis.correlations} />
        <WeekdayPanel cells={analysis.weekday} />
        <RecordsPanel extremes={analysis.extremes} streak={analysis.streak} />
      </div>

      <div id="night" className="grid grid-cols-[minmax(0,1fr)] gap-4 scroll-mt-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <SleepDebtPanel {...analysis.debt} />
        <CircadianPanel {...analysis.circadian} />
      </div>

      <div id="load" className="grid grid-cols-[minmax(0,1fr)] gap-4 scroll-mt-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <ZonePanel {...analysis.zones} />
        <WorkoutPanel workouts={view.workouts} />
      </div>
    </div>
  );
}

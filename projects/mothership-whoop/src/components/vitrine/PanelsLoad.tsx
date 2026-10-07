'use client';

import { sportName, type Extreme, type Streak, type ZoneTotals } from '@/lib/analysis';
import type { WorkoutRow } from '@/lib/vitrine';
import { Label, Panel, Unavailable } from './primitives';

const ZONE_COLORS = ['#2E2D33', '#3B3A6B', '#5B3FFF', '#9B44E8', '#FF4FD8', '#FF7A6B'];

/* ------------------------------------------------------------------ zones */

export function ZonePanel({
  totals,
  totalMin,
  available,
}: {
  totals: ZoneTotals;
  totalMin: number;
  available: boolean;
}) {
  return (
    <Panel title="Heart rate zones" meta={available ? `${totalMin} min logged` : undefined}>
      {!available ? (
        <Unavailable>No zone data in this window</Unavailable>
      ) : (
        <>
          <div className="flex h-[26px] w-full overflow-hidden border border-vit-rule">
            {totals.map((z) => (
              <i
                key={z.zone}
                title={`Zone ${z.zone} · ${z.minutes} min`}
                style={{
                  flex: `${z.minutes} 0 0`,
                  background: ZONE_COLORS[z.zone],
                  opacity: 0.9,
                }}
              />
            ))}
          </div>

          <div className="mt-3 grid gap-2">
            {totals
              .filter((z) => z.minutes > 0)
              .reverse()
              .map((z) => (
                <div
                  key={z.zone}
                  className="flex items-center gap-3 font-vit-mono text-[10px] tracking-[0.12em] text-vit-dim uppercase"
                >
                  <i
                    className="block h-2 w-4 shrink-0"
                    style={{ background: ZONE_COLORS[z.zone] }}
                  />
                  <span className="text-vit-bone">Zone {z.zone}</span>
                  <span>{z.label}</span>
                  <span className="ml-auto tabular-nums text-vit-bone">{z.minutes}m</span>
                  <span className="w-10 text-right tabular-nums">
                    {totalMin ? Math.round((z.minutes / totalMin) * 100) : 0}%
                  </span>
                </div>
              ))}
          </div>
        </>
      )}
    </Panel>
  );
}

/* --------------------------------------------------------------- workouts */

export function WorkoutPanel({ workouts }: { workouts: WorkoutRow[] }) {
  const rows = [...workouts]
    .filter((w) => typeof w.strain === 'number')
    .sort((a, b) => +new Date(b.start) - +new Date(a.start))
    .slice(0, 8);

  return (
    <Panel title="Workout log" meta={rows.length ? `${rows.length} shown` : undefined}>
      {rows.length === 0 ? (
        <Unavailable>No workouts in this window</Unavailable>
      ) : (
        <div className="-mx-1 overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse font-vit-mono text-[10px] tracking-[0.1em] uppercase">
            <thead>
              <tr className="text-vit-dim">
                <th className="border-b border-vit-rule px-1 py-2 text-left font-normal">Date</th>
                <th className="border-b border-vit-rule px-1 py-2 text-left font-normal">Sport</th>
                <th className="border-b border-vit-rule px-1 py-2 text-right font-normal">Strain</th>
                <th className="border-b border-vit-rule px-1 py-2 text-right font-normal">Avg</th>
                <th className="border-b border-vit-rule px-1 py-2 text-right font-normal">Max</th>
                <th className="border-b border-vit-rule px-1 py-2 text-right font-normal">kcal</th>
              </tr>
            </thead>
            <tbody className="text-vit-bone">
              {rows.map((w, i) => (
                <tr key={`${w.start}-${i}`}>
                  <td className="border-b border-vit-rule px-1 py-2 text-vit-dim">
                    {new Date(w.start).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </td>
                  <td className="border-b border-vit-rule px-1 py-2">{sportName(w.sportId)}</td>
                  <td
                    className="border-b border-vit-rule px-1 py-2 text-right tabular-nums"
                    style={{ color: (w.strain ?? 0) >= 14 ? '#FF4FD8' : '#8FFFE0' }}
                  >
                    {(w.strain ?? 0).toFixed(1)}
                  </td>
                  <td className="border-b border-vit-rule px-1 py-2 text-right tabular-nums">
                    {w.averageHeartRate ?? '—'}
                  </td>
                  <td className="border-b border-vit-rule px-1 py-2 text-right tabular-nums">
                    {w.maxHeartRate ?? '—'}
                  </td>
                  <td className="border-b border-vit-rule px-1 py-2 text-right tabular-nums text-vit-dim">
                    {/* WHOOP reports energy in kilojoules. */}
                    {w.kilojoules ? Math.round(w.kilojoules / 4.184) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

/* -------------------------------------------------------- records + streak */

export function RecordsPanel({
  extremes,
  streak,
}: {
  extremes: Extreme[];
  streak: Streak;
}) {
  return (
    <Panel title="Records" meta={`green ≥ ${streak.threshold}%`}>
      <div className="grid grid-cols-2 gap-px border border-vit-rule bg-vit-rule">
        <div className="bg-vit-bg px-3 py-3">
          <Label>Current streak</Label>
          <b
            className="mt-1 block font-vit-serif text-3xl font-normal tabular-nums"
            style={{ color: streak.current > 0 ? '#8FFFE0' : '#7C7A85' }}
          >
            {streak.current}
          </b>
        </div>
        <div className="bg-vit-bg px-3 py-3">
          <Label>Best streak</Label>
          <b className="mt-1 block font-vit-serif text-3xl font-normal tabular-nums">
            {streak.best}
          </b>
        </div>
      </div>

      {extremes.length === 0 ? (
        <Unavailable>Nothing recorded yet</Unavailable>
      ) : (
        <dl className="mt-3 grid gap-px border border-vit-rule bg-vit-rule">
          {extremes.map((e) => (
            <div
              key={e.label}
              className="flex items-baseline justify-between gap-3 bg-vit-bg px-3 py-2 font-vit-mono text-[10px] tracking-[0.12em] uppercase"
            >
              <dt className="text-vit-dim">{e.label}</dt>
              <dd className="ml-auto tabular-nums text-vit-bone">{e.value}</dd>
              <dd className="w-14 text-right text-vit-dim">{e.when}</dd>
            </div>
          ))}
        </dl>
      )}
    </Panel>
  );
}

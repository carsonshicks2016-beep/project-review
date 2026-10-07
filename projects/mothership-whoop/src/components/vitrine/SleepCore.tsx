'use client';

import { hhmm } from '@/lib/vitrine';
import { Label } from './primitives';

/**
 * Sleep composition as a core sample.
 *
 * Proportional, not chronological. WHOOP's sleep record stores per-stage
 * TOTALS, so the interleaved strata a real hypnogram would show cannot be
 * drawn from this data — inventing an order would be fabricating structure.
 * Bands run shallow to deep: awake is a void punched through the column.
 */

type Stage = { key: string; label: string; minutes: number; color: string | null };

export function SleepCore({
  awakeMin,
  lightMin,
  remMin,
  deepMin,
  inBedMin,
  efficiency,
}: {
  awakeMin: number;
  lightMin: number;
  remMin: number;
  deepMin: number;
  inBedMin: number;
  efficiency: number;
}) {
  const stages: Stage[] = [
    { key: 'awake', label: 'Awake', minutes: awakeMin, color: null },
    { key: 'light', label: 'Light', minutes: lightMin, color: '#5B3FFF' },
    { key: 'rem', label: 'REM', minutes: remMin, color: '#FF4FD8' },
    { key: 'deep', label: 'Deep', minutes: deepMin, color: '#8FFFE0' },
  ];

  const total = stages.reduce((sum, s) => sum + s.minutes, 0);

  return (
    <>
      <Label>
        Sleep composition &middot;{' '}
        <b className="font-normal text-vit-bone">{hhmm(inBedMin)} in bed</b>
      </Label>

      <div className="mt-3 flex min-h-[200px] gap-4">
        <div className="flex w-[52px] shrink-0 flex-col border border-vit-rule bg-[#040407]">
          {total > 0 &&
            stages.map((s) => (
              <div
                key={s.key}
                title={`${s.label} · ${hhmm(s.minutes)}`}
                style={{
                  flex: `${s.minutes} 0 0`,
                  background: s.color ?? 'transparent',
                  opacity: s.color ? 0.78 : 1,
                  filter: s.color ? 'blur(0.4px)' : undefined,
                  boxShadow: s.color
                    ? undefined
                    : 'inset 0 1px 0 #4E4C58, inset 0 -1px 0 #4E4C58',
                }}
              />
            ))}
        </div>

        <dl className="grid min-w-0 flex-1 content-start gap-2.5 font-vit-mono text-[10px] tracking-[0.13em] text-vit-dim uppercase">
          {stages.map((s) => (
            <div key={s.key} className="flex items-center gap-2.5">
              <i
                className="block h-2 w-4 shrink-0"
                style={
                  s.color
                    ? { background: s.color }
                    : { border: '1px solid var(--color-vit-rule)' }
                }
              />
              <dt>{s.label}</dt>
              <dd className="ml-auto tabular-nums text-vit-bone">{hhmm(s.minutes)}</dd>
            </div>
          ))}
          <div className="flex items-center gap-2.5 text-vit-cy">
            <i className="block h-2 w-4 shrink-0 bg-vit-cy" />
            <dt>Efficiency</dt>
            <dd className="ml-auto tabular-nums">{efficiency.toFixed(1)}%</dd>
          </div>
        </dl>
      </div>
    </>
  );
}

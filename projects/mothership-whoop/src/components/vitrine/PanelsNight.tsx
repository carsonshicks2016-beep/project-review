'use client';

import {
  clockLabel,
  type CircadianPoint,
  type DebtPoint,
} from '@/lib/analysis';
import { hhmm } from '@/lib/vitrine';
import { Panel, Unavailable } from './primitives';

/* ------------------------------------------------------------- sleep debt */

/**
 * Running sleep debt. The zero line is where need was exactly met; the fill
 * grows downward into debt, which is the direction it actually feels.
 */
export function SleepDebtPanel({
  points,
  totalMin,
  available,
}: {
  points: DebtPoint[];
  totalMin: number;
  available: boolean;
}) {
  const inDebt = totalMin > 0;

  return (
    <Panel
      title="Sleep debt"
      meta={available ? `${inDebt ? '+' : ''}${hhmm(Math.abs(totalMin))} ${inDebt ? 'owed' : 'ahead'}` : undefined}
    >
      {!available ? (
        <Unavailable>Sleep need not synced yet — press Sync</Unavailable>
      ) : (
        <>
          <svg
            viewBox="0 0 600 130"
            preserveAspectRatio="none"
            className="block h-[130px] w-full"
            role="img"
            aria-label={`Cumulative sleep debt, ${hhmm(Math.abs(totalMin))} ${inDebt ? 'owed' : 'ahead'}`}
          >
            {(() => {
              const vals = points.map((p) => p.cumulativeMin);
              const hi = Math.max(...vals, 0);
              const lo = Math.min(...vals, 0);
              const span = hi - lo || 1;
              const y = (v: number) => 10 + ((hi - v) / span) * 110;
              const x = (i: number) =>
                points.length > 1 ? (i / (points.length - 1)) * 600 : 300;
              const zero = y(0);
              const line = points.map((p, i) => `${x(i)},${y(p.cumulativeMin)}`).join(' ');

              return (
                <>
                  <line x1="0" y1={zero} x2="600" y2={zero} stroke="#2E2D33" strokeWidth="1" />
                  <polygon
                    points={`0,${zero} ${line} 600,${zero}`}
                    fill={inDebt ? '#FF4FD8' : '#8FFFE0'}
                    fillOpacity="0.16"
                  />
                  <polyline
                    points={line}
                    fill="none"
                    stroke={inDebt ? '#FF4FD8' : '#8FFFE0'}
                    strokeWidth="1.5"
                  />
                  {points.map((p, i) => (
                    <circle
                      key={p.dayKey}
                      cx={x(i)}
                      cy={y(p.cumulativeMin)}
                      r={i === points.length - 1 ? 4 : 2}
                      fill={inDebt ? '#FF4FD8' : '#8FFFE0'}
                    />
                  ))}
                </>
              );
            })()}
          </svg>

          <div className="mt-3 grid gap-px border border-vit-rule bg-vit-rule">
            {points.slice(-4).reverse().map((p) => (
              <div
                key={p.dayKey}
                className="flex items-baseline justify-between gap-3 bg-vit-bg px-3 py-2 font-vit-mono text-[10px] tracking-[0.12em] text-vit-dim uppercase"
              >
                <span className="text-vit-bone">{p.label}</span>
                <span className="tabular-nums">
                  {hhmm(p.asleepMin)} of {hhmm(p.neededMin)}
                </span>
                <span
                  className="w-14 text-right tabular-nums"
                  style={{ color: p.deficitMin > 0 ? '#FF4FD8' : '#8FFFE0' }}
                >
                  {p.deficitMin > 0 ? '−' : '+'}
                  {hhmm(Math.abs(p.deficitMin))}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}

/* -------------------------------------------------------------- circadian */

/**
 * Every night as a bar from lights-out to wake, on one shared clock axis.
 * Ragged left edges mean an irregular bedtime, which is the thing worth seeing.
 */
export function CircadianPanel({
  points,
  onsetSd,
  wakeSd,
  available,
}: {
  points: CircadianPoint[];
  onsetSd: number;
  wakeSd: number;
  available: boolean;
}) {
  if (!available) {
    return (
      <Panel title="Circadian band">
        <Unavailable>Needs at least two nights</Unavailable>
      </Panel>
    );
  }

  const lo = Math.min(...points.map((p) => p.onsetMin)) - 30;
  const hi = Math.max(...points.map((p) => p.wakeMin)) + 30;
  const span = hi - lo || 1;
  const pos = (m: number) => ((m - lo) / span) * 100;

  // Hour gridlines across whatever range the nights actually cover.
  const ticks: number[] = [];
  for (let m = Math.ceil(lo / 60) * 60; m <= hi; m += 60) ticks.push(m);

  return (
    <Panel
      title="Circadian band"
      meta={`onset ±${Math.round(onsetSd)}m · wake ±${Math.round(wakeSd)}m`}
    >
      <div className="relative grid gap-1">
        {ticks.map((t) => (
          <i
            key={t}
            className="pointer-events-none absolute inset-y-0 w-px bg-vit-rule"
            style={{ left: `${pos(t)}%` }}
          />
        ))}

        {points.slice(-10).map((p) => (
          <div key={p.dayKey} className="relative flex items-center gap-3">
            <span className="w-8 shrink-0 font-vit-mono text-[9px] tracking-[0.1em] text-vit-dim uppercase">
              {p.label}
            </span>
            <div className="relative h-[13px] flex-1">
              <i
                className="absolute inset-y-0 rounded-none"
                style={{
                  left: `${pos(p.onsetMin)}%`,
                  width: `${Math.max(1, pos(p.wakeMin) - pos(p.onsetMin))}%`,
                  background: 'linear-gradient(90deg,#5B3FFF,#8FFFE0)',
                  opacity: 0.8,
                }}
              />
            </div>
            <span className="w-24 shrink-0 text-right font-vit-mono text-[9px] tabular-nums text-vit-dim">
              {clockLabel(p.onsetMin)}–{clockLabel(p.wakeMin)}
            </span>
          </div>
        ))}

        <div className="mt-2 flex justify-between font-vit-mono text-[9px] tracking-[0.1em] text-vit-dim tabular-nums">
          <span>{clockLabel(lo)}</span>
          <span>{clockLabel(hi)}</span>
        </div>
      </div>
    </Panel>
  );
}

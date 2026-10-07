'use client';

import type { Baseline, Correlation, WeekdayCell } from '@/lib/analysis';
import { recoveryColor } from '@/lib/vitrine';
import { Label, Panel, Unavailable } from './primitives';

/* -------------------------------------------------------------- baselines */

export type BaselineRow = {
  key: string;
  label: string;
  unit: string;
  current: number;
  stat: Baseline;
  /** RHR is the one where a rise is bad, so the colour has to flip. */
  higherIsBetter: boolean;
};

/**
 * Where each metric sits against its own trailing baseline, in standard
 * deviations. The bar is centred: left of centre is below baseline.
 */
export function BaselinePanel({ rows }: { rows: BaselineRow[] }) {
  const usable = rows.filter((r) => r.stat.available);

  return (
    <Panel
      title="Baseline deviation"
      meta={usable.length ? `n=${usable[0].stat.n}` : undefined}
    >
      {usable.length === 0 ? (
        <Unavailable>Needs at least four days of history</Unavailable>
      ) : (
        <div className="grid gap-4">
          {usable.map((r) => {
            // Clamp the drawn bar at 3σ; the number still tells the truth.
            const z = r.stat.z;
            const pct = Math.min(Math.abs(z) / 3, 1) * 50;
            const good = r.higherIsBetter ? z >= 0 : z <= 0;
            const color = Math.abs(z) < 0.5 ? '#5B3FFF' : good ? '#8FFFE0' : '#FF4FD8';

            return (
              <div key={r.key} className="grid gap-1.5">
                <div className="flex items-baseline justify-between gap-3 font-vit-mono text-[10px] tracking-[0.14em] text-vit-dim uppercase">
                  <span>{r.label}</span>
                  <span className="tabular-nums text-vit-bone">
                    {r.current}
                    {r.unit} · base {r.stat.mean.toFixed(1)}
                    {r.unit}
                  </span>
                </div>

                <div className="relative h-[9px] border border-vit-rule">
                  <i className="absolute inset-y-0 left-1/2 w-px bg-vit-rule" />
                  <i
                    className="absolute inset-y-0"
                    style={{
                      background: color,
                      boxShadow: `0 0 10px ${color}66`,
                      left: z >= 0 ? '50%' : `${50 - pct}%`,
                      width: `${pct}%`,
                    }}
                  />
                </div>

                <div className="flex justify-between font-vit-mono text-[9px] tracking-[0.12em] text-vit-dim uppercase tabular-nums">
                  <span>
                    {z >= 0 ? '+' : ''}
                    {r.stat.delta.toFixed(1)}
                    {r.unit}
                  </span>
                  <span style={{ color }}>
                    {z >= 0 ? '+' : ''}
                    {z.toFixed(2)} &sigma;
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------ correlations */

/** r is signed, so the bar grows from the centre in the direction of the sign. */
export function CorrelationPanel({ rows }: { rows: Correlation[] }) {
  const usable = rows.filter((r) => r.r !== null);

  return (
    <Panel title="Relationships" meta="correlation, not cause">
      {usable.length === 0 ? (
        <Unavailable>Needs at least three paired days</Unavailable>
      ) : (
        <div className="grid gap-3.5">
          {usable.map((c) => {
            const r = c.r as number;
            const pct = Math.abs(r) * 50;
            const strong = Math.abs(r) >= 0.5;
            const color = !strong ? '#5B3FFF' : r > 0 ? '#8FFFE0' : '#FF4FD8';
            return (
              <div key={c.key} className="grid gap-1.5">
                <div className="flex min-w-0 items-baseline justify-between gap-3 font-vit-mono text-[10px] tracking-[0.13em] text-vit-dim uppercase">
                  <span className="min-w-0">{c.label}</span>
                  <span className="shrink-0 tabular-nums">n={c.n}</span>
                </div>
                <div className="relative h-[7px] border border-vit-rule">
                  <i className="absolute inset-y-0 left-1/2 w-px bg-vit-rule" />
                  <i
                    className="absolute inset-y-0"
                    style={{
                      background: color,
                      left: r >= 0 ? '50%' : `${50 - pct}%`,
                      width: `${pct}%`,
                    }}
                  />
                </div>
                <div className="flex min-w-0 justify-between gap-3 font-vit-mono text-[9px] tracking-[0.1em] text-vit-dim">
                  <span className="min-w-0 truncate uppercase">{c.detail}</span>
                  <span className="shrink-0 tabular-nums" style={{ color }}>
                    r = {r >= 0 ? '+' : ''}
                    {r.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
          <p className="mt-1 font-vit-mono text-[9px] leading-relaxed tracking-[0.1em] text-vit-dim uppercase">
            {/* Each row has its own n, so quote the range rather than one row's. */}
            {(() => {
              const ns = usable.map((u) => u.n);
              const lo = Math.min(...ns);
              const hi = Math.max(...ns);
              const span = lo === hi ? `${hi}` : `${lo}–${hi}`;
              return `n = ${span} paired days. Too few to be conclusive — read as a hint, not a finding.`;
            })()}
          </p>
        </div>
      )}
    </Panel>
  );
}

/* ---------------------------------------------------------------- rhythm */

/** Average recovery by weekday — the pattern a chronological chart hides. */
export function WeekdayPanel({ cells }: { cells: WeekdayCell[] }) {
  const any = cells.some((c) => c.n > 0);

  return (
    <Panel title="Weekly rhythm" meta="mean recovery">
      {!any ? (
        <Unavailable>No scored recoveries yet</Unavailable>
      ) : (
        <div className="grid grid-cols-7 gap-px border border-vit-rule bg-vit-rule">
          {cells.map((c) => (
            <div
              key={c.weekday}
              title={c.n ? `${c.weekday}: ${c.mean.toFixed(0)}% over ${c.n} day(s)` : 'No data'}
              className="relative flex h-[92px] flex-col justify-end bg-vit-bg"
            >
              {c.n > 0 && (
                <i
                  className="block w-full"
                  style={{ height: `${c.mean}%`, background: recoveryColor(c.mean), opacity: 0.85 }}
                />
              )}
              <span className="absolute inset-x-0 top-1.5 text-center font-vit-mono text-[9px] tracking-[0.08em] text-vit-dim uppercase">
                {c.weekday}
              </span>
              <span className="absolute inset-x-0 bottom-1.5 text-center font-vit-mono text-[9px] tabular-nums text-vit-bone mix-blend-difference">
                {c.n > 0 ? Math.round(c.mean) : '—'}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

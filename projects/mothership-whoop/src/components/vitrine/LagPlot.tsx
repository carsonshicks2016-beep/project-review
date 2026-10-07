'use client';

import {
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from 'recharts';

/**
 * HRV(n) against HRV(n+1).
 *
 * This is a lag-1 recurrence plot over days, NOT a Poincare plot — WHOOP
 * exposes one HRV scalar per recovery, never the beat-to-beat RR intervals a
 * true Poincare plot needs. It reads the same way: points hugging the diagonal
 * in a tight cluster mean stable autonomics, a wide scatter means volatility.
 */

type DotProps = { cx?: number; cy?: number; index?: number };

/** Two stacked circles instead of an SVG blur filter — cheaper, and it survives
 *  Recharts re-rendering the shape on resize. */
function GlowDot({ cx, cy, index = 0 }: DotProps) {
  if (cx == null || cy == null) return null;
  const color = index % 3 === 0 ? '#FF4FD8' : '#8FFFE0';
  return (
    <g>
      <circle cx={cx} cy={cy} r={5} fill={color} opacity={0.16} />
      <circle cx={cx} cy={cy} r={2.2} fill={color} opacity={0.75} />
    </g>
  );
}

export function LagPlot({ points }: { points: { x: number; y: number }[] }) {
  if (points.length < 2) {
    return (
      <div className="flex h-[196px] items-center justify-center font-vit-mono text-[10px] tracking-[0.2em] text-vit-dim uppercase">
        Not enough recoveries yet
      </div>
    );
  }

  const values = points.flatMap((p) => [p.x, p.y]);
  const pad = (Math.max(...values) - Math.min(...values)) * 0.12 || 4;
  const lo = Math.min(...values) - pad;
  const hi = Math.max(...values) + pad;

  return (
    <div className="h-[196px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <XAxis type="number" dataKey="x" domain={[lo, hi]} hide />
          <YAxis type="number" dataKey="y" domain={[lo, hi]} hide />
          {/* Identity line: structure, so it stays bone. */}
          <ReferenceLine
            segment={[
              { x: lo, y: lo },
              { x: hi, y: hi },
            ]}
            stroke="rgba(233,228,216,0.16)"
            strokeDasharray="2 4"
            ifOverflow="visible"
          />
          <Scatter
            data={points}
            shape={<GlowDot />}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

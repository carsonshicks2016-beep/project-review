'use client';

/**
 * Autonomic density band. Every bar is a real HRV sample — the series is
 * linearly resampled to a fixed bar count so a 7-day window and a 30-day
 * window read at the same density. Nothing is added to the data; the only
 * treatment is interpolation between measured points.
 */
export function NoiseBand({ band, bars = 160 }: { band: number[]; bars?: number }) {
  if (band.length < 2) return <div className="h-[34px] border-y border-vit-rule" />;

  const height = 34;
  const step = 600 / bars;

  const samples = Array.from({ length: bars }, (_, i) => {
    const t = (i / (bars - 1)) * (band.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(band.length - 1, lo + 1);
    return band[lo] + (band[hi] - band[lo]) * (t - lo);
  });

  return (
    <svg
      viewBox={`0 0 600 ${height}`}
      preserveAspectRatio="none"
      className="block h-[34px] w-full border-y border-vit-rule"
      role="img"
      aria-label="Heart rate variability across the window"
    >
      {samples.map((v, i) => {
        const h = Math.max(2, v * (height - 2));
        const color = v >= 0.55 ? '#8FFFE0' : v >= 0.3 ? '#5B3FFF' : '#FF4FD8';
        return (
          <rect
            key={i}
            x={i * step}
            y={(height - h) / 2}
            width={step * 0.5}
            height={h}
            fill={color}
            opacity={0.3 + v * 0.5}
          />
        );
      })}
    </svg>
  );
}

'use client';

import { useId } from 'react';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * The organism. Three nested elements each own exactly one transform so the
 * animations never fight:
 *
 *   positioning wrapper  translate      (static)
 *   breathing wrapper    scale          <- sleep.respiratoryRate
 *   drift svg            rotate         <- ambient, never repeats
 *
 * The turbulence itself is computed once from recovery and never animated.
 * Animating feTurbulence forces the filter to re-rasterise every frame, which
 * is expensive enough to be felt on a page that is always open; rotating an
 * already-filtered layer is GPU-composited and free. The edge still never sits
 * still, because drift and breath run on incommensurable periods.
 */
export function Membrane({
  recovery,
  respiratoryRate,
  restingHr,
}: {
  recovery: number;
  respiratoryRate: number;
  restingHr: number;
}) {
  // useId() emits colons, which are not safe inside url(#...).
  const uid = useId().replace(/:/g, '');
  const turb = `vit-turb-${uid}`;
  const fill = `vit-fill-${uid}`;

  const depleted = 100 - clamp(recovery, 0, 100);

  // Low recovery tears the edge apart; high recovery resolves it to a circle.
  const displacement = 22 + depleted * 0.55;
  const frequency = (0.01 + depleted * 0.00006).toFixed(5);

  // High recovery pushes the violet stop outward, leaving more cyan.
  const midStop = 20 + clamp(recovery, 0, 100) * 0.45;

  // No measured rate means no motion. Falling back to a plausible default would
  // make the membrane breathe at a number nobody recorded.
  const breathes = respiratoryRate > 0;
  const pulses = restingHr > 0;
  const breathSeconds = 60 / clamp(respiratoryRate, 8, 25);
  const pulseSeconds = 60 / clamp(restingHr, 35, 110);

  return (
    <div
      className="pointer-events-none absolute top-1/2 left-[53%] aspect-square h-[142%] -translate-x-1/2 -translate-y-1/2"
      style={
        {
          '--vit-breath': `${breathSeconds.toFixed(2)}s`,
          '--vit-pulse': `${pulseSeconds.toFixed(2)}s`,
        } as React.CSSProperties
      }
    >
      <div className={`relative h-full w-full ${breathes ? 'vit-breathe' : ''}`}>
        <svg
          className="vit-drift h-full w-full"
          viewBox="0 0 400 400"
          role="img"
          aria-label={`Recovery membrane at ${Math.round(recovery)} percent`}
        >
          <defs>
            <filter id={turb} x="-30%" y="-30%" width="160%" height="160%">
              <feTurbulence
                type="fractalNoise"
                baseFrequency={frequency}
                numOctaves={3}
                seed={41}
                result="noise"
              />
              <feDisplacementMap
                in="SourceGraphic"
                in2="noise"
                scale={displacement}
                xChannelSelector="R"
                yChannelSelector="G"
              />
              <feGaussianBlur stdDeviation={1.1} />
            </filter>
            <radialGradient id={fill} cx="44%" cy="40%" r="70%">
              <stop offset="0%" stopColor="#8FFFE0" stopOpacity={0.88} />
              <stop offset={`${midStop}%`} stopColor="#5B3FFF" stopOpacity={0.62} />
              <stop offset="100%" stopColor="#FF4FD8" stopOpacity={0.22} />
            </radialGradient>
          </defs>
          <g filter={`url(#${turb})`}>
            <circle cx={200} cy={200} r={150} fill={`url(#${fill})`} />
            <circle
              cx={200}
              cy={200}
              r={150}
              fill="none"
              stroke="#8FFFE0"
              strokeOpacity={0.8}
              strokeWidth={1.2}
            />
            <circle
              cx={200}
              cy={200}
              r={116}
              fill="none"
              stroke="#FF4FD8"
              strokeOpacity={0.42}
              strokeWidth={1}
            />
            <circle
              cx={200}
              cy={200}
              r={78}
              fill="none"
              stroke="#8FFFE0"
              strokeOpacity={0.3}
              strokeWidth={1}
            />
          </g>
        </svg>

        {/*
          The cardiac pulse. Kept outside the SVG on purpose: animating opacity
          on a plain element is composited, while animating anything inside the
          filtered group would re-run the turbulence.
        */}
        <div
          className={`absolute inset-0 ${pulses ? 'vit-pulse' : ''}`}
          style={{
            background:
              'radial-gradient(circle at 44% 40%, rgba(143,255,224,0.42), transparent 38%)',
          }}
        />
      </div>
    </div>
  );
}

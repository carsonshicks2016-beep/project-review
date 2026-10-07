'use client';

import { useEffect, useRef, useState } from 'react';

/** Tracks the live value of prefers-reduced-motion. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => setReduced(mq.matches);
    sync();
    mq.addEventListener('change', sync);
    return () => mq.removeEventListener('change', sync);
  }, []);
  return reduced;
}

/**
 * Counts to `target` once, on mount.
 *
 * The target is frozen at first render on purpose: scrubbing must not restart
 * the animation, and the caller renders scrubbed values directly instead. Note
 * there is no "already ran" ref — under StrictMode the effect runs, is torn
 * down, and runs again, so a latch would cancel the only real frame loop and
 * leave the number at zero.
 */
export function useCountUp(target: number, duration = 1100): number {
  const reduced = useReducedMotion();
  const frozen = useRef(target);
  const [value, setValue] = useState(0);

  useEffect(() => {
    const to = frozen.current;

    // requestAnimationFrame is suspended in a background tab, so a page mounted
    // there would sit at zero and then crawl the moment it is looked at.
    if (reduced || document.visibilityState === 'hidden') {
      setValue(to);
      return;
    }

    let raf = 0;
    let start: number | null = null;
    const step = (ts: number) => {
      start ??= ts;
      const p = Math.min(1, (ts - start) / duration);
      setValue(Math.round(to * (1 - Math.pow(1 - p, 3))));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [duration, reduced]);

  return value;
}

/** Structure is always mono, always bone or dim. Never coloured. */
export function Label({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`font-vit-mono text-[10px] uppercase tracking-[0.2em] text-vit-dim ${className}`}
    >
      {children}
    </div>
  );
}

/** A mono readout cell: dim label, serif numeral. */
export function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-vit-bg px-3 py-2.5">
      <span className="block font-vit-mono text-[9px] uppercase tracking-[0.17em] text-vit-dim">
        {label}
      </span>
      <b className="mt-0.5 block font-vit-serif text-2xl leading-tight font-normal tabular-nums">
        {value}
      </b>
    </div>
  );
}

/** A bordered instrument panel: mono title, optional right-hand meta, content. */
export function Panel({
  title,
  meta,
  children,
  className = '',
}: {
  title: React.ReactNode;
  meta?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`min-w-0 border border-vit-rule bg-vit-bg p-5 ${className}`}>
      <div className="flex items-baseline justify-between gap-4">
        <Label>{title}</Label>
        {meta ? <Label>{meta}</Label> : null}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** Shown in place of a panel body when the inputs genuinely are not there. */
export function Unavailable({ children }: { children: React.ReactNode }) {
  return (
    <p className="py-6 text-center font-vit-mono text-[10px] tracking-[0.18em] text-vit-dim uppercase">
      {children}
    </p>
  );
}

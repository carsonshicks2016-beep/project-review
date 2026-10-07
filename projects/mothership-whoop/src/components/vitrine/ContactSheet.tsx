'use client';

import { recoveryColor, type Vitals } from '@/lib/vitrine';

/**
 * Seven days as seven glyphs. Doubles as the scrub control: pointing at a cell
 * re-reads the whole hero to that day, so there is no tooltip anywhere.
 * Buttons rather than divs so the same thing works from the keyboard.
 */
export function ContactSheet({
  week,
  activeIndex,
  onScrub,
}: {
  week: Vitals[];
  activeIndex: number | null;
  onScrub: (index: number | null) => void;
}) {
  return (
    <div
      className="mt-3 grid gap-px border border-vit-rule bg-vit-rule"
      style={{ gridTemplateColumns: `repeat(${Math.max(week.length, 1)}, minmax(0,1fr))` }}
      onPointerLeave={() => onScrub(null)}
      onKeyDown={(e) => {
        // Left/right walk the week, escape drops back to today.
        const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (step === 0) {
          if (e.key === 'Escape') onScrub(null);
          return;
        }
        e.preventDefault();
        const buttons = Array.from(
          e.currentTarget.querySelectorAll<HTMLButtonElement>('button'),
        );
        const from = buttons.indexOf(document.activeElement as HTMLButtonElement);
        const next = Math.min(week.length - 1, Math.max(0, (from < 0 ? 0 : from) + step));
        buttons[next]?.focus();
      }}
    >
      {week.map((day, i) => {
        const active = activeIndex === i;
        return (
          <button
            key={day.dayKey}
            type="button"
            onPointerEnter={() => onScrub(i)}
            onFocus={() => onScrub(i)}
            onBlur={() => onScrub(null)}
            aria-label={`${day.label}, recovery ${day.recovery} percent`}
            aria-pressed={active}
            className={`relative grid justify-items-center gap-3 px-0 pt-3.5 pb-3 transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-vit-cy ${
              active ? 'bg-[#101018]' : 'bg-vit-bg'
            }`}
          >
            <span
              className="block aspect-square w-[40%] max-w-[34px] rounded-full blur-[3px] transition-transform duration-300"
              style={{
                background: recoveryColor(day.recovery),
                opacity: 0.34 + day.recovery / 165,
                transform: `scale(${(0.5 + day.recovery / 180) * (active ? 1.18 : 1)})`,
              }}
            />
            <span className="font-vit-mono text-[9px] tracking-[0.1em] text-vit-dim uppercase">
              {day.label} {day.recovery}
            </span>
          </button>
        );
      })}
    </div>
  );
}

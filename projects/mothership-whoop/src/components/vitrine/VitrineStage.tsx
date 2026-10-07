'use client';

import { useState } from 'react';
import { hhmm, type VitrineData } from '@/lib/vitrine';
import { ContactSheet } from './ContactSheet';
import { LagPlot } from './LagPlot';
import { Membrane } from './Membrane';
import { NoiseBand } from './NoiseBand';
import { SleepCore } from './SleepCore';
import { Field, Label, useCountUp } from './primitives';

const MAX_STRAIN = 21;

export function VitrineStage({ data }: { data: VitrineData }) {
  if (!data.latest) {
    return (
      <div className="border border-vit-rule bg-vit-bg p-16 text-center">
        <Label>No recoveries synced</Label>
        <p className="mt-3 font-vit-serif text-2xl text-vit-bone">
          Nothing to look at yet.
        </p>
      </div>
    );
  }
  return <Stage data={data} latest={data.latest} />;
}

function Stage({
  data,
  latest,
}: {
  data: VitrineData;
  latest: NonNullable<VitrineData['latest']>;
}) {
  const [scrub, setScrub] = useState<number | null>(null);

  const scrubbed = scrub === null ? null : (data.week[scrub] ?? null);
  const counted = useCountUp(latest.recovery);

  // Count up once on mount; scrubbing reads exact values immediately.
  const recovery = scrubbed ? scrubbed.recovery : latest.recovery;
  const heroValue = scrubbed ? scrubbed.recovery : counted;
  const hrv = scrubbed ? scrubbed.hrv : latest.hrv;
  const restingHr = scrubbed ? scrubbed.restingHr : latest.restingHr;
  const strain = scrubbed ? scrubbed.strain : latest.strain;
  const strainSource = scrubbed ? scrubbed.strainSource : latest.strainSource;

  // Day strain and peak workout strain are different measures, so the label has
  // to say which one is on screen.
  const strainLabel =
    strainSource === 'cycle'
      ? 'Day strain'
      : strainSource === 'workout'
        ? 'Peak workout strain'
        : 'Strain';

  const coherence = (recovery / 100).toFixed(2);
  const tone =
    recovery >= 67
      ? 'Parasympathetic dominant'
      : recovery >= 34
        ? 'Mixed autonomic tone'
        : 'Sympathetic dominant';

  return (
    <section className="overflow-hidden border border-vit-rule bg-vit-bg font-vit-mono text-vit-bone">
      {/* status bar */}
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-vit-rule px-3.5 py-2.5 text-[10px] tracking-[0.13em] text-vit-dim uppercase">
        <span
          className="vit-pulse block h-[7px] w-[7px] shrink-0 bg-vit-cy shadow-[0_0_9px_#8FFFE0]"
          style={{ '--vit-pulse': `${(60 / Math.max(35, restingHr)).toFixed(2)}s` } as React.CSSProperties}
        />
        <span>Mothership &middot; Specimen</span>
        <span className="ml-auto">
          {scrubbed ? `Reading ${scrubbed.label}` : latest.label}
        </span>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        {/* ---------------- hero ---------------- */}
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-3.5 border-b border-vit-rule p-5 lg:border-r lg:border-b-0">
          <Label>
            Recovery index /{' '}
            <b className="font-normal text-vit-bone">
              {scrubbed ? scrubbed.label : 'Today'}
            </b>
          </Label>

          {/* the case: a hard 1px crop around something alive */}
          <div className="relative aspect-[4/3] w-full overflow-hidden border border-vit-rule bg-[#040407] lg:aspect-[16/10]">
            <Membrane
              recovery={recovery}
              respiratoryRate={latest.respiratoryRate}
              restingHr={restingHr}
            />

            {/* registration: structure laid over the specimen */}
            <div className="pointer-events-none absolute inset-0">
              <i className="absolute inset-x-0 top-1/2 h-px bg-vit-rule opacity-55" />
              <i className="absolute inset-y-0 left-[62%] w-px bg-vit-rule opacity-55" />
              {[20, 40, 60, 80].map((left) => (
                <i
                  key={left}
                  className="absolute top-0 h-[7px] w-px bg-vit-rule opacity-90"
                  style={{ left: `${left}%` }}
                />
              ))}
            </div>

            <div className="absolute bottom-6 left-4 z-20 font-vit-serif text-[clamp(88px,17vw,192px)] leading-[0.76] tracking-[-0.02em] tabular-nums mix-blend-difference">
              {heroValue}
              <sup className="ml-[0.12em] align-super font-vit-mono text-[0.13em] tracking-[0.16em]">
                %
              </sup>
            </div>

            <div className="absolute inset-x-0 bottom-0 z-30 flex justify-between gap-3 border-t border-vit-rule bg-vit-bg/70 px-2.5 py-1.5 text-[9px] tracking-[0.16em] text-vit-dim uppercase backdrop-blur-sm">
              <span className="min-w-0 truncate">
                Membrane coherence <b className="font-normal text-vit-cy">{coherence}</b>
              </span>
              <span className="hidden min-w-0 truncate sm:block">{tone}</span>
            </div>
          </div>

          <NoiseBand band={data.band} />

          <div className="grid grid-cols-4 gap-px border border-vit-rule bg-vit-rule">
            <Field label="HRV" value={`${hrv}`} />
            <Field label="RHR" value={`${restingHr}`} />
            <Field
              label="Strain"
              value={strainSource === 'none' ? '\u2014' : strain.toFixed(1)}
            />
            <Field label="Sleep" value={`${latest.sleepPerformance}`} />
          </div>

          <div className="grid gap-2">
            <Label className="flex justify-between">
              <span>{strainLabel}</span>
              <span>
                {strainSource === 'none'
                  ? 'Cycle not scored'
                  : `${strain.toFixed(1)} / ${MAX_STRAIN.toFixed(1)}`}
              </span>
            </Label>
            <div className="relative h-[7px] overflow-hidden border border-vit-rule">
              {strainSource !== 'none' && (
                <i
                  className="block h-full bg-linear-to-r from-vit-vi to-vit-mg shadow-[0_0_14px_rgba(255,79,216,0.55)]"
                  style={{ width: `${Math.min(100, (strain / MAX_STRAIN) * 100)}%` }}
                />
              )}
            </div>
          </div>
        </div>

        {/* ---------------- side ---------------- */}
        <div className="grid grid-rows-[auto_auto]">
          <div className="border-b border-vit-rule px-5 py-5">
            <div className="flex items-baseline justify-between gap-4">
              <Label>HRV lag &middot; n / n+1</Label>
              <Label>n={data.lag.length}</Label>
            </div>
            <LagPlot points={data.lag} />
          </div>
          <div className="px-5 py-5">
            <SleepCore
              awakeMin={latest.awakeMin}
              lightMin={latest.lightMin}
              remMin={latest.remMin}
              deepMin={latest.deepMin}
              inBedMin={latest.inBedMin}
              efficiency={latest.efficiency}
            />
          </div>
        </div>
      </div>

      <div className="border-t border-vit-rule px-5 py-5">
        <Label>Contact sheet &middot; last seven</Label>
        <ContactSheet week={data.week} activeIndex={scrub} onScrub={setScrub} />
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-2 border-t border-vit-rule px-5 py-3 text-[10px] tracking-[0.16em] text-vit-dim uppercase">
        <span>
          Respiratory
          <b className="ml-2 font-normal text-vit-bone">
            {latest.respiratoryRate > 0
              ? `${latest.respiratoryRate.toFixed(1)} rpm`
              : '\u2014'}
          </b>
        </span>
        <span>
          Asleep
          <b className="ml-2 font-normal text-vit-bone">{hhmm(latest.asleepMin)}</b>
        </span>
        <span>
          Needed
          <b className="ml-2 font-normal text-vit-bone">
            {latest.neededMin > 0 ? hhmm(latest.neededMin) : '\u2014'}
          </b>
        </span>
      </div>
    </section>
  );
}

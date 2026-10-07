// Template voice: the drone's own running commentary.
//
// Grammar-based rather than generative, which for this job is a feature -- every number
// it says is one that was measured. It exists behind the same interface a real model
// would use, so swapping in WebLLM or a hosted API later changes nothing upstream.

const pick = (arr, seed) => arr[Math.abs(seed) % arr.length];
const kmh = ms => Math.round(ms * 3.6);
const s1 = v => v.toFixed(1);
// Gates are 0-indexed internally. Nobody says "gate zero".
const gnum = i => i + 1;

// Each entry returns a line. `n` is a per-event seed so repeated event types vary.
const LINES = {
  launch: (e, n) => pick([
    e.gate > 0 ? `Starting from gate ${gnum(e.gate)}. Let's see how far I get.` : `Armed. ${e.totalGates} gates ahead.`,
    e.gate > 0 ? `Dropped in at gate ${gnum(e.gate)}.` : `Here we go — full lap.`,
    e.gate > 0 ? `Picking it up from ${gnum(e.gate)}.` : `Clean board. ${e.totalGates} to clear.`
  ], n),

  gate: (e, n) => {
    const base = pick([
      `Gate ${gnum(e.gate)} — ${kmh(e.speed)}.`,
      `Through ${gnum(e.gate)}, carrying ${kmh(e.speed)}.`,
      `${gnum(e.gate)} done. ${kmh(e.speed)} k.`,
      `That's ${e.ordinal} of ${e.totalGates}.`
    ], n);
    if (e.delta !== null && e.delta !== undefined && Math.abs(e.delta) > 0.15) {
      return `${base} ${e.delta < 0 ? `${s1(-e.delta)}s up` : `${s1(e.delta)}s down`} on my best.`;
    }
    if (e.split > 4.5) return `${base} Slow leg, that one.`;
    return base;
  },

  nearMiss: (e, n) => pick([
    `Whoa — ${s1(e.clearance)} metres ${e.side}.`,
    `Tight ${e.side}. ${s1(e.clearance)} to spare.`,
    `Nearly wore that one — ${s1(e.clearance)}m ${e.side}.`,
    `Threading it. ${s1(e.clearance)} metres ${e.side}.`
  ], n),

  bank: (e, n) => pick([
    `Laying it right over — ${Math.round(e.degrees)} degrees.`,
    `Full commit through there — ${Math.round(e.degrees)} degrees of bank.`,
    `On its side at ${kmh(e.speed)}.`
  ], n),

  topSpeed: (e, n) => pick([
    `${kmh(e.speed)} — quickest I've been.`,
    `Peak of ${kmh(e.speed)} on that straight.`,
    `Topping out at ${kmh(e.speed)}.`
  ], n),

  stall: (e, n) => pick([
    `I've lost all my speed here.`,
    `Hanging in the air — ${s1(e.seconds)} seconds of nothing.`,
    `Dead in the air. Need to get moving.`
  ], n),

  wallHit: (e, n) => pick([
    `Clipped the wall. That's me done at ${e.cleared}.`,
    `Into the scenery. ${e.cleared} gates.`,
    `Found a wall on the way to ${gnum(e.nextGate)}.`
  ], n),

  groundHit: (e, n) => pick([
    `Put it in the ground. ${e.cleared} gates.`,
    `Down hard. ${e.cleared} of ${e.totalGates}.`,
    `Lost it and hit the deck at gate ${gnum(e.nextGate)}.`
  ], n),

  timeout: (e, n) => pick([
    `Out of time at gate ${gnum(e.nextGate)}. ${e.cleared} cleared.`,
    `Clock beat me — ${e.cleared} of ${e.totalGates}.`,
    `Ran the tank dry. ${e.cleared} ${e.cleared === 1 ? 'gate' : 'gates'} in ${s1(e.time)}s.`
  ], n),

  finish: (e, n) => pick([
    `Lap complete. ${e.cleared} for ${e.totalGates}, ${s1(e.time)} seconds.`,
    `All ${e.totalGates}. ${s1(e.time)} on the clock.`,
    `Full lap, ${s1(e.time)}s, peak ${kmh(e.topSpeed)}.`
  ], n),

  record: (e, n) => pick([
    `And that's a new best.`,
    `Personal record. I'll take that.`,
    `Best run yet.`
  ], n)
};

export const templateVoice = {
  name: 'template',
  // Synchronous: the scheduler can call this exactly when an event comes due.
  async say(event) {
    const fn = LINES[event.type];
    if (!fn) return null;
    return fn(event, Math.round(event.frame * 7 + (event.gate || 0)));
  }
};

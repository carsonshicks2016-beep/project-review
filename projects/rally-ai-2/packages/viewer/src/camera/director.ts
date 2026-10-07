/**
 * Deterministic replay-time camera director.
 *
 * Cuts between the existing presentation cameras the way a rally broadcast
 * would: sideline for jumps and crests, tight bonnet through hairpins, and the
 * chase boom held through fast sweepers and everything else.
 *
 * Presentational only. The cut list is computed once, up front, purely from
 * the recorded stage pace notes and the recorded frames (arc length → stage
 * time), so the same replay always cuts at the same timestamps — there is no
 * randomness and nothing feeds back into playback or simulation. The director
 * only selects which of the existing `CameraMode`s is active; vibration,
 * slip lag and landing kick continue to run inside `ChaseCamera.apply()`
 * untouched.
 */

import type { CameraMode } from "../camera";
import type { LoadedReplay } from "../replay";

export interface CameraCut {
  /** Stage-time window, seconds. */
  t0: number;
  t1: number;
  mode: CameraMode;
  /** Why the shot exists — surfaced on the dev handle for debugging. */
  reason: string;
}

/** Seconds of shot before/after the note's arc position. */
const JUMP_LEAD = 1.3;
const JUMP_TAIL = 2.1;
const TIGHT_LEAD = 0.9;
const TIGHT_TAIL = 1.7;
const SWEEPER_LEAD = 0.8;
const SWEEPER_TAIL = 1.4;

/** No shot shorter than this survives clamping to the replay's ends. */
const MIN_SHOT = 0.9;
/** Breathing room between accepted cuts so the edit never strobes. */
const MIN_GAP = 1.6;

type Candidate = CameraCut & { priority: number };

export class CameraDirector {
  /** Accepted cuts, sorted by start time. Gaps between them are chase. */
  readonly cuts: readonly CameraCut[];

  constructor(replay: LoadedReplay) {
    this.cuts = buildCuts(replay);
  }

  /** Camera mode at a stage time. Pure — same input, same answer. */
  modeAt(time: number): CameraMode {
    const cut = this.cuts[this.shotIndexAt(time)];
    return cut ? cut.mode : "chase";
  }

  /**
   * Identity of the shot at a stage time; -1 is the default chase hold.
   * A change of shot index is a hard cut, even chase → chase, so the caller
   * can re-seed camera damping instead of swinging the boom across the cut.
   */
  shotIndexAt(time: number): number {
    let lo = 0;
    let hi = this.cuts.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const cut = this.cuts[mid]!;
      if (time < cut.t0) hi = mid - 1;
      else if (time >= cut.t1) lo = mid + 1;
      else return mid;
    }
    return -1;
  }
}

function buildCuts(replay: LoadedReplay): CameraCut[] {
  const duration = replay.frames[replay.frames.length - 1]!.t;
  const notes = [...replay.stage.pace_notes].sort((a, b) => a.s - b.s);

  // Arc length → stage time from the recorded frames. `s` is monotonic on a
  // point-to-point stage, so one forward pointer covers all sorted notes.
  const frames = replay.frames;
  let fi = 0;
  const candidates: Candidate[] = [];
  for (const note of notes) {
    while (fi < frames.length && (frames[fi]!.s ?? 0) < note.s) fi++;
    if (fi >= frames.length) break; // The run ended before reaching this note.
    const t = frames[fi]!.t;

    if (note.dir === "jump" || note.dir === "crest") {
      candidates.push({
        t0: t - JUMP_LEAD,
        t1: t + JUMP_TAIL,
        mode: "sideline",
        reason: `${note.dir} @ ${note.s.toFixed(0)} m`,
        priority: 3,
      });
    } else if (
      (note.dir === "left" || note.dir === "right") &&
      note.severity <= 2
    ) {
      candidates.push({
        t0: t - TIGHT_LEAD,
        t1: t + TIGHT_TAIL,
        mode: "bonnet",
        reason: `tight ${note.dir} ${note.severity} @ ${note.s.toFixed(0)} m`,
        priority: 2,
      });
    } else if (
      (note.dir === "left" || note.dir === "right") &&
      note.severity >= 5
    ) {
      // Chase is the default anyway; an explicit low-priority hold keeps a
      // neighbouring bonnet/sideline shot from bleeding into the sweeper.
      candidates.push({
        t0: t - SWEEPER_LEAD,
        t1: t + SWEEPER_TAIL,
        mode: "chase",
        reason: `sweeper ${note.dir} ${note.severity} @ ${note.s.toFixed(0)} m`,
        priority: 1,
      });
    }
  }

  // Clamp to the replay, drop what clamping starved, then resolve conflicts:
  // highest priority wins, ties go to the earlier shot. Order of iteration is
  // fully determined by (priority, t0, s), so the edit is reproducible.
  const clamped = candidates
    .map((c) => ({ ...c, t0: Math.max(0, c.t0), t1: Math.min(duration, c.t1) }))
    .filter((c) => c.t1 - c.t0 >= MIN_SHOT)
    .sort((a, b) => b.priority - a.priority || a.t0 - b.t0);

  const accepted: Candidate[] = [];
  for (const c of clamped) {
    const collides = accepted.some(
      (o) => c.t0 < o.t1 + MIN_GAP && o.t0 < c.t1 + MIN_GAP,
    );
    if (!collides) accepted.push(c);
  }
  accepted.sort((a, b) => a.t0 - b.t0);
  return accepted.map(({ t0, t1, mode, reason }) => ({ t0, t1, mode, reason }));
}

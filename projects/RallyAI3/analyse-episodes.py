#!/usr/bin/env python3
"""
What actually ended the episodes, and why.

WHY THIS EXISTS

Three training runs plateaued against the same two walls. Obstacle strikes stayed at
45-53 % of failures through every configuration change and a sensor upgrade that tripled
detection range. Rollovers stayed at 15-20 % across every run and every difficulty. Both
were completely undiagnosed, for the same reason: the log recorded one word per episode,
and every competing explanation produces the same word.

    "HitObstacle"  — a rock in the racing line? a tree met after already running wide?
                     avoidable, or three rocks abreast with no line through?
    "RolledOver"   — too fast for the corner? a bad landing off the racing line?
                     something structural about where the centre of mass sits?

Those need opposite fixes. The agent now records the state at the moment things went
wrong, and this reads it back.

    .venv/bin/python analyse-episodes.py                    # newest run in results/
    .venv/bin/python analyse-episodes.py results/eval/x     # a specific directory
    .venv/bin/python analyse-episodes.py --last 2000        # only the last N episodes

Old logs, written before the forensic fields existed, still parse — the outcome table
works on them and the diagnosis sections say plainly that there is nothing to read.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

EPISODE_DIRS = ("results/eval", "results/episodes")

# Fields the forensics added. Their absence means an old log, not a corrupt one.
FORENSIC_FIELDS = ("station", "speed", "curve", "air", "wheels")


# ══════════════════════════════════════════════════════════════
#  LOADING
# ══════════════════════════════════════════════════════════════

def find_logs(where: str | None) -> list[str]:
    """Every .jsonl under a directory, or the newest batch if none was named."""
    if where:
        if os.path.isfile(where):
            return [where]
        found = sorted(glob.glob(os.path.join(where, "**", "*.jsonl"), recursive=True))
        return found

    for directory in EPISODE_DIRS:
        found = sorted(glob.glob(os.path.join(directory, "**", "*.jsonl"), recursive=True))
        if not found:
            continue
        # The newest session only. Mixing two runs' episodes into one table is how you
        # conclude a change helped when what actually changed was the curriculum.
        newest = max(found, key=os.path.getmtime)
        session = os.path.basename(newest).rsplit("-", 2)[-2:]
        stamp = "-".join(session).removesuffix(".jsonl")
        same_session = [f for f in found if f.endswith(f"{stamp}.jsonl")]
        return same_session or [newest]
    return []


def load(paths: list[str]) -> list[dict]:
    episodes = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # A partial last line is normal: the file is append-only and the run
                    # may still be going. Anything else is worth saying out loud.
                    print(f"  ! {path}:{line_number} unreadable ({exc})", file=sys.stderr)
    episodes.sort(key=lambda e: e.get("t", 0))
    return episodes


# ══════════════════════════════════════════════════════════════
#  PRESENTATION
# ══════════════════════════════════════════════════════════════

def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("─" * max(46, len(title)))


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f} %" if whole else "    — "


def spread(values: list[float], unit: str = "", places: int = 1) -> str:
    """Median and quartiles. A mean would hide exactly the tail that matters here."""
    if not values:
        return "no data"
    ordered = sorted(values)
    n = len(ordered)

    def at(fraction):
        return ordered[min(n - 1, int(fraction * n))]

    # "max", not "worst": the same helper describes rollover speeds (where high is bad)
    # and stage progress (where high is the goal), and a judgemental label would be wrong
    # half the time.
    return (f"median {at(0.5):.{places}f}{unit}   "
            f"quartiles {at(0.25):.{places}f}-{at(0.75):.{places}f}{unit}   "
            f"max {ordered[-1]:.{places}f}{unit}")


def histogram(pairs: list[tuple[str, int]], total: int, width: int = 28) -> None:
    if not total:
        return
    biggest = max((count for _, count in pairs), default=0) or 1
    for label, count in pairs:
        bar = "█" * max(0, round(width * count / biggest))
        print(f"  {label:<22} {count:6d}  {pct(count, total)}  {bar}")


# ══════════════════════════════════════════════════════════════
#  THE OVERALL PICTURE
# ══════════════════════════════════════════════════════════════

def outcomes(episodes: list[dict]) -> None:
    rule(f"Outcomes — {len(episodes)} episodes")

    counted = Counter(e.get("outcome", "?") for e in episodes)
    order = ["Finished", "TimedOut", "HitObstacle", "HardImpact",
             "RolledOver", "FellOff", "Stalled"]
    ranked = [(name, counted[name]) for name in order if counted[name]]
    ranked += [(name, count) for name, count in counted.items() if name not in order]
    histogram(ranked, len(episodes))

    progress = [e["waypoints"] / e["target"] for e in episodes
                if e.get("target")]
    if progress:
        print(f"\n  stage completed      {spread([p * 100 for p in progress], ' %', 0)}")

    rewards = [e["reward"] for e in episodes if "reward" in e]
    if rewards:
        print(f"  reward               {spread(rewards)}")

    seconds = [e["seconds"] for e in episodes if "seconds" in e]
    if seconds:
        longest = max(seconds)
        print(f"  episode length       {spread(seconds, ' s')}")
        # An episode that stops dead on the clock is a different failure from one that
        # crashes, and the two used to be indistinguishable in this log.
        near_clock = sum(1 for s in seconds if s > longest * 0.98)
        if counted.get("TimedOut"):
            print(f"  ran the clock out    {near_clock} episodes reached ~{longest:.0f} s")


# ══════════════════════════════════════════════════════════════
#  ROLLOVERS
#
#  The three live hypotheses, and what separates them:
#
#    too fast for the corner   high speed AND meaningful curvature, wheels on the ground
#    a bad landing             airborne time > 0 just before, often off the racing line
#    structural / centre of mass
#                              moderate speed, on the road, straight, no air — the car
#                              simply falling over, which no amount of driving fixes
# ══════════════════════════════════════════════════════════════

def rollovers(episodes: list[dict]) -> None:
    rolls = [e for e in episodes if e.get("outcome") == "RolledOver"]
    rule(f"Rollovers — {len(rolls)} of {len(episodes)} episodes ({pct(len(rolls), len(episodes)).strip()})")

    if not rolls:
        print("  none")
        return
    if not any(f in rolls[0] for f in FORENSIC_FIELDS):
        print("  This log predates the forensic fields, so there is nothing to diagnose.")
        print("  Re-run with the current build: ./evaluate.sh --episodes 200")
        return

    speeds = [e.get("speed", 0.0) for e in rolls]
    curves = [abs(e.get("curve", 0.0)) for e in rolls]
    offsets = [abs(e.get("off", 0.0)) for e in rolls]
    flights = [e.get("flight", 0.0) for e in rolls]

    print("  State at the last moment the car was still level:")
    print(f"    speed              {spread(speeds, ' m/s')}")
    print(f"    corner ahead       {spread([c * 57.3 for c in curves], ' deg/25 m')}")
    print(f"    off the centreline {spread(offsets, ' m')}")
    print(f"    longest flight     {spread(flights, ' s', 2)}")

    # ── Attribution. Deliberately crude thresholds, stated in the output so the reader
    #    can disagree with them: the point is the SHAPE of the split, not the cut points.
    ON_ROAD = 6.0        # metres; the road is 12 m wide
    A_CORNER = 0.15      # radians per 25 m, about 9 degrees
    FAST = 20.0          # m/s, about 72 km/h
    AIRBORNE = 0.15      # seconds off all four wheels

    causes = Counter()
    for e in rolls:
        airborne = e.get("flight", 0.0) >= AIRBORNE
        cornering = abs(e.get("curve", 0.0)) >= A_CORNER
        fast = e.get("speed", 0.0) >= FAST
        on_road = abs(e.get("off", 0.0)) <= ON_ROAD

        if airborne and not on_road:
            causes["landed off the road"] += 1
        elif airborne:
            causes["landed on the road"] += 1
        elif cornering and fast:
            causes["too fast for the corner"] += 1
        elif not on_road:
            causes["off the road, no air"] += 1
        elif cornering:
            causes["cornering, not fast"] += 1
        else:
            causes["upright, level, slow — fell over"] += 1

    print(f"\n  Attributed (on road = within {ON_ROAD:.0f} m, corner = {A_CORNER * 57.3:.0f} deg/25 m, "
          f"fast = {FAST:.0f} m/s, air = {AIRBORNE:.2f} s):")
    histogram(causes.most_common(), len(rolls))

    print("\n  Read it like this:")
    print("    landed off the road       -> a cornering problem. The roll is the symptom;")
    print("                                 the car was already in the scenery.")
    print("    too fast for the corner   -> reward or observation: it is not braking.")
    print("    upright, level, slow      -> structural. Check the centre of mass height")
    print("                                 and the anti-roll rates; no policy fixes this.")


# ══════════════════════════════════════════════════════════════
#  OBSTACLES
#
#  Two questions the old log could not tell apart:
#    1. WHAT was hit — a rock in the road, or a tree only reachable by leaving it?
#    2. Was there anywhere else to go, and was there time to go there?
# ══════════════════════════════════════════════════════════════

def obstacles(episodes: list[dict]) -> None:
    hits = [e for e in episodes if e.get("outcome") in ("HitObstacle", "HardImpact")]
    rule(f"Impacts — {len(hits)} of {len(episodes)} episodes ({pct(len(hits), len(episodes)).strip()})")

    if not hits:
        print("  none")
        return
    if "hit" not in hits[0]:
        print("  This log predates the forensic fields, so there is nothing to diagnose.")
        print("  Re-run with the current build: ./evaluate.sh --episodes 200")
        return

    # ── 1. What was hit.
    struck = Counter(e.get("hit") or "(unnamed)" for e in hits)
    print("  What was struck:")
    histogram(struck.most_common(), len(hits))

    rocks = [e for e in hits if e.get("hit") == "RoadRock"]
    scenery = [e for e in hits if e.get("hit") in ("Tree", "Boulder")]

    if scenery:
        off = [abs(e.get("off", 0.0)) for e in scenery]
        print(f"\n  Scenery strikes happened {spread(off, ' m')} from the centreline.")
        print("  The road is 12 m wide, so anything past 6 m was already off it: those are")
        print("  lane-keeping failures wearing an obstacle's clothes, not obstacle failures.")

    # ── 2. Was there a gap, and was there time?
    if not rocks:
        print("\n  No road-rock strikes to judge for avoidability.")
        return

    gapped = [e for e in rocks if "gap" in e]
    rule(f"Road rocks — {len(rocks)} strikes, {len(gapped)} with a measured gap")

    if gapped:
        gaps = [e["gap"] for e in gapped]
        print(f"  Widest clear corridor at the strike: {spread(gaps, ' m')}")

        # A rally car is about 1.75 m wide. Under ~2.5 m of gap there is no line through
        # at speed; over ~4 m there plainly was one.
        CAR = 1.75
        blocked = sum(1 for g in gaps if g < CAR * 1.4)
        roomy = sum(1 for g in gaps if g >= CAR * 2.3)
        print(f"    no line through (< {CAR * 1.4:.1f} m)   {blocked:4d}  {pct(blocked, len(gaps))}")
        print(f"    room to spare  (>= {CAR * 2.3:.1f} m)   {roomy:4d}  {pct(roomy, len(gaps))}")

        missed = [e for e in gapped if e["gap"] >= CAR * 2.3]
        if missed:
            wrong_side = [abs(e.get("off", 0.0) - e.get("gapCentre", 0.0)) for e in missed]
            print(f"    when there was room, the car was {spread(wrong_side, ' m')} from the")
            print("    middle of it — that distance is the size of the mistake.")

    # ── 3. How much warning.
    warned = [e["warning"] for e in rocks if "warning" in e]
    unseen = sum(1 for w in warned if w < 0)
    seen = [w for w in warned if w >= 0]

    if warned:
        print(f"\n  Warning the sensor gave (optimistic — occlusion ignored):")
        print(f"    never in the ray fan   {unseen:4d}  {pct(unseen, len(warned))}")
        if seen:
            print(f"    when it was visible    {spread(seen, ' s', 2)}")
            hopeless = sum(1 for w in seen if w < 0.4)
            print(f"    under 0.4 s of warning {hopeless:4d}  {pct(hopeless, len(seen))}")

    print("\n  Read it like this:")
    print("    mostly blocked gaps    -> the stage is unfair. Lower the rock density or")
    print("                              stop them landing abreast.")
    print("    mostly roomy gaps      -> a skill ceiling, not a sensing one. More training")
    print("                              or better reward shaping, not more rays.")
    print("    mostly never in view   -> it is hitting rocks while sideways or off the road,")
    print("                              which is again a cornering problem.")


# ══════════════════════════════════════════════════════════════
#  PER-STAGE, for replayed evaluations
# ══════════════════════════════════════════════════════════════

def per_stage(episodes: list[dict], minimum: int = 3) -> None:
    by_seed = defaultdict(list)
    for e in episodes:
        by_seed[e.get("seed", 0)].append(e)

    repeated = {seed: eps for seed, eps in by_seed.items() if len(eps) >= minimum}
    if len(repeated) < 2:
        return

    rule(f"By stage — {len(repeated)} seeds driven {minimum}+ times")
    print(f"  {'seed':>12}  {'n':>4}  {'finished':>9}  {'progress':>9}  most common ending")

    rows = []
    for seed, eps in repeated.items():
        finished = sum(1 for e in eps if e.get("outcome") == "Finished")
        progress = sum(e["waypoints"] / e["target"] for e in eps if e.get("target")) / len(eps)
        common = Counter(e.get("outcome") for e in eps).most_common(1)[0][0]
        rows.append((progress, seed, len(eps), finished, common))

    # The hardest handful only. A training run touches well over a thousand seeds and a
    # full listing buries the one line worth acting on.
    WORST = 12
    for progress, seed, n, finished, common in sorted(rows)[:WORST]:
        print(f"  {seed:>12}  {n:>4}  {pct(finished, n):>9}  {progress * 100:8.1f} %  {common}")
    if len(rows) > WORST:
        print(f"  … {len(rows) - WORST} more, easier")

    print("\n  A stage at the top of this table that every policy fails is a property of")
    print("  the generator, not of the driver. Replay it: ./evaluate.sh --seeds <seed>")


# ══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="episode .jsonl file or a directory of them")
    parser.add_argument("--last", type=int, default=0,
                        help="only the most recent N episodes")
    args = parser.parse_args()

    paths = find_logs(args.path)
    if not paths:
        print("No episode logs found. Train something, or run ./evaluate.sh", file=sys.stderr)
        return 1

    episodes = load(paths)
    if not episodes:
        print("Episode logs are empty.", file=sys.stderr)
        return 1

    if args.last:
        episodes = episodes[-args.last:]

    print(f"\n{len(paths)} file(s): {', '.join(os.path.basename(p) for p in paths[:4])}"
          f"{' …' if len(paths) > 4 else ''}")

    outcomes(episodes)
    rollovers(episodes)
    obstacles(episodes)
    per_stage(episodes)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

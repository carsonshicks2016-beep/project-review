#!/usr/bin/env python3
"""
Live digest of a training run, in the terminal.

TensorBoard shows every metric ML-Agents emits, which is about fifteen charts and no
guidance on which of them mean anything. This prints the four that actually tell you
whether the run is working, plus a read of what the numbers are saying.

    .venv/bin/python watch-training.py            # latest run
    .venv/bin/python watch-training.py rally05    # a specific run
"""
import glob
import os
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS = "results"
REFRESH = 20  # seconds


def latest_run() -> str | None:
    runs = [d for d in glob.glob(f"{RESULTS}/*") if os.path.isdir(d)]
    return max(runs, key=os.path.getmtime) if runs else None


def series(acc, tag, n=1):
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [e.value for e in acc.Scalars(tag)][-n:]


def spark(values, width=28):
    """A one-line trend. Enough to see direction, which is all that matters here."""
    blocks = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    if len(vals) < 2:
        return "collecting…"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return blocks[0] * len(vals)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.99))] for v in vals)


def trend(values, n=6):
    """+1 improving, -1 getting worse, 0 flat or not enough data yet."""
    if len(values) < 2 * n:
        return 0
    before = sum(values[-2 * n:-n]) / n
    after = sum(values[-n:]) / n
    margin = abs(before) * 0.05 + 1e-6
    return 1 if after > before + margin else -1 if after < before - margin else 0


def verdict(frac, timed, fell, hit, rolled, finished, stalled, rising=0):
    """The one sentence worth acting on.

    `rising` is whether reward is actually improving, and it exists because this function
    got it badly wrong without it. At 400k steps of a healthy run it read 72 % "left
    stage" with a low stage fraction and advised raising failurePenalty — while reward was
    climbing -3 -> +28 and waypoints 0 -> 5. Crashing a lot IS the intermediate state: the
    car has learnt to move and not yet to steer. Acting on that advice would have pushed it
    straight back toward the parking behaviour it had just escaped.

    A high crash rate only means "too aggressive" if progress has STOPPED improving.
    """
    if finished > 0.25:
        return "Finishing regularly, on stages it has never seen. That is real driving — let the curriculum raise the rocks."
    if stalled > 0.60:
        return "Mostly parking. Normal for an untrained policy sitting on the brake; if it persists past ~1M steps the drive axis is not finding throttle, so raise beta."
    if rising > 0:
        return "Reward and progress are both climbing. This is working — leave it alone, whatever the crash rate looks like."
    if timed > 0.80 and frac < 0.10:
        return "Running out the clock without progress. Check it is actually moving — mean length at the cap means it is not."
    if frac < 0.10 and (fell + hit + rolled) > 0.70 and rising < 0:
        return "Crashing early and no longer improving. Now it is worth raising failurePenalty."
    if frac > 0.10:
        return "Making real progress down the stage. Leave it alone."
    return "Too early to read. Give it another twenty minutes."


def render(run):
    files = sorted(glob.glob(f"{run}/*/events.out.tfevents.*"))
    if not files:
        return f"  {run}: no data yet"

    acc = EventAccumulator(files[-1])
    acc.Reload()
    if "Environment/Cumulative Reward" not in acc.Tags().get("scalars", []):
        return f"  {run}: waiting for the first summary…"

    rewards = series(acc, "Environment/Cumulative Reward", 40)
    fracs = series(acc, "Progress/StageFraction", 40)
    waypoints = series(acc, "Progress/Waypoints", 40)
    step = acc.Scalars("Environment/Cumulative Reward")[-1].step

    def last(tag):
        v = series(acc, tag, 1)
        return v[0] if v else 0.0

    frac = fracs[-1] if fracs else 0.0
    timed, fell = last("Outcome/TimedOut"), last("Outcome/FellOff")
    hit, rolled = last("Outcome/HitObstacle"), last("Outcome/RolledOver")
    finished = last("Outcome/Finished")
    stalled = last("Outcome/Stalled")

    # Waypoints in absolute terms as well as as a fraction. The percentage alone is hard to
    # act on — "6 %" and "one waypoint out of seventeen" are the same fact, but only one of
    # them tells you what the car is actually doing.
    wp = waypoints[-1] if waypoints else 0.0
    target = round(wp / frac) if frac > 0.01 and wp > 0 else 0
    wp_line = f"  waypoints        {wp:7.2f}" + (f" of {target}" if target else "") + f"   {spark(waypoints)}"

    rising = trend(rewards)
    arrow = {1: "climbing", -1: "falling", 0: "flat"}[rising]

    out = [
        f"  {os.path.basename(run)}   {step:,} steps        reward {arrow}",
        "",
        f"  stage progress   {frac * 100:5.1f} %   {spark(fracs)}",
        wp_line,
        f"  reward           {rewards[-1]:7.1f}   {spark(rewards)}",
        "",
        "  how episodes end",
    ]
    for name, v in [("finished", finished), ("stalled", stalled), ("timed out", timed),
                    ("left stage", fell), ("hit something", hit), ("rolled", rolled)]:
        out.append(f"    {name:<14} {v * 100:4.0f} %  {'█' * int(v * 26)}")
    out += ["", f"  {verdict(frac, timed, fell, hit, rolled, finished, stalled, rising)}"]
    return "\n".join(out)


def main():
    run = f"{RESULTS}/{sys.argv[1]}" if len(sys.argv) > 1 else latest_run()
    if not run:
        print("No runs in results/.")
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear
            print(f"  rally training — {time.strftime('%H:%M:%S')}   (ctrl-c to stop)\n")
            print(render(run))
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

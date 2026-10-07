"""
Aegis M5 — ablation runner. Launches several mlagents-learn runs with different settings and
prints the final metric for each, so you can answer research questions (AMP on/off, shield in
observations on/off, reward-term sweeps).

SOLID: this launches real runs and parses real TensorBoard event files.

Usage example:
  python tools/ablation.py \
      --base config/duel_ppo_selfplay.yaml \
      --runs amp_on:--env-args,style=1  amp_off:--env-args,style=0 \
      --metric "Self-play/ELO"
Requires: mlagents, tensorboard.  Set extra env knobs via Unity Environment Parameters.
"""
import argparse, subprocess, glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def launch(base_cfg, run_id, extra_args):
    cmd = ["mlagents-learn", base_cfg, f"--run-id={run_id}",
           "--time-scale=20", "--no-graphics", "--force"] + extra_args
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def final_scalar(run_id, metric):
    # results/<run_id>/<behavior or run>/events.out.tfevents.*
    paths = glob.glob(os.path.join("results", run_id, "**", "events.out.tfevents.*"), recursive=True)
    best = None
    for p in paths:
        acc = EventAccumulator(os.path.dirname(p))
        acc.Reload()
        if metric not in acc.Tags().get("scalars", []):
            continue
        vals = acc.Scalars(metric)
        if vals:
            best = vals[-1].value
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base trainer YAML")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="entries like name:arg1,arg2 (args passed to mlagents-learn)")
    ap.add_argument("--metric", default="Environment/Cumulative Reward")
    ap.add_argument("--no-launch", action="store_true", help="only parse existing results")
    a = ap.parse_args()

    parsed = []
    for spec in a.runs:
        name, _, args = spec.partition(":")
        extra = [x for x in args.split(",") if x]
        if not a.no_launch:
            launch(a.base, name, extra)
        parsed.append(name)

    print(f"\n=== ablation: {a.metric} ===")
    for name in parsed:
        v = final_scalar(name, a.metric)
        print(f"{name:20s} {('%.3f' % v) if v is not None else 'n/a'}")


if __name__ == "__main__":
    main()

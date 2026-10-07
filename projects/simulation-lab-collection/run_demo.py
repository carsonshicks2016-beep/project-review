"""
Quick launcher for ML Drift Master.
Usage:
    python3 run_demo.py             # Playable Human Drive Mode
    python3 run_demo.py --ai        # AI Autonomous Mode
    python3 run_demo.py --eval      # Benchmark Telemetry
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="ML Drift Master Launcher")
    parser.add_argument("--track", type=str, default="gymkhana", choices=["touge", "gymkhana", "stadium"], help="Track layout")
    parser.add_argument("--ai", action="store_true", help="Launch in AI mode (uses best trained model)")
    parser.add_argument("--eval", action="store_true", help="Run telemetry benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes for evaluation")
    args = parser.parse_args()

    if args.eval:
        from drift_rl.evaluate import evaluate_agent
        evaluate_agent(track=args.track, episodes=args.episodes)
    else:
        from drift_rl.visualizer import DriftVisualizer
        model_path = "models/drift_ppo_touge_final.zip" if args.ai else None
        vis = DriftVisualizer(track_name=args.track, model_path=model_path)
        vis.run()

if __name__ == "__main__":
    main()

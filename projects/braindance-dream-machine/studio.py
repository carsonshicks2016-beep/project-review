#!/usr/bin/env python3
"""
AI Audio Dream Studio - Master CLI
Unconditional neural audio hallucinations inspired by Aphex Twin & LTJ Bukem.
"""

import sys
import os
import argparse
import glob
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.preprocessor import AudioDatasetManager
from engine.synthesizer import generate_starter_pack
from engine.dreamer import Dreamer
from engine.transformer import TimbreTransformer
from engine.trainer import Trainer

def find_default_model() -> str:
    """Finds an available exported model or downloaded checkpoint."""
    # 1. Check local models/exported
    exported = sorted(glob.glob("models/exported/*.ts"))
    if exported:
        return exported[-1]
    
    # 2. Check local models/
    local = sorted(glob.glob("models/*.ts"))
    if local:
        return local[-1]

    # 3. Check huggingface cache
    hf_cached = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/*.ts"), recursive=True))
    if hf_cached:
        return hf_cached[0]

    return ""

def main():
    parser = argparse.ArgumentParser(
        description="AI Audio Dream Studio: Neural audio hallucination engine."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: demo
    sub_demo = subparsers.add_parser("demo", help="Generate synthetic starter pack (Bukem pads, Aphex breaks, acid).")
    sub_demo.add_argument("--duration", type=float, default=12.0, help="Duration of clips in seconds")

    # Command: stats
    sub_stats = subparsers.add_parser("stats", help="Show audio dataset statistics (tracks, duration, formats).")

    # Command: prep
    sub_prep = subparsers.add_parser("prep", help="Preprocess audio files in data/raw into RAVE database.")
    sub_prep.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100)")

    # Command: dream
    sub_dream = subparsers.add_parser("dream", help="Hallucinate a new audio track from scratch.")
    sub_dream.add_argument("--duration", type=float, default=8.0, help="Duration of dream in seconds")
    sub_dream.add_argument("--temp", type=float, default=1.0, help="Latent temperature / chaos (default: 1.0)")
    sub_dream.add_argument("--momentum", type=float, default=0.92, help="Trajectory smoothness (default: 0.92)")
    sub_dream.add_argument("--glitch", type=float, default=0.15, help="Aphex stutter probability (default: 0.15)")
    sub_dream.add_argument("--model", type=str, default="", help="Path to .ts model (optional)")
    sub_dream.add_argument("--out", type=str, default=None, help="Output wav filename")

    # Command: morph
    sub_morph = subparsers.add_parser("morph", help="Neural timbre transfer / morph input audio with dream state.")
    sub_morph.add_argument("--input", type=str, required=True, help="Input audio file path")
    sub_morph.add_argument("--blend", type=float, default=0.25, help="Dream blend factor (0=pure timbre, 1=pure dream)")
    sub_morph.add_argument("--temp", type=float, default=1.0, help="Temperature variation")
    sub_morph.add_argument("--model", type=str, default="", help="Path to .ts model (optional)")
    sub_morph.add_argument("--out", type=str, default=None, help="Output wav filename")

    # Command: train
    sub_train = subparsers.add_parser("train", help="Train RAVE model on preprocessed dataset.")
    sub_train.add_argument("--name", type=str, default="aphex_bukem_run", help="Run name")
    sub_train.add_argument("--config", type=str, default="v2", help="RAVE config (v2, v1, etc.)")
    sub_train.add_argument("--batch", type=int, default=4, help="Batch size")
    sub_train.add_argument("--steps", type=int, default=50000, help="Max steps")

    # Command: ui
    sub_ui = subparsers.add_parser("ui", help="Launch interactive Gradio Web Studio.")
    sub_ui.add_argument("--port", type=int, default=7860, help="Port to run on")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "demo":
        print("--- Generating Starter Audio Pack ---")
        generate_starter_pack(duration=args.duration)

    elif args.command == "stats":
        print("--- Dataset Statistics ---")
        mgr = AudioDatasetManager()
        stats = mgr.get_dataset_stats()
        print(f"Total Tracks Found: {stats['count']}")
        print(f"Total Duration:     {stats['total_seconds']} seconds ({stats['total_hours']} hours)")
        print(f"Source Directory:   {'data/demo (Starter Pack)' if stats['is_demo'] else 'data/raw (User Catalog)'}")
        print("\nFiles:")
        for f in stats["files"]:
            print(f"  * {f['name']} ({f.get('duration_sec', '?')}s, {f.get('samplerate', '?')} Hz, {f.get('channels', '?')}ch)")

    elif args.command == "prep":
        print("--- Preprocessing Dataset for RAVE ---")
        mgr = AudioDatasetManager(target_sr=args.sr)
        mgr.prepare_dataset_for_rave()

    elif args.command == "dream":
        model_path = args.model or find_default_model()
        if not model_path:
            print("No model found! Train a model first, or supply --model path/to/model.ts")
            sys.exit(1)

        dreamer = Dreamer(model_path)
        audio, out_file = dreamer.dream(
            duration_sec=args.duration,
            temperature=args.temp,
            momentum=args.momentum,
            glitch_probability=args.glitch,
            out_path=args.out
        )
        print(f"Dream completed! Listen with: afplay {out_file}")

    elif args.command == "morph":
        model_path = args.model or find_default_model()
        if not model_path:
            print("No model found! Train a model first, or supply --model path/to/model.ts")
            sys.exit(1)

        transformer = TimbreTransformer(model_path)
        y, out_file = transformer.transform(
            audio_path=args.input,
            dream_blend=args.blend,
            temperature=args.temp,
            out_path=args.out
        )
        print(f"Morph completed! Listen with: afplay {out_file}")

    elif args.command == "train":
        trainer = Trainer()
        proc = trainer.train_vae(
            run_name=args.name,
            config=args.config,
            batch_size=args.batch,
            max_steps=args.steps
        )
        proc.wait()

    elif args.command == "ui":
        print("Launching Web Studio...")
        os.system(f"{sys.executable} app.py --port {args.port}")

if __name__ == "__main__":
    main()

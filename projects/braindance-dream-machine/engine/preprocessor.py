"""
Audio Preprocessor & Dataset Pipeline for RAVE
Scans input music files (Aphex Twin, LTJ Bukem, etc.), analyzes dataset metrics,
normalizes audio, and builds the RAVE training database.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import soundfile as sf
import librosa
import numpy as np

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}

class AudioDatasetManager:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", target_sr: int = 44100):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_sr = target_sr
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def scan_files(self, include_demo_if_empty: bool = True) -> List[Path]:
        """Finds all audio files in raw directory, falling back to demo if empty."""
        files = [
            p for p in self.raw_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
        ]
        if not files and include_demo_if_empty:
            demo_dir = Path("data/demo")
            if demo_dir.exists():
                files = [
                    p for p in demo_dir.rglob("*")
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
                ]
        return sorted(files)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Calculates dataset summary: total tracks, duration, sample rates, file sizes."""
        files = self.scan_files(include_demo_if_empty=True)
        if not files:
            return {
                "count": 0,
                "total_seconds": 0.0,
                "total_hours": 0.0,
                "files": [],
                "is_demo": False
            }

        total_sec = 0.0
        file_details = []
        is_demo = any("data/demo" in str(f) for f in files)

        for f in files:
            try:
                info = sf.info(str(f))
                total_sec += info.duration
                file_details.append({
                    "name": f.name,
                    "path": str(f),
                    "duration_sec": round(info.duration, 2),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format
                })
            except Exception as e:
                file_details.append({
                    "name": f.name,
                    "path": str(f),
                    "error": str(e)
                })

        return {
            "count": len(files),
            "total_seconds": round(total_sec, 2),
            "total_hours": round(total_sec / 3600.0, 3),
            "files": file_details,
            "is_demo": is_demo
        }

    def prepare_dataset_for_rave(self, output_db: str = "data/processed/rave_db") -> bool:
        """
        Executes RAVE's native preprocess command to construct the LMDB database.
        """
        files = self.scan_files(include_demo_if_empty=True)
        if not files:
            print("No audio files found to preprocess! Add audio files to data/raw/")
            return False

        # Input directory to feed rave preprocess
        is_demo = any("data/demo" in str(f) for f in files)
        input_dir = "data/demo" if is_demo and not list(self.raw_dir.glob("*")) else str(self.raw_dir)

        print(f"Preprocessing audio from '{input_dir}' into '{output_db}' at {self.target_sr} Hz...")

        cmd = [
            "rave", "preprocess",
            "--input_path", input_dir,
            "--output_path", output_db,
            "--sampling_rate", str(self.target_sr)
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("RAVE preprocessing completed successfully!")
            print(res.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error during RAVE preprocess: {e}")
            print(e.stderr)
            return False

if __name__ == "__main__":
    manager = AudioDatasetManager()
    stats = manager.get_dataset_stats()
    print("Dataset Stats:", stats)

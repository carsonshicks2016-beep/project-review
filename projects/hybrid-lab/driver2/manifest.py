from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_spec_hash(spec) -> str:
    """Compute SHA-256 hash of the repr of a CarSpec's fields."""
    h = hashlib.sha256()
    h.update(repr(asdict(spec)).encode('utf-8'))
    return h.hexdigest()

def create_run_id() -> str:
    """Return a timestamped UUID: YYYYMMDD_HHMMSS_<short-uuid>"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


@dataclass
class RunManifest:
    """Versioned run manifest for Driver 2.0 training runs."""
    # Identity
    run_id: str                    # UUID for this run
    timestamp: str                 # ISO 8601 start time
    driver2_version: str           # from driver2.__version__
    
    # Configuration
    car: str                       # e.g. 'porsche_919evo'
    track: str                     # e.g. 'nordschleife'
    brain_config: dict             # serialized BrainConfig
    ga_config: dict                # serialized GAConfig  
    curriculum_config: dict        # serialized CurriculumConfig
    
    # Physics fingerprint
    simulator_fingerprint: str     # hash of supra/physics.py content
    car_spec_hash: str             # hash of CarSpec fields
    
    # Training state
    curriculum_level: int = 0
    generation: int = 0
    best_fitness: float = -1e9
    segments_mastered: int = 0
    total_segments: int = 64
    
    # Evaluation results (filled in when eval runs)
    eval_result: dict | None = None  # lap_time, clean_sectors, etc.
    
    # Labeling
    label: str = 'legacy experimental simulation'
    notes: str = ''


class ManifestWriter:
    """Writes manifests, telemetry, genomes and logs for a training run."""
    
    def __init__(self, base_dir: str, run_id: str):
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def save_manifest(self, manifest: RunManifest) -> Path:
        """Write manifest.json atomically."""
        manifest_path = self.run_dir / "manifest.json"
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            dir=self.run_dir, 
            delete=False, 
            suffix='.tmp'
        )
        try:
            json.dump(asdict(manifest), temp_file, indent=2, sort_keys=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_file.name, manifest_path)
        except Exception:
            os.unlink(temp_file.name)
            raise
            
        return manifest_path
        
    def save_genome(self, genome: np.ndarray, tag: str = 'best') -> Path:
        """Save a numpy genome to npz compressed format."""
        path = self.run_dir / f"{tag}.npz"
        np.savez_compressed(path, genome=genome)
        return path
        
    def save_telemetry(self, data: dict, tag: str = 'latest') -> Path:
        """Save a telemetry dict to json."""
        path = self.run_dir / f"telemetry_{tag}.json"
        with open(path, 'w') as f:
            json.dump(data, f)
        return path
        
    def save_generation_log(self, gen: int, stats: dict) -> None:
        """Append a JSON line to generations.jsonl."""
        path = self.run_dir / "generations.jsonl"
        log_entry = {
            "gen": gen,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **stats
        }
        with open(path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def artifact_path(self, filename: str) -> Path:
        """Return a path inside the run directory, asserting safety."""
        path = (self.run_dir / filename).resolve()
        run_dir_resolved = self.run_dir.resolve()
        assert path.is_relative_to(run_dir_resolved), f"Path {filename} escapes run directory!"
        return path

"""
Braindance Dream Machine - FastAPI Studio Server
Full-featured backend API for audio generation, timbre transfer, dataset management,
training monitoring, and model management.
"""

import os
import sys
import glob
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque

import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Project imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engine.preprocessor import AudioDatasetManager
from engine.dreamer import Dreamer
from engine.transformer import TimbreTransformer
from engine.synthesizer import generate_starter_pack
from engine.trainer import Trainer

app = FastAPI(title="Braindance Dream Machine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
TRAINING_PROCESS: Optional[subprocess.Popen] = None
TRAINING_LOGS = deque(maxlen=1000)
TRAINING_LOCK = threading.Lock()
ACTIVE_MODEL_PATH: Optional[str] = None
CURRENT_DREAMER: Optional[Dreamer] = None
CURRENT_TRANSFORMER: Optional[TimbreTransformer] = None

def get_available_models() -> List[Dict[str, str]]:
    """Collects local exported models, repo models, and HF cache models."""
    results = []
    
    # 1. Local exported
    for p in sorted(glob.glob("models/exported/*.ts")):
        results.append({
            "name": f"Local: {Path(p).stem}",
            "path": p,
            "type": "exported"
        })
        
    # 2. Local models
    for p in sorted(glob.glob("models/*.ts")):
        results.append({
            "name": f"Local: {Path(p).stem}",
            "path": p,
            "type": "local"
        })

    # 3. HF Cached
    for p in sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/*.ts"), recursive=True)):
        results.append({
            "name": f"HuggingFace: {Path(p).stem}",
            "path": p,
            "type": "cached"
        })

    return results

def get_or_load_dreamer(model_path: Optional[str] = None) -> Dreamer:
    global ACTIVE_MODEL_PATH, CURRENT_DREAMER
    models = get_available_models()
    target_path = model_path or (models[0]["path"] if models else None)
    
    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="No valid neural model found or specified.")

    if CURRENT_DREAMER is None or ACTIVE_MODEL_PATH != target_path:
        ACTIVE_MODEL_PATH = target_path
        CURRENT_DREAMER = Dreamer(target_path)

    return CURRENT_DREAMER

def get_or_load_transformer(model_path: Optional[str] = None) -> TimbreTransformer:
    global ACTIVE_MODEL_PATH, CURRENT_TRANSFORMER
    models = get_available_models()
    target_path = model_path or (models[0]["path"] if models else None)
    
    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="No valid neural model found or specified.")

    if CURRENT_TRANSFORMER is None or ACTIVE_MODEL_PATH != target_path:
        ACTIVE_MODEL_PATH = target_path
        CURRENT_TRANSFORMER = TimbreTransformer(target_path)

    return CURRENT_TRANSFORMER

# ----------------- API Endpoints -----------------

@app.get("/api/status")
def get_system_status():
    mps_available = torch.backends.mps.is_available()
    models = get_available_models()
    
    return {
        "device": "Apple Silicon (Metal / MPS)" if mps_available else "CPU",
        "mps_available": mps_available,
        "torch_version": torch.__version__,
        "models": models,
        "active_model": ACTIVE_MODEL_PATH or (models[0]["path"] if models else None),
        "is_training": TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None
    }

@app.get("/api/library")
def get_library():
    mgr = AudioDatasetManager()
    stats = mgr.get_dataset_stats()
    
    # Check if rave_db is ready
    db_exists = os.path.exists("data/processed/rave_db/data.mdb")
    db_size_mb = 0
    if db_exists:
        db_size_mb = round(os.path.getsize("data/processed/rave_db/data.mdb") / (1024 * 1024), 2)

    return {
        "stats": stats,
        "db_ready": db_exists,
        "db_size_mb": db_size_mb
    }

@app.post("/api/library/upload")
async def upload_audio_files(files: List[UploadFile] = File(...)):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for file in files:
        target_path = raw_dir / file.filename
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(file.filename)

    return {"status": "success", "uploaded": saved_files}

@app.post("/api/library/generate-demo")
def generate_demo():
    created = generate_starter_pack()
    return {"status": "success", "files": created}

@app.post("/api/library/preprocess")
def preprocess_dataset(background_tasks: BackgroundTasks):
    def run_preprocess_task():
        mgr = AudioDatasetManager()
        mgr.prepare_dataset_for_rave()

    background_tasks.add_task(run_preprocess_task)
    return {"status": "started", "message": "Preprocessing running in background"}

@app.get("/api/audio/{category}/{filename}")
def stream_audio(category: str, filename: str):
    """Safe streaming endpoint for previewing audio files."""
    valid_dirs = {
        "raw": Path("data/raw"),
        "demo": Path("data/demo"),
        "dreams": Path("dreams")
    }
    if category not in valid_dirs:
        raise HTTPException(status_code=400, detail="Invalid audio category")

    target = valid_dirs[category] / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(target, media_type="audio/wav")

@app.post("/api/dream")
def generate_dream(
    duration: float = Form(8.0),
    temperature: float = Form(1.0),
    momentum: float = Form(0.92),
    glitch: float = Form(0.15),
    seed: Optional[str] = Form(None),
    model_path: Optional[str] = Form(None)
):
    dreamer = get_or_load_dreamer(model_path)
    seed_val = int(seed) if seed and seed.strip().isdigit() else None

    t0 = time.time()
    audio, out_path = dreamer.dream(
        duration_sec=duration,
        temperature=temperature,
        momentum=momentum,
        glitch_probability=glitch,
        seed=seed_val
    )
    elapsed = time.time() - t0

    filename = Path(out_path).name
    return {
        "status": "success",
        "filename": filename,
        "audio_url": f"/api/audio/dreams/{filename}",
        "duration": duration,
        "elapsed_sec": round(elapsed, 2)
    }

@app.get("/api/dreams")
def list_dreams():
    dream_files = sorted(glob.glob("dreams/*.wav"), key=os.path.getmtime, reverse=True)
    results = []
    for f in dream_files:
        p = Path(f)
        try:
            info = sf.info(f)
            dur = round(info.duration, 2)
        except:
            dur = 0.0
        results.append({
            "name": p.name,
            "url": f"/api/audio/dreams/{p.name}",
            "duration": dur,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "timestamp": int(p.stat().st_mtime)
        })
    return {"dreams": results}

@app.post("/api/transform")
async def transform_audio(
    file: UploadFile = File(...),
    blend: float = Form(0.25),
    temperature: float = Form(1.0),
    model_path: Optional[str] = Form(None)
):
    transformer = get_or_load_transformer(model_path)
    
    # Save input temp file
    temp_dir = Path("data/temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = temp_dir / file.filename
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    t0 = time.time()
    audio, out_path = transformer.transform(
        audio_path=str(input_path),
        dream_blend=blend,
        temperature=temperature
    )
    elapsed = time.time() - t0

    filename = Path(out_path).name
    return {
        "status": "success",
        "filename": filename,
        "audio_url": f"/api/audio/dreams/{filename}",
        "elapsed_sec": round(elapsed, 2)
    }

CURRENT_COMPOSER = None

def get_or_load_composer():
    global CURRENT_COMPOSER
    if CURRENT_COMPOSER is None:
        from engine.musicgen_engine import MusicGenComposer
        CURRENT_COMPOSER = MusicGenComposer()
    return CURRENT_COMPOSER

@app.post("/api/compose")
def compose_music(
    prompt: str = Form("165 bpm atmospheric jungle, rolling amen breakbeats, lush ambient rhodes chords, deep sub bass, LTJ Bukem style"),
    duration: float = Form(8.0),
    temperature: float = Form(1.0)
):
    composer = get_or_load_composer()
    out_path, elapsed = composer.compose(
        prompt=prompt,
        duration_sec=duration,
        temperature=temperature
    )
    filename = Path(out_path).name
    return {
        "status": "success",
        "filename": filename,
        "audio_url": f"/api/audio/dreams/{filename}",
        "duration": duration,
        "elapsed_sec": round(elapsed, 2)
    }

# ----------------- Training Endpoints -----------------

def log_reader(pipe):
    try:
        with pipe:
            for line in iter(pipe.readline, ''):
                with TRAINING_LOCK:
                    TRAINING_LOGS.append(line.strip())
    except:
        pass

@app.post("/api/training/start")
def start_training(
    run_name: str = Form("aphex_bukem_run"),
    config: str = Form("v2"),
    batch_size: int = Form(4),
    max_steps: int = Form(50000)
):
    global TRAINING_PROCESS
    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        return {"status": "error", "message": "Training is already running!"}

    if not os.path.exists("data/processed/rave_db/data.mdb"):
        raise HTTPException(status_code=400, detail="Preprocessed dataset not found! Run preprocess first.")

    run_name = run_name.strip().replace(" ", "_")
    cmd = [
        "rave", "train",
        "--config", config,
        "--db_path", "data/processed/rave_db",
        "--name", run_name,
        "--channels", "1",
        "--batch", str(batch_size),
        "--max_steps", str(max_steps),
        "--val_every", "2000"
    ]

    with TRAINING_LOCK:
        TRAINING_LOGS.clear()
        TRAINING_LOGS.append(f"Starting training run '{run_name}' with command: {' '.join(cmd)}")

    TRAINING_PROCESS = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t = threading.Thread(target=log_reader, args=(TRAINING_PROCESS.stdout,), daemon=True)
    t.start()

    return {"status": "started", "pid": TRAINING_PROCESS.pid, "run_name": run_name}

@app.post("/api/training/stop")
def stop_training():
    global TRAINING_PROCESS
    if TRAINING_PROCESS is None or TRAINING_PROCESS.poll() is not None:
        return {"status": "idle", "message": "No training is running."}

    TRAINING_PROCESS.terminate()
    time.sleep(1)
    if TRAINING_PROCESS.poll() is None:
        TRAINING_PROCESS.kill()

    with TRAINING_LOCK:
        TRAINING_LOGS.append("🛑 Training process stopped by user.")

    return {"status": "stopped", "message": "Training process terminated."}

@app.get("/api/training/logs")
def get_training_logs():
    with TRAINING_LOCK:
        logs = list(TRAINING_LOGS)
    
    is_running = TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None
    return {
        "is_running": is_running,
        "logs": logs[-100:] # Last 100 lines
    }

@app.post("/api/models/export")
def export_model(run_name: str = Form("aphex_bukem_run")):
    run_dir = Path("runs") / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory '{run_dir}' not found.")

    trainer = Trainer()
    exported_file = trainer.export_model(str(run_dir), out_name=f"{run_name}.ts")
    return {"status": "success", "exported_path": exported_file}

# Serve web frontend
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = Path("web/templates/index.html")
    if not index_path.exists():
        return HTMLResponse("<h1>Loading Braindance Dream Studio...</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=7860, reload=False)

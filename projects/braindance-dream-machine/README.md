# 🌌 BRAINDANCE DREAM MACHINE
### *Neural Audio Hallucination & Timbre Synthesis (Aphex Twin × LTJ Bukem)*

Inspired by the concept of feeding a language model thousands of scripts to let it hallucinate its own bizarre stories, **Braindance Dream Machine** feeds a deep neural acoustic model raw discographies of **Aphex Twin** (IDM, drill 'n' bass, micro-edited breakbeats) and **LTJ Bukem** (atmospheric jungle, lush ambient pads, rolling Amen breaks) to **hallucinate completely new music from scratch**.

---

## ⚡ Quickstart

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Generate Starter Pack (Demo Audio)
If you don't have your MP3 files on hand yet, generate the synthetic Bukem atmospheric pads, Aphex breakbeats, and 303 acid sequence:
```bash
python studio.py demo
```

### 3. Launch the Interactive Web Studio
Open the local tactile web interface with interactive audio waveform players:
```bash
python studio.py ui
# Or directly:
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 🎧 CLI Commands

### 💭 Dream Mode (Unconditional Hallucination)
Sample random continuous trajectories through the latent space and decode them into raw `.wav` audio:
```bash
# Hallucinate an 8-second audio track
python studio.py dream --duration 8.0

# Push the chaos / temperature higher for wilder textures
python studio.py dream --duration 12.0 --temp 1.4 --glitch 0.25

# Listen immediately on macOS
afplay dreams/dream_*.wav
```

**Controls**:
* `--duration`: Length in seconds.
* `--temp`: Latent energy / variance (0.5 = calm/ambient, 1.0 = normal, 1.8 = high chaos).
* `--momentum`: Trajectory memory (0.92 = smooth liquid evolution, 0.5 = jittery).
* `--glitch`: Probability of triggering Aphex-style micro-stutters and temporal repeats.

---

### 🎛️ Timbre Transformer (Style Morphing)
Feed any external drum loop, vocal, or synth riff through the model. It preserves the rhythmic/harmonic skeleton while resynthesizing the timbre in the neural model's voice:
```bash
# Re-synthesize an audio file
python studio.py morph --input data/demo/aphex_braindance_breakbeat.wav --blend 0.35
```
* `--blend`: `0.0` = pure timbre resynthesis; `1.0` = completely dissolved into an unconditional dream.

---

## 🏋️ Training on Your Own Music (Aphex Twin & LTJ Bukem)

To train the AI specifically on your own catalog:

### 1. Drop Audio Files
Drop any `.mp3`, `.wav`, `.flac`, or `.aif` files into the `data/raw/` folder:
```bash
cp /path/to/aphex_twin_tracks/*.mp3 data/raw/
cp /path/to/ltj_bukem_tracks/*.mp3 data/raw/
```

### 2. Check Dataset Stats
```bash
python studio.py stats
```
This will report total tracks, duration (in seconds and hours), sample rates, and channel layouts.

### 3. Preprocess Dataset
Builds the high-speed LMDB audio database formatted for RAVE:
```bash
python studio.py prep
```

### 4. Train with Metal (MPS) GPU Acceleration
Run training optimized for your Apple Silicon M2 Pro:
```bash
python studio.py train --name aphex_bukem_run --batch 4 --steps 50000
```

### 5. Export Model for Live Dreaming
Once training has run to your desired fidelity, export the checkpoint to a standalone TorchScript model:
```bash
rave export --run runs/aphex_bukem_run --output models/exported/aphex_bukem.ts
```
The model will automatically show up in the Web Studio dropdown and CLI!

---

## 🔬 How the Architecture Works

```
                     ┌────────────────────────────────────────┐
                     │          Aphex Twin & LTJ Bukem        │
                     │          Raw Audio Tracks (WAV/MP3)    │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼ [Preprocess & Chunk]
                     ┌────────────────────────────────────────┐
                     │          LMDB Audio Database           │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼ [RAVE Multiband GAN VAE]
       ┌─────────────────────────────────┴─────────────────────────────────┐
       │                                                                   │
       ▼ [Phase 1: Autoencoder]                                            ▼ [Phase 2: Dream Engine]
┌──────────────────────────────┐                           ┌──────────────────────────────┐
│  Audio (48kHz / 44.1kHz)     │                           │  Unconditional Latent Walk   │
│              │               │                           │  (Ornstein-Uhlenbeck Drift   │
│              ▼               │                           │   + Micro-Glitch Injection)  │
│    Encoder (Cached Conv)     │                           └──────────────┬───────────────┘
│              │               │                                          │
│              ▼               │                                          │
│ Latent Space z (8-16 dim)    │◄─────────────────────────────────────────┘
│              │               │
│              ▼               │
│  Decoder (Multiband PQMF)    │
│              │               │
│              ▼               │
│  High-Fidelity Raw Waveform  │
└──────────────────────────────┘
```

1. **RAVE (Realtime Audio Variational autoEncoder)**:
   Compresses continuous audio into a low-dimensional latent manifold using Pseudo Quadrature Mirror Filters (PQMF) and dilated convolutional stacks, reconstructing audio with multi-period and multiband spectral discriminators.
2. **Latent Space Hallucination**:
   Because raw audio has 44,100 samples every second, sampling raw waveforms directly creates static. Sampling trajectories in RAVE's compact latent space and decoding them ensures every frame is acoustically coherent and rich with timbre.
3. **Apple Silicon Metal (MPS)**:
   Fully leverages the 16-core GPU of your M2 Pro with Unified Memory for fast training and real-time generation.

---

## 📁 Project Structure

```
radiant-heisenberg/
├── app.py                  # Gradio Web Studio application
├── studio.py               # Master CLI interface
├── engine/                 # Core audio and ML engine
│   ├── dreamer.py          # Latent trajectory hallucination & decoding
│   ├── transformer.py      # Timbre transfer & style morphing
│   ├── preprocessor.py     # Audio scanner, metrics & RAVE DB builder
│   ├── synthesizer.py      # Demo audio generator (pads, breaks, acid)
│   └── trainer.py          # Training orchestrator (MPS acceleration)
├── data/
│   ├── raw/                # Drop your Aphex Twin & LTJ Bukem music here
│   ├── processed/          # Cached LMDB database for RAVE
│   └── demo/               # Synthetic starter pack
├── models/
│   ├── exported/           # Exported .ts models ready for dreaming
│   └── checkpoints/        # Training checkpoints
├── dreams/                 # Output folder for generated .wav files
└── .venv/                  # Python 3.10 virtual environment
```

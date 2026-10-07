# 🎙️ LectureCanvas: Classroom Audio Transcriber & Artistic Word Visualizer

**LectureCanvas** records or imports lecture audio, transcribes it, and maps the
words a lecturer actually repeats into an interactive word diagram — so the shape
of a lecture's vocabulary is visible while it is still being given.

---

## 🚀 Quick Start

```bash
python3 -m pip install -r requirements.txt   # Flask + Whisper + PyTorch
brew install ffmpeg                          # Whisper decodes audio through ffmpeg
./run.sh
```

`run.sh` checks your dependencies, starts the Whisper server on
`http://127.0.0.1:5175`, waits for it to answer, then opens the web app on
`http://localhost:5174`.

The web app runs without Python — live microphone transcription uses the browser's
own Web Speech API. The server is only needed to import recorded audio files.

---

## ✨ Features

### 🎧 Live transcription
- Continuous Web Speech API recognition with an auto-restart watchdog, so a long
  lecture keeps transcribing through pauses.
- **Mic boost (1.0×–5.0×)** via a Web Audio `GainNode` and `DynamicsCompressorNode`,
  for a lecturer who is quiet or far from the laptop. Adjustable mid-recording.

### 📁 Importing recorded lectures
- Drag and drop audio (`.mp3`, `.wav`, `.m4a`, `.webm`, `.flac`, `.aac`, `.mp4`)
  or an existing transcript (`.txt`, `.vtt`, `.srt`).
- Subtitle files are parsed as subtitles: cue timings and sequence numbers are
  stripped, and each caption keeps its real position in the recording.
- Audio goes to the local Whisper server, which transcribes in windows and streams
  each batch of segments back over SSE. Progress and text appear as the work
  happens, not in one dump at the end — a two-hour lecture reports where it is
  throughout.
- Runs on Apple Silicon (MPS) or CUDA when available, CPU otherwise.

### ⏯️ Synchronized playback
Docked player with play/pause, scrubbing, playback speed (1×–2×), and mute.

### 🎨 Three visualizers
1. **Kinetic Constellation** — a d3 force-directed graph. Words repel and collide,
   recently spoken words pulse, and any word can be dragged to a new position.
2. **Editorial Mosaic** — typographic poster packing via `d3-cloud`. Each word keeps
   a stable orientation between layouts, so the poster does not reshuffle as the
   lecture goes on.
3. **Celestial Galaxy** — a golden-ratio spiral with core themes at the centre and
   emerging concepts on the outer arms.

Hover any word for its count and share of the lecture; click to exclude it (with
one-click undo); drag it in the Kinetic view.

### 🧠 Lexical processing
- English stopword filtering, plus spoken classroom fillers (*"um"*, *"basically"*,
  *"you know"*).
- Conservative plural consolidation: *algorithms → algorithm*, *theories → theory*,
  *classes → class* — while leaving *series*, *physics*, *analysis* and *bias* alone.
- Acronyms keep their casing (*GPU*, *DNA*).
- Words are tallied once, as they are spoken, and keep the time they were actually
  said, which is what drives the recency glow.

### ⏱️ Metrics HUD
Session length, speaking pace (WPM), total spoken words, unique vocabulary, and the
dominant topic. Imported lectures derive their duration from the recording's own
timeline rather than from how long the import took.

### 📜 Live transcript drawer
Collapsible timestamped transcript with an interim preview of the current phrase,
keyword search, keyword highlighting, copy, and `.txt` export.

### 💾 Exports
PNG of the diagram, the recorded audio, and the timestamped transcript.

### 🧪 Demo lectures
Two pre-loaded samples (Astrophysics & Black Holes, Deep Learning & Neural Networks)
that stream in as if spoken live.

---

## ⌨️ Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Start / stop listening |
| `T` | Toggle the transcript drawer |
| `W` | Toggle the word frequency sidebar |
| `Esc` | Close the open dialog or menu |

Shortcuts are ignored while typing in a text field.

---

## 🎨 Aesthetic palettes

**Cyber Neon** · **Warm Editorial** · **Bioluminescent** · **Sunset Horizon** ·
**Minimal Titanium**

---

## ⚙️ Configuration

| Variable | Where | Default | Purpose |
| --- | --- | --- | --- |
| `VITE_WHISPER_SERVER_URL` | web app | `http://127.0.0.1:5175` | Point the UI at a server on another host |
| `PORT` | server | `5175` | Whisper server port |
| `UI_PORT` | `run.sh` | `5174` | Vite dev server port |
| `LECTURECANVAS_DEVICE` | server | auto | Force `cpu`, `mps`, or `cuda` |
| `LECTURECANVAS_WINDOW_SECONDS` | server | `120` | Transcription window size. Smaller streams more often but re-primes the decoder more |

Half precision is used only on CUDA. MPS and CPU run in fp32, where decoding is
reliable — MPS still does the compute.

---

## 🧰 Whisper models

| Model | Trade-off |
| --- | --- |
| `tiny.en` | Fastest |
| `base.en` | Balanced (default) |
| `small.en` | Most accurate |

Weights are cached in memory after first use, so switching models the first time
costs a download and a load.

---

## 📦 Offline processing

`offline_processor.py` transcribes a file and prints ranked keywords without
running the web app:

```bash
python3 offline_processor.py lecture.mp3
```

It writes `<file>.transcript.txt` and `<file>.keywords.json` alongside the input.

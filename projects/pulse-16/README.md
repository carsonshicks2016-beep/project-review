# ⚡ PULSE-16: Procedural Studio DAW & Synthesizer

**PULSE-16** is a desktop music workstation featuring **26 procedurally synthesized instruments**, a 16-step sequencer, an interactive 25-key chromatic piano keyboard with hardware arpeggiator, multi-pattern song arranger, master DSP rack, and multi-track stems exporter.

Every sound is synthesized mathematically in NumPy in real time with **zero external audio sample files**.

---

## ✨ Features & Sound Architecture

### 1. 🎻 Orchestral Section (7 Instruments)
- **Timpani (Kettle Drum)**: Physical circular membrane modal ratios ($1.0, 1.59, 2.14, 2.30$ Bessel harmonics) with felt mallet attack and hall resonance.
- **Full Orchestra Hit**: Symphonic power hit combining timpani thud, brass horn stabs, string clusters, and crash wash.
- **Cinematic Brass Stab**: Detuned French Horn and Trombone ensemble with a brass swell envelope and resonant filter sweep.
- **Pizzicato Strings (C4, Eb4, G4)**: Crisp fingertip pluck transients with progressive acoustic string harmonic damping.
- **Tubular Bells / Chimes**: Inharmonic cylindrical tube partials ($1.0, 2.76, 5.40, 8.93$) with shimmering decay.

### 2. 🥁 Drum Core & Latin Percussion (11 Instruments)
- **808 Sub-Kick**, **Crisp Snare**, **808/909 Handclap**, **Side-stick Rimshot**
- **Closed Hi-Hat**, **Open Hi-Hat**, **Latin Shaker**, **Crash Cymbal**
- **Resonant Conga**, **Mid/Low Tom**, **808 Cowbell**

### 3. 🎹 Melodic Bass & Lead Synths (8 Instruments)
- **FM Bass Synthesizer ($C2, E\flat2, F2, G2$)**: Modulating carrier waves with dynamic filter envelope decay.
- **Lead Synth ($C3, E\flat3, G3, B\flat3$)**: Bright analog saw/pulse waveforms with warm filter sweep.

### 4. 🎹 25-Key Chromatic Virtual Piano Keyboard & Arpeggiator
- Playable 2-octave chromatic keyboard ($C3$ to $C5$) with ivory and obsidian keys and real-time press animations.
- **Beat-Synced Arpeggiator**:
  - Modes: **UP**, **DOWN**, **RANDOM**.
  - **LATCH Mode**: Holds chords in memory to arpeggiate hands-free.
  - Automatically synchronizes with the sequencer playback clock.

### 5. 🎼 Song Mode Arranger
- Chain pattern banks into a complete song timeline: `[ A ] [ A ] [ B ] [ B ] [ C ] [ C ] [ D ] [ D ]`.
- Seamless automated pattern bank switching as the song progresses.

### 6. 📦 Multi-Track Stems Exporter
- Click **`[STEMS WAV]`** to render individual separated 44.1kHz 16-bit stereo tracks into `stems/`:
  - `drums_stem.wav`
  - `orchestra_stem.wav`
  - `synth_stem.wav`
  - `full_master_mix.wav`

---

## 🎛️ Studio Hardware Controls

- **Master FX Rack**:
  - **DRIVE** (0–100%): Analog tape saturation and tube overdrive.
  - **LP FILTER** (100–0%): Resonant DJ club low-pass sweep down to 500 Hz.
  - **DELAY** (0–60%): Tempo-synced stereo ping-pong echo.
- **Multi-Pattern Memory Banks**:
  - 4 independent pattern slots: **Bank A**, **Bank B**, **Bank C**, and **Bank D**.
- **3-State Velocity Accents**:
  - Click any step pad to cycle: `OFF` $\to$ `NORMAL (●)` $\to$ `ACCENT (◆)` $\to$ `OFF`.
- **Dual-Mode Visualizer**:
  - Toggle between **16-Band Real-Time FFT Spectrum Analyzer** (with floating peak needles) and **Phosphor Oscilloscope**.
- **Category Filter Tabs**:
  - `[ORCHESTRA (7)]`, `[DRUMS (11)]`, `[SYNTH & BASS (8)]`, `[ALL TRACKS (26)]`.

---

## ⌨️ Tactile Keyboard Shortcuts

| Category | Keys | Instruments / Actions |
| :--- | :--- | :--- |
| **Transport** | `Spacebar` (Play/Pause), `Esc` (Stop/Rewind) | — |
| **Drums** | `1` to `0`, `-` | Kick, Snare, Clap, Rimshot, Hats, Shaker, Crash, Conga, Tom, Cowbell |
| **Orchestra** | `Z`, `X`, `C`, `V`, `B`, `N`, `M` | Timpani, Orch Hit, Brass Stab, Pizz Strings, Tubular Bell |
| **Bass Synth** | `Q`, `W`, `E`, `R` | Bass C2, Eb2, F2, G2 |
| **Lead Synth** | `A`, `S`, `D`, `F` | Lead C3, Eb3, G3, Bb3 |
| **Virtual Piano** | `Shift + P` | Toggle Virtual Piano & Arpeggiator view |
| **Visualizer** | `Shift + V` | Toggle FFT Spectrum Analyzer / Oscilloscope |
| **Groove Roll** | `G` | Generate algorithmic procedural groove |
| **Screenshot** | `F12` | Save high-res PNG screenshot |

---

## 🚀 Running the Workstation

```bash
python3 main.py
```

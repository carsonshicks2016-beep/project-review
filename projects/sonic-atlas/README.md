# Sonic Atlas

A browser-based collection of interactive sound experiments. Choose an experiment, adjust its parameters, listen to the generated audio, and watch a visualization of the underlying sound behavior. Audio is synthesized in the browser after a user gesture; playback level is not calibrated to real-world sound pressure.

## Experiments

- **Jet flyby:** The original Mach Lab simulator, with retarded-time Doppler sampling, expanding wavefronts, shock-cone timing, and an illustrative N-wave pressure profile. It also includes trajectory and environment controls.
- **Doppler pass:** A car, train, or siren moves past a ground listener. Retarded source timing produces the pitch shift.
- **Echo room:** A source and listener sit in a configurable rectangular room. Image-source early reflections feed the impulse response, and a Sabine estimate sets the reverberant tail.
- **Underwater:** Whale, sonar, or ship signals travel through seawater. Frequency-dependent Thorp absorption and propagation delay shape what arrives at the hydrophone.
- **Beats & waves:** Two nearby tones combine at a movable listener; the spatial wave pattern shows reinforcement and cancellation.
- **Thunder distance:** A lightning flash precedes a filtered, rolling thunder signal by the sound-travel time over the selected distance.
- **Sound designer:** Build a tone with a selected oscillator, filter cutoff, and envelope; inspect its waveform and harmonic spectrum.

These are educational, idealized models rather than field recordings or calibrated predictions. Room reverberation, ocean propagation, thunder, and flyby synthesis each use simplified assumptions described in the in-app controls and physics notes.

## Local use

From this directory:

```sh
python3 -m http.server 4317 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:4317`. ES modules and the audio worker need HTTP; opening the HTML with a file URL is unsupported. There is no build or package installation. Fonts use Google Fonts with local fallbacks; the experiments have no external audio dependencies.

## Verification

```sh
node --check dist/main.mjs
node --check dist/experiments.mjs
node --check dist/experiment-physics.mjs
node --check dist/experiment-worker.mjs
node --test tests/*.test.mjs
```

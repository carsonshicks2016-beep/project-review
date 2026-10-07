# Ink Current Reactor

A local MP3-powered aquarium visualizer built with React, TypeScript, and the Web Audio API.

It renders a minimal ink-pen aquarium where:

- bass thickens the fish and pulls the sand bed
- mids bend the kelp and stretch the swim path
- treble drives bubbles and scratches motion into the waterline

## Run It

```bash
npm install
npm run dev
```

Open the local Vite URL, load an `.mp3`, and press play.

## Build and Verify

```bash
npm run build
npm run lint
```

## Stack

- React 19
- TypeScript
- Vite
- Web Audio API
- Canvas 2D rendering

## Notes

- The audio file stays local in the browser.
- The app expects MP3 input for the cleanest browser support.
- The aquarium keeps a gentle idle animation even before a track is loaded.

# Command Center v1 — archived 2026-09-10

Full snapshot of the original dashboard taken immediately before the
condense/rebuild pass. Nothing here is referenced by the running server;
it exists so the old UI can be diffed against or restored wholesale.

Contents:
  static/            index.html, style.css, pitwall.css, app.js, pitwall.js, charts.js
  server.py          Flask backend as of the snapshot
  observatory_api.py
  SUPRA.command      launcher
  fable_track_cache.json

## Restore
    cd command-center
    rm -rf static && cp -Rp legacy/v1_20260910/static static
    cp -p legacy/v1_20260910/server.py server.py

## Why it was rebuilt
  - 12 top-level tabs with heavy functional overlap (4 separate training tabs,
    3 separate "run the sim" tabs, 2 separate analysis tabs)
  - Two competing stylesheets (style.css + pitwall.css) = two design systems
  - Chassis picker overflowed its column; porsche_919_evo unselectable on Train

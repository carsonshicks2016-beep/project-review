# Stage 10 — Map Zoom Layers + Living-Map UI Remake

**Status: complete** — 2026-06-11 (verified in-browser)

## Delivered
- `src/ui/components/WorldMap.tsx` replaces the hand-placed SVG town view:
  - **Canvas terrain layer**, 16-bit pixel style (per-tile shade jitter, pixel tree/peak/reed stamps, `image-rendering: pixelated`), repainted as the sim advances so path wear appears as trails → paths → roads
  - **Claims canvas**: household-colored territory fills with hard borders and red-stippled disputed ground; opacity fades as you zoom in
  - **Three zoom tiers via scrolling** (wheel anchored under the cursor, drag to pan, center-anchored ± buttons):
    - *Region* — biomes, claim territories, named landmarks ("the Wide Water", "Ember Lake", "Storm Tor", "Tangle Wood")
    - *Settlement* — buildings (ghost-outline **planned**, scaffold + progress bar **under construction**, full sprite **complete**), roads, agent dots, indoor headcount chips
    - *Street* — pixel people sized relative to the houses, color-coded speech bubbles, thought bubbles for pressing needs, lit windows at night when someone's home
- `PhraseCard.tsx`: click any bubble → the line, speaker → listener, motive chip (premeditated / genuine / deceitful / performative / venting / bonding / desperate), the numeric reasons from both agents' state, and a jump into the speaker's decision inspector
- `BuildingCard.tsx`: click any building → construction state + adapted material recipe progress, owner household, and **everyone inside and outside with what each is doing** (from live decision traces)
- Day-phase tint overlay; save/load buttons (localStorage) in the top bar
- Old `TownMap.tsx`, `PlaceCard.tsx`, `mapLayout.ts` deleted; shared visual constants live in `palette.ts`

## Verified in-browser (preview, 1440×900)
- All three tiers render and transition by scroll/buttons; landmark labels at region zoom; claims + disputed hatch visible; bubbles clickable → motive card (e.g. *"My mother swore Tangle Wood was full of voices at night…"* → BONDING, belonging 86/100); building card shows Bray home occupants + activities; save → step → load rewinds correctly; zero console errors. `npm run build` + 54/54 tests green.

## Notes
- Thought bubbles are derived presentation (needs + traces), never stored as events — by design.
- `vite.config.ts` now respects the preview harness's assigned port; `.claude/launch.json` uses `autoPort`.

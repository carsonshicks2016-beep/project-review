# Stage 0 — Spec Ledger and Project Scaffold

**Status: complete** (reinforced 2026-06-11; scaffold predates the protocol, planning artifacts added retroactively)

## Delivered
- React/Vite + TypeScript project with isolated sim core (`src/sim/` has no UI imports)
- Vitest test harness, headless runner (`npm run simulate`)
- Planning artifacts: [source-spec.md](../source-spec.md), [coverage-ledger.md](../coverage-ledger.md), [living-map-design.md](../living-map-design.md), this reports directory
- Git repository initialized (2026-06-11)

## Deviations
- No separate Node server yet; the app is browser-only. The server arrives with the Ollama gateway (stage 29). Noted in the ledger under persistence.
- The spec lives as `civilia-brainstorm.md`; `source-spec.md` is an index into it rather than a copy (avoids a 2,500-line duplicate drifting out of sync).

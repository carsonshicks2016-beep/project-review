# Stage 7 — Debug Inspector

**Status: complete** (audited 2026-06-11)

## Delivered
- Decision inspector inside the agent drawer: candidate actions, utility scores, seeded noise, reasons, selected action highlighted
- Event causality visible via timeline + ticker; rumors carried per agent inspectable
- Headless `npm run simulate` prints a 30-day town report for debugging without the UI

## Reinforcement notes
- "Why did Mara steal bread?" is answerable today from stored traces. The spec's deeper debug questions (belief updates, memory retrieval) land with stages 14–15.

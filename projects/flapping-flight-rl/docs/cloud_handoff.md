# Claude Cloud handoff

This repository snapshot is prepared for source-level work in Claude Code on
the web / cloud sessions. Start by reading
`research/hawkmoth_hover/docs/agent_takeover.md`, then the latest status in
`docs/stage_a_report.md`, `docs/validation_protocol.md`, and
`docs/feasibility_log.md`.

## Current boundary

Stage A is **FAILED / UNVALIDATED**. The next work is to diagnose and qualify
the moving-boundary method. Do not run the prescribed hawkmoth hover benchmark,
claim physical validation, or train PPO. The cloud session is useful for code
review, analysis of the preserved summaries, documentation, and proposing
small source changes. It does not have the local M2 Pro solver process or the
omitted local run archive.

## Included and omitted evidence

Included: tracked source, benchmark configuration, protocols, chronological
reports, agent takeover, generated result inventory, and the compact
`research/hawkmoth_hover/results/stage_a_summary.json` summary.

Omitted from this repository snapshot: the large per-run `results/` archive and
the binary user-supplied `DPIV_DATA.mat`. These remain on the owner's local
machine. The inventory and provenance notes describe their local paths and
limitations. Do not treat the summary/inventory as a replacement for those
files; request a specific artifact if an analysis needs it.

## Suggested first cloud task

Ask for a read-only, evidence-cited review of the latest Stage A failure,
checking code paths against the report and inventory, and a ranked list of
diagnostic experiments that remain within the agreed local limits. Require the
agent to distinguish directly observed evidence from inference and to identify
which conclusions cannot be checked without the omitted raw run artifacts.

Return proposed code changes as a branch/PR for local review. Run solver
experiments only from the local checkout with its recorded IBAMR build unless a
separately approved environment qualification is performed.

# Bellwether — Plan

> **Codename:** "Bellwether" (a bellwether leads the flock — a leading indicator). Placeholder; rename freely.
>
> **This document is authoritative.** When in doubt, follow PLAN.md. Update it when decisions change.
> **Build order lives in [BUILD_ROADMAP.md](BUILD_ROADMAP.md)** — the granular, step-by-step "how we build it." PLAN = what & why; ROADMAP = how & in what order.
> **Status:** Pre-build. No code yet. Last updated 2026-06-18.

---

## 1. One-sentence thesis

Build a system that **reads public regulatory and financial filings the instant they're published, for small companies no professional analyst bothers to cover, and flags material information before the market has priced it in.**

We are not predicting prices. We are exploiting an **attention gap**, not a speed gap or a smarts gap.

---

## 2. Why this works (the bet)

- **Big companies** (Apple, Exxon) have armies of analysts. Any news is priced in within seconds. No edge — we can't out-read Wall Street there.
- **Small, ignored companies** ($300M–$3B, ≤3 analysts) have *nobody* reading their filings. When one files "we signed a major contract," there's a real lag before the price reflects it.
- The document is **public** — the edge isn't secrecy, it's that reading paperwork for a tiny company was never worth a human's time.
- **What changed:** an LLM reads that paperwork instantly for pennies. The thing that was never economical to do by hand is now cheap to do at scale — and it's most valuable exactly where attention is lowest.

This is the "neglected firm effect," academically documented. It persists because it was never industrializable. Now it is.

---

## 3. Scope & non-goals

### In scope (v1)
- A **personal tool** for the local users. Private use only.
- A defined **universe** ("the pond") of ~150–300 neglected-but-investable companies.
- **Near-real-time ingestion** of SEC filings for that universe.
- **LLM extraction** that scores each new document for materiality and direction.
- **Alerts** + a simple dashboard.
- An **honest backtest** proving (or disproving) the edge before any real money.

### Explicit non-goals
- ❌ **No price-prediction model / no PPO on OHLCV.** That's the crowded, arbitraged game. (RL has a legitimate but *narrow* slot later — see §11.)
- ❌ **No auto-trading in v1.** Human decides. Flag, don't trade.
- ❌ **No high-frequency / latency arms race.** We compete on attention, not milliseconds.
- ❌ **No early-stage biotech.** Binary FDA coin-flips you can't handicap. Excluded.
- ❌ **No OTC / pink sheets / sub-$5 penny stocks.** Liquidity and fraud risk.
- ❌ **No recommendations to third parties** until the regulatory fork (§4) is decided deliberately.

---

## 4. The regulatory fork — DECIDE BEFORE CODING

| | **Personal tool** | **Product for others** |
|---|---|---|
| Who sees signals | Just us | Strangers / subscribers |
| Regulatory status | Unregulated | Likely an **RIA** (Registered Investment Adviser) — registration, fiduciary duty, compliance |
| Build implication | Optimize for usefulness | Entire product/legal shape changes |

**v1 decision: PERSONAL TOOL.** Full stop. Do not let it drift across the line by accident (e.g. "let me just text my buddy the alerts"). Dad knows this terrain — he makes the call if/when we ever consider going public.

---

## 5. Architecture (the whole system at a glance)

```
┌─────────────────┐
│ UNIVERSE SCREEN │   Define the pond: ~150-300 neglected-but-investable names
│  (§7, Hudson)   │   → companies table (cik, ticker, sector, mktcap, analyst_count…)
└────────┬────────┘
         │ watchlist (CIKs)
         ▼
┌─────────────────┐
│  FILING WATCHER │   Poll SEC EDGAR near-real-time, filter to our CIKs
│   (§8, the developer)  │   → filings table, stamped with ACCEPTANCE datetime
└────────┬────────┘
         │ new filing
         ▼
┌─────────────────┐
│ TRIAGE (cheap)  │   Haiku: is this even potentially material? (cuts 90% of noise)
└────────┬────────┘
         │ maybe-material
         ▼
┌─────────────────┐
│ EXTRACT (deep)  │   Sonnet/Opus: read it. Return structured JSON:
│   (§9, the developer)  │   catalyst_type, materiality 0-10, direction, confidence, summary
└────────┬────────┘
         │ signal
         ▼
┌─────────────────┐
│  SIGNAL STORE   │   signals table (point-in-time, immutable)
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌──────────────┐
│ ALERTS │ │   BACKTEST   │   Replay history at acceptance timestamps,
│ + dash │ │  (§10, both) │   measure abnormal returns. THE credibility test.
└────────┘ └──────────────┘
```

---

## 6. Tech stack

Keep it boring and cheap. This is a low-volume, read-heavy pipeline — no GPU, no cluster.

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3** (`python3`/`pip3`) | the developer's default; best filing/finance libs |
| Storage | **SQLite** v1 → Postgres later | Point-in-time, zero-ops to start |
| Filings | **SEC EDGAR** APIs (`data.sec.gov`, `efts.sec.gov`) | Free, official, the crown jewel |
| Universe screen (cap, price, ADV, **analyst count**) | **yfinance** (Yahoo `numberOfAnalystOpinions` etc.) | ⚠️ FMP free tier **paywalls analyst coverage for micro-caps** (402) — exactly our universe — so yfinance is the analyst-count source, not FMP |
| Cross-check / live fundamentals | **Financial Modeling Prep** free tier (`stable/profile` works for all symbols) | Secondary; key already set up |
| **Backtest** price + constituent history | **Sharadar** (Nasdaq Data Link, ~$30–50/mo) or **Tiingo** — point-in-time, *includes delisted names* | yfinance is NOT survivorship-safe; see §10 |
| LLM extraction | **Provider-agnostic client** (one-line model swap). **Triage → Gemini Flash (free tier)**; **deep read → eval-decided** (Gemini Flash vs `claude-sonnet-4-6`, picked by the Step 3.10 eval) | Cost is ~$0 at this volume either way, so choose on signal quality, not price. Don't lock to one provider |
| Scheduling | cron / APScheduler / simple loop | Low frequency (neglected names file rarely) |
| Alerts | Email/Telegram to start | Don't over-build the UI |
| Dashboard | CLI → Streamlit later | Defer until signals are proven |
| Backtest | pandas | Standard |

**Cost reality:** neglected names file infrequently; you're making a small number of cheap API calls per day. Monthly LLM + data cost is dollars, not thousands. This is the cheap architecture *and* the better one — opposite of the GPU-hungry PPO path.

---

## 7. The universe / "the pond" (Hudson's lane)

The screen doesn't find winners — **it defines the pond.** The edge is the agent reading inside it. Three layers (from Dad's framework):

**Layer 1 — Investability floor (so we can actually trade it)**
- Exchange-listed (NYSE/Nasdaq), no OTC/pink sheets
- Price > $5
- Market cap ~$300M–$3B
- Avg daily dollar volume > ~$1–2M

**Layer 2 — The neglect signal (the actual thesis)**
- Sell-side analyst count ≤ 3 (sweet spot 2–3; 0 is interesting but riskier)
- Institutional ownership ~20–60% (some validation, not crowded)
- **Bonus sub-universe:** recently lost coverage, spun off, emerged from bankruptcy, or de-SPAC'd → structurally orphaned, richest hunting ground

**Layer 3 — Not-a-trap gate (neglected is often neglected for a reason)**
- Positive/improving revenue or cash flow, or clear path to it
- Not in financial distress (manageable debt)
- Plus signal: insider buying / meaningful insider ownership

**How to set thresholds:** don't theorize — **pull the list.** Set Layers 1–2 loosely in Finviz/FMP, run it, look at what comes back. 800 names → tighten. 12 names → loosen. Calibrate against real output until you have ~150–300 genuinely interesting companies, each with a CIK (EDGAR's company ID). That CIK list is the handoff to the watcher.

**Sector note:** don't hard-limit to energy, but Dad's edge is sharpest there (FERC/ERCOT/PUCT filings are leading indicators he can read). Energy is the highest-confidence slice; start weighted toward it, expand as the backtest earns trust.

**Energy × orphaned are orthogonal axes — aim at the intersection.** Sector tilt (energy = Dad's edge) and situation tilt (structurally orphaned = where mispricing is violent) are independent. The bullseye is a name that's *both* — a post-bankruptcy E&P, a spun-off midstream unit, a de-SPAC'd energy-tech. **Make the first tranche (~first 20 names) heavily that overlap** — prove Dad's highest-conviction edge first, then broaden. Don't treat "energy" and "orphaned" as competing quotas.

**Sharpest neglect trigger — *recently lost* coverage.** A name whose analyst count just dropped (e.g. 4→2 in 60 days) is often a richer signal than one that's been quietly uncovered for years: coverage vanishes and an attention vacuum opens right as the story is changing. Treat a recent coverage *drop* as a priority flag. (Caveat: it cuts both ways — analysts also flee before bad news — so it's a flag to *watch*, not a buy. Requires time-series analyst data, which the FMP free tier may not provide; if not, this is a manual/Hudson signal until we have a better source.)

---

## 8. Filing watcher (the developer's lane)

**Goal:** hear the market in near-real-time for our CIKs, and record *exactly when* each filing became public.

- **Source:** SEC EDGAR.
  - Per-company submissions: `https://data.sec.gov/submissions/CIK##########.json`
  - Real-time "latest filings" feed: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent`
  - Full-text search: `https://efts.sec.gov/LATEST/search-index?q=...`
  - Set a proper `User-Agent` header (EDGAR requires it) and respect rate limits (~10 req/s).
- **Filings that ARE the thesis:**
  - **8-K** — material events (contracts, impairments, exec changes) ← highest value
  - **Form 4** — insider buys (cluster of these = strong signal)
  - **SC 13D/13D-A** — activist stakes
  - **10-Q / 10-K** — guidance, going-concern language
  - Energy add-ons: **FERC eLibrary, ERCOT, PUCT** dockets (scrape; v1.5)
- **CRITICAL — store the `acceptance-datetime`.** Every filing carries the exact moment it went public. This single field is what makes an honest backtest possible (no look-ahead). Store it immutably.
- Watcher does **no AI** — it's pure plumbing. Detect → download → record → enqueue.

---

## 9. Extraction (the developer's lane)

Built behind a **provider-agnostic LLM client** — the model is a config switch, never hard-wired. Cost at our volume (~a few filings/day live; one ~440-filing backfill) is a few dollars *or* free either way, so the provider is chosen on **signal quality, not price**. Two-stage to keep cost and noise down:

1. **Triage → Gemini Flash (free tier):** "Could this filing plausibly be material? yes/no + one-line why." High-volume, low-stakes, fits easily inside the free daily caps. Kills ~90% of routine filings.
2. **Deep read → eval-decided:** For survivors, return strict JSON. The model is chosen empirically by the Step 3.10 eval (Gemini Flash vs `claude-sonnet-4-6`), because the entire Phase 4 verdict rests on this stage's faithfulness:

```json
{
  "catalyst_type": "new_contract | insider_buy_cluster | guidance_change | litigation | activist_13d | mgmt_change | regulatory_decision | impairment | other",
  "direction": "bullish | bearish | neutral",
  "materiality": 0,            // 0-10, how market-moving
  "confidence": 0.0,           // 0-1, model's own certainty
  "summary": "≤2 sentences, plain English",
  "evidence_quote": "the exact sentence from the filing that drives this"
}
```

- **Force the `evidence_quote`** — it grounds the score in real text and makes false positives auditable.
- Store every signal **immutably** with the model ID + timestamp. Never overwrite; this is the backtest's ground truth. (Storing the model ID matters more now that the provider can vary.)
- **Free-tier data caveat:** Gemini's free tier may use prompts/responses to improve Google's products. The filings are public (fine), but our *prompt design* (triage criteria, scoring rubric, catalyst taxonomy) is arguably the edge — acceptable for personal v1, revisit if we ever productize.

---

## 10. The backtest (both — the credibility of the whole project)

If the backtest is clean and *still* shows an edge, you have something real. If not, you found out for ~$0 instead of with Dad's money. **This milestone matters more than the dashboard.**

Three traps to defeat:
1. **Look-ahead bias** — only ever use info available at the filing's `acceptance-datetime`. The whole point of storing that field.
2. **Survivorship bias** — use a *point-in-time* universe (companies as they existed then, including ones later delisted), not today's survivors. ⚠️ **This rules out yfinance for backtest prices** — it purges delisted micro-caps, and the names that vanish are disproportionately the post-catastrophe ones (bankruptcy, going-concern), which would make bearish catalysts look artificially predictive. Use a PIT source (Sharadar/Tiingo).
3. **Slippage/liquidity** — thin coverage and thin liquidity travel together. A backtest ignoring slippage looks gorgeous and dies in reality. Model realistic fills (method below); respect the ADV floor from Layer 1.

**Method:** for each historical high-materiality signal, compute forward abnormal return (vs a sector ETF benchmark) at +1d / +5d / +20d. Question: did high-materiality bullish flags precede positive abnormal returns more than chance? Bucket by `catalyst_type` and `materiality` to see *which* signals carry the edge.

**Slippage model (v1, designed for OHLC-only data — no tick/quote feed):**
1. **Fill at the *next* bar after `acceptance_datetime`, never the signal-bar close.** If an 8-K drops at 4:05pm you fill at next open — often *after* the gap. This single rule kills most of the look-ahead fantasy.
2. **Cap assumed size ≤1% of 20-day ADV** — bounds market impact and keeps capacity claims honest.
3. **Half-spread cost via a high-low spread estimator** (Corwin-Schultz or Abdi-Ranaldo) — recovers effective bid-ask from daily OHLC alone, no tick data needed.
4. **Square-root market-impact term**: `impact ∝ σ·√(size/ADV)`.
5. **Stress-test at 2× and 3× modeled slippage** (and **5× for the thinnest names, ADV < $2M**). Edge must survive the stress to be called real; if it dies at 1.5×, it never existed.
6. **Non-linearity:** the √-impact model comes from large/mid-cap studies and *understates* impact for thin micro-caps (a 0.5%-ADV order can move price 5%+). Apply a harsher multiplier for ADV < $5M, and log the gap between assumed fill and a cheap VWAP proxy `(O+H+L+C)/4` as a sanity check.

**The intellectual-honesty guardrail (read before running M3):** the biggest risk in this whole project is *not* technical — it's the temptation, when the backtest comes back flat, to quietly relax a slippage assumption, loosen the look-ahead rule, or re-cut the universe until an edge "appears." **Do not yield.** Decide the slippage model, the PIT rules, and the universe *before* you see the result, and don't touch them after. The system's job isn't to make money — it's to tell you *honestly* whether it can, then be believed. A clean "no edge" that saves Dad's capital is a win, not a failure.

---

## 11. Where RL fits later (the developer's interest, correctly placed)

RL does **not** belong in signal generation (that's forecasting, and PPO-on-price competes with Renaissance — a losing game). But once signals exist, **position sizing / portfolio allocation** is a genuine sequential decision problem: given N flagged catalysts with conviction scores and liquidity limits, how much to allocate and when to exit. That's a real MDP. **Parked as M5+, after signals are proven.** Don't build it until the backtest validates the signals it would size.

---

## 12. Milestones

| # | Name | Deliverable | Done when… |
|---|---|---|---|
| **M0** | Pond v0 | Hardcoded ~50 names passing Layers 1–2 (manual Finviz pull), with CIKs, in `companies` table | A CSV/DB of 50 tickers + CIKs exists; energy-weighted |
| **M1** | Watcher | EDGAR poller logs new filings for those CIKs with acceptance timestamps; no AI | A real filing appears in `filings` within minutes of EDGAR publishing it |
| **M2** | Extractor | Triage + deep-read pipeline emits structured signals for each new filing | New filing → JSON signal in `signals`, manually spot-checked as sensible |
| **M3** | Honest backtest | Replay historical filings at acceptance timestamps; abnormal-return analysis with slippage | A clean report answering "is there an edge, and in which catalyst types?" |
| **M4** | Alerts + dashboard | High-materiality signals push to email/Telegram; simple Streamlit board | Dad gets a real alert and can see the reasoning + evidence quote |
| **M5** | (Optional) RL sizing | RL allocator over proven signals | Only after M3 validates the signals |

Each milestone is shippable and resume-legible on its own. M1 alone ("real-time SEC ingestion pipeline") is already a credible portfolio piece.

---

## 13. Division of labor

- **the developer (infrastructure):** M1 watcher, M2 extraction pipeline, storage/schema, M4 alerts/dashboard, M5 RL. Owns the code.
- **Hudson (data/universe):** M0 pond definition, threshold calibration, maintaining the watchlist, sourcing analyst-coverage / institutional-ownership data, energy-docket sources. Owns *what we look at*.
- **Dad (domain + regulatory):** judges signal vs. noise (esp. energy), owns the RIA fork decision, sanity-checks the backtest.

---

## 14. Data schema (v1, SQLite)

```
companies(cik PK, ticker, name, sector, mktcap, adv_usd,
          analyst_count, inst_own_pct, in_universe, added_at, notes)

filings(id PK, cik FK, form_type, accession_no UNIQUE,
        filed_date, acceptance_datetime,   -- ← the point-in-time anchor
        url, raw_path, fetched_at)

signals(id PK, filing_id FK, catalyst_type, direction,
        materiality, confidence, summary, evidence_quote,
        model, created_at)                 -- immutable, never overwrite

prices(ticker, date, open, high, low, close, volume)   -- for backtest

outcomes(signal_id FK, fwd_ret_1d, fwd_ret_5d, fwd_ret_20d,
         abnormal_ret_5d, benchmark, computed_at)
```

---

## 15. Open questions to resolve before / during M0

1. ~~**Analyst-coverage data source**~~ **RESOLVED (2026-06-18):** FMP free tier paywalls analyst coverage for micro-caps (402 on `grades-consensus`), so the source is **yfinance** (Yahoo `numberOfAnalystOpinions`), which covers them free. Step 1.0 still gates: validate yfinance counts against ~10 human-known values before loading; >30% discrepancy on low-coverage names → build the universe manually.
2. **Universe refresh cadence** — how often do we re-run the screen and add/drop names? (Monthly to start?)
3. **What counts as an "alert-worthy" materiality threshold** — set after we see M2 output, not before.
4. ~~**Point-in-time data for backtest**~~ **RESOLVED:** backtest prices/constituents come from a PIT source (Sharadar ~$30–50/mo or Tiingo) incl. delisted names; yfinance is live-only. PIT *fundamentals* (vs prices) deferred — filing-text + PIT price is enough for v1.
5. **Energy docket scraping** — defer to v1.5, or is it valuable enough to pull into M1?

---

## 16. The honest summary

We're building a tireless intern that reads tedious public paperwork on companies nobody watches, and taps Dad on the shoulder when something real shows up — in the one corner of the market where reading faster than everyone else is actually an edge. The make-or-break isn't the code; it's whether the backtest (§10) survives contact with reality. Build toward that test first.

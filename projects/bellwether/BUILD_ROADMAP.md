# Bellwether — Build Roadmap

> Tactical companion to **PLAN.md** (which holds the strategy & decisions). This file is the **step-by-step build order**: many small, verifiable steps that compound into the full system. PLAN.md = *what & why*. This = *how & in what order*.
>
> **Status:** Pre-build. Last updated 2026-06-18.

---

## How to use this document

**The philosophy you asked for:** lots of tiny steps, each one shippable and verifiable, that stack into something massive — not a few big leaps that land you at a hollow demo. Every step below follows the same contract:

- **Atomic** — one concept, ideally < a few hours. If a step feels big, split it.
- **Runnable** — at the end of every step *something works and you can see it work*. No step leaves the repo broken.
- **Verifiable** — each has a **✓ Done when** line: a concrete check, not a vibe.
- **Committed** — one git commit per step (or smaller). The history becomes the story.
- **Vertical when possible** — prefer a thin slice through the whole stack (1 company, 1 filing, end-to-end) over building one layer fully before the next. You want the pipe connected early, then widen it.

**The North Star (read this whenever a step feels pointless):** see *Desired End Product* below. Every step is a brick in that building.

**Working rhythm:** do a step → run it → see it work → commit → check it off here. Don't batch. The momentum comes from a long chain of small green checks.

---

## Desired End Product (the "massive" version we're walking toward)

A self-running system that, every day with zero human babysitting:

1. **Maintains a living universe** of ~150–300 neglected-but-investable companies, auto-refreshed as names enter/leave the criteria.
2. **Listens to SEC EDGAR in near-real-time**, catching every new filing for those companies within minutes of it going public — and recording the exact public-availability timestamp.
3. **Reads each filing with an LLM**, classifying it (contract win, insider-buying cluster, guidance change, activist stake, litigation, impairment…), scoring materiality 0–10 and direction, and quoting the exact sentence that drives the call.
4. **Pushes high-conviction signals** to Dad/the developer — a clean alert with the plain-English summary, the score, the evidence quote, and a link — within minutes of the filing.
5. **Shows everything on a dashboard**: the universe, the live signal feed, per-company history, and the running track record.
6. **Knows whether it actually works**, because a rigorous point-in-time backtest (defeating look-ahead, survivorship, and slippage) continuously measures whether its high-materiality flags precede real abnormal returns — broken down by catalyst type.
7. **(Stretch)** sizes a paper portfolio from validated signals via an RL allocator, logging hypothetical performance.

The end product is **a tireless analyst covering the companies Wall Street ignores, that can prove its own edge.** Everything below builds to exactly that, one brick at a time.

---

## Phase map (the whole journey)

| Phase | Theme | Maps to PLAN milestone | Gate to pass |
|---|---|---|---|
| **0** | Foundations & scaffolding | (pre-M0) | Repo runs, DB exists, tests pass |
| **1** | The Pond (universe) | M0 | ~50 real names + CIKs in DB |
| **2** | The Watcher (ingestion) | M1 | Real filing captured within minutes |
| **3** | The Extractor (LLM signals) | M2 | New filing → sensible JSON signal |
| **4** | The Backtest (truth) | M3 | A clean yes/no on the edge |
| **5** | Alerts & Dashboard | M4 | Dad gets a real alert + can browse |
| **6** | Hardening & scale | (M4.5) | Runs unattended for a week |
| **7** | RL sizing (optional) | M5 | Only unlocked if Phase 4 says "edge real" |

Phases are sequential, but inside each phase steps are small. **Do not skip Phase 0** — boring scaffolding is what lets the next 100 steps go fast.

---

## Phase 0 — Foundations & scaffolding

**Goal:** a clean, runnable skeleton so every later step has a home. **Why first:** cheap now, brutal to retrofit later.

- **0.1 — Repo & git.** `git init`, add `.gitignore` (Python, `.env`, `data/`, `*.db`), commit a one-line README.
  ✓ Done when: `git log` shows the first commit.
- **0.2 — Python env.** Create a venv, `requirements.txt` (start empty), document `python3 -m venv` setup in README.
  ✓ Done when: `source .venv/bin/activate` works and is documented.
- **0.3 — Folder structure.** `src/bellwether/`, `data/`, `tests/`, `scripts/`, `notebooks/`. Add empty `__init__.py`.
  ✓ Done when: `python3 -c "import bellwether"` succeeds.
- **0.4 — Config & secrets.** `config.py` reading from `.env` (use `python-dotenv`). Keys: `ANTHROPIC_API_KEY`, `FMP_API_KEY`, `SEC_USER_AGENT`. Provide `.env.example`.
  ✓ Done when: a tiny script prints a loaded config value (no secrets committed).
- **0.5 — Logging.** One `logging` setup helper used everywhere (level from config, timestamped).
  ✓ Done when: `log.info("hello")` prints a formatted line.
- **0.6 — Database bootstrap.** SQLite file + a `schema.sql` creating the tables from PLAN.md §14 (`companies`, `filings`, `signals`, `prices`, `outcomes`). A `db.py` with connect + init.
  ✓ Done when: running `scripts/init_db.py` creates `bellwether.db` with all 5 tables (verify with `.tables`).
- **0.7 — Test harness.** `pytest` installed; one trivial passing test; a `make test` (or `scripts/test.sh`).
  ✓ Done when: `pytest` is green.
- **0.8 — Task runner.** `Makefile` (or `tasks.py`) with `init-db`, `test`, `run-watcher`, `lint` stubs.
  ✓ Done when: `make test` runs the suite.

**🚪 Phase 0 gate:** fresh clone → follow README → DB initializes → tests pass. The skeleton is alive.

---

## Phase 1 — The Pond (universe) · M0

**Goal:** ~50 real, vetted companies with CIKs in the `companies` table. **Why:** the watcher needs a target list; everything downstream is scoped to the pond. *(Hudson co-owns the criteria; the developer builds the loaders.)*

- **1.0 — Universe sanity check (GATE — do this before loading anything).** The whole pond rests on `analyst_count`, and the FMP free tier is unreliable for micro-caps. Pull ~10 names matching Layers 1–2; for each, compare FMP's analyst count to a manual check (Yahoo Finance "Analysis" tab, or a name Dad knows cold). Flag mismatches.
  ✓ Done when: <30% discrepancy for names with true count ≤3. **If higher → the universe is built manually (Hudson's lane) until FMP proves reliable.** Don't load a rotten pond — garbage in here poisons every downstream phase.
- **1.1 — Manual seed list.** Hudson pulls ~50 tickers from Finviz/Koyfin matching Layers 1–2 (energy-weighted). **Heavily weight the *structurally orphaned* sub-universe** — recent spin-offs, de-SPACs, post-bankruptcy emergences, lost-coverage names. That's where mispricing is most violent and the attention gap widest. Save raw as `data/seed_universe.csv` (ticker + why-interesting note + orphan-type tag).
  ✓ Done when: the CSV has ~50 tickers, with the orphaned sub-universe over-represented.
- **1.2 — Ticker→CIK map.** Download SEC's official `company_tickers.json`, build a lookup, resolve each seed ticker to its 10-digit CIK.
  ✓ Done when: every seed ticker has a CIK (flag any that don't resolve).
- **1.3 — Load companies.** Insert seeds (ticker, name, CIK, sector) into `companies`.
  ✓ Done when: `SELECT count(*) FROM companies` ≈ 50.
- **1.4 — Price/liquidity enrich.** Pull mkt cap, price, avg daily $ volume (yfinance) into each row.
  ✓ Done when: every company has mktcap + adv_usd populated.
- **1.5 — Neglect enrich.** Pull analyst_count from **yfinance** (`numberOfAnalystOpinions`) — *not* FMP, whose free tier paywalls analyst data for micro-caps. Institutional ownership: yfinance `heldPercentInstitutions` if available. Into each row.
  ✓ Done when: rows carry analyst_count + inst_own_pct (note any missing source coverage).
- **1.6 — Apply the 3-layer screen in code.** Function that marks `in_universe` true/false per PLAN §7 thresholds.
  ✓ Done when: running it sets `in_universe` and prints kept/dropped counts.
- **1.7 — Calibrate.** ✅ DONE 2026-06-18. ≤3 analysts left only 8; **calibrated to ≤4 → 11-name pond** (7 energy, 3 materials, 1 tech). Finding: Hudson's coverage estimates ran systematically low (many "≤3" names actually carry 4–11). Threshold lives in `screen.py` (`MAX_ANALYSTS=4`). 3 stale tickers dropped for Hudson: VTLE (→CRGY), BOWL (→LUCK), BRY (confirm).
  ✓ Done when: the kept list looks like "real companies, thinly covered," and thresholds are written down.
- **1.8 — Refresh script.** `scripts/refresh_universe.py` re-runs 1.4–1.6 idempotently.
  ✓ Done when: re-running it updates metrics without duplicating rows.

**🚪 Phase 1 gate (= M0):** a queryable table of ~50 in-universe companies with CIKs, fundamentals, and neglect metrics. Energy-weighted. **This is the pond.**

---

## Phase 2 — The Watcher (ingestion) · M1

**Goal:** detect every new filing for the pond in near-real-time and store it with its public timestamp. **No AI in this phase** — pure, reliable plumbing. *(the developer's core lane.)*

- **2.1 — EDGAR client.** HTTP client with the **exactly-formatted** `User-Agent` the SEC requires (`Your Name user@example.com` — a missing/malformed UA gets you IP-banned, not throttled). Hard-code `time.sleep(0.15)` between *every* EDGAR call (~6.6 req/s, safely under the 10/s limit) — no exceptions, no parallel bursts. Retries with backoff on 429/503.
  ✓ Done when: it fetches one EDGAR URL successfully, every call is spaced ≥0.15s, and the UA matches SEC format.
- **2.2 — One company, raw.** Fetch `data.sec.gov/submissions/CIK##########.json` for a single company; print its 5 most recent filings.
  ✓ Done when: real recent filings print for one ticker.
- **2.3 — Parse & normalize.** Extract per filing: form_type, accession_no, filed_date, **acceptance_datetime**, primary-doc URL.
  ✓ Done when: a clean dict per filing, with acceptance_datetime correctly parsed.
- **2.4 — Store + dedupe.** Insert into `filings`, unique on accession_no (re-runs insert nothing new).
  ✓ Done when: first run inserts N, second run inserts 0.
- **2.5 — Whole pond, backfill.** Loop 2.2–2.4 over all in-universe CIKs to seed recent history.
  ✓ Done when: `filings` holds recent filings across the whole pond.
- **2.6 — Real-time feed.** Poll EDGAR's "latest filings" current feed, filter to our CIKs, capture new ones.
  ✓ Done when: a brand-new filing for a pond company lands in the DB on a poll.
- **2.7 — Fetch raw document.** Download the filing's primary document to `data/filings/<accession>.html` and record `raw_path`.
  ✓ Done when: the document file exists on disk and the row points to it.
- **2.8 — New-filing event.** Emit a clean "new filing" signal/log (and a row flag) when something is first seen — the hook Phase 3 listens to.
  ✓ Done when: only genuinely new filings fire the event.
- **2.9 — Scheduler.** Run the poller every N minutes (APScheduler or cron). Config-driven interval.
  ✓ Done when: it runs unattended on a timer and keeps capturing.
- **2.10 — Latency metric.** Log capture latency = now − acceptance_datetime per filing.
  ✓ Done when: you can see "captured X minutes after publish" in logs.
- **2.11 — 48-hour latency + noise audit (GATE before Phase 3).** Let the watcher run 48h with no AI. Log every captured filing; hand-label the first ~20: material? why? This gives you real samples to tune triage *before burning a single LLM credit*, and proves the watcher pulls signal, not just noise.
  ✓ Done when: median latency < 15 min (near-real-time confirmed), < 50% of 8-K/Form-4/13D are material (confirms triage is needed), and you've personally seen ≥1 genuinely material filing come through the pipe.

**🚪 Phase 2 gate (= M1):** leave it running; a real filing for a pond company appears in the DB within minutes of EDGAR publishing it, with its document on disk and acceptance timestamp recorded. **You can now hear the market.** *(This alone is a resume-worthy artifact.)*

---

## Phase 3 — The Extractor (LLM signals) · M2

**Goal:** turn each new filing into a structured, scored signal. Two-stage to control noise & cost. *(the developer's lane; consult the `claude-api` skill before writing the API calls.)*

- **3.1 — Structure-preserving extraction.** Do **NOT** use naive `BeautifulSoup.get_text()` — EDGAR uses ancient nested HTML tables, and a flat strip turns financials into unreadable number-soup that hallucinates the model. Convert HTML→**Markdown preserving table boundaries** (e.g. a table-aware converter / `sec-parsers`). **Form 4 and 13D are structured XML, not HTML** — parse those as XML directly and never route them through the text path.
  ✓ Done when: a real 8-K renders with its tables intact and legible, and a Form 4's transaction fields parse from XML.
- **3.2 — Chunking.** Split oversized filings into model-sized chunks with light overlap.
  ✓ Done when: a long 10-K splits into ordered chunks under the token budget.
- **3.3 — Provider-agnostic LLM client.** One thin interface (`complete(prompt, schema) -> dict`) with swappable backends — **Gemini** (free tier) and **Anthropic** (`claude-*`) — selected by config, plus structured-output (JSON/tool) support, token/cost logging, retries. No provider hard-wired anywhere downstream.
  ✓ Done when: the same call returns schema-valid JSON from *either* backend by flipping one config value.
- **3.4 — Triage stage (recall-first, not precision-first).** The whole value is the rare catalyst buried in boilerplate — lawyers are paid to make bad news sound routine — so triage must be tuned for **high recall** (a ~30% false-positive rate is fine; a 1% false-negative on a real catalyst is not). Two layers:
  - **(a) Deterministic routing first — no LLM.** 8-Ks carry SEC **item codes**; route high-value items (1.01 material agreement, 2.06 impairment, 5.02 exec departure, 3.02 unregistered equity, 1.03 bankruptcy…) and **all Form 4 / 13D** straight to deep-read, *bypassing triage entirely*.
  - **(b) Gemini Flash (free-tier) triage only for the ambiguous middle**, prompted against an explicit **trigger-keyword list** (litigation, termination, resignation, going-concern, impairment, dilution…) rather than a vague "is this material?". High-volume/low-stakes → free tier is genuinely sufficient here.
  ✓ Done when: high-value item codes + Form 4/13D skip triage; a catalyst buried in a generic-sounding 8-K is *not* dropped by the keyword triage on a hand-picked adversarial sample.
- **3.5 — Extraction schema.** Lock the JSON shape from PLAN §9 (catalyst_type, direction, materiality, confidence, summary, evidence_quote) as a tool/JSON schema.
  ✓ Done when: the schema is defined and validated against a sample object.
- **3.6 — Deep-read stage (eval-decided model).** Prompt the deep-read model (Gemini Flash or `claude-sonnet-4-6` — chosen by 3.10) on triage survivors → the structured JSON, **forcing a real evidence_quote** from the text. Wire through the 3.3 client so the choice is config, not code.
  ✓ Done when: a real filing yields a populated, schema-valid signal with a genuine quote on whichever model is configured.
- **3.7 — Store signals (immutable).** Write to `signals` with model id + timestamp; never overwrite.
  ✓ Done when: a signal row links back to its filing and is append-only.
- **3.8 — Wire to the watcher.** Phase 2's new-filing event triggers triage → (maybe) deep-read → store. The pipe is now end-to-end.
  ✓ Done when: a freshly captured filing produces a signal automatically, untouched by hand.
- **3.9 — Form-4 cluster logic (POST-M2 refinement — defer).** Aggregating multiple Form 4s per company/window into one "insider-buying cluster" signal is an *optimization*, not needed to prove signal existence. Ship single-filing signals for M2; add clustering only once the core pipe is validated. *(This is the one piece of extraction sophistication worth deferring — item-code routing in 3.4 stays, because it's a dict lookup, not complexity.)*
  ✓ Done when (later): several insider buys roll up into one cluster signal, not N noisy ones.
- **3.10 — Eval set + provider decision.** Hand-label ~20 historical filings (material? type? direction?); script measures model agreement **and runs both candidate deep-read models** (Gemini Flash vs `claude-sonnet-4-6`) against the labels, comparing **evidence_quote faithfulness** (real quote vs hallucination) and **materiality calibration**. The winner becomes the configured deep-read model (3.6). Don't let "free" decide this — the data does.
  ✓ Done when: accuracy/agreement prints per provider, a winner is chosen and set in config, and failure modes are noted.
- **3.11 — Cost dashboard line.** Sum tokens/$ per day from logs.
  ✓ Done when: you can state "$X/day at current volume" (should be tiny).

**🚪 Phase 3 gate (= M2):** a new filing flows in untouched and comes out as a sensible, evidence-grounded signal you'd trust to glance at. **The product exists.**

---

## Phase 4 — The Backtest (truth) · M3

**Goal:** find out whether the signals actually precede real moves — honestly. **This is the most important phase in the whole project.** A clean test that says "no edge" is a success; it saves Dad's money. *(Both; Dad sanity-checks.)*

- **4.1 — Price history (point-in-time).** Pull daily OHLCV from a **PIT source (Sharadar / Tiingo) that includes delisted names** into `prices`. **Do NOT use yfinance here** — it purges delisted micro-caps and will make the backtest look artificially brilliant (you'd only be testing survivors). yfinance is live-monitoring only.
  ✓ Done when: each pond ticker has multi-year daily prices *and* at least one known-delisted name still has its full pre-delisting history.
- **4.2 — Benchmark map.** Assign each company a sector/benchmark ETF for abnormal-return calc.
  ✓ Done when: every company maps to a benchmark.
- **4.3 — Forward returns.** Function: given (ticker, date) → +1d/+5d/+20d returns.
  ✓ Done when: spot-checked against a hand calc.
- **4.4 — Abnormal returns.** Subtract benchmark return over the same window.
  ✓ Done when: abnormal return computes and ties out on a sample.
- **4.5 — Point-in-time discipline.** Hard rule + test: a signal may only "know" things at/before its `acceptance_datetime`. No future leakage anywhere.
  ✓ Done when: a deliberate look-ahead attempt is caught by a guard/test.
- **4.6 — Survivorship handling.** Use a point-in-time universe including later-delisted names; document the data approach and its limits.
  ✓ Done when: delisted names aren't silently dropped, and the limitation is written down.
- **4.7 — Slippage model (OHLC-only, no tick data).** Per PLAN §10: (a) **fill at the *next* bar after `acceptance_datetime`, never the signal-bar close**; (b) cap assumed size **≤1% of 20-day ADV**; (c) charge a **half-spread via a high-low estimator (Corwin-Schultz / Abdi-Ranaldo)** — recovers effective spread from daily OHLC, no tick feed needed; (d) add a **√-impact term** `∝ σ·√(size/ADV)`, **with a harsher multiplier for ADV < $5M** (√-impact is calibrated on bigger names and *understates* micro-cap impact, where our edge lives); (e) **stress at 2× and 3× — and 5× for ADV < $2M names**; (f) log the gap between assumed fill and a **VWAP proxy `(O+H+L+C)/4`** as a sanity check.
  ✓ Done when: results show pre-/post-slippage, the thinnest names are penalized hardest, and the report states whether the edge survives 3× (and the sub-$2M names at 5×).
- **4.8 — Join signals→outcomes.** Populate `outcomes` linking each historical signal to its forward/abnormal returns.
  ✓ Done when: `outcomes` is populated for all backtestable signals.
- **4.9 — Aggregate analysis.** Group by catalyst_type × materiality bucket: hit rate, mean abnormal return, dispersion.
  ✓ Done when: a table shows which signal types carry edge, if any.
- **4.10 — Significance check (a panel, not a single guillotine).** With n often < 30 and fat tails, one p-value shouldn't kill or crown a signal. Report a **panel** per bucket: (i) **permutation/bootstrap p-value** (not a t-test — normality doesn't hold); (ii) **hit rate with a binomial test** (e.g. "12/18 positive at +5d; 67% hit, p=0.18 — suggestive, underpowered"); (iii) **effect size** (mean abnormal return). Then look for **coherent drift across +1d/+5d/+20d** — a real catalyst trends consistently. ⚠️ Do **not** "accept if positive at +1d *or* +5d *or* +20d" — OR-ing windows is a multiple-testing trap that *inflates* false positives; you want a consistent pattern, not a lucky window.
  ✓ Done when: each bucket shows n + the three-stat panel + a cross-window read, and no signal is judged on a single number.
- **4.10b — Dad's manual false-pos/neg review.** Have Dad eyeball 5–10 of the model's misses and false alarms. Does the LLM lack context only a domain expert has? ("This impairment is routine for shale." "This resignation is the CFO fleeing fraud.") This is the human-in-the-loop check no statistic can replace — and Dad's whole reason for being on the project.
  ✓ Done when: Dad has reviewed a sample and noted any systematic blind spots to feed back into the extraction prompts.
- **4.11 — Backtest report.** A notebook/markdown report telling the honest story with charts.
  ✓ Done when: the report renders and a non-coder (Dad) can follow the conclusion.

**🚪 Phase 4 gate (= M3) — THE DECISION GATE:** a credible answer to "is there an edge, and in which catalyst types?" If yes → proceed and lean into the winning catalysts. If no → either re-aim (different signals/universe) or stop, having spent ~$0 instead of real capital. **Do not skip or soften this gate.**

---

## Phase 5 — Alerts & Dashboard · M4

**Goal:** make the proven signals usable by humans. *Build only what survived Phase 4* — wire alerts to the catalyst types that showed edge. *(the developer.)*

> ⏩ **Dashboard scaffolded early (2026-06-18)** — `dashboard.py` (Streamlit) built ahead of schedule via a phase-registry: each phase is a `status()`+`render()` section that auto-activates when its DB table fills. Phases 1–2 live now; 3–5 show "coming" placeholders that light up automatically when `signals`/`outcomes` get rows. Steps 5.5–5.8 (universe view, signal feed, drilldown, track record) get fleshed out as those phases land. `make dashboard`. Alerts (5.1–5.4) still deferred to post-M4 per the goal above.

- **5.1 — Alert threshold config.** Configurable materiality/confidence cutoffs (informed by Phase 4 results).
  ✓ Done when: thresholds live in config, not hardcoded.
- **5.2 — Alert sender.** Email or Telegram delivery.
  ✓ Done when: a test alert arrives on Dad's phone.
- **5.3 — Alert format.** Summary + score + direction + evidence_quote + filing link, clean and skimmable.
  ✓ Done when: a real signal renders as a readable alert.
- **5.4 — Throttle/dedupe.** No duplicate or spammy alerts for the same event.
  ✓ Done when: re-processing a filing doesn't re-alert.
- **5.5 — Dashboard: universe view.** Streamlit page listing the pond with metrics.
  ✓ Done when: the pond renders in a browser.
- **5.6 — Dashboard: signal feed.** Reverse-chron live signals with filters (type, score, ticker).
  ✓ Done when: recent signals show and filter.
- **5.7 — Dashboard: company drilldown.** Per-company filing/signal history + price chart.
  ✓ Done when: clicking a ticker shows its history.
- **5.8 — Dashboard: track record.** Surface the backtest stats live.
  ✓ Done when: edge-by-catalyst is visible in the UI.
- **5.9 — Run docs.** README section: how to start watcher + dashboard from a clean machine.
  ✓ Done when: someone else can start it from the docs.

**🚪 Phase 5 gate (= M4):** Dad receives a real, well-formatted alert minutes after a filing, and can open a dashboard to explore the universe, the signal, and the track record.

---

## Phase 6 — Hardening & scale

**Goal:** make it run unattended and reliably; widen the moat. Pull items in as the system earns trust.

- **6.1 — Process supervision** (auto-restart the watcher; healthcheck).
- **6.2 — Error alerting** (it tells *you* when ingestion/LLM breaks).
- **6.3 — Universe auto-refresh** on a schedule (Phase 1.8 on a timer; add/drop names).
- **6.4 — Postgres migration** when SQLite strains.
- **6.5 — Energy dockets:** FERC eLibrary / ERCOT / PUCT scrapers feeding the same pipeline — Dad's sharpest edge. ⭐ **Candidate to promote to a "Phase 3.5 / v1.1" right after M2** (these notices often lead 8-Ks by hours/days = pure alpha for neglected energy names). **Gated:** only after M2 proves the core EDGAR pipe, and note dockets are a strong *live* feed but *harder to backtest* cleanly (no tidy filing→price-reaction mapping), so they don't go on the M3 critical path.
- **6.6 — Paper-trading log:** record hypothetical positions from live signals; compare to the backtest's promise.
- **6.7 — Backtest-as-monitor:** re-run the edge analysis on accumulating *live* signals to catch decay.

**🚪 Phase 6 gate:** runs a full week unattended, self-heals, refreshes its own universe, and logs a live paper track record.

---

## Phase 7 — RL position sizing (optional, gated) · M5

**Unlocked only if Phase 4 validated an edge.** This is where your RL interest legitimately lives — sizing/allocation, never price prediction.

- **7.1 — Frame the MDP** (state = open signals + scores + liquidity + cash; action = allocate/hold/exit).
- **7.2 — Sim environment** over historical signals + prices.
- **7.3 — Baselines first** (equal-weight, score-weighted) — beat these before any RL.
- **7.4 — PPO allocator** trained in the sim.
- **7.5 — Walk-forward eval** vs. baselines, slippage-aware.
- **7.6 — Paper-only deployment** (never auto-executes real trades in v1).

**🚪 Phase 7 gate:** the RL allocator beats the simple baselines out-of-sample, after slippage — otherwise keep the baseline and move on.

---

## Guardrails that apply to every phase

- **Never use post-filing information** in any signal or backtest. The `acceptance_datetime` is sacred (PLAN §10).
- **Signals are immutable.** Append, never overwrite — they're the backtest's ground truth.
- **Personal tool only** until the RIA fork (PLAN §4) is deliberately decided. Don't drift across it.
- **Each step ends runnable and committed.** If the repo is broken, the step isn't done.
- **PLAN.md wins** on any conflict; update it when a decision changes, and update this roadmap's checkboxes as you go.
- **The intellectual-honesty rule (the one that actually decides success):** freeze the slippage model, look-ahead rules, and universe definition *before* M3 runs — and don't touch them after seeing the result. When the backtest comes back flat, you will feel the pull to relax an assumption until an edge "appears." That's self-deception, and it's the failure mode that kills these projects. A clean "no edge" that saves Dad's capital is a win. (PLAN §10.)

---

## Immediate next 3 steps (start here)

1. **0.1–0.3** — repo, venv, structure (≈ 30 min).
2. **0.6** — DB + schema from PLAN §14 (≈ 45 min).
3. **1.0 (GATE)** — universe sanity check: FMP vs. manual analyst count on ~10 names. *Don't skip — this decides whether the pond is trustworthy at all.*
4. **1.1–1.3** — Hudson's 50-name seed list → CIKs → loaded into `companies`.

After these, you'll have a living database with a *validated* pond in it — and Phase 2's watcher has something to point at.

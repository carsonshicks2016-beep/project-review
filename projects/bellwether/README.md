# Bellwether

An AI tool that reads SEC filings in near-real-time for small, analyst-neglected
companies and flags material catalysts the market hasn't priced yet — exploiting an
**attention gap**, not a speed or smarts gap. Personal tool (v1).

See **[PLAN.md](PLAN.md)** (strategy / why) and **[BUILD_ROADMAP.md](BUILD_ROADMAP.md)**
(step-by-step build order). PLAN.md is authoritative.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env        # then fill in keys (not needed until Phase 1/3)
make init-db                # create bellwether.db with the schema
make test                   # should be green
```

## Layout

```
src/bellwether/   # package code (config, db, logging, schema.sql)
scripts/          # runnable entry points (init_db, …)
tests/            # pytest suite
data/             # local data + downloaded filings (gitignored)
notebooks/        # backtest / analysis notebooks
```

## Status

Phases 1–5 complete. Live EDGAR watcher, LLM extraction pipeline, backtest engine, and email alert system are all operational.

## Running from Scratch

```bash
# 1. One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # Fill in: GEMINI_API_KEY, TIINGO_API_KEY, SEC_USER_AGENT,
                           #          ALERT_FROM_EMAIL, ALERT_TO_EMAIL, ALERT_GMAIL_APP_PASSWORD
make init-db               # Creates bellwether.db

# 2. Load the universe
python scripts/load_universe.py

# 3. Sync historical prices (for backtesting)
python -m bellwether.pricing

# 4. Backfill historical filings
python scripts/run_watcher.py backfill

# 5. Run the LLM extraction pipeline over the backlog
python scripts/process_backlog.py --limit 50

# 6. Run the backtest engine and generate the report
python -m bellwether.backtest
python -m bellwether.analysis

# 7. Start the watcher (keeps polling EDGAR every 15 min)
make run-watcher

# 8. Open the dashboard (in a separate terminal)
make dashboard             # Opens at http://localhost:8501
                           # or: streamlit run dashboard.py --server.port 8505
```

> **Alert setup:** You need a [Gmail App Password](https://myaccount.google.com/apppasswords)
> (not your main password) to enable email alerts. Set `ALERT_FROM_EMAIL`, `ALERT_TO_EMAIL`,
> and `ALERT_GMAIL_APP_PASSWORD` in `.env`.


## External audit scheduler

GitHub's native `*/15` schedule remains enabled as a fallback. For a more
consistent 15-minute cadence, cron-job.org can trigger the existing
`workflow_dispatch` endpoint at minutes 07, 22, 37, and 52 each hour.

1. Create a GitHub fine-grained personal access token restricted to the
   `bellwether` repository with only **Actions: Read and write** permission.
   Give it a short expiry that covers the audit window.
2. Create a cron-job.org API key under **Settings > API keys**. Do not add an IP
   restriction if this command will be run from a changing home IP.
3. Run `make setup-external-pinger`. Both keys are requested with hidden prompts
   and are not saved locally. Re-running the command updates the existing job.

Use `.venv/bin/python scripts/setup_external_pinger.py --dry-run` to inspect
the redacted request configuration. Successful external triggers appear in
GitHub Actions as `watcher-audit (cron-job.org)`.

cron-job.org must store the GitHub token to send the authenticated dispatch.
When the audit ends, disable/delete the cron job and revoke the GitHub token.

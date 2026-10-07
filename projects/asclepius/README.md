<img src="assets/asclepius-logo-1024.png" alt="" width="120" align="left" hspace="18" vspace="4">

# Asclepius

An MCP server that puts your WHOOP data in front of Claude, plus a statistics
layer that answers "what actually drives my recovery" more honestly than the
app's built-in insights.

<br clear="left">


Data is synced once into local SQLite and analyzed offline. Nothing leaves your
machine except the API calls to WHOOP.

## Setup

### 1. Create a WHOOP app

Go to the [WHOOP developer dashboard](https://developer-dashboard.whoop.com/)
and create an app named **Asclepius**, using `logo.png` from this repository as
the app logo. Tick all six scopes the dashboard offers:

```
read:recovery  read:cycles  read:workout  read:sleep  read:profile  read:body_measurement
```

The dashboard has no checkbox for `offline`, but that is the scope that matters
most — without it WHOOP never issues a refresh token and you re-authorize every
hour. Asclepius appends it to the authorization request itself (`config.SCOPES`),
which is why it shows up on the consent screen but not here.

Register this redirect URI:

```
https://localhost:8177/callback
```

Note the `https`. WHOOP's dashboard rejects `http://` redirects, so a plain
localhost callback cannot be registered and the automatic browser handoff is
not available. `https://localhost` is still the right choice: the browser
cannot connect, so it shows an error page — but the authorization code is in
the address bar, which is all the paste-based flow needs. The code never leaves
your machine, which would not be true of a redirect pointing at a domain
someone else operates.

The dashboard also requires a **privacy policy URL**. `PRIVACY.md` in this
repository is written for exactly this app; publish it somewhere public (a
GitHub repo or gist works) and use that link. The URL has to resolve before
WHOOP will accept the form.

### 2. Save your credentials

```bash
mkdir -p ~/.asclepius && cat > ~/.asclepius/app.json <<'EOF'
{
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET",
  "redirect_uri": "https://localhost:8177/callback"
}
EOF
chmod 600 ~/.asclepius/app.json
```

The `redirect_uri` here must match what you registered character for character,
`https` included — WHOOP rejects the exchange otherwise.

`WHOOP_CLIENT_ID` / `WHOOP_CLIENT_SECRET` / `WHOOP_REDIRECT_URI` environment
variables override the file if you prefer.

### 3. Authorize

```bash
uv run asclepius-auth
```

This opens your browser. After you approve, the page will fail to load — that
is expected and means the code stayed local. Copy the full URL from the address
bar and paste it at the prompt. Tokens are written to
`~/.asclepius/credentials.json` (mode 0600).

The command detects a non-`http://` redirect and switches to this paste flow on
its own; `--manual` forces it explicitly.

### 4. Backfill

```bash
uv run python -m asclepius.cli sync --full
```

Three years by default, a few minutes at WHOOP's 100 req/min ceiling. After
that, `whoop_sync` from inside Claude does fast incremental pulls.

### 5. Connect to Claude Code

`.mcp.json` in this directory already registers the server. Restart Claude Code
and approve it when prompted. Verify with:

```bash
uv run python -m asclepius.cli status
```

## Tools

| Tool | What it does |
| --- | --- |
| `whoop_status` | Cache coverage, record counts, last sync |
| `whoop_sync` | Pull new records (`full=true` to re-backfill) |
| `whoop_metrics` | Every available metric with units and coverage |
| `whoop_daily` | Day-by-day values for chosen metrics |
| `whoop_correlate` | One driver vs one outcome across time lags |
| `whoop_rank_drivers` | Screen all metrics against one outcome |
| `whoop_regress` | Isolate an effect with confounders held fixed |
| `whoop_compare_periods` | Before/after comparison of two date windows |
| `whoop_anomalies` | Vitals departing from their trailing baseline |

## What you can ask

> Why has my recovery been bad for the last two weeks?

> Does lifting hurt my next-day recovery more than running does?

> Is my late bedtime actually the problem, or is it just that late nights are short nights?

> Compare my recovery during the January training block against February.

> Am I coming down with something?

## Dashboard

```bash
uv run python -m asclepius.cli dashboard
```

Writes `~/.asclepius/dashboard.html` and opens it. The entire daily frame is
embedded in the file and every chart and statistic is recomputed in the browser,
so it is fully interactive with no server behind it.

| Panel | What you can do |
| --- | --- |
| Range | Presets, or drag across the mini-trace to zoom. Everything else re-renders against the selection. |
| Vitals | Four dials; the pale tick is the range mean, the delta compares halves of the range. Click one to plot it. |
| Metrics | Click any of the 30 metrics to plot; shift-click to overlay up to three. |
| Explorer | Smoothing slider from 1–28 days, hover crosshair, click to inspect a day. |
| Correlation lab | Any driver against any outcome, lag from −7 to +7, live scatter with fit and a bar profile of r at every lag. |
| Day | Full readout for the selected day plus that day's workouts. Arrow keys step. |
| Calendar | 216 days coloured by any metric; click to select. |
| Sports | Activity breakdown for the current range. |

Correlations in the browser use a **Bartlett-adjusted effective n** and a
Fisher-z confidence interval, so autocorrelation is not quietly inflating them.
What the browser does *not* do is FDR correction across a family of tests, or
regression with confounders held fixed — for those use `whoop_rank_drivers` and
`whoop_regress`, and the panel says so.

The file makes no network requests of any kind: data, styles, script and logo
are all inlined. That is deliberate — it keeps the dashboard consistent with the
claim in `PRIVACY.md` that nothing leaves the machine. It lands in
`~/.asclepius/` rather than the repo so a snapshot of your own physiology can
never be committed by accident.

Markup, styling and behaviour live in `src/asclepius/web/` as real
`.html`/`.css`/`.js` files and are inlined at build time, so edit those and
re-run the command.

## How the analysis works

Wearable data invites bad statistics. Three things are handled deliberately:

**Autocorrelation.** Recovery today strongly resembles recovery yesterday, so
two such series correlate impressively by accident — the number of genuinely
independent observations is far below the number of days. Correlation p-values
are computed against a Bartlett-adjusted effective sample size, and regressions
use Newey-West (HAC) standard errors. Expect weaker, more honest results than a
naive correlation would give.

**Multiple comparisons.** `whoop_rank_drivers` tests ~30 metrics at once, where
a handful will clear p<0.05 by chance alone. Every family of tests reports
Benjamini-Hochberg q-values; read those, not raw p.

**Confounding.** A raw association between late bedtimes and poor recovery
mostly restates that late nights are short nights. `whoop_regress` holds
covariates fixed so you can see what each variable contributes on its own, and
flags collinear predictors by VIF when they carry overlapping information.

### Alignment

One convention underpins everything. A row is one physiological cycle, keyed by
its local date. On that row, `recovery_*` and `sleep_*` describe the night that
**ended that morning**, while `strain` and `workout_*` describe the day that
**followed it**. So within a row, sleep precedes strain.

That means `strain(t) → recovery(t+1)` is lag 1, and positive lag always means
the driver leads the outcome.

This is subtler than it looks: a sleep record's own `cycle_id` refers to the
cycle it happened *during*, which is the previous one. The engine therefore
links sleep through `recovery.sleep_id`, not through `cycle_id`. Joining the
obvious way inverts every lag result. `tests/test_pipeline.py` plants a known
effect (`recovery[d] = 60 − 1.8·strain[d−1] + 4.0·sleep_hours[d]`) and asserts
the engine recovers both coefficients at the correct lag, which is what keeps
this honest.

## What the API does not expose

Worth knowing before planning anything on top of this. The public API serves
cycles, recovery, sleep, workouts, and body measurements. It does **not** serve
ECG waveforms, blood pressure, journal entries, or continuous heart rate — the
MG-exclusive medical data lives only in the app, the PDF health report, and the
CSV export. Anything built on those has to go through the export path.

## Development

```bash
uv run pytest tests/ -q
```

Tests run entirely on synthetic data with planted effects — no credentials or
network needed.

CLI output is drawn as an amber console panel. It detects a terminal and
degrades to plain text when piped, so `| grep` and `> file` stay clean.
`NO_COLOR=1` forces plain, `ASCLEPIUS_COLOR=always` forces styled, and
`drivers --json` gives the raw result object instead of the panel.

## Files

| Path | Purpose |
| --- | --- |
| `src/asclepius/config.py` | Paths, endpoints, scopes |
| `src/asclepius/auth.py` | OAuth flow, token rotation |
| `src/asclepius/client.py` | Paginated API client, rate limiting |
| `src/asclepius/store.py` | SQLite schema and mappers |
| `src/asclepius/sync.py` | Incremental sync |
| `src/asclepius/frame.py` | Daily feature frame |
| `src/asclepius/analyze.py` | Statistics |
| `src/asclepius/server.py` | MCP tools |
| `src/asclepius/cli.py` | Terminal commands |
| `src/asclepius/console.py` | ANSI styling for the CLI |
| `src/asclepius/dashboard.py` | Builds the dashboard; embeds the frame as JSON |
| `src/asclepius/web/` | Dashboard markup, styling and browser-side statistics |
| `assets/make_logo.py` | Renders the mark; run it to regenerate `logo.png` |

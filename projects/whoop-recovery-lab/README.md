# WHOOP Personal Recovery Lab

A local-only personal dashboard for exploring WHOOP recovery patterns and browsing the fields returned by WHOOP's official API.

## Run it

```sh
npm install
npm run dev
```

Open [http://localhost:8787](http://localhost:8787). The local WHOOP OAuth credentials and token encryption key are in `.env`; this file is ignored by Git. The app binds only to `127.0.0.1` and stores the SQLite database under `data/`.

The first local run can import an existing OAuth token file through `WHOOP_TOKEN_BOOTSTRAP_FILE`. It encrypts the access and refresh tokens before saving them in SQLite. If the token can no longer refresh, use **Connect WHOOP**. The callback URL shown by the server must be registered in your WHOOP Developer App settings before OAuth reconnection can complete.

## Views

- **Primary:** daily recovery, HRV, resting heart rate, sleep and strain trends; a personal rolling baseline; possible deviations; event notes with an exploratory recovery window; and simple Yes/No experiment comparisons.
- **Max data:** paginated field-level records, nested field paths, provider JSON, date and type filters, search, unit hints, indexed metric overlays, and CSV/JSON export.

Deviations and event windows are descriptive signals. They do not identify the cause of a physiological change or provide medical advice. WHOOP's API is cycle-oriented and does not expose every raw sensor sample.

## Checks

```sh
npm test
npm run build
```

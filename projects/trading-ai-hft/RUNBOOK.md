# Trading AI Runbook

## Install Or Refresh Dependencies

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 -m pip install -r requirements.txt
```

## Run Safety Tests

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 -m unittest discover -s tests
```

## Backtest One Existing PPO Bot

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 backtest_ppo.py --model models/ppo_bot_1.zip --ticker BTC-USD
```

By default, serious validation does not use synthetic sentiment. To intentionally enable the old price-derived synthetic sentiment bootstrap mode, add:

```bash
--use-synthetic-sentiment
```

## Train A Better Crypto Universe Ensemble

Fast smoke run:

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 train_crypto_universe.py --discover-alpaca --interval 5m --max-assets 8 --candidates 3 --timesteps 5000 --no-promote
```

Serious paper-trading candidate run:

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 train_crypto_universe.py --discover-alpaca --interval 5m --max-assets 12 --candidates 9 --timesteps 100000 --promote-count 5
```

Deeper historical research run:

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 train_crypto_universe.py --discover-alpaca --interval 1d --start 2020-01-01 --max-assets 12 --candidates 9 --timesteps 500000 --promote-count 5
```

The trainer ranks candidates on later sequential validation windows and copies only winners into `models/active/`. The live trader loads `models/active/` first when it exists.

## Run The Ensemble Against Alpaca Paper Trading

Your current `.env` has `ALPACA_PAPER=True`, so this submits paper orders, not real-money orders.

One-shot smoke test:

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 live_ensemble.py --once --tickers BTC-USD ETH-USD SOL-USD --interval 300
```

Continuous paper trading:

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
python3 live_ensemble.py --tickers BTC-USD ETH-USD SOL-USD --interval 300 --max-order-value 15 --min-trade-value 1
```

The live scripts refuse real-money mode when `ALPACA_PAPER=False` unless you pass `--allow-live`.

## Dashboard

```bash
cd /Users/REVIEW_USER/Desktop/trading-ai-hft
streamlit run dashboard_app.py
```

## What To Trust First

Trust the simulator tests first, then paper-trading logs, then backtests. Do not treat model returns as meaningful unless the tests pass and the run uses non-leaked sentiment.

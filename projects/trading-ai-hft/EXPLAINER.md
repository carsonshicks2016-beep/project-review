# Trading AI HFT Explainer

## What This Program Is Supposed To Do

This project is meant to build an AI-assisted trading system that learns when to hold cash, buy an asset, sell an asset, or allocate a percentage of a portfolio to an asset. It combines market data, technical indicators, sentiment scores, reinforcement learning, evolutionary neural networks, backtesting, dashboards, and Alpaca order execution.

Despite the folder name, this is not true high-frequency trading in the professional sense. True HFT usually means colocated infrastructure, exchange-level market data, order book logic, microsecond/millisecond latency, and carefully engineered execution. This project works mostly with daily or 5-minute candles from yfinance, so it is better described as an AI trading research sandbox.

The intended workflow is:

1. Download historical price data.
2. Compute normalized technical indicators.
3. Add sentiment features from news or synthetic sentiment.
4. Build state vectors that represent each market moment.
5. Train AI agents in a simulated trading environment.
6. Backtest trained agents against unseen periods or assets.
7. Visualize model behavior in dashboards.
8. Optionally connect the model signals to Alpaca paper/live trading.

## The Big Picture

The project has two main model families:

- NEAT models: Evolved neural networks that choose one of three actions: HOLD, BUY, or SELL.
- PPO models: Reinforcement learning models from Stable-Baselines3 that output a continuous target allocation from 0.0 to 1.0.

The center of the system is `TradingEnv`, a Gymnasium environment. Training code feeds observations into an agent, the agent chooses an action, and `TradingEnv` simulates portfolio changes, transaction costs, drawdown, and performance metrics.

Conceptually:

```text
yfinance prices
    -> technical indicators
    -> optional sentiment
    -> state matrix
    -> TradingEnv
    -> NEAT or PPO agent
    -> simulated trades
    -> metrics, charts, dashboards, or Alpaca orders
```

## Main Components

### Price Data

`data/fetch_prices.py` downloads OHLCV candles with yfinance and stores them in `data/cache/prices.db`.

Default crypto tickers include BTC, ETH, DOGE, SOL, ADA, XRP, SHIB, and AVAX. Other scripts also use stock symbols like SPY, NVDA, and AAPL.

The cache is useful because repeated training runs should not redownload the same data. However, the current cache logic is too simple: it does not distinguish daily data from 5-minute data, does not verify that cached date ranges satisfy the current request, and replaces the whole table whenever new data is saved.

### Technical Indicators

`features/technical_indicators.py` computes:

- RSI
- MACD line
- MACD signal
- MACD histogram
- Bollinger Band position
- Bollinger Band width
- Volume z-score
- 1-day return
- 5-day return
- 20-day return

The features are normalized to roughly 0 to 1 so they are easier for neural networks to consume.

### Sentiment

There are two sentiment paths:

- Real or semi-real news ingestion through CryptoPanic or FNSPID.
- Synthetic sentiment generated from price moves.

`data/sentiment_scorer.py` uses FinBERT to score headlines from -1.0 to +1.0. `features/state_builder.py` then normalizes sentiment to 0.0 to 1.0 and appends it to the technical indicators.

The synthetic sentiment path is useful for prototyping, but it is dangerous for evaluation because it derives sentiment from the same price movement the model is trying to trade. That can create data leakage: the model may appear to understand sentiment when it is actually being handed a disguised version of price direction.

### State Builder

`features/state_builder.py` combines indicators and sentiment into a state matrix.

Each base state has 11 features:

```text
rsi, macd, macd_signal, macd_hist,
bb_position, bb_width, vol_zscore,
ret_1d, ret_5d, ret_20d,
sentiment
```

At runtime, `TradingEnv` appends two portfolio-state features:

```text
position_flag, unrealized_pnl
```

That gives NEAT 13 inputs total.

### Trading Environment

`env/trading_env.py` simulates trading.

Current design:

- Observation: market features plus portfolio state.
- Action space: continuous target allocation from 0.0 to 1.0.
- Backward compatibility: integer actions still map to HOLD, BUY, SELL.
- Portfolio: cash balance plus long-only asset position.
- Costs: transaction cost defaults to 0.1%.
- Metrics: total return, Sharpe ratio, drawdown, trades, win rate, buy-and-hold return, alpha.

This file is the most important part of the project. If the simulator is wrong, training and backtesting results become misleading even if the model code is fancy.

### NEAT Training

`train.py` trains a NEAT agent. NEAT evolves neural network topology and weights over generations.

The NEAT configuration lives at `agents/config-trader`.

The evolved network has:

- 13 inputs
- 3 outputs: HOLD, BUY, SELL
- population size 80

Fitness rewards return and Sharpe while penalizing drawdown and too few trades.

### PPO Training

`train_ppo.py` trains a Stable-Baselines3 PPO model. PPO outputs a target allocation instead of a discrete action.

`train_ensemble_factory.py` trains multiple PPO models with different seeds and hyperparameters. This is the source of the current `models/ppo_bot_1.zip` through `models/ppo_bot_9.zip` files.

The ensemble concept is good: different models may make different mistakes, and averaging them can reduce single-model overconfidence. But ensemble voting only helps if the underlying simulation and validation are reliable.

### Backtesting

There are several backtesting scripts:

- `backtest.py`: NEAT out-of-sample backtest.
- `backtest_ppo.py`: PPO backtest.
- `walk_forward.py`: rolling train/test validation for NEAT.
- `utils/backtest_engine.py`: lightweight dashboard backtest helper.

Walk-forward validation is the best idea here because it tests whether a strategy survives different market regimes. A simple train/test split can be too optimistic.

### Dashboards

There are two dashboard styles:

- `live_dashboard.py`: Flask dashboard for live NEAT training progress.
- `dashboard_app.py`: Streamlit dashboard for model signals and short-horizon backtests.

The dashboards are mostly for research visibility. They show model consensus, technical conditions, generated model explanations, and backtest leaderboards.

### Live Trading

`live_trader.py` connects a NEAT genome to Alpaca.

`live_ensemble.py` loads all available NEAT and PPO models and creates an ensemble council. It averages model allocations, then trades only when conviction crosses thresholds:

- 70% or more: buy/full allocation.
- 30% or less: sell/zero allocation.
- between 30% and 70%: hold current allocation.

This is pointed at Alpaca using keys from `.env`. It can run in dry-run mode, which should be the default until the simulator and validation issues are fixed.

## What Is Currently Present

Current artifacts:

- Nine PPO model zips in `models/`.
- No `models/ppo_trader.zip`.
- No NEAT `checkpoints/best_genome.pkl`.
- Cached BTC-USD 5-minute-ish data in `data/cache/prices.db`.
- Backtest result images in `backtest_results/`.

This means the ensemble dashboard can load the nine PPO bots, but scripts expecting `ppo_trader.zip` or `best_genome.pkl` will fail unless models are trained or filenames are adjusted.

## Key Problems And How To Make It Better

### 1. Fix The PPO Reward Timing

Current problem:

`TradingEnv.step()` computes portfolio value before and after the trade using the same candle price. That means price movement from one candle to the next does not directly produce reward. PPO mostly learns from transaction costs and rebalancing side effects, while final metrics still reflect later price changes.

Why this matters:

PPO learns from step-by-step reward. If reward does not reflect market movement, the model receives a poor learning signal. It may learn not to trade, churn strangely, or optimize quirks in the simulator.

Better design:

At step `t`, execute the action at price `t`, advance to `t + 1`, then calculate new portfolio value using price `t + 1`.

Specific change:

- Store `old_value` before action at current price.
- Execute trade at current price.
- Increment `current_step`.
- Compute `new_value` at next price.
- Reward should be based on `(new_value - old_value) / old_value`.

### 2. Record Real Trade Logs

Current problem:

`TradingEnv` creates `self.trade_log = []`, but never appends trades to it.

Why this matters:

Backtest charts and dashboards expect trade markers. Without logs, you can see total trade counts but not inspect where the model bought or sold. That makes debugging much harder.

Better design:

Whenever a buy or sell executes, append a structured record:

```python
{
    "step": self.current_step,
    "date": self.dates[self.current_step],
    "action": "BUY",
    "price": current_price,
    "qty": new_qty,
    "value": value_delta,
    "fee": cost,
    "portfolio_value": self.portfolio_value,
}
```

For sells, include realized P&L.

### 3. Fix End-Of-Episode Liquidation

Current problem:

When the episode ends, forced selling sets `self.balance = net`. If the strategy was partially allocated, existing cash can be overwritten instead of preserved.

Why this matters:

Final metrics can be wrong, especially for PPO models that use fractional allocations.

Better design:

On final liquidation:

```python
self.balance += net
self.portfolio_value = self.balance
```

Do not replace cash with only the sale proceeds.

### 4. Make Live Observations Match Training Observations

Current problem:

During training, unrealized P&L is sigmoid-normalized and flat position uses `0.5`. In live scripts, unrealized P&L is raw percentage return and flat position uses `0.0`.

Why this matters:

Models are sensitive to input scale. If the live input distribution differs from training, the model can behave unpredictably.

Better design:

Create one shared helper, for example:

```python
build_portfolio_features(position_qty, entry_price, current_price)
```

Use it in `TradingEnv`, `live_trader.py`, `live_ensemble.py`, dashboards, and backtest helpers.

### 5. Replace Synthetic Sentiment For Serious Evaluation

Current problem:

Synthetic sentiment is generated from same-day price returns.

Why this matters:

This leaks the answer into the feature set. It is fine for bootstrapping code paths, but not for measuring real predictive performance.

Better design:

Use only timestamp-valid sentiment:

- Headline published before the trading decision.
- No same-day close-derived labels.
- Aggregate sentiment using only data available before the candle being traded.

For crypto, this likely means either paid historical news APIs, exchange/social data with timestamps, or removing sentiment until real data exists.

### 6. Improve The Price Cache

Current problem:

The SQLite cache does not separate intervals, does not verify requested start/end ranges, and replacing the table can wipe unrelated tickers.

Why this matters:

Training can silently use the wrong timeframe or incomplete data. That corrupts experiments.

Better design:

Add `interval` to the primary key:

```sql
PRIMARY KEY (date, ticker, interval)
```

Use upserts instead of replacing the whole table. When loading from cache, verify:

- ticker coverage
- interval match
- start date coverage
- end date freshness

### 7. Add Tests Around The Simulator

Current problem:

There are no automated tests.

Why this matters:

Trading simulations are easy to get subtly wrong. A small accounting bug can create fake alpha.

Best first tests:

- Holding cash in a rising market stays flat.
- Buying once in a rising market increases value.
- Buying once in a falling market decreases value.
- Transaction costs reduce value.
- Partial allocation preserves cash.
- Forced liquidation preserves cash plus sale proceeds.
- `trade_log` length matches executed trades.
- PPO action `0.5` produces about 50% allocation.
- Integer NEAT action `1` maps to buy, `2` maps to sell.

### 8. Clean Up Model Artifacts And Script Expectations

Current problem:

The repo has `ppo_bot_1.zip` through `ppo_bot_9.zip`, but `backtest_ppo.py` expects `models/ppo_trader.zip`. NEAT scripts expect `checkpoints/best_genome.pkl`, but that file is absent.

Why this matters:

The project looks more broken than it is because scripts and artifacts disagree.

Better design:

Support explicit model paths:

```bash
python backtest_ppo.py --model models/ppo_bot_1.zip --ticker BTC-USD
```

Also add a `models/manifest.json` describing model type, training data, interval, date range, feature set, and environment version.

### 9. Make Requirements Complete And Reproducible

Current problem:

`requirements.txt` has duplicates and missing packages.

Missing imports include:

- flask
- streamlit
- streamlit-autorefresh
- datasets

Duplicate or questionable entries include:

- duplicate `torch`
- duplicate `stable-baselines3[extra]`
- `pandas-ta`, which appears unused

Why this matters:

A trading research repo needs reproducibility. If dependencies drift, model behavior and results can change.

Better design:

Use pinned or bounded dependencies, and split optional groups:

- core training
- dashboards
- FinBERT/FNSPID
- live trading

### 10. Strengthen Risk Controls Before Any Live Trading

Current problem:

Live trading uses simple cash/position state inside the process. It does not fully reconcile with Alpaca positions each loop, does not enforce robust max loss rules, and uses simplified execution assumptions.

Why this matters:

A local process can restart, lose state, or disagree with broker state. That is dangerous even in paper trading and unacceptable in real trading.

Better design:

Before every trading decision:

- Pull account equity.
- Pull current positions.
- Pull open orders.
- Cancel stale orders if needed.
- Compute portfolio exposure from broker truth.
- Enforce max position size.
- Enforce daily loss limit.
- Enforce max drawdown stop.
- Enforce no-trade mode if data is stale.
- Log every signal and order.

### 11. Improve Validation Standards

Current problem:

The project has walk-forward validation, which is good, but the simulator and data leakage issues make current results hard to trust.

Better design:

After fixing the simulator:

- Compare against buy-and-hold.
- Compare against cash.
- Compare against simple moving-average or RSI strategies.
- Report transaction-cost sensitivity.
- Report slippage sensitivity.
- Report performance by regime.
- Use out-of-sample assets.
- Avoid repeated tuning on the same test period.

### 12. Rename Or Reframe The Project

Current problem:

The name says HFT, but the implementation is candle-based AI trading research.

Why this matters:

Names shape expectations. Calling it HFT invites the wrong engineering standard and can hide the fact that the actual idea is an experimental model-driven allocation system.

Better framing:

- `trading-ai-lab`
- `ai-trading-research`
- `rl-trading-sandbox`
- `crypto-allocation-agents`

## Suggested Development Roadmap

### Phase 1: Make The Simulator Trustworthy

Do this before caring about model performance.

- Fix PPO reward timing.
- Fix end liquidation.
- Add trade logging.
- Add simulator tests.
- Make live/backtest observation features use the same helper.

Reasoning:

The simulator defines reality for the agent. If that reality is wrong, all training results are noise.

### Phase 2: Make Experiments Reproducible

- Clean requirements.
- Add README run commands.
- Add model path CLI options.
- Add model metadata manifests.
- Fix cache interval/date correctness.

Reasoning:

You need to know what data, feature set, model, and environment version produced each result.

### Phase 3: Remove Data Leakage

- Disable synthetic sentiment for real evaluation.
- Use only timestamp-valid external sentiment.
- Add a mode that trains with technical indicators only.

Reasoning:

If the model cannot beat baselines without leaked information, it does not have a deployable edge.

### Phase 4: Improve Backtesting Realism

- Add slippage.
- Add spread assumptions.
- Add exchange/trading-hour constraints for equities.
- Add market order fill assumptions.
- Add fees by asset class.
- Add benchmark strategies.

Reasoning:

Small simulated edges often disappear after realistic costs.

### Phase 5: Harden Live Trading

- Always reconcile state from Alpaca.
- Add risk limits.
- Add stale-data checks.
- Add structured logs.
- Add dry-run/paper/live mode separation.
- Add alerting.

Reasoning:

Live trading is not just model prediction. It is state management, risk management, and failure handling.

## Bottom Line

This project has a strong prototype skeleton: data ingestion, features, RL environment, NEAT, PPO, ensembles, backtests, visualizations, and Alpaca integration. The most valuable next step is not training bigger models. It is making the environment and data pipeline correct enough that model results mean something.

The highest-impact fixes are:

1. Fix `TradingEnv.step()` reward timing.
2. Add real trade logging.
3. Fix final liquidation accounting.
4. Make live observations match training observations.
5. Remove synthetic sentiment from serious evaluation.
6. Fix cache interval/date handling.
7. Add simulator tests.

#### CLI Flags for Ensemble Factory
- `--cash-penalty` (default 0.0): per‑step cash penalty.
- `--benchmark-weight` (default 0.1): weight for benchmark‑return reward in the environment.


Once those are done, the project becomes much more useful: it turns from a cool AI trading demo into a research system where backtest results can be interpreted with a straight face.

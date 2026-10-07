# 🛡️ AegisMM: Adaptive Microstructure Liquidity Provision Engine

An institutional-grade, high-frequency market making and liquidity provision engine combining **classical closed-form stochastic control (Avellaneda-Stoikov & Guéant-Tapia-Manziadi)** with **Deep Reinforcement Learning (PPO)** on Apple Silicon (MPS).

AegisMM is specifically engineered to overcome the two structural killers of retail market making:
1. **Adverse Selection (Informed Flow Toxicity)** — solved via real-time Order Flow Imbalance (OFI), micro-price skew, and dynamic spread widening.
2. **Negative Expected Value under Retail Fees** — solved by targeting maker-rebate environments (e.g. crypto perpetual CLOBs with negative maker fees) and asymmetric inventory-reducing quoting.

---

## 📐 Mathematical Architecture

AegisMM uses **Residual Policy Learning** (Silver et al.). Rather than forcing a neural network to learn basic arithmetic from scratch, the policy is anchored to the analytical closed-form solution of Avellaneda & Stoikov (2008) and learns non-linear micro-corrections based on order flow dynamics.

### 1. Classical Reservation Price & Optimal Spread
Given asset mid-price $S_t$, inventory $q_t$, risk aversion $\gamma$, volatility $\sigma$, and time horizon $\tau = T - t$:

$$\text{Reservation Price: } \quad r(s, q, t) = s - q \gamma \sigma^2 \tau$$

$$\text{Optimal Total Spread: } \quad s^*(t) = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$

Optimal bid and ask quotes:
$$p^a(t) = r(s, q, t) + \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$
$$p^b(t) = r(s, q, t) - \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$

### 2. High-Frequency Microstructure Metrics
* **Order Flow Imbalance (OFI)**: Cont, Kukanov & Stoikov (2014) discrete delta in bid/ask queue sizes and prices.
* **Micro-Price**: Volume-weighted mid-price:
  $$P_{micro} = \frac{Q_b P_a + Q_a P_b}{Q_b + Q_a}$$
* **Order Flow Toxicity Proxy**: Quantifies adverse selection probability from order book depth imbalance and signed trade flow.

### 3. PPO Residual Policy
The Actor-Critic network observes a 16-dimensional microstructure state vector and outputs continuous quote offsets:
$$\text{Bid}^* = p^b(t) + \delta_{bid}(\theta) \cdot \Delta_{\text{tick}}$$
$$\text{Ask}^* = p^a(t) + \delta_{ask}(\theta) \cdot \Delta_{\text{tick}}$$

---

## 🏛️ System Structure

```
aegis_mm/
├── src/
│   ├── config.py                 # Central typed configuration (fees, risk, models)
│   ├── core/
│   │   ├── orderbook.py          # Level-2 Order Book with fast NumPy caching
│   │   └── matching_engine.py    # FIFO queue-priority matching with queue ahead & adverse selection
│   ├── math/
│   │   ├── avellaneda_stoikov.py # Analytical AS & Guéant-Tapia-Manziadi equations
│   │   └── microstructure.py     # Streaming OFI, micro-price, volatility & toxicity estimators
│   ├── env/
│   │   └── market_making_env.py  # Gymnasium-compatible RL environment
│   ├── models/
│   │   ├── policy.py             # PyTorch Gaussian Actor-Critic with orthogonal init
│   │   └── agent.py              # PPO training loop with GAE & MPS acceleration
│   ├── risk/
│   │   └── risk_manager.py       # Hard circuit breakers: Max inventory, drawdown halt, one-sided quoting
│   ├── execution/
│   │   ├── base_exchange.py      # Abstract async exchange interface
│   │   ├── alpaca_exchange.py    # Alpaca live/paper adapter (equities & crypto)
│   │   └── hyperliquid_exchange.py # Hyperliquid crypto perps adapter (negative maker fees)
│   └── dashboard/
│       └── terminal_ui.py        # Rich interactive terminal dashboard
├── scripts/
│   ├── generate_market_data.py   # Jump-diffusion + Hawkes order book simulator
│   ├── train.py                  # PPO policy training pipeline
│   ├── backtest.py               # Head-to-head strategy benchmark (Naive vs. AS vs. AegisMM)
│   └── run_paper.py              # Live paper trading runner with dashboard
├── tests/                        # Full unit test suite (pytest)
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### 1. Run Unit Tests
```bash
.venv/bin/pytest tests/
```

### 2. Train the PPO Policy
```bash
.venv/bin/python scripts/train.py
```
Checkpoints are automatically saved to `checkpoints/best_policy.pt`.

### 3. Run Strategy Benchmark (Backtest)
```bash
.venv/bin/python scripts/backtest.py
```
Compares **Naive Fixed Spread**, **Classical Avellaneda-Stoikov**, and **AegisMM PPO**.

### 4. Run Live Paper Trading with Terminal Dashboard
```bash
.venv/bin/python scripts/run_paper.py --mode simulated --symbol BTC/USD
```

---

## 🛡️ Risk Management & Circuit Breakers

AegisMM enforces hard, non-negotiable institutional risk limits:
1. **Max Inventory Cap**: Clamps base asset exposure to `max_inventory`.
2. **One-Sided Quoting Skew**: When inventory exceeds 70% of max threshold, the engine immediately suppresses quotes on the increasing side and quotes exclusively on the reducing side.
3. **Max Drawdown Breaker**: If equity drops by $\ge 5\%$, all active orders are canceled and open inventory is flattened at market.
4. **Stale Order Canceler**: Automatically purges unrefreshed limit orders older than 10 seconds.

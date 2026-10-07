"""
Live Training Dashboard — Real-time web UI showing NEAT evolution progress.
Run: python3 live_dashboard.py
Then open http://localhost:5050 in your browser.
"""
import sys, os, json, time, threading, queue
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, Response, render_template_string
import neat

from data.fetch_prices import fetch_all
from data.sentiment_loader import get_sentiment_for_ticker
from features.state_builder import build_states
from env.trading_env import TradingEnv

app = Flask(__name__)
event_queue = queue.Queue(maxsize=500)

# ── Training state ──
training_state = {
    "status": "idle",
    "generation": 0,
    "max_generations": 100,
    "best_fitness": 0,
    "best_return": 0,
    "best_sharpe": 0,
    "best_trades": 0,
    "best_winrate": 0,
    "best_drawdown": 0,
    "best_alpha": 0,
    "buy_hold": 0,
    "gen_history": [],
    "genome_log": [],
    "trade_log": [],
    "portfolio_history": [],
    "ticker": "BTC-USD",
    "data_rows": 0,
    "pop_size": 80,
    "news_source": "Synthetic",
    "news_count": 0,
}

def push_event(event_type, data):
    msg = json.dumps({"type": event_type, **data})
    try:
        event_queue.put_nowait(msg)
    except queue.Full:
        try:
            event_queue.get_nowait()
            event_queue.put_nowait(msg)
        except:
            pass

def run_training(tickers=None, generations=100, start="2020-01-01", balance=10000.0,
                 allow_synthetic_sentiment=False):
    global training_state
    if tickers is None:
        tickers = ["BTC-USD"]
    if isinstance(tickers, str):
        tickers = [tickers]

    time.sleep(2)  # Let Flask bind first
    training_state["status"] = "loading_data"
    training_state["max_generations"] = generations
    training_state["ticker"] = ", ".join(tickers)
    push_event("status", {"message": f"Fetching price data for {len(tickers)} assets..."})

    # Fetch all tickers
    prices = fetch_all(tickers, start=start, force_refresh=False)
    push_event("status", {"message": "Loading sentiment data..."})

    if allow_synthetic_sentiment:
        push_event("status", {"message": "Synthetic sentiment enabled for bootstrap mode..."})
    else:
        push_event("status", {"message": "Synthetic sentiment disabled; missing sentiment will be neutral."})

    # Build state matrices for EACH ticker
    push_event("status", {"message": f"Computing features for {len(tickers)} assets..."})
    ticker_data = {}  # {ticker: (train_states, train_prices, train_dates, test_states, test_prices, test_dates)}

    for tk in tickers:
        try:
            tk_sent, sent_source = get_sentiment_for_ticker(
                prices, ticker=tk, allow_synthetic=allow_synthetic_sentiment
            )
            training_state["news_source"] = sent_source

            state_matrix, dates, close_prices = build_states(prices, sentiment_df=tk_sent, ticker=tk)

            # 80/20 split
            split_idx = int(len(close_prices) * 0.8)
            ticker_data[tk] = {
                "train": (state_matrix[:split_idx], close_prices[:split_idx], dates[:split_idx]),
                "test": (state_matrix[split_idx:], close_prices[split_idx:], dates[split_idx:]),
            }
            push_event("status", {"message": f"  ✅ {tk}: {len(close_prices)} candles → {split_idx} train / {len(close_prices)-split_idx} test"})
        except Exception as e:
            push_event("status", {"message": f"  ❌ {tk}: {e}"})

    if not ticker_data:
        push_event("status", {"message": "No valid tickers! Aborting."})
        training_state["status"] = "error"
        return

    # Use first ticker for primary display
    primary_ticker = list(ticker_data.keys())[0]
    primary_train = ticker_data[primary_ticker]["train"]
    training_state["data_rows"] = sum(len(d["train"][1]) for d in ticker_data.values())
    training_state["status"] = "training"
    push_event("status", {"message": f"Training on {len(ticker_data)} assets × {generations} generations..."})

    config_path = os.path.join(PROJECT_ROOT, "agents", "config-trader")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    pop = neat.Population(config)
    training_state["pop_size"] = config.pop_size

    best_genome_overall = None
    best_fitness_overall = float("-inf")

    def eval_single(net, states, prices_arr, dates_arr, bal):
        """Evaluate a net on a single asset, return (fitness, metrics, env)."""
        env = TradingEnv(states, prices_arr, dates_arr, initial_balance=bal)
        obs, _ = env.reset()
        for _ in range(len(prices_arr) - 1):
            output = net.activate(obs.tolist())
            action = int(np.argmax(output))
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        m = env.get_final_metrics()
        fitness = m["total_return_pct"]
        fitness += min(20, max(0, m["sharpe_ratio"] * 5))
        fitness -= m["max_drawdown_pct"] * 0.3
        if m["total_trades"] < 3:
            fitness -= 50
        return fitness, m, env

    def eval_genomes(genomes, cfg):
        nonlocal best_genome_overall, best_fitness_overall
        gen_num = len(training_state["gen_history"]) + 1
        training_state["generation"] = gen_num
        gen_fitnesses = []
        genome_details = []

        for idx, (gid, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)

            # Evaluate across ALL tickers and average
            ticker_fitnesses = []
            ticker_metrics = []
            best_env = None
            best_m = None
            for tk, tdata in ticker_data.items():
                train_s, train_p, train_d = tdata["train"]
                f, m, env = eval_single(net, train_s, train_p, train_d, balance)
                ticker_fitnesses.append(f)
                ticker_metrics.append(m)
                if best_m is None or f > (best_m.get("_fitness", float("-inf"))):
                    best_m = m
                    best_m["_fitness"] = f
                    best_env = env

            # Average fitness across all assets
            fitness = float(np.mean(ticker_fitnesses))
            genome.fitness = fitness
            gen_fitnesses.append(fitness)

            # Use averaged metrics for display
            avg_return = np.mean([m["total_return_pct"] for m in ticker_metrics])
            avg_sharpe = np.mean([m["sharpe_ratio"] for m in ticker_metrics])
            avg_winrate = np.mean([m["win_rate"] for m in ticker_metrics])
            avg_trades = int(np.mean([m["total_trades"] for m in ticker_metrics]))
            avg_drawdown = np.mean([m["max_drawdown_pct"] for m in ticker_metrics])

            detail = {
                "id": gid, "fitness": round(fitness, 2),
                "return_pct": round(avg_return, 2),
                "trades": avg_trades, "winrate": round(avg_winrate * 100, 1),
                "sharpe": round(avg_sharpe, 3),
                "drawdown": round(avg_drawdown, 2),
            }
            genome_details.append(detail)

            if fitness > best_fitness_overall:
                best_fitness_overall = fitness
                best_genome_overall = genome
                training_state["best_fitness"] = round(fitness, 2)
                training_state["best_return"] = detail["return_pct"]
                training_state["best_sharpe"] = detail["sharpe"]
                training_state["best_trades"] = detail["trades"]
                training_state["best_winrate"] = detail["winrate"]
                training_state["best_drawdown"] = detail["drawdown"]
                training_state["best_alpha"] = round(np.mean([m["alpha_pct"] for m in ticker_metrics]), 2)
                training_state["buy_hold"] = round(np.mean([m["buy_hold_return_pct"] for m in ticker_metrics]), 2)
                if best_env:
                    training_state["trade_log"] = [{**t, "date": str(t["date"])[:10]} for t in best_env.trade_log[-30:]]
                    training_state["portfolio_history"] = [round(v, 2) for v in best_env.portfolio_history[::max(1, len(best_env.portfolio_history)//200)]]

            if idx % 10 == 0:
                push_event("genome", {"gen": gen_num, "idx": idx + 1,
                                      "total": len(genomes), "detail": detail})

        gen_stat = {
            "gen": gen_num, "best": round(max(gen_fitnesses), 2),
            "avg": round(np.mean(gen_fitnesses), 2),
            "worst": round(min(gen_fitnesses), 2),
            "std": round(np.std(gen_fitnesses), 2),
        }
        training_state["gen_history"].append(gen_stat)
        training_state["genome_log"] = sorted(genome_details, key=lambda x: -x["fitness"])[:10]

        push_event("generation", {**gen_stat, "best_overall": training_state["best_fitness"],
                                   "best_return": training_state["best_return"],
                                   "portfolio": training_state["portfolio_history"],
                                   "trades": training_state["trade_log"]})

    pop.run(eval_genomes, generations)

    # Save best
    import pickle
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "checkpoints", "best_genome.pkl"), "wb") as f:
        pickle.dump(best_genome_overall, f)

    # Test on out-of-sample (primary ticker)
    if best_genome_overall:
        test_s, test_p, test_d = ticker_data[primary_ticker]["test"]
        net = neat.nn.FeedForwardNetwork.create(best_genome_overall, config)
        env = TradingEnv(test_s, test_p, test_d, initial_balance=balance)
        obs, _ = env.reset()
        for _ in range(len(test_p) - 1):
            output = net.activate(obs.tolist())
            obs, _, done, _, _ = env.step(int(np.argmax(output)))
            if done:
                break
        test_m = env.get_final_metrics()
        push_event("complete", {
            "test_return": round(test_m["total_return_pct"], 2),
            "test_sharpe": round(test_m["sharpe_ratio"], 3),
            "test_trades": test_m["total_trades"],
            "test_winrate": round(test_m["win_rate"] * 100, 1),
            "test_alpha": round(test_m["alpha_pct"], 2),
            "test_drawdown": round(test_m["max_drawdown_pct"], 2),
        })

    training_state["status"] = "complete"

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading AI — Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e17;--card:#111827;--card2:#1a2332;--border:#1e293b;--text:#e2e8f0;--dim:#64748b;--green:#10b981;--red:#ef4444;--blue:#3b82f6;--purple:#8b5cf6;--amber:#f59e0b;--cyan:#06b6d4}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.header{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100;backdrop-filter:blur(20px)}
.header h1{font-size:20px;font-weight:700;background:linear-gradient(135deg,var(--cyan),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .status{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim)}
.header .status .dot{width:8px;height:8px;border-radius:50%;background:var(--dim);animation:pulse 2s infinite}
.header .status .dot.active{background:var(--green)}
.header .status .dot.training{background:var(--amber);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;padding:16px 24px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--blue),transparent)}
.stat-card .label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:4px}
.stat-card .value{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700}
.stat-card .value.text-md{font-size:16px;padding-top:4px}
.stat-card .sub{font-size:11px;color:var(--dim);margin-top:2px}
.pos{color:var(--green)}.neg{color:var(--red)}
.main{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:0 24px 16px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;overflow:hidden}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.panel h2::before{content:'';width:3px;height:14px;border-radius:2px;background:var(--purple)}
.chart-container{position:relative;height:250px}
.feed{max-height:420px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.feed::-webkit-scrollbar{width:4px}
.feed::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.feed-item{padding:8px 10px;border-bottom:1px solid #1e293b22;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.6;display:flex;gap:8px;transition:background .2s}
.feed-item:hover{background:#ffffff06}
.feed-item .time{color:var(--dim);min-width:52px}
.feed-item .tag{padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600}
.tag-gen{background:#3b82f620;color:var(--blue)}
.tag-buy{background:#10b98120;color:var(--green)}
.tag-sell{background:#ef444420;color:var(--red)}
.tag-best{background:#f59e0b20;color:var(--amber)}
.tag-info{background:#8b5cf620;color:var(--purple)}
.bottom-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 24px 24px}
.genome-table{width:100%;font-size:12px;font-family:'JetBrains Mono',monospace}
.genome-table th{text-align:left;color:var(--dim);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.genome-table td{padding:6px 8px;border-bottom:1px solid #1e293b33}
.genome-table tr:first-child td{color:var(--amber);font-weight:600}
.progress-bar{width:100%;height:4px;background:var(--border);border-radius:2px;margin-top:4px;overflow:hidden}
.progress-bar .fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));border-radius:2px;transition:width .5s ease}
.trade-row{display:grid;grid-template-columns:80px 50px 1fr 80px;gap:4px;padding:4px 0;font-size:11px;font-family:'JetBrains Mono',monospace;border-bottom:1px solid #1e293b22}
</style>
</head>
<body>
<div class="header">
  <h1>🧠 Trading AI — Live Evolution</h1>
  <div class="status">
    <div class="dot" id="statusDot"></div>
    <span id="statusText">Connecting...</span>
  </div>
</div>

<div class="grid">
  <div class="stat-card"><div class="label">Generation</div><div class="value" id="genNum">0</div><div class="sub"><span id="genMax">/ 100</span><div class="progress-bar"><div class="fill" id="genBar" style="width:0%"></div></div></div></div>
  <div class="stat-card"><div class="label">Best Return</div><div class="value" id="bestReturn">—</div><div class="sub">Buy&Hold: <span id="buyHold">—</span></div></div>
  <div class="stat-card"><div class="label">Best Fitness</div><div class="value" id="bestFitness">—</div><div class="sub">Sharpe: <span id="bestSharpe">—</span></div></div>
  <div class="stat-card"><div class="label">Best Win Rate</div><div class="value" id="bestWinrate">—</div><div class="sub">Trades: <span id="bestTrades">—</span> | DD: <span id="bestDD">—</span></div></div>
  <div class="stat-card"><div class="label">News Source</div><div class="value text-md" id="newsSource" style="color:var(--cyan)">Loading...</div><div class="sub">Articles: <span id="newsCount">0</span></div></div>
</div>

<div class="main">
  <div class="panel">
    <h2>Evolution Progress</h2>
    <div class="chart-container"><canvas id="evoChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>Live Feed</h2>
    <div class="feed" id="feed"></div>
  </div>
</div>

<div class="bottom-grid">
  <div class="panel">
    <h2>Best Portfolio Equity Curve</h2>
    <div class="chart-container"><canvas id="eqChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>Top Genomes This Generation</h2>
    <div style="max-height:250px;overflow-y:auto">
      <table class="genome-table">
        <thead><tr><th>#</th><th>Fitness</th><th>Return</th><th>Sharpe</th><th>Trades</th><th>Win%</th></tr></thead>
        <tbody id="genomeTable"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const feed = document.getElementById('feed');
const evoCtx = document.getElementById('evoChart').getContext('2d');
const eqCtx = document.getElementById('eqChart').getContext('2d');

const evoChart = new Chart(evoCtx, {
  type:'line',
  data:{labels:[],datasets:[
    {label:'Best',data:[],borderColor:'#f59e0b',borderWidth:2,pointRadius:0,tension:.3},
    {label:'Average',data:[],borderColor:'#3b82f6',borderWidth:1.5,pointRadius:0,tension:.3},
    {label:'Worst',data:[],borderColor:'#ef4444',borderWidth:1,pointRadius:0,tension:.3,borderDash:[4,4]}
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},
    scales:{x:{grid:{color:'#1e293b'},ticks:{color:'#64748b',font:{size:10}}},
            y:{grid:{color:'#1e293b'},ticks:{color:'#64748b',font:{size:10}}}},
    plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}}}
});

const eqChart = new Chart(eqCtx, {
  type:'line',
  data:{labels:[],datasets:[{label:'Portfolio $',data:[],borderColor:'#10b981',borderWidth:2,pointRadius:0,tension:.2,fill:{target:'origin',above:'#10b98110'}}]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},
    scales:{x:{display:false},y:{grid:{color:'#1e293b'},ticks:{color:'#64748b',font:{size:10},callback:v=>'$'+v.toLocaleString()}}},
    plugins:{legend:{display:false}}}
});

function addFeed(tag, tagClass, msg) {
  const now = new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const div = document.createElement('div');
  div.className = 'feed-item';
  div.innerHTML = `<span class="time">${now}</span><span class="tag ${tagClass}">${tag}</span><span>${msg}</span>`;
  feed.prepend(div);
  if (feed.children.length > 200) feed.removeChild(feed.lastChild);
}

function colorVal(v, suffix='%') {
  const n = parseFloat(v);
  const cls = n > 0 ? 'pos' : n < 0 ? 'neg' : '';
  return `<span class="${cls}">${n > 0 ? '+' : ''}${v}${suffix}</span>`;
}

const evtSource = new EventSource('/stream');
evtSource.onopen = () => {
  document.getElementById('statusDot').className = 'dot active';
  document.getElementById('statusText').textContent = 'Connected';
};
evtSource.onerror = () => {
  document.getElementById('statusDot').className = 'dot';
  document.getElementById('statusText').textContent = 'Reconnecting...';
};
evtSource.onmessage = (e) => {
  const d = JSON.parse(e.data);

  if (d.type === 'status') {
    document.getElementById('statusText').textContent = d.message;
    document.getElementById('statusDot').className = 'dot training';
    addFeed('SYS', 'tag-info', d.message);
  }

  if (d.type === 'genome') {
    const det = d.detail;
    addFeed(`G${d.gen}`, 'tag-gen', `Genome ${d.idx}/${d.total} → fitness ${det.fitness} | return ${det.return_pct}% | ${det.trades} trades`);
  }

  if (d.type === 'generation') {
    document.getElementById('genNum').textContent = d.gen;
    document.getElementById('genBar').style.width = (d.gen / parseInt(document.getElementById('genMax').textContent.replace('/ ','')) * 100) + '%';
    document.getElementById('bestFitness').innerHTML = colorVal(d.best_overall, '');
    document.getElementById('bestReturn').innerHTML = colorVal(d.best_return, '%');

    evoChart.data.labels.push(d.gen);
    evoChart.data.datasets[0].data.push(d.best);
    evoChart.data.datasets[1].data.push(d.avg);
    evoChart.data.datasets[2].data.push(d.worst);
    evoChart.update();

    if (d.portfolio && d.portfolio.length) {
      eqChart.data.labels = d.portfolio.map((_,i)=>i);
      eqChart.data.datasets[0].data = d.portfolio;
      const last = d.portfolio[d.portfolio.length-1];
      eqChart.data.datasets[0].borderColor = last >= 10000 ? '#10b981' : '#ef4444';
      eqChart.update();
    }

    addFeed(`GEN ${d.gen}`, 'tag-best', `Best: ${d.best} | Avg: ${d.avg} | Worst: ${d.worst}`);

    if (d.trades) {
      d.trades.slice(-3).forEach(t => {
        const tag = t.action === 'BUY' ? 'tag-buy' : 'tag-sell';
        const pnl = t.pnl ? ` P&L: $${t.pnl.toFixed(2)}` : '';
        const sent = t.sentiment !== undefined ? ` (Sent: ${t.sentiment > 0 ? '+' : ''}${t.sentiment.toFixed(2)})` : '';
        addFeed(t.action, tag, `${String(t.date).slice(0,10)} @ $${t.price.toFixed(2)}${sent}${pnl}`);
      });
    }
  }

  if (d.type === 'state') {
    document.getElementById('genNum').textContent = d.generation;
    document.getElementById('genMax').textContent = '/ ' + d.max_generations;
    document.getElementById('bestFitness').innerHTML = colorVal(d.best_fitness, '');
    document.getElementById('bestReturn').innerHTML = colorVal(d.best_return, '%');
    document.getElementById('bestSharpe').textContent = d.best_sharpe;
    document.getElementById('bestTrades').textContent = d.best_trades;
    document.getElementById('bestWinrate').innerHTML = d.best_winrate + '%';
    document.getElementById('bestDD').textContent = d.best_drawdown + '%';
    document.getElementById('buyHold').innerHTML = colorVal(d.buy_hold, '%');
    
    if (d.news_source) {
      document.getElementById('newsSource').textContent = d.news_source;
      document.getElementById('newsCount').textContent = (d.news_count || 0).toLocaleString();
    }
    if (d.genome_log) {
      const tb = document.getElementById('genomeTable');
      tb.innerHTML = d.genome_log.map((g,i) =>
        `<tr><td>${i+1}</td><td>${g.fitness}</td><td class="${g.return_pct>=0?'pos':'neg'}">${g.return_pct}%</td><td>${g.sharpe}</td><td>${g.trades}</td><td>${g.winrate}%</td></tr>`
      ).join('');
    }
    if (d.status === 'training') {
      document.getElementById('statusDot').className = 'dot training';
      document.getElementById('statusText').textContent = `Training Gen ${d.generation}/${d.max_generations}`;
    }
    if (d.gen_history) {
      evoChart.data.labels = d.gen_history.map(g=>g.gen);
      evoChart.data.datasets[0].data = d.gen_history.map(g=>g.best);
      evoChart.data.datasets[1].data = d.gen_history.map(g=>g.avg);
      evoChart.data.datasets[2].data = d.gen_history.map(g=>g.worst);
      evoChart.update();
    }
  }

  if (d.type === 'complete') {
    document.getElementById('statusDot').className = 'dot active';
    document.getElementById('statusText').textContent = 'Training Complete!';
    addFeed('DONE', 'tag-best', `Test Return: ${d.test_return}% | Alpha: ${d.test_alpha}% | Sharpe: ${d.test_sharpe} | Win: ${d.test_winrate}%`);
  }
};

// Poll full state every 2s
setInterval(async () => {
  try {
    const r = await fetch('/state');
    const d = await r.json();
    d.type = 'state';
    evtSource.dispatchEvent(new MessageEvent('message', {data: JSON.stringify(d)}));
  } catch(e) {}
}, 2000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/state")
def get_state():
    return json.dumps(training_state)

@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                msg = event_queue.get(timeout=5)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type':'ping'})}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="Single ticker (legacy)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Multiple tickers for multi-asset training (e.g. --tickers BTC-USD ETH-USD SOL-USD)")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--use-synthetic-sentiment", action="store_true",
                        help="Opt into price-derived synthetic sentiment for bootstrapping only")
    args = parser.parse_args()

    # Resolve tickers list
    if args.tickers:
        tickers = args.tickers
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = ["BTC-USD"]

    t = threading.Thread(
        target=run_training,
        args=(tickers, args.generations, args.start, args.balance, args.use_synthetic_sentiment),
        daemon=True,
    )
    t.start()

    print(f"\n  🌐 Dashboard: http://localhost:{args.port}")
    print(f"  📊 Training {', '.join(tickers)} for {args.generations} generations\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

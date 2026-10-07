document.addEventListener('DOMContentLoaded', () => {
    const assetSelect = document.getElementById('asset-select');
    const daysSelect = document.getElementById('days-select');
    const daysVal = document.getElementById('days-val');
    const refreshBtn = document.getElementById('refresh-btn');
    const backtestBtn = document.getElementById('backtest-btn');
    
    // UI Elements
    const statusText = document.getElementById('system-status');
    const valPrice = document.getElementById('val-price');
    const valRsi = document.getElementById('val-rsi');
    const valMacd = document.getElementById('val-macd');
    const valSentiment = document.getElementById('val-sentiment');
    const consensusVal = document.getElementById('consensus-val');
    const consensusBar = document.getElementById('consensus-bar');
    const convictionLabel = document.getElementById('conviction-label');
    const councilGrid = document.getElementById('council-grid');
    const sentienceOutput = document.getElementById('sentience-output'); // New Element
    
    let priceChart = null;
    
    // Setup listeners
    daysSelect.addEventListener('input', (e) => {
        daysVal.textContent = e.target.value;
    });
    
    refreshBtn.addEventListener('click', () => {
        fetchState();
    });
    
    backtestBtn.addEventListener('click', () => {
        runBacktest();
    });
    
    assetSelect.addEventListener('change', () => {
        fetchState();
        document.getElementById('backtest-section').style.display = 'none';
    });

    // Color helpers
    const getColorClass = (val, thresholds) => {
        if (val > thresholds[1]) return 'val-green';
        if (val < thresholds[0]) return 'val-red';
        return '';
    };

    const getMonologue = (alloc, rsi, macd, sentiment) => {
        if (alloc > 0.7) {
            if (rsi < 40) return "RSI is oversold. Scaling in heavily to catch the bounce.";
            if (macd > 0.5) return "MACD momentum is strongly positive. Riding the trend.";
            if (sentiment > 0.2) return "News sentiment is highly positive. Front-running the retail volume.";
            return "Mathematical edge detected. Aggressive long.";
        } else if (alloc < 0.3) {
            if (rsi > 60) return "RSI is overbought. Locking in profits and reducing risk.";
            if (macd < -0.5) return "MACD crossover indicates bearish momentum. Exiting position.";
            if (sentiment < -0.2) return "Negative NLP sentiment spike. Protecting capital.";
            return "Risk metrics elevated. Rotating into cash for capital preservation.";
        } else {
            if (Math.abs(macd) < 0.2) return "Volatility compressing. Waiting for a clear directional breakout.";
            if (rsi > 40 && rsi < 60) return "Market chopping in a tight range. Maintaining a neutral hedge.";
            return "Conflicting momentum and sentiment signals. Balancing risk at 50%.";
        }
    };

    // Sentience Engine Logic
    async function fetchBrainState() {
        try {
            const res = await fetch('/api/brain_state');
            const data = await res.json();
            if (sentienceOutput) {
                sentienceOutput.textContent = `> ${data.thought}`;
            }
        } catch (e) {
            console.error("Brain sync error:", e);
        }
    }

    async function fetchState() {
        statusText.textContent = "Fetching live market data...";
        
        try {
            const ticker = assetSelect.value;
            const days = daysSelect.value;
            const res = await fetch(`/api/state?ticker=${ticker}&days=${days}`);
            
            if (!res.ok) throw new Error("Failed to fetch state");
            
            const data = await res.json();
            
            // Update Tech Panel
            valPrice.textContent = `$${data.current_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            
            valRsi.textContent = data.indicators.rsi.toFixed(1);
            valRsi.className = `tech-val ${getColorClass(data.indicators.rsi, [30, 70])}`;
            
            valMacd.textContent = data.indicators.macd.toFixed(2);
            valMacd.className = `tech-val ${getColorClass(data.indicators.macd, [0, 0])}`;
            
            valSentiment.textContent = data.indicators.sentiment.toFixed(2);
            valSentiment.className = `tech-val ${getColorClass(data.indicators.sentiment, [-0.1, 0.1])}`;
            
            // Update Consensus
            const consensusPct = data.consensus * 100;
            consensusVal.textContent = `${consensusPct.toFixed(1)}%`;
            consensusBar.style.width = `${consensusPct}%`;
            
            if (consensusPct >= 70) {
                convictionLabel.textContent = "🟢 HIGH (BUY)";
                consensusVal.style.background = "linear-gradient(to right, #00FFCC, #ffffff)";
                consensusVal.style.webkitBackgroundClip = "text";
            } else if (consensusPct <= 30) {
                convictionLabel.textContent = "🔴 HIGH (SELL)";
                consensusVal.style.background = "linear-gradient(to right, #EF4444, #ffffff)";
                consensusVal.style.webkitBackgroundClip = "text";
            } else {
                convictionLabel.textContent = "🟡 LOW (HOLD)";
                consensusVal.style.background = "linear-gradient(to right, #F59E0B, #ffffff)";
                consensusVal.style.webkitBackgroundClip = "text";
            }
            
            // Update Grid
            councilGrid.innerHTML = '';
            data.allocations.forEach(agent => {
                const allocPct = agent.allocation * 100;
                let colorClass = "";
                if (allocPct > 60) colorClass = "val-green";
                else if (allocPct < 40) colorClass = "val-red";
                
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.innerHTML = `
                    <div class="agent-header">
                        <div class="agent-name">${agent.name}</div>
                        <div class="agent-alloc ${colorClass}">${allocPct.toFixed(0)}%</div>
                    </div>
                    <div class="agent-monologue">
                        "${getMonologue(agent.allocation, data.indicators.rsi, data.indicators.macd, data.indicators.sentiment)}"
                    </div>
                `;
                councilGrid.appendChild(card);
            });
            
            statusText.textContent = "Live data synced";
            
        } catch (e) {
            console.error(e);
            statusText.textContent = "Error fetching data";
        }
    }

    async function runBacktest() {
        backtestBtn.innerHTML = `<span class="icon">⌛</span> Crunching...`;
        backtestBtn.disabled = true;
        
        try {
            const ticker = assetSelect.value;
            const days = daysSelect.value;
            
            const res = await fetch(`/api/backtest`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ticker, days})
            });
            
            if (!res.ok) throw new Error("Backtest failed");
            
            const data = await res.json();
            
            document.getElementById('backtest-section').style.display = 'flex';
            
            // Render Table
            const tbody = document.querySelector('#leaderboard-table tbody');
            tbody.innerHTML = '';
            data.results.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.name}</td>
                    <td>${row.type}</td>
                    <td class="${row.return_pct > 0 ? 'val-green' : 'val-red'}">${row.return_pct.toFixed(2)}%</td>
                    <td class="${row.alpha_pct > 0 ? 'val-green' : 'val-red'}">${row.alpha_pct.toFixed(2)}%</td>
                `;
                tbody.appendChild(tr);
            });
            
            renderChart(data.chart_data);
            
            setTimeout(() => {
                document.getElementById('backtest-section').scrollIntoView({behavior: 'smooth'});
            }, 100);
            
        } catch (e) {
            console.error(e);
            alert("Error running backtest: " + e.message);
        } finally {
            backtestBtn.innerHTML = `<span class="icon">📈</span> Run Backtest`;
            backtestBtn.disabled = false;
        }
    }
    
    function renderChart(chartData) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        if (priceChart) priceChart.destroy();
        
        const buyPoints = chartData.trades.filter(t => t.type === 'BUY').map(t => ({x: t.date, y: t.price}));
        const sellPoints = chartData.trades.filter(t => t.type === 'SELL').map(t => ({x: t.date, y: t.price}));
        
        Chart.defaults.color = '#94A3B8';
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.dates,
                datasets: [
                    { label: 'Price', data: chartData.prices, borderColor: 'rgba(255, 255, 255, 0.5)', pointRadius: 0 },
                    { type: 'scatter', label: 'Buy', data: buyPoints, backgroundColor: '#10B981', pointRadius: 8 },
                    { type: 'scatter', label: 'Sell', data: sellPoints, backgroundColor: '#EF4444', pointRadius: 8 }
                ]
            }
        });
    }

    // Initialize
    fetchState();
    fetchBrainState(); // Initial run
    
    setInterval(fetchState, 30000); // UI Refresh
    setInterval(fetchBrainState, 10000); // "Sentient" thought refresh
});

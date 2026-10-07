import streamlit as st
import pandas as pd
import time
import os
from alpaca.trading.client import TradingClient

# --- CONFIGURATION ---
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

st.set_page_config(page_title="AI Ensemble Command Center", layout="wide", page_icon="🤖")

st.title("🤖 10-Agent AI Ensemble Command Center")
st.markdown("Live tracking of Proximal Policy Optimization (PPO) neural network votes and portfolio execution.")

@st.cache_resource
def get_alpaca_client():
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

trading_client = get_alpaca_client()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🏦 Portfolio Status")
    balance_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("🧠 Neural Network Voting (1-Min Ticks)")
    majority_placeholder = st.empty()
    votes_placeholder = st.empty()

# Add a placeholder for our new history graph at the bottom
st.markdown("---")
st.subheader("📊 Ensemble Consensus History (Last 60 Minutes)")
history_placeholder = st.empty()

vote_map = {
    "0": {"label": "HOLD ⏸", "color": "gray"},
    "1": {"label": "BUY 🟢", "color": "green"},
    "2": {"label": "SELL 🔴", "color": "red"}
}

while True:
    try:
        # 1. Update Portfolio Balance
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        portfolio_val = float(account.portfolio_value)
        
        with balance_placeholder.container():
            st.metric(label="Paper Portfolio Value", value=f"${portfolio_val:,.2f}")
            st.metric(label="Available Buying Power", value=f"${buying_power:,.2f}")
        
        # 2. Read latest individual votes
        if os.path.exists("live_votes.log"):
            with open("live_votes.log", "r") as f:
                data = f.read().strip().split(",")
            
            if len(data) >= 11:
                votes = data[:-1]
                majority = data[-1]
                
                with majority_placeholder.container():
                    st.markdown(f"### Majority Decision: <span style='color:{vote_map[majority]['color']}'>{vote_map[majority]['label']}</span>", unsafe_allow_html=True)
                
                with votes_placeholder.container():
                    grid_cols = st.columns(5)
                    for i, vote in enumerate(votes):
                        col_index = i % 5
                        with grid_cols[col_index]:
                            st.markdown(f"**Agent {i+1}**<br><span style='color:{vote_map[vote]['color']}'>{vote_map[vote]['label']}</span>", unsafe_allow_html=True)
                            if i == 4:
                                st.write("---")
            
            with status_placeholder.container():
                st.success("Live Data Feed Active 🟢")
        else:
            with status_placeholder.container():
                st.warning("Waiting for the AI to make its first move...")

        # 3. GRAPH THE HISTORY
        if os.path.exists("vote_history.csv"):
            try:
                # Read the CSV log
                cols = ["Time"] + [f"Agent_{i}" for i in range(1, 11)] + ["Majority"]
                df = pd.read_csv("vote_history.csv", names=cols)
                
                # Keep only the last 60 ticks so the graph doesn't get crushed
                df = df.tail(60)
                
                # Count the votes for each category per minute
                agent_cols = [f"Agent_{i}" for i in range(1, 11)]
                df['Buy 🟢'] = (df[agent_cols] == 1).sum(axis=1)
                df['Hold ⏸'] = (df[agent_cols] == 0).sum(axis=1)
                df['Sell 🔴'] = (df[agent_cols] == 2).sum(axis=1)
                
                # Format for Streamlit charting
                chart_data = df.set_index("Time")[['Buy 🟢', 'Hold ⏸', 'Sell 🔴']]
                
                # Draw the stacked bar chart
                with history_placeholder.container():
                    st.bar_chart(chart_data, color=["#10B981", "#6B7280", "#EF4444"])
            except Exception as e:
                pass # Silently pass if reading the file at the exact millisecond it's being written
                
    except Exception as e:
        with status_placeholder.container():
            st.error(f"Connection Error: {e}")
            
    time.sleep(2)

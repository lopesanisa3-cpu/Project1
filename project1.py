"""
Streamlit Stock Dashboard
- Single-file Streamlit app to display world & Indian indices with basic analysis
- Requirements: streamlit, yfinance, pandas, numpy, plotly

How to run locally:
1. pip install -r requirements.txt
2. streamlit run streamlit_stock_dashboard.py

How to deploy on Streamlit Cloud:
- Push this file and requirements.txt to a GitHub repo
- Connect repo to Streamlit Cloud and deploy

This file includes simple CSS injected via st.markdown.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# -----------------------------
# Page config & CSS
# -----------------------------
st.set_page_config(page_title="Global & Indian Indices Dashboard", layout="wide")

# Minimal CSS to tidy UI (inserted via Python)
st.markdown(
    """
    <style>
    .reportview-container .main header {visibility: hidden}
    .css-1v3fvcr {padding-top: 0rem;} /* compact header spacing */
    .stApp { background: linear-gradient(180deg, #f7f9fc, #ffffff);} 
    .sidebar .sidebar-content {background-image: linear-gradient(#f8fafc, #ffffff);} 
    .card {background: white; border-radius: 10px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);} 
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Tickers configuration
# -----------------------------
WORLD_INDICES = {
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'Nasdaq Composite': '^IXIC',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI',
    'Shanghai Composite': '000001.SS',
    'ASX 200': '^AXJO',
}

INDIA_INDICES = {
    'NIFTY 50': '^NSEI',
    'BSE SENSEX': '^BSESN',
    'NIFTY Bank': '^NSEBANK',
}

ALL = {**WORLD_INDICES, **INDIA_INDICES}

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end, interval='1d'):
    """Download price data for tickers and return adjusted close DataFrame."""
    data = yf.download(list(tickers), start=start, end=end, interval=interval, progress=False)
    if data.empty:
        return pd.DataFrame()
    # yfinance returns multiindex if multiple columns. We want Adj Close
    if ('Adj Close' in data.columns.levels[0]) if hasattr(data.columns, 'levels') else ('Adj Close' in data.columns):
        adj = data['Adj Close'].copy()
    elif 'Adj Close' in data.columns:
        adj = data['Adj Close']
    else:
        # fallback to Close
        adj = data['Close']
    # For single ticker, ensure DataFrame
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    adj.columns = [tickers[i] if isinstance(tickers, list) and i < len(tickers) else c for i, c in enumerate(adj.columns)]
    return adj


def compute_returns(df):
    return df.pct_change().dropna()


def moving_average(df, window=50):
    return df.rolling(window).mean()


def compute_drawdown(series):
    cummax = series.cummax()
    drawdown = (series / cummax) - 1
    return drawdown

# -----------------------------
# UI - sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")
    default_selection = ['S&P 500', 'NIFTY 50']
    indices = st.multiselect("Select indices to analyze", options=list(ALL.keys()), default=default_selection)
    start_date = st.date_input("Start date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End date", value=pd.Timestamp.today())
    interval = st.selectbox("Interval", options=['1d', '1wk', '1mo'], index=0)
    ma1 = st.number_input("Short MA window", min_value=5, max_value=200, value=50, step=5)
    ma2 = st.number_input("Long MA window", min_value=20, max_value=400, value=200, step=10)
    show_corr = st.checkbox("Show correlation heatmap", value=True)
    show_drawdown = st.checkbox("Show drawdown", value=True)
    download_csv = st.checkbox("Provide CSV download", value=True)

# Validate selection
if not indices:
    st.warning("Please select at least one index from the sidebar.")
    st.stop()

selected_tickers = [ALL[name] for name in indices]

# -----------------------------
# Fetch data
# -----------------------------
with st.spinner("Fetching data..."):
    prices = fetch_data(selected_tickers, start=start_date, end=end_date + pd.Timedelta(days=1), interval=interval)

if prices.empty:
    st.error("No data returned for the selected tickers / date range. Try expanding the range or picking different indices.")
    st.stop()

# Rename columns to friendly names
col_map = {ALL[k]: k for k in ALL}
prices = prices.rename(columns=col_map)

# -----------------------------
# Main layout
# -----------------------------
st.title("🌏 Global & Indian Indices Dashboard")
st.markdown("A simple Streamlit dashboard showing prices, returns, moving averages, volatility, correlations, and drawdowns.")

# Price chart
st.subheader("Price chart")
fig = go.Figure()
for col in prices.columns:
    fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode='lines', name=col))
fig.update_layout(height=400, margin=dict(l=40,r=20,t=40,b=20))
st.plotly_chart(fig, use_container_width=True)

# Moving averages overlay for first selected
st.subheader("Moving averages (per selected index)")
cols = st.columns(len(prices.columns))
for i, col in enumerate(prices.columns):
    with cols[i]:
        st.markdown(f"**{col}**")
        ma_short = moving_average(prices[col], ma1)
        ma_long = moving_average(prices[col], ma2)
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=prices.index, y=prices[col], name=col))
        fig_ma.add_trace(go.Scatter(x=ma_short.index, y=ma_short, name=f"MA {ma1}"))
        fig_ma.add_trace(go.Scatter(x=ma_long.index, y=ma_long, name=f"MA {ma2}"))
        fig_ma.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=10))
        st.plotly_chart(fig_ma, use_container_width=True)

# Returns & Volatility
st.subheader("Returns & Volatility")
returns = compute_returns(prices)
ann_factor = {'1d':252, '1wk':52, '1mo':12}[interval]
annualized_return = returns.mean() * ann_factor
annualized_vol = returns.std() * np.sqrt(ann_factor)
summary = pd.DataFrame({'Annualized Return': annualized_return, 'Annualized Volatility': annualized_vol})
summary['Sharpe (0 rf)'] = summary['Annualized Return'] / summary['Annualized Volatility']
st.dataframe(summary.round(4))

# Correlation heatmap
if show_corr and len(prices.columns) > 1:
    st.subheader("Correlation heatmap (returns)")
    corr = returns.corr()
    fig_corr = px.imshow(corr, text_auto=True)
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

# Drawdown
if show_drawdown:
    st.subheader("Max Drawdowns")
    dd_df = pd.DataFrame()
    for col in prices.columns:
        dd = compute_drawdown(prices[col])
        dd_df[col] = dd
    # show worst drawdown per index
    worst = dd_df.min().sort_values()
    st.bar_chart(worst)

# Raw data & download
st.subheader("Raw adjusted close prices")
st.dataframe(prices.tail(20))

if download_csv:
    csv = prices.to_csv().encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='indices_prices.csv', mime='text/csv')

# Small analysis text
st.markdown("---")
st.subheader("Quick analysis notes")
st.markdown(
    "- Use the moving averages to eyeball trend changes.
- Check the correlation heatmap to understand how indexes move relative to each other.
- Large negative values in the drawdown chart indicate significant historical declines.")

# Footer / credits
st.markdown("---")
st.caption("Data via Yahoo Finance (yfinance). Built with Streamlit.")

# -----------------------------
# End of file
# -----------------------------

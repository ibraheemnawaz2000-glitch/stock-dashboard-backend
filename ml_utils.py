import joblib
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import ta 

# Load model
def load_model(model_path="ml_stock_model.pkl"):
    return joblib.load(model_path)

# Score a new signal using the ML model
def score_signal(reason, model):
    reason = reason.lower()
    features = {
        "rsi_triggered": int("rsi" in reason),
        "ema_crossover": int("ema" in reason),
        "breakout": int("breakout" in reason)
    }
    X = [[features["rsi_triggered"], features["ema_crossover"], features["breakout"]]]
    probability = model.predict_proba(X)[0][1]  # probability of success
    return round(probability * 100, 2)

# Generate interactive candlestick chart using Plotly
def generate_chart(ticker, days=30, save_html=False):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    if df.empty:
        return None

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    fig.update_layout(title=f"{ticker} Price Chart", xaxis_title='Date', yaxis_title='Price')

    if save_html:
        path = f"charts/{ticker}_chart.html"
        fig.write_html(path)
        return path

    return fig

def calculate_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["ema5"] = ta.trend.EMAIndicator(close=df["Close"], window=5).ema_indicator()
    df["ema20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()

    bb = ta.volatility.BollingerBands(close=df["Close"], window=20)
    df["bb_upper"] = bb.bollinger_hband().squeeze()  # <-- flatten if needed
    df["bb_lower"] = bb.bollinger_lband().squeeze()

    df["volume"] = df["Volume"]

    # Also flatten any others that might return 2D arrays
    df["macd"] = df["macd"].squeeze()
    df["macd_signal"] = df["macd_signal"].squeeze()
    df["rsi"] = df["rsi"].squeeze()

    return df

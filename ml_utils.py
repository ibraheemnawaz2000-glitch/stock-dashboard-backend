# ml_utils.py

import joblib
import plotly.graph_objs as go
from polygon_api import get_ohlcv
import os

def load_model(model_path="ml_stock_model.pkl"):
    """Loads the pre-trained machine learning model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train a model first and ensure it is in your GitHub repository.")
    return joblib.load(model_path)

def generate_chart(ticker, days=60, save_path=None):
    """Generates an interactive candlestick chart and saves it to a given path."""
    df = get_ohlcv(ticker, days=days)
    if df.empty:
        print(f"Could not retrieve chart data for {ticker}")
        return None

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    
    fig.update_layout(
        title=f"{ticker} Price Chart ({days} Days)",
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False
    )

    # --- FIX: Save the chart to the specified path ---
    if save_path:
        # The main.py script will ensure the directory exists
        fig.write_html(save_path)
        return save_path

    return fig

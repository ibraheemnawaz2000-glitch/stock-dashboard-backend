# ml_utils.py

import joblib
import plotly.graph_objs as go
from polygon_api import get_ohlcv # Use our existing function
import os

def load_model(model_path="ml_stock_model.pkl"):
    """Loads the pre-trained machine learning model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train a model first.")
    return joblib.load(model_path)

def generate_chart(ticker, days=60, save_html=False):
    """Generates an interactive candlestick chart using Polygon data."""
    # Correction: Use the same data source as the scanner for consistency
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
        xaxis_rangeslider_visible=False # Cleaner look
    )

    if save_html:
        # Ensure the 'charts' directory exists
        os.makedirs("charts", exist_ok=True)
        path = f"charts/{ticker}_chart.html"
        fig.write_html(path)
        return path

    return fig
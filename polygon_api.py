# polygon_api.py

import requests
import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

def get_top_tickers_by_volume(limit=500):
    """
    Fetches the top US stock tickers sorted by the most recent day's trading volume.
    """
    if not API_KEY:
        raise ValueError("🚨 POLYGON_API_KEY not found. Please set it in your .env file.")

    # Start from yesterday to ensure we get a day with complete data.
    date_to_check = datetime.now() - timedelta(days=1)

    # Loop backwards to find the last valid trading day (in case of weekend/holiday)
    for i in range(7):
        day_str = (date_to_check - timedelta(days=i)).strftime('%Y-%m-%d')
        print(f"📡 Checking for market data on {day_str}...")

        # Use the Grouped Daily Bars endpoint for market-wide data
        url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{day_str}?adjusted=true&apiKey={API_KEY}"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            if "results" in data and data["resultsCount"] > 0:
                print(f"✅ Found data for {day_str}. Processing...")
                df = pd.DataFrame(data["results"])

                # 'T' is the ticker symbol, 'v' is volume
                df.rename(columns={'T': 'ticker', 'v': 'volume'}, inplace=True)

                # Sort by volume and get the top tickers
                sorted_df = df.sort_values(by="volume", ascending=False)
                top_tickers = sorted_df.head(limit)['ticker'].tolist()

                print(f"📦 Loaded {len(top_tickers)} US tickers sorted by volume.")
                return top_tickers

        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            continue # Try the previous day

    print("❌ Could not find any trading data in the last 7 days.")
    return []

def get_ohlcv(ticker, days=90):
    """Fetches OHLCV data for a single ticker."""
    end = datetime.now()
    start = end - timedelta(days=days)
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit={days}&apiKey={API_KEY}"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get("results", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume"
    }, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]
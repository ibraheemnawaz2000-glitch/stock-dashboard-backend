# polygon_api.py

import requests
from datetime import datetime, timedelta
import pandas as pd

API_KEY = "u2BVZaKDdPX8QTUD3TR31yMl1ZtbdnDB"
BASE_URL = "https://api.polygon.io"


def get_last_trading_day():
    """
    Returns the most recent trading day (skipping weekends).
    """
    today = datetime.now()
    while today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        today -= timedelta(days=1)
    return today.strftime("%Y-%m-%d")


def get_top_movers(limit=25):
    """
    Fetches the top daily gainers by percent from Polygon.
    """
    today = get_last_trading_day()
    url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{today}?adjusted=true&apiKey={API_KEY}"

    response = requests.get(url)
    results = response.json().get("results", [])

    sorted_by_gain = sorted(results, key=lambda x: (x['c'] - x['o']) / x['o'] if x['o'] > 0 else 0, reverse=True)
    tickers = [r['T'] for r in sorted_by_gain[:limit]]
    print(f"📈 Top Movers from Polygon: {tickers}")
    return tickers


def get_ohlcv(ticker, days=60):
    """
    Fetches historical OHLCV data for a stock from Polygon.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=120&apiKey={API_KEY}"
    )

    response = requests.get(url)
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

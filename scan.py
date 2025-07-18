import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd
from ml_utils import calculate_indicators, load_model

WATCHLIST = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT"]
SIGNALS_FILE = "data/signals.json"
MODEL_FILE = "ml_stock_model.pkl"

def main():
    signals = []
    model = load_model(MODEL_FILE)

    for ticker in WATCHLIST:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)

            if len(df) < 30:
                continue

            df = calculate_indicators(df)
            latest = df.iloc[-1:]

            X = latest[["rsi", "macd", "macd_signal", "ema5", "ema20", "volume"]]
            proba = model.predict_proba(X)[0][1]

            if proba >= 0.9:
                signal = {
                    "ticker": ticker,
                    "confidence": round(proba * 100, 2),
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "reason": "ML model triggered high confidence signal",
                    "chart_url": f"https://www.tradingview.com/symbols/{ticker}/"
                }
                signals.append(signal)
        except Exception as e:
            print(f"Error with {ticker}: {e}")

    os.makedirs("data", exist_ok=True)
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=4)
    print("✅ Signals updated.")

if __name__ == "__main__":
    main()

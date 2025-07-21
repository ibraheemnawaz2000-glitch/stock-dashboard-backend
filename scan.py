# scan.py

import os
import json
import time # <-- Import the time module
from datetime import datetime
from dotenv import load_dotenv
from polygon_api import get_top_tickers_by_volume, get_ohlcv
from indicator_utils import calculate_indicators, detect_strategies, calculate_support_resistance, detect_candles
from gpt_utils import generate_gpt_reasoning
from ml_utils import load_model
import pandas as pd

load_dotenv()
MODEL_FILE = "ml_stock_model.pkl"
# --- Change: Point to persistent storage ---
DATA_DIR = "/var/data" # Render's path for persistent disks
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.json")
os.makedirs(DATA_DIR, exist_ok=True)

def run_scan(): # <-- Move the main logic into a function
    print("--- Starting a new scan cycle ---")
    model = load_model(MODEL_FILE)
    signals = []

    tickers = get_top_tickers_by_volume(limit=500)
    if not tickers:
        print("No tickers found to scan. Exiting cycle.")
        return

    print(f"🧠 Scanning {len(tickers)} tickers...")

    for ticker in tickers:
        try:
            df = get_ohlcv(ticker, days=60)
            if df.empty or len(df) < 30:
                continue

            df = calculate_indicators(df)
            support, resistance = calculate_support_resistance(df)
            candle_tags = detect_candles(df)

            if not candle_tags:
                continue

            latest = df.iloc[-1]
            strategy_tags = detect_strategies(latest)
            tags = strategy_tags + candle_tags
            
            features = {
                "rsi": latest["rsi"], "macd": latest["macd"],
                "macd_signal": latest["macd_signal"], "ema5": latest["ema5"],
                "ema20": latest["ema20"], "volume": latest["Volume"]
            }
            
            X = pd.DataFrame([features])
            proba = model.predict_proba(X)[0][1]

            if proba >= 0.75:
                print(f"📈 Found potential signal for {ticker} with confidence {proba:.2f}")
                gpt_reason = generate_gpt_reasoning(ticker, tags, support, resistance)
                signals.append({
                    "ticker": ticker, "confidence": round(proba * 100, 2),
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "reason": gpt_reason, "tags": list(set(tags)),
                    "chart_url": f"charts/{ticker}_chart.html"
                })
        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")

    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=4)

    print(f"✅ Scan complete. {len(signals)} signals saved to {SIGNALS_FILE}")

if __name__ == "__main__":
    # --- Change: Create an infinite loop ---
    while True:
        run_scan()
        # Sleep for 1 hour (3600 seconds) before the next run
        print("--- Scan cycle finished. Sleeping for 30minutes. ---")
        time.sleep(1800)
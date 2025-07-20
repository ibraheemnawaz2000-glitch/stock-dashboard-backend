import os
import json
from datetime import datetime
from dotenv import load_dotenv
from polygon_api import get_all_us_tickers, get_ohlcv
from indicator_utils import calculate_indicators, detect_strategies, calculate_support_resistance, detect_candles
from gpt_utils import generate_gpt_reasoning
from ml_utils import load_model

load_dotenv()
MODEL_FILE = "ml_stock_model.pkl"
SIGNALS_FILE = "data/signals.json"
os.makedirs("data", exist_ok=True)

def main():
    model = load_model(MODEL_FILE)
    signals = []

    tickers = get_all_us_tickers(limit=500)  # Adjustable for performance
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
                "rsi": latest["rsi"],
                "macd": latest["macd"],
                "macd_signal": latest["macd_signal"],
                "ema5": latest["ema5"],
                "ema20": latest["ema20"],
                "volume": latest["volume"]
            }

            import pandas as pd
            X = pd.DataFrame([features])
            proba = model.predict_proba(X)[0][1]

            if proba >= 0.75:  # Tunable threshold
                gpt_reason = generate_gpt_reasoning(ticker, tags, support, resistance)
                signals.append({
                    "ticker": ticker,
                    "confidence": round(proba * 100, 2),
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "reason": gpt_reason,
                    "tags": tags,
                    "chart_url": f"https://www.tradingview.com/symbols/{ticker}/"
                })

        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")

    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ {len(signals)} signals saved to {SIGNALS_FILE}")

if __name__ == "__main__":
    main()
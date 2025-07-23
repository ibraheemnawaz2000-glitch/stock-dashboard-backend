# scan.py (DB version)

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from models import Signal, SessionLocal, engine
from polygon_api import get_top_tickers_by_volume, get_ohlcv
from indicator_utils import calculate_indicators, detect_strategies, calculate_support_resistance, detect_candles
from gpt_utils import generate_gpt_reasoning
from ml_utils import load_model
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

load_dotenv()
MODEL_FILE = "ml_stock_model.pkl"

# Create DB tables
Signal.__table__.create(bind=engine, checkfirst=True)

def run_scan():
    print("--- Starting a new scan cycle ---")
    db = SessionLocal()
    model = load_model(MODEL_FILE)
    tickers = get_top_tickers_by_volume(limit=500)
    signals = []

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
            tag_str = ",".join(set(tags))

            features = {
                "rsi": latest["rsi"], "macd": latest["macd"],
                "macd_signal": latest["macd_signal"], "ema5": latest["ema5"],
                "ema20": latest["ema20"], "volume": latest["Volume"]
            }

            X = pd.DataFrame([features])
            proba = model.predict_proba(X)[0][1]

            if proba >= 0.6:
                print(f"📈 Signal: {ticker} | Confidence: {proba:.2f}")
                gpt_reason = generate_gpt_reasoning(ticker, tags, support, resistance)

                signal = Signal(
                    ticker=ticker,
                    confidence=round(proba * 100, 2),
                    date=datetime.utcnow().date(),
                    reason=gpt_reason,
                    tags=tag_str,
                    chart_url=f"charts/{ticker}_chart.html"
                )
                db.add(signal)
                signals.append(signal)

        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")
    
    db.commit()
    db.close()
    print(f"✅ Scan complete. Saved {len(signals)} signals to DB.")

if __name__ == "__main__":
    while True:
        run_scan()
        print("💤 Sleeping 30 minutes...")
        time.sleep(1800)
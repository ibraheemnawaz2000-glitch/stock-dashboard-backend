import os
import time
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from models import Signal, SessionLocal, engine
from polygon_api import get_top_tickers_by_volume, get_ohlcv
from indicator_utils import calculate_indicators, detect_strategies, calculate_support_resistance, detect_candles
from gpt_utils import generate_gpt_reasoning
from ml_utils import load_model

load_dotenv()
MODEL_FILE = "ml_stock_model.pkl"
SIGNALS_DIR = "signals_data" # Directory to store signal files

# --- Create DB tables and signals directory ---
Signal.__table__.create(bind=engine, checkfirst=True)
if not os.path.exists(SIGNALS_DIR):
    os.makedirs(SIGNALS_DIR)
    print(f"Created directory: {SIGNALS_DIR}")

def save_signals_to_csv(signals_list, filename):
    """
    Saves a list of signal objects to a CSV file.
    """
    if not signals_list:
        return
    data = [
        {
            "date": s.date.strftime('%Y-%m-%d %H:%M:%S'), "ticker": s.ticker,
            "confidence": s.confidence, "tags": s.tags, "reason": s.reason,
            "chart_url": s.chart_url,
        } for s in signals_list
    ]
    df = pd.DataFrame(data)
    file_exists = os.path.isfile(filename)
    try:
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
    except IOError as e:
        print(f"❌ Error writing to file {filename}: {e}")

def run_scan():
    """
    Runs a scan for stock signals and saves them to the database and a local CSV file.
    """
    print("--- Starting a new scan cycle ---")

    if not os.path.exists(MODEL_FILE):
        print(f"❌ CRITICAL: Model file not found at '{MODEL_FILE}'")
        return
    else:
        print(f"✅ Found model file at '{MODEL_FILE}'")
        
    db = SessionLocal()
    model = load_model(MODEL_FILE)
    
    # Reduced limit to avoid memory issues on free cloud hosting tiers.
    tickers = get_top_tickers_by_volume(limit=250) 
    print(f"Found {len(tickers)} tickers to analyze.")
    signals = []

    for i, ticker in enumerate(tickers):
        try:
            if (i + 1) % 10 == 0: # Log every 10 tickers
                print(f"--> Processing ticker {i+1}/{len(tickers)}: {ticker}")

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

            if proba > 0.5:
                print(f"    Ticker: {ticker}, Probability: {proba:.2f}")

            if proba >= 0.8:
                print(f"📈 Signal FOUND: {ticker} | Confidence: {proba:.2f}")
                gpt_reason = generate_gpt_reasoning(ticker, tags, support, resistance)

                signal = Signal(
                    ticker=ticker,
                    confidence=round(proba * 100, 2),
                    date=datetime.now(timezone.utc), # Use timezone.utc for compatibility
                    reason=gpt_reason,
                    tags=tag_str,
                    chart_url=f"charts/{ticker}_chart.html"
                )
                db.add(signal)
                signals.append(signal)

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    if signals:
        db.commit()
        print(f"✅ Scan complete. Saved {len(signals)} signals to the database.")
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d') # Use timezone.utc for compatibility
        filename = os.path.join(SIGNALS_DIR, f"signals_{today_str}.csv")
        save_signals_to_csv(signals, filename)
        print(f"📄 Also saved {len(signals)} signals to local file: {filename}")
    else:
        print("✅ Scan complete. No new signals found.")

    db.close()

if __name__ == "__main__":
    print("🚀 Script starting up...") # Added to confirm script execution starts
    while True:
        run_scan()
        print("💤 Sleeping for 30 minutes...")
        time.sleep(1800)
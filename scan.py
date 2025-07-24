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

    This function converts the list of signal objects into a pandas DataFrame
    and appends it to the specified CSV file. If the file doesn't exist,
    it will be created with a header.

    Args:
        signals_list (list): A list of Signal objects to save.
        filename (str): The path to the CSV file.
    """
    if not signals_list:
        return

    # Convert the list of Signal objects into a list of dictionaries
    data = [
        {
            "date": s.date.strftime('%Y-%m-%d %H:%M:%S'),
            "ticker": s.ticker,
            "confidence": s.confidence,
            "tags": s.tags,
            "reason": s.reason,
            "chart_url": s.chart_url,
        }
        for s in signals_list
    ]
    df = pd.DataFrame(data)

    # Check if the file already exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)

    # Append to the CSV file. Use mode='a' for append.
    # The header is only written if the file does not exist.
    try:
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
    except IOError as e:
        print(f"❌ Error writing to file {filename}: {e}")


def run_scan():
    """
    Runs a scan for stock signals and saves them to the database and a local CSV file.
    """
    print("--- Starting a new scan cycle ---")
    db = SessionLocal()
    model = load_model(MODEL_FILE)
    tickers = get_top_tickers_by_volume(limit=500)
    signals = []

    for ticker in tickers:
        try:
            # Fetch and prepare data
            df = get_ohlcv(ticker, days=60)
            if df.empty or len(df) < 30:
                continue

            # Calculate indicators and features
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

            # If confidence is high, generate reasoning and create a signal
            if proba >= 0.8:
                print(f"📈 Signal: {ticker} | Confidence: {proba:.2f}")
                gpt_reason = generate_gpt_reasoning(ticker, tags, support, resistance)

                signal = Signal(
                    ticker=ticker,
                    confidence=round(proba * 100, 2),
                    date=datetime.utcnow(), # Use datetime object for now
                    reason=gpt_reason,
                    tags=tag_str,
                    chart_url=f"charts/{ticker}_chart.html"
                )
                db.add(signal)
                signals.append(signal)

        except Exception as e:
            # It's good practice to catch specific exceptions, but this is a fallback
            print(f"❌ Error processing {ticker}: {e}")
    
    # --- Save signals to Database and Local File ---
    if signals:
        db.commit()
        print(f"✅ Scan complete. Saved {len(signals)} signals to the database.")

        # Save to local file BEFORE closing the session
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        filename = os.path.join(SIGNALS_DIR, f"signals_{today_str}.csv")
        save_signals_to_csv(signals, filename)
        print(f"📄 Also saved {len(signals)} signals to local file: {filename}")
    else:
        print("✅ Scan complete. No new signals found.")

    # Close the session after all operations are complete
    db.close()


if __name__ == "__main__":
    while True:
        run_scan()
        print("💤 Sleeping for 30 minutes...")
        time.sleep(1800)
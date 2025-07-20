
import json
import os
from datetime import datetime
import pandas as pd
from polygon_api import get_ohlcv, get_top_movers
from ml_utils import load_model
import ta

SIGNALS_FILE = "data/signals.json"
MODEL_FILE = "ml_stock_model.pkl"

def detect_bullish_reversal(df):
    patterns = []
    latest = df.iloc[-1]

    body = abs(latest['Close'] - latest['Open'])
    candle_range = latest['High'] - latest['Low']
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])

    if body < candle_range * 0.3 and lower_shadow > 2 * body:
        patterns.append("Hammer")
    if abs(latest['Close'] - latest['Open']) < candle_range * 0.1:
        patterns.append("Doji")

    return patterns

def calculate_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi().values.ravel()
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd().values.ravel()
    df['macd_signal'] = macd.macd_signal().values.ravel()
    df['ema5'] = ta.trend.EMAIndicator(close=df['Close'], window=5).ema_indicator().values.ravel()
    df['ema20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator().values.ravel()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['bb_upper'] = bb.bollinger_hband().values.ravel()
    df['bb_lower'] = bb.bollinger_lband().values.ravel()
    return df

def main():
    signals = []
    os.makedirs("data", exist_ok=True)
    model = load_model(MODEL_FILE)
    tickers = get_top_movers(20)

    for ticker in tickers:
        try:
            df = get_ohlcv(ticker)
            if df.empty or len(df) < 30:
                continue
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            reversals = detect_bullish_reversal(df)

            features = {
                "rsi": latest["rsi"],
                "macd": latest["macd"],
                "macd_signal": latest["macd_signal"],
                "ema5": latest["ema5"],
                "ema20": latest["ema20"],
                "volume": latest["Volume"]
            }

            X = pd.DataFrame([features])
            proba = model.predict_proba(X)[0][1]

            if reversals and proba > 0.8:
                signals.append({
                    "ticker": ticker,
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "confidence": round(proba * 100, 2),
                    "reason": ", ".join(reversals),
                    "chart_url": f"https://www.tradingview.com/symbols/{ticker}/"
                })

        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")

    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ {len(signals)} signals saved to {SIGNALS_FILE}")

if __name__ == "__main__":
    main()

import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import ta
from ml_utils import load_model
from finviz_scraper import fetch_finviz_reversals

SIGNALS_FILE = "data/signals.json"
MODEL_FILE = "ml_stock_model.pkl"

# Dynamically pulled watchlist from Finviz
WATCHLIST = fetch_finviz_reversals()
print(f"🔍 Tickers from Finviz: {WATCHLIST}")
if not WATCHLIST:
    print("⚠️ Using fallback watchlist.")
    WATCHLIST = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT"]

def calculate_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema5'] = ta.trend.EMAIndicator(close=df['Close'], window=5).ema_indicator()
    df['ema20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['volume'] = df['Volume'].rolling(window=5).mean().fillna(df['Volume'])

    for col in ['rsi', 'macd', 'macd_signal', 'ema5', 'ema20', 'bb_upper', 'bb_lower']:
        df[col] = df[col].values.ravel()

    return df

def detect_strategies(row):
    tags = []
    if row['rsi'] < 35:
        tags.append("RSI_Oversold")
    if row['rsi'] > 70:
        tags.append("RSI_Overbought")
    if row['ema5'] > row['ema20']:
        tags.append("EMA_Bullish_Crossover")
    if row['macd'] > row['macd_signal']:
        tags.append("MACD_Bullish")
    if row['Close'] > row['bb_upper']:
        tags.append("Breakout_BB_Upper")
    if row['volume'] > 1.5 * row['volume'].mean():
        tags.append("Volume_Spike")
    return tags

def main():
    signals = []
    model = load_model(MODEL_FILE)

    for ticker in WATCHLIST:
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if len(df) < 30:
                continue

            df = calculate_indicators(df)
            latest = df.iloc[-1]

            features = {
                "rsi": latest["rsi"],
                "macd": latest["macd"],
                "macd_signal": latest["macd_signal"],
                "ema5": latest["ema5"],
                "ema20": latest["ema20"],
                "volume": latest["volume"]
            }

            X = pd.DataFrame([features])
            proba = model.predict_proba(X)[0][1]

            strategy_tags = detect_strategies(latest)
            if proba >= 0.85:
                signals.append({
                    "ticker": ticker,
                    "confidence": round(proba * 100, 2),
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "reason": ", ".join(strategy_tags),
                    "strategy_tags": strategy_tags,
                    "chart_url": f"https://www.tradingview.com/symbols/{ticker}/"
                })

        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")

    os.makedirs("data", exist_ok=True)
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ {len(signals)} signals saved to {SIGNALS_FILE}")

if __name__ == "__main__":
    main()

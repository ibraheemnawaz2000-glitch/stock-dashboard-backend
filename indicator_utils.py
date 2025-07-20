import pandas as pd
import ta

def calculate_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi().fillna(50)
    df["ema5"] = ta.trend.EMAIndicator(close=df["Close"], window=5).ema_indicator().fillna(method="bfill")
    df["ema20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator().fillna(method="bfill")
    df["macd"] = ta.trend.MACD(close=df["Close"]).macd().fillna(0)
    df["macd_signal"] = ta.trend.MACD(close=df["Close"]).macd_signal().fillna(0)
    return df

def detect_strategies(row):
    tags = []
    if row["rsi"] < 35:
        tags.append("RSI_Oversold")
    if row["ema5"] > row["ema20"]:
        tags.append("EMA_Crossover")
    if row["macd"] > row["macd_signal"]:
        tags.append("MACD_Bullish")
    return tags

def calculate_support_resistance(df):
    support = df["Low"].rolling(window=5).min().iloc[-1]
    resistance = df["High"].rolling(window=5).max().iloc[-1]
    return round(support, 2), round(resistance, 2)

def detect_candles(df):
    """
    Basic candle pattern detection logic.
    Returns a list of pattern names from the last candle.
    """
    patterns = []
    latest = df.iloc[-1]

    body = abs(latest["Close"] - latest["Open"])
    range_ = latest["High"] - latest["Low"]
    upper_shadow = latest["High"] - max(latest["Close"], latest["Open"])
    lower_shadow = min(latest["Close"], latest["Open"]) - latest["Low"]

    # Avoid divide-by-zero
    if range_ == 0:
        return patterns

    # Doji
    if body / range_ < 0.1:
        patterns.append("Doji")

    # Hammer
    if lower_shadow > 2 * body and upper_shadow < 0.2 * body:
        patterns.append("Hammer")

    # Inverted Hammer
    if upper_shadow > 2 * body and lower_shadow < 0.2 * body:
        patterns.append("Inverted Hammer")

    # Bullish Engulfing
    if len(df) >= 2:
        prev = df.iloc[-2]
        if prev["Close"] < prev["Open"] and latest["Close"] > latest["Open"]:
            if latest["Open"] < prev["Close"] and latest["Close"] > prev["Open"]:
                patterns.append("Bullish Engulfing")

    return patterns

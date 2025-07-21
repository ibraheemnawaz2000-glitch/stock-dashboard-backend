# indicator_utils.py

import pandas as pd
import ta

def calculate_indicators(df):
    """Calculates all technical indicators and adds them to the DataFrame."""
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["ema5"] = ta.trend.EMAIndicator(close=df["Close"], window=5).ema_indicator()
    df["ema20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
    
    macd_indicator = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()

    # --- FIX: Update `fillna` to the modern syntax ---
    # Fill any missing indicator values at the beginning of the series
    df.bfill(inplace=True) # Back-fill first to handle leading NaNs
    df.ffill(inplace=True) # Then forward-fill to catch any remaining NaNs

    return df

def detect_strategies(row):
    """Detects bullish trading strategies based on the latest data row."""
    tags = []
    # Ensure there's a previous row to compare against
    if pd.isna(row.shift(1)["ema5"]):
        return tags

    if row["rsi"] < 35:
        tags.append("RSI Oversold")
    if row["ema5"] > row["ema20"] and row.shift(1)["ema5"] <= row.shift(1)["ema20"]:
         tags.append("EMA Bullish Crossover")
    if row["macd"] > row["macd_signal"] and row.shift(1)["macd"] <= row.shift(1)["macd_signal"]:
        tags.append("MACD Bullish Crossover")
    return tags

def calculate_support_resistance(df):
    """Calculates basic support and resistance from recent price action."""
    support = df["Low"].rolling(window=14, min_periods=1).min().iloc[-1]
    resistance = df["High"].rolling(window=14, min_periods=1).max().iloc[-1]
    return round(support, 2), round(resistance, 2)

def detect_candles(df):
    """Detects basic bullish candlestick patterns on the latest candle."""
    patterns = []
    if len(df) < 2:
        return patterns

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if (prev["Close"] < prev["Open"]) and \
       (latest["Close"] > latest["Open"]) and \
       (latest["Close"] > prev["Open"]) and \
       (latest["Open"] < prev["Close"]):
        patterns.append("Bullish Engulfing")

    body = abs(latest["Close"] - latest["Open"])
    if body == 0: return patterns # Avoid division by zero for Doji-like candles
    
    lower_shadow = min(latest["Open"], latest["Close"]) - latest["Low"]
    upper_shadow = latest["High"] - max(latest["Open"], latest["Close"])
    if lower_shadow > 2 * body and upper_shadow < body:
        patterns.append("Hammer")

    return patterns
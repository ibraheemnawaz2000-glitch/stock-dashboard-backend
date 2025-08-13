# app/indicator_utils.py
import pandas as pd
import ta

# ==============================
# Indicator calculation
# ==============================
def calculate_indicators(df: pd.DataFrame, bb_window: int = 20, bb_std: float = 2.0) -> pd.DataFrame:
    """
    Calculates core technical indicators and adds them to the DataFrame.
    Works on any timeframe (15m, 30m, daily, etc.).

    Adds columns:
      - rsi (14)
      - ema5, ema20
      - macd, macd_signal
      - bb_mid, bb_high, bb_low (Bollinger Bands, window=20, std=2 by default)
    """
    close = df["Close"]

    # Momentum / trend
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["ema5"] = ta.trend.EMAIndicator(close=close, window=5).ema_indicator()
    df["ema20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()

    macd_indicator = ta.trend.MACD(close=close)
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()

    # Bollinger Bands (for quick reversal setups)
    bb = ta.volatility.BollingerBands(close=close, window=bb_window, window_dev=bb_std)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"]  = bb.bollinger_lband()

    # Fill small gaps
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df


# ==============================
# Strategy / pattern detection
# ==============================
def detect_strategies(row: pd.Series) -> list[str]:
    """
    Detects bullish trading strategy tags for the *latest* row.
    NOTE: Because only the latest row is provided, true crossovers that
    require the previous candle are approximated (same as existing logic).

    Adds Bollinger-based tags that don't require the previous candle:
      - "BB Lower Touch" (price <= lower band)
      - "BB Upper Touch" (price >= upper band)
      - "BB Lower + RSI Oversold" (price <= lower & RSI<35)
      - "BB Upper + RSI Overbought" (price >= upper & RSI>65)
    """
    tags: list[str] = []

    # Guard for the crossover checks (kept for backward compatibility)
    try:
        prev_ema5 = row.shift(1)["ema5"]  # will be NaN when only latest is passed
        prev_macd = row.shift(1)["macd"]
        prev_macd_sig = row.shift(1)["macd_signal"]
    except Exception:
        prev_ema5 = prev_macd = prev_macd_sig = pd.NA

    if pd.isna(prev_ema5):
        # can't reliably check crossovers without prior row; fall through to BB/RSI tags
        pass
    else:
        # Existing tags (unchanged)
        if row.get("rsi", 50) < 35:
            tags.append("RSI Oversold")
        if row.get("ema5", 0) > row.get("ema20", 0) and prev_ema5 <= row.shift(1)["ema20"]:
            tags.append("EMA Bullish Crossover")
        if row.get("macd", 0) > row.get("macd_signal", 0) and prev_macd <= prev_macd_sig:
            tags.append("MACD Bullish Crossover")

    # --- New: Bollinger/RSI quick-reversal cues (do not need previous bar) ---
    close = row.get("Close")
    bb_low = row.get("bb_low")
    bb_high = row.get("bb_high")
    rsi = row.get("rsi")

    if pd.notna(close) and pd.notna(bb_low) and close <= bb_low:
        tags.append("BB Lower Touch")
        if pd.notna(rsi) and rsi < 35:
            tags.append("BB Lower + RSI Oversold")

    if pd.notna(close) and pd.notna(bb_high) and close >= bb_high:
        tags.append("BB Upper Touch")
        if pd.notna(rsi) and rsi > 65:
            tags.append("BB Upper + RSI Overbought")

    return list(dict.fromkeys(tags))  # de-dupe while preserving order


def calculate_support_resistance(df: pd.DataFrame) -> tuple[float, float]:
    """
    Basic support/resistance from recent price action (last 14 bars).
    """
    support = df["Low"].rolling(window=14, min_periods=1).min().iloc[-1]
    resistance = df["High"].rolling(window=14, min_periods=1).max().iloc[-1]
    return round(float(support), 2), round(float(resistance), 2)


def detect_candles(df: pd.DataFrame) -> list[str]:
    """
    Detect basic bullish candlestick patterns on the latest candle.
    Currently: Bullish Engulfing, Hammer.
    """
    patterns: list[str] = []
    if len(df) < 2:
        return patterns

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish Engulfing
    if (prev["Close"] < prev["Open"]) and \
       (latest["Close"] > latest["Open"]) and \
       (latest["Close"] > prev["Open"]) and \
       (latest["Open"] < prev["Close"]):
        patterns.append("Bullish Engulfing")

    body = abs(latest["Close"] - latest["Open"])
    if body == 0:
        return patterns

    lower_shadow = min(latest["Open"], latest["Close"]) - latest["Low"]
    upper_shadow = latest["High"] - max(latest["Open"], latest["Close"])
    if lower_shadow > 2 * body and upper_shadow < body:
        patterns.append("Hammer")

    return patterns


# ==============================
# Optional: Daily trend bias helpers (for later wiring in scan.py)
# ==============================
def daily_trend_bias(daily_df: pd.DataFrame, lookback: int = 3) -> str | None:
    """
    Returns a simple daily trend bias string based on EMA slopes:
      - "Daily Uptrend"   if EMA20 today > EMA20 N bars ago (default 3)
      - "Daily Downtrend" if EMA20 today < EMA20 N bars ago
      - None otherwise

    Useful to filter 15m longs to only those aligned with a rising daily trend.
    """
    if daily_df is None or len(daily_df) < max(lookback + 1, 25):
        return None

    tmp = daily_df.copy()
    tmp = calculate_indicators(tmp)  # ensures ema20 exists on daily timeframe
    ema20_now = float(tmp["ema20"].iloc[-1])
    ema20_prev = float(tmp["ema20"].iloc[-1 - lookback])

    if ema20_now > ema20_prev:
        return "Daily Uptrend"
    if ema20_now < ema20_prev:
        return "Daily Downtrend"
    return None

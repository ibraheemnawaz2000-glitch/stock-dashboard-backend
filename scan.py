# ==============================
# FILE: app/scan.py (updated)
# ==============================
import os
from datetime import datetime, timedelta, timezone, time as dtime, date
from typing import List, Set

import pandas as pd
from dotenv import load_dotenv

from app.models import SessionLocal, engine, Base, Signal, Outcome
from app.polygon_api import (
    get_ohlcv, get_ohlcv_intraday, get_company_name, get_diversified_universe,
)
from app.indicator_utils import (
    calculate_indicators, detect_strategies,
    calculate_support_resistance, detect_candles,
)
from app.ml_utils import load_model
from app.ranker import ask_gpt_top5

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "ml_stock_model.pkl")
SIGNALS_DIR = "signals_data"
HORIZON_DAYS_DEFAULT = 10

# --- selection knobs ---
USE_INTRADAY = True           # short-term mode on 15/30-min bars
INTRADAY_MULTIPLIER = 15      # 15-minute bars as requested
BATCH_SIZE = 800              # 500–1000; tune to infra limits
MIN_PER_SECTOR = 40           # sector spread (min)
MAX_PER_SECTOR = 120          # sector spread (prep cap used by get_diversified_universe)

# London tz
try:
    import zoneinfo
    LONDON = zoneinfo.ZoneInfo("Europe/London")
except Exception:
    from pytz import timezone as _tz
    LONDON = _tz("Europe/London")


def in_trading_window(now_utc: datetime) -> tuple[bool, str]:
    now_local = now_utc.astimezone(LONDON)
    is_weekday = now_local.weekday() < 5
    start = dtime(15, 0)   # 3:00pm BST per your preference
    end = dtime(21, 0)
    within = is_weekday and (start <= now_local.time() <= end)
    window_tag = now_local.strftime("%Y-%m-%d-%H:%M")
    return within, window_tag


def ensure_dirs():
    if not os.path.exists(SIGNALS_DIR):
        os.makedirs(SIGNALS_DIR)


def save_signals_to_csv(signals_rows: List[dict], filename: str):
    if not signals_rows:
        return
    df = pd.DataFrame(signals_rows)
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode="a", header=not file_exists, index=False)


def _tickers_scanned_today(db) -> Set[str]:
    today = date.today()
    start_utc = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    rows = db.execute(
        """
        SELECT DISTINCT ticker FROM signals
        WHERE created_at >= :start
        """,
        {"start": start_utc}
    ).fetchall()
    return {r[0] for r in rows}


def pick_next_batch(db, batch_size=BATCH_SIZE, min_per_sector=MIN_PER_SECTOR, max_per_sector=MAX_PER_SECTOR) -> pd.DataFrame:
    """
    1) Build diversified, ranked universe (ETF-excluded, mcap/price filtered),
       with true relative volume and true gap.
    2) Remove tickers scanned earlier today.
    3) Keep sector spread (min per sector), then fill by score to batch_size.

    RETURNS: DataFrame with at least columns [ticker, sector].
    """
    uni = get_diversified_universe(sector_cap=max_per_sector, use_history=True, history_limit=1500)
    if not isinstance(uni, pd.DataFrame) or uni.empty:
        return pd.DataFrame(columns=["ticker", "sector"])  # empty frame for safety

    already = _tickers_scanned_today(db)
    fresh = uni[~uni["ticker"].isin(already)].copy()

    # Sector spread by score
    picks = []
    for sector, g in fresh.groupby("sector"):
        g = g.sort_values(["score", "pct_change_abs"], ascending=[False, False])
        take = min(min_per_sector, len(g))
        if take > 0:
            picks.append(g.head(take)[["ticker", "sector", "score"]])

    selected = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=["ticker", "sector", "score"])

    # Fill remainder by score across all sectors
    if len(selected) < batch_size:
        leftovers = fresh[~fresh["ticker"].isin(selected["ticker"])].sort_values("score", ascending=False)
        want = batch_size - len(selected)
        if want > 0:
            selected = pd.concat([selected, leftovers.head(want)[["ticker", "sector", "score"]]], ignore_index=True)

    return selected.reset_index(drop=True)


def run_scan():
    now_utc = datetime.now(timezone.utc)
    within, window_tag = in_trading_window(now_utc)
    if not within:
        print("Outside London trading window. Skipping this run.")
        return

    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found at '{MODEL_FILE}'")
        return

    ensure_dirs()
    model = load_model(MODEL_FILE)
    db = SessionLocal()

    # diversified & rotating universe (no same-day duplicates)
    selected = pick_next_batch(db, batch_size=BATCH_SIZE)
    print(f"Universe size this cycle: {len(selected)}")

    candidates = []
    saved_rows_for_csv = []
    added_signals = []

    for _, row in selected.iterrows():
        ticker = row["ticker"]
        sector_from_universe = row.get("sector", None)
        try:
            df = get_ohlcv_intraday(ticker, multiplier=INTRADAY_MULTIPLIER, lookback_days=5) if USE_INTRADAY else get_ohlcv(ticker, days=60)
            if df.empty or len(df) < 30:
                continue

            df = calculate_indicators(df)
            support, resistance = calculate_support_resistance(df)
            candle_tags = detect_candles(df)
            if not candle_tags:
                continue

            latest = df.iloc[-1]
            strategy_tags = detect_strategies(latest)
            all_tags = list(set(strategy_tags + candle_tags))

            features = {
                "rsi": float(latest["rsi"]),
                "macd": float(latest["macd"]),
                "macd_signal": float(latest["macd_signal"]),
                "ema5": float(latest["ema5"]),
                "ema20": float(latest["ema20"]),
                "volume": float(latest["Volume"]),
            }
            X = pd.DataFrame([features])
            proba = float(model.predict_proba(X)[0][1])
            price_now = float(latest["Close"])

            indicators_json = {
                "features": features,
                "strategy_tags": strategy_tags,
                "candle_tags": candle_tags,
                "all_tags": all_tags,
                "support": support,
                "resistance": resistance,
                "ml_proba": proba,
                "timeframe": (f"{INTRADAY_MULTIPLIER}m" if USE_INTRADAY else "1d"),
            }

            # --- NEW: ensure Sector is always present on the card ---
            # Prefer sector from the diversified universe; fall back to any existing indicator hint.
            sector = sector_from_universe or indicators_json.get("sector")
            if sector:
                indicators_json["sector"] = sector

            sig = Signal(
                ticker=ticker,
                price_at_signal=price_now,
                indicators_json=indicators_json,
                is_top_pick=False,
                gpt_rank=None,
                stars=None,
                horizon_days=HORIZON_DAYS_DEFAULT,
                window_tag=window_tag,
            )
            db.add(sig)
            db.flush()
            added_signals.append(sig)

            saved_rows_for_csv.append({
                "created_at": now_utc.isoformat(),
                "window_tag": window_tag,
                "ticker": ticker,
                "sector": sector or "Unknown",
                "price_at_signal": price_now,
                "ml_proba": round(proba, 4),
                "tags": ",".join(all_tags),
                "support": support,
                "resistance": resistance,
                "timeframe": (f"{INTRADAY_MULTIPLIER}m" if USE_INTRADAY else "1d"),
            })

            if proba >= 0.60:
                candidates.append({
                    "ticker": ticker,
                    "price_at_signal": price_now,
                    "support": support,
                    "resistance": resistance,
                    "indicators_json": indicators_json,
                    "ml_confidence": proba,
                    "signal_id": str(sig.id),
                })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    top5 = ask_gpt_top5(candidates, horizon_days=HORIZON_DAYS_DEFAULT) if candidates else []
    print(f"GPT selected {len(top5)} top picks.")

    by_ticker = {s.ticker: s for s in added_signals}
    for i, pick in enumerate(top5, start=1):
        t = pick.get("ticker")
        sig = by_ticker.get(t)
        if not sig:
            continue
        sig.is_top_pick = True
        sig.gpt_rank = int(pick.get("rank", i))

        def _to_float(x):
            return float(str(x).replace("$", "").replace(",", "").strip()) if x is not None else None

        tp = _to_float(pick.get("target_price"))
        sl = _to_float(pick.get("stop_loss"))
        if tp is None or sl is None:
            continue
        sig.target_price = tp
        sig.stop_loss = sl

        try:
            gpt_stars = int(pick.get("stars", 3) or 3)
            sig.stars = max(1, min(5, gpt_stars))
        except Exception:
            sig.stars = 3

        sig.reason_json = {
            "direction": pick.get("direction", "long"),
            "rationale": pick.get("rationale"),
        }

        # Enrich once for Top Picks: company name from Polygon (cached).
        try:
            cname = get_company_name(sig.ticker)
            if cname:
                ind = sig.indicators_json or {}
                ind["company_name"] = cname
                # also make sure sector key is present even if None/Unknown
                if "sector" not in ind:
                    ind["sector"] = "Unknown"
                sig.indicators_json = ind
        except Exception:
            pass

        deadline = datetime.now(timezone.utc) + timedelta(days=sig.horizon_days or HORIZON_DAYS_DEFAULT)
        outcome = Outcome(signal_id=sig.id, status="PENDING", deadline=deadline)
        db.add(outcome)

    db.commit()
    print(f"Saved {len(added_signals)} signals; flagged {len(top5)} as Top 5.")

    if saved_rows_for_csv:
        fname = os.path.join(SIGNALS_DIR, f"signals_{now_utc.strftime('%Y-%m-%d')}.csv")
        save_signals_to_csv(saved_rows_for_csv, fname)

    db.close()


if __name__ == "__main__":
    run_scan()

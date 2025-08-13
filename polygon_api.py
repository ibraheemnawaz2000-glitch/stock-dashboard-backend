# ==============================
# FILE: app/polygon_api.py
# ==============================
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# Optional Redis cache (set REDIS_URL). Falls back to in-process cache.
try:
    import redis  # type: ignore
    _HAS_REDIS = True
except Exception:
    redis = None
    _HAS_REDIS = False

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"
REDIS_URL = os.getenv("REDIS_URL")

# ---- tiny on-disk cache: ticker -> name (company name lookup)
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "ticker_name_cache.json")
try:
    with open(_CACHE_PATH, "r", encoding="utf-8") as f:
        _NAME_CACHE: dict[str, str] = json.load(f)
except Exception:
    _NAME_CACHE = {}

def _save_name_cache():
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_NAME_CACHE, f)
    except Exception:
        pass

# ---- lightweight in-process cache for universe + metrics ----
_MEMO: Dict[str, Any] = {}


def _auth_params(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = {"apiKey": API_KEY}
    if extra:
        p.update(extra)
    return p


def _redis_client():
    if _HAS_REDIS and REDIS_URL:
        try:
            return redis.from_url(REDIS_URL)
        except Exception:
            return None
    return None


def _cache_set(key: str, payload: Any, ttl_seconds: int = 600) -> None:
    r = _redis_client()
    if r is not None:
        try:
            r.setex(key, ttl_seconds, json.dumps(payload))
            return
        except Exception:
            pass
    _MEMO[key] = payload


def _cache_get(key: str) -> Optional[Any]:
    r = _redis_client()
    if r is not None:
        try:
            v = r.get(key)
            if v:
                return json.loads(v)
        except Exception:
            pass
    return _MEMO.get(key)


# --------------------
# Reference tickers
# --------------------

def _get_reference_page(cursor: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY not found. Set it in your environment.")
    params = _auth_params({"market": "stocks", "active": "true", "limit": limit})
    url = f"{BASE_URL}/v3/reference/tickers"
    if cursor:
        params["cursor"] = cursor
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json() or {}


def _get_all_reference_tickers(max_pages: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        data = _get_reference_page(cursor=cursor)
        results = data.get("results", [])
        out.extend(results)
        cursor_url = data.get("next_url")
        if not cursor_url:
            break
        # Polygon v3 uses next_url; extract cursor token if present
        try:
            cursor = cursor_url.split("cursor=")[-1]
        except Exception:
            cursor = None
    return out


# --------------------
# Grouped daily bars
# --------------------

def get_grouped_for(date_str: str) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    params = _auth_params({"adjusted": "true"})
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def last_trade_day_str(today: Optional[datetime] = None) -> str:
    base = today or datetime.now()
    for i in range(1, 10):
        day_str = (base - timedelta(days=i)).strftime("%Y-%m-%d")
        results = get_grouped_for(day_str)
        if results:
            return day_str
    return (base - timedelta(days=1)).strftime("%Y-%m-%d")


# --------------------
# OHLCV (daily & intraday)
# --------------------

def get_ohlcv(ticker: str, days: int = 90) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = _auth_params({"adjusted": "true", "sort": "asc", "limit": days})
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("results", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_ohlcv_intraday(ticker: str, multiplier: int = 15, timespan: str = "minute", lookback_days: int = 5) -> pd.DataFrame:
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY not found. Set it in your environment.")
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = _auth_params({"adjusted": "true", "sort": "asc", "limit": 50000})
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("results", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# --------------------
# Company name lookup (cached)
# --------------------

def get_company_name(ticker: str) -> Optional[str]:
    if not ticker:
        return None
    t = ticker.upper()
    name = _NAME_CACHE.get(t)
    if name:
        return name
    if not API_KEY:
        return None
    url = f"{BASE_URL}/v3/reference/tickers/{t}"
    params = _auth_params()
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        result = data.get("results") or {}
        name = result.get("name")
        if name:
            _NAME_CACHE[t] = name
            _save_name_cache()
            return name
    except requests.RequestException:
        pass
    return None

# Optional alias
get_ticker_name = get_company_name


# --------------------
# Patch 3: True Relative Volume & True Gap helpers
# --------------------

def _avg_volume_30d_for_ticker(ticker: str, days: int = 45) -> Optional[float]:
    """Fetch last ~45 sessions and compute avg Volume over last 30 sessions."""
    end = datetime.now()
    start = end - timedelta(days=days)
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = _auth_params({"adjusted": "true", "sort": "desc", "limit": days})
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        res = r.json().get("results", [])
        if not res:
            return None
        vols = [row.get("v", 0) for row in res[:30] if row.get("v")]
        if not vols:
            return None
        return float(sum(vols)) / float(len(vols))
    except requests.RequestException:
        return None


def _prev_close_map(date_str: Optional[str] = None) -> Dict[str, float]:
    day = date_str or last_trade_day_str()
    cache_key = f"prevclose:{day}"
    cached = _cache_get(cache_key)
    if cached:
        try:
            return {d["ticker"]: float(d["close"]) for d in cached}
        except Exception:
            pass
    results = get_grouped_for(day)
    if not results:
        return {}
    df = pd.DataFrame(results).rename(columns={"T": "ticker", "c": "close"})
    payload = json.loads(df[["ticker", "close"]].to_json(orient="records"))
    _cache_set(cache_key, payload, ttl_seconds=6 * 3600)
    return {row["ticker"]: float(row["close"]) for row in payload}


def _avg_vol_map(tickers: List[str], day_tag: Optional[str] = None, max_tickers: int = 1500) -> Dict[str, float]:
    """Compute/Cache avg_volume_30d for up to max_tickers; returns ticker->avgVol."""
    day = day_tag or datetime.now().strftime("%Y-%m-%d")
    base_key = f"avgvol30:{day}"
    existing = _cache_get(base_key) or {}

    out: Dict[str, float] = {}
    to_fetch: List[str] = []
    for t in tickers[:max_tickers]:
        if t in existing:
            out[t] = existing[t]
        else:
            to_fetch.append(t)

    # Fetch missing tickers one by one (Polygon rate limits apply). Consider batching or background job in prod.
    for t in to_fetch:
        avgv = _avg_volume_30d_for_ticker(t)
        if avgv is not None:
            out[t] = avgv
            existing[t] = avgv

    # Save merged cache
    _cache_set(base_key, existing, ttl_seconds=12 * 3600)
    return out


# --------------------
# Diversified universe with true rel-vol & gap
# --------------------

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_diversified_universe(
    sector_cap: int = 120,
    min_mktcap: int = 300_000_000,
    min_price: float = 2.0,
    max_price: float = 500.0,
    avg_vol_30d_floor: int = 500_000,
    cache_minutes: int = 10,
    use_history: bool = True,
    history_limit: int = 1500,
) -> pd.DataFrame:
    """
    Build a diversified pool of liquid, non-ETF US stocks with sector metadata,
    merged with today's grouped bars. If `use_history` is True, computes
    **true relative volume** (today volume / 30d avg) and **true gap** (open vs prev close).
    """
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY not found. Set it in your environment.")

    day = _today_str()
    cache_key = f"universe:{day}:{int(use_history)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        try:
            return pd.read_json(pd.io.json.dumps(cached))
        except Exception:
            pass

    # 1) Reference metadata (ETF exclusion, market cap, sector)
    meta = _get_all_reference_tickers(max_pages=10)
    rows = []
    for r in meta:
        t = r.get("ticker")
        if not t:
            continue
        ttype = (r.get("type") or "").upper()
        if "ETF" in ttype:
            continue
        mcap = r.get("market_cap") or 0
        if not mcap or mcap < min_mktcap:
            continue
        exch = (r.get("primary_exchange") or "").upper()
        if "OTC" in exch:
            continue
        rows.append({
            "ticker": t,
            "sector": r.get("sic_sector") or r.get("sector") or "Unknown",
            "market_cap": float(mcap),
        })
    if not rows:
        return pd.DataFrame()

    ref_df = pd.DataFrame(rows)

    # 2) Today’s grouped bars
    grouped = get_grouped_for(day)
    if not grouped:
        return pd.DataFrame()

    gdf = pd.DataFrame(grouped).rename(columns={
        "T": "ticker", "o": "open", "c": "close", "v": "volume", "h": "high", "l": "low", "vw": "vwap"
    })
    gdf = gdf[(gdf["close"] >= min_price) & (gdf["close"] <= max_price)]

    # Merge
    uni = ref_df.merge(gdf, on="ticker", how="inner")
    uni["sector"] = uni["sector"].fillna("Unknown")

    # 3) History-backed metrics
    if use_history and not uni.empty:
        # Prev close map for true gap
        prev_map = _prev_close_map()
        uni["prev_close"] = uni["ticker"].map(prev_map)
        uni["gap_abs"] = (uni["open"] - uni["prev_close"]).abs() / uni["prev_close"].replace(0, pd.NA)

        # Avg vol 30d (true rel-vol) for top names only to cap API usage
        # Choose candidates by raw volume to focus on active names
        vol_sorted = uni.sort_values("volume", ascending=False).head(history_limit)
        avg_map = _avg_vol_map(vol_sorted["ticker"].tolist())
        uni["avg_vol_30d"] = uni["ticker"].map(avg_map)
        uni["rel_vol"] = uni["volume"] / uni["avg_vol_30d"].replace(0, pd.NA)
    else:
        # Proxies if history disabled
        uni["gap_abs"] = (uni["open"] - uni["vwap"]).abs() / uni["vwap"].replace(0, pd.NA)
        uni["rel_vol"] = 1.0

    # Basic intraday move
    uni["pct_change_abs"] = (uni["close"] - uni["open"]).abs() / uni["open"].replace(0, pd.NA)

    # Liquidity filter using true avg vol if available
    if "avg_vol_30d" in uni.columns:
        uni = uni[(uni["avg_vol_30d"].fillna(0) >= avg_vol_30d_floor)]

    # Early per-sector cap to avoid domination
    uni = uni.groupby("sector").head(sector_cap).reset_index(drop=True)

    # 4) Composite score (tunable). Favor movers & rel-vol, then gap.
    w_move, w_relvol, w_gap = 0.55, 0.35, 0.10
    uni["score"] = (
        w_move * uni["pct_change_abs"].fillna(0).clip(lower=0) +
        w_relvol * uni["rel_vol"].fillna(1.0).clip(lower=0, upper=10) / 10.0 +
        w_gap  * uni["gap_abs"].fillna(0).clip(lower=0)
    )

    uni["sector_rank"] = uni.groupby("sector")["score"].rank(ascending=False, method="first")

    try:
        payload = json.loads(uni.to_json(orient="records"))
        _cache_set(cache_key, payload, ttl_seconds=cache_minutes * 60)
    except Exception:
        pass

    return uni.sort_values(["sector", "sector_rank", "score"], ascending=[True, True, False]).reset_index(drop=True)

# app/main.py (v1.3.4)
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, desc, func, and_
import zoneinfo

from app.models import get_db, Signal, Outcome, PriceCheck
from app.polygon_api import get_ohlcv

# Optional chart generation
try:
    from app.ml_utils import generate_chart
except Exception:
    generate_chart = None

LONDON = zoneinfo.ZoneInfo("Europe/London")

app = FastAPI(title="Tradia Signals API", version="1.3.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Redirect '/' -> '/docs' ---
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

CHARTS_DIR = os.getenv("CHARTS_DIR", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


# ---------- helpers ----------
def _num(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _latest_outcome(db, signal_id):
    q = (
        select(Outcome)
        .where(Outcome.signal_id == signal_id)
        .order_by(desc(Outcome.created_at))
        .limit(1)
    )
    return db.execute(q).scalars().first()


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def _infer_direction(entry: Optional[float], tp: Optional[float], fallback: str = "long") -> str:
    if entry is None or tp is None:
        return fallback
    try:
        return "long" if tp >= entry else "short"
    except Exception:
        return fallback


def _risk_reward(entry: Optional[float], tp: Optional[float], sl: Optional[float], direction: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (rr, reward_pct, risk_pct) where rr = reward/risk as a multiple.
    Handles long/short and guards invalid math.
    """
    try:
        if entry is None or tp is None or sl is None or entry <= 0:
            return None, None, None
        d = (direction or "long").lower()
        if d == "short":
            reward = max(entry - tp, 0.0)
            risk = max(sl - entry, 0.0)
            reward_pct = (reward / entry) * 100.0
            risk_pct = (risk / entry) * 100.0 if risk > 0 else None
        else:  # long
            reward = max(tp - entry, 0.0)
            risk = max(entry - sl, 0.0)
            reward_pct = (reward / entry) * 100.0
            risk_pct = (risk / entry) * 100.0 if risk > 0 else None
        rr = (reward / risk) if risk and risk > 0 else None
        # Clean improbable negatives
        if rr is not None and rr < 0:
            rr = None
        return (round(rr, 3) if rr is not None else None,
                round(reward_pct, 2) if reward_pct is not None else None,
                round(risk_pct, 2) if risk_pct is not None else None)
    except Exception:
        return None, None, None


def _serialize_signal(sig: Signal, outcome: Optional[Outcome]) -> Dict[str, Any]:
    indicators = sig.indicators_json or {}

    # Lift commonly-used fields to top-level for convenience
    sector     = indicators.get("sector")
    timeframe  = indicators.get("timeframe")            # e.g., "15m" or "1d"
    ml_proba   = _num(indicators.get("ml_proba"))
    support    = _num(indicators.get("support"))
    resistance = _num(indicators.get("resistance"))

    # Tags at top-level
    strategy_tags = _as_list(indicators.get("strategy_tags"))
    candle_tags   = _as_list(indicators.get("candle_tags"))
    all_tags      = _as_list(indicators.get("all_tags")) or (strategy_tags + [t for t in candle_tags if t not in strategy_tags])

    # Company name normalized at top-level (fallback to indicators.name if present)
    company_name = indicators.get("company_name") or indicators.get("name")

    # Risk/Reward computation
    entry = _num(sig.price_at_signal)
    tp    = _num(sig.target_price)
    sl    = _num(sig.stop_loss)
    # Direction (prefer reason.direction; infer from tp vs entry else default long)
    direction = None
    try:
        direction = (sig.reason_json or {}).get("direction")
    except Exception:
        direction = None
    if not direction:
        direction = _infer_direction(entry, tp, fallback="long")

    rr, reward_pct, risk_pct = _risk_reward(entry, tp, sl, direction)

    return {
        "id": str(sig.id),
        "created_at": sig.created_at.isoformat() if sig.created_at else None,
        "ticker": sig.ticker,
        "company_name": company_name,    # normalized top-level
        "sector": sector,
        "timeframe": timeframe,
        "ml_proba": ml_proba,
        "support": support,
        "resistance": resistance,
        "all_tags": all_tags,
        "strategy_tags": strategy_tags,
        "candle_tags": candle_tags,
        "direction": direction,          # convenient for UI/filtering
        "risk_reward": rr,               # RR multiple (reward/risk)
        "reward_pct": reward_pct,        # % to TP from entry
        "risk_pct": risk_pct,            # % to SL from entry
        "price_at_signal": entry,
        "target_price": tp,
        "stop_loss": sl,
        "stars": sig.stars,
        "gpt_rank": sig.gpt_rank,
        "is_top_pick": sig.is_top_pick,
        "horizon_days": sig.horizon_days,
        "indicators": indicators,
        "reason": sig.reason_json,
        "window_tag": sig.window_tag,
        "outcome": {
            "status": outcome.status if outcome else "PENDING",
            "deadline": outcome.deadline.isoformat() if outcome and outcome.deadline else None,
            "target_met_at": outcome.target_met_at.isoformat() if outcome and outcome.target_met_at else None,
        } if outcome else None,
    }


# ---------- base endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "now": datetime.utcnow().isoformat() + "Z"}

# Optional alias for frontend that calls /signals/latest
@app.get("/signals")
def signals_alias(db=Depends(get_db), limit: int = Query(50, ge=1, le=500)):
    return latest_signals(limit=limit, db=db)


@app.get("/signals/latest")
def latest_signals(
    limit: int = Query(50, ge=1, le=500),
    only_top: bool = Query(False),
    db=Depends(get_db)
):
    q = select(Signal)
    if only_top:
        q = q.where(Signal.is_top_pick == True)  # noqa: E712
    q = q.order_by(desc(Signal.created_at)).limit(limit)
    rows = db.execute(q).scalars().all()
    return [_serialize_signal(s, _latest_outcome(db, s.id)) for s in rows]


@app.get("/signals/top5")
def top5(db=Depends(get_db)):
    q = (
        select(Signal)
        .where(Signal.is_top_pick == True)  # noqa: E712
        .order_by(desc(Signal.created_at), Signal.gpt_rank)
        .limit(5)
    )
    rows = db.execute(q).scalars().all()
    return [_serialize_signal(s, _latest_outcome(db, s.id)) for s in rows]


@app.get("/stats/summary")
def stats_summary(db=Depends(get_db)):
    total_signals = db.execute(select(func.count(Signal.id))).scalar() or 0
    total_top = db.execute(select(func.count(Signal.id)).where(Signal.is_top_pick == True)).scalar() or 0  # noqa

    met = db.execute(select(func.count(Outcome.id)).where(Outcome.status == "MET")).scalar() or 0
    not_met = db.execute(select(func.count(Outcome.id)).where(Outcome.status == "NOT_MET")).scalar() or 0
    pending = db.execute(select(func.count(Outcome.id)).where(Outcome.status == "PENDING")).scalar() or 0

    # avg days to target (MET only)
    oq = (
        select(Outcome.signal_id, Outcome.target_met_at)
        .where(and_(Outcome.status == "MET", Outcome.target_met_at.is_not(None)))
    )
    rows = db.execute(oq).all()

    ids = [r[0] for r in rows]
    created_lookup: Dict[Any, datetime] = {}
    if ids:
        sq = select(Signal.id, Signal.created_at).where(Signal.id.in_(ids))
        for sid, created_at in db.execute(sq).all():
            created_lookup[str(sid)] = created_at

    diffs_days: List[float] = []
    for sig_id, met_at in rows:
        created_at = created_lookup.get(str(sig_id))
        if created_at and met_at:
            dt = (met_at - created_at).total_seconds() / 86400.0
            if dt >= 0:
                diffs_days.append(dt)

    avg_days_to_target = round(sum(diffs_days) / len(diffs_days), 3) if diffs_days else None
    win_rate = round(met / max(1, (met + not_met)) * 100, 2) if (met + not_met) else None

    return {
        "totals": {"signals": total_signals, "top_picks": total_top},
        "outcomes": {"MET": met, "NOT_MET": not_met, "PENDING": pending},
        "win_rate_pct": win_rate,
        "avg_days_to_target": avg_days_to_target,
    }


# ---------- Day filter (London date) ----------
@app.get("/signals/day")
def signals_by_day(
    date: str = Query(..., description="London date, format YYYY-MM-DD"),
    only_top: bool = Query(True),
    limit: int = Query(600, ge=1, le=2000),
    db=Depends(get_db),
):
    q = select(Signal).where(Signal.window_tag.like(f"{date}%"))
    if only_top:
        q = q.where(Signal.is_top_pick == True)  # noqa: E712
    q = q.order_by(desc(Signal.created_at)).limit(limit)
    rows = db.execute(q).scalars().all()
    return [_serialize_signal(s, _latest_outcome(db, s.id)) for s in rows]


# ---------- Successful signals ----------
@app.get("/signals/success")
def successful_signals(
    days: int = Query(10, ge=1, le=120),
    limit: int = Query(200, ge=1, le=1000),
    db=Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    oq = (
        select(Outcome)
        .where(and_(Outcome.status == "MET", Outcome.target_met_at.is_not(None), Outcome.target_met_at >= since))
        .order_by(desc(Outcome.target_met_at))
        .limit(limit)
    )
    outs = db.execute(oq).scalars().all()

    result = []
    for o in outs:
        s = db.get(Signal, o.signal_id)
        if s:
            result.append(_serialize_signal(s, o))
    return result


# ---------- Progress movers since signal ----------
@app.get("/signals/progress")
def top_progress(
    hours: int = Query(48, ge=1, le=240),
    limit: int = Query(30, ge=1, le=200),
    only_top: bool = Query(True),
    db=Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    q = select(Signal).where(Signal.created_at >= since)
    if only_top:
        q = q.where(Signal.is_top_pick == True)  # noqa: E712
    q = q.order_by(desc(Signal.created_at)).limit(limit * 3)  # oversample before sorting

    rows = db.execute(q).scalars().all()
    out = []
    for s in rows:
        try:
            df = get_ohlcv(s.ticker, days=5)
            if df.empty:
                continue
            last = float(df["Close"].iloc[-1])
            base = float(s.price_at_signal or 0.0) or 1e-9
            chg_pct = (last - base) / base * 100.0

            item = _serialize_signal(s, _latest_outcome(db, s.id))
            item["progress"] = {"last_price": round(last, 4), "change_pct": round(chg_pct, 2)}
            out.append(item)
        except Exception:
            continue

    out.sort(key=lambda x: x.get("progress", {}).get("change_pct", -999), reverse=True)
    return out[:limit]


# ---------- search/history ----------
@app.get("/signals/search")
def search_signals(
    ticker: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=2000),
    top_only: bool = Query(False),
    db=Depends(get_db)
):
    q = select(Signal).where(Signal.ticker == ticker)
    if top_only:
        q = q.where(Signal.is_top_pick == True)  # noqa
    q = q.order_by(desc(Signal.created_at)).limit(limit)
    rows = db.execute(q).scalars().all()
    return [_serialize_signal(s, _latest_outcome(db, s.id)) for s in rows]


@app.get("/signals/history")
def signal_history(
    ticker: str = Query(..., min_length=1),
    db=Depends(get_db)
):
    q = select(Signal).where(Signal.ticker == ticker).order_by(desc(Signal.created_at))
    rows = db.execute(q).scalars().all()
    return [_serialize_signal(s, _latest_outcome(db, s.id)) for s in rows]


@app.get("/outcomes/open")
def open_outcomes(db=Depends(get_db)):
    oq = select(Outcome).where(Outcome.status == "PENDING")
    open_rows = db.execute(oq).scalars().all()

    result = []
    for o in open_rows:
        s = db.get(Signal, o.signal_id)
        if not s:
            continue
        result.append(_serialize_signal(s, o))
    result.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return result


# ---------- keep dynamic route LAST ----------
@app.get("/signals/{signal_id}")
def get_signal(signal_id: str, db=Depends(get_db)):
    s = db.get(Signal, signal_id)
    if not s:
        raise HTTPException(status_code=404, detail="Signal not found")
    return _serialize_signal(s, _latest_outcome(db, s.id))


# ---------- optional chart endpoint ----------
@app.get("/charts/{ticker}_chart.html")
def get_chart(ticker: str):
    chart_path = os.path.join(CHARTS_DIR, f"{ticker}_chart.html")

    if not os.path.exists(chart_path):
        if generate_chart is None:
            raise HTTPException(status_code=404, detail="Chart not found and generator unavailable.")
        try:
            generate_chart(ticker, save_path=chart_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate chart: {e}")

    return FileResponse(chart_path)




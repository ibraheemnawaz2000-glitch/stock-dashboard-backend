# main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models import SessionLocal, Signal
from ml_utils import generate_chart
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

# Define chart storage path
CHARTS_DIR = os.getenv("CHARTS_DIR", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

@app.get("/signals")
def get_signals():
    """
    Reads signals from PostgreSQL and ensures charts are available.
    """
    db = SessionLocal()
    signals = db.query(Signal).order_by(Signal.date.desc()).limit(50).all()
    db.close()

    result = []
    for s in signals:
        ticker = s.ticker
        chart_filename = f"{ticker}_chart.html"
        chart_path = os.path.join(CHARTS_DIR, chart_filename)

        # Ensure chart exists
        if not os.path.exists(chart_path):
            print(f"Chart for {ticker} not found, generating now...")
            try:
                generate_chart(ticker, save_path=chart_path)
            except Exception as e:
                print(f"Could not generate chart for {ticker}: {e}")
                chart_path = None

        result.append({
            "ticker": ticker,
            "confidence_score": s.confidence,
            "date": s.date.strftime("%Y-%m-%d"),
            "reason": s.reason,
            "tags": s.tags,
            "chart_url": f"/charts/{ticker}_chart.html" if chart_path else None
        })

    return result


@app.get("/charts/{ticker}_chart.html")
def get_chart(ticker: str):
    """
    Serves the generated HTML chart file for a specific ticker.
    """
    chart_path = os.path.join(CHARTS_DIR, f"{ticker}_chart.html")
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Chart not found.")
    
    return FileResponse(chart_path)

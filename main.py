# main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from ml_utils import generate_chart

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

# --- FIX: Read the data directory from environment variables ---
# On Render, this will be '/var/data'. Locally, it will be 'data'.
DATA_DIR = os.getenv("DATA_DIR", "data")
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.json")
CHARTS_DIR = "charts"


@app.get("/signals")
def get_signals():
    """
    Reads signals from the JSON file and ensures charts are generated.
    """
    if not os.path.exists(SIGNALS_FILE):
        return {"message": "No signals file found. The scanner may not have run yet."}

    with open(SIGNALS_FILE, "r") as f:
        signals = json.load(f)

    for signal in signals:
        ticker = signal["ticker"]
        chart_path = os.path.join(CHARTS_DIR, f"{ticker}_chart.html")
        if not os.path.exists(chart_path):
            print(f"Chart for {ticker} not found, generating now...")
            try:
                generate_chart(ticker, save_html=True)
            except Exception as e:
                print(f"Could not generate chart for {ticker}: {e}")
                signal["chart_url"] = None
    
    return signals


@app.get("/charts/{ticker}_chart.html")
def get_chart(ticker: str):
    """
    Serves the generated HTML chart file for a specific ticker.
    """
    chart_path = os.path.join(CHARTS_DIR, f"{ticker}_chart.html")
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Chart not found.")
    
    return FileResponse(chart_path)

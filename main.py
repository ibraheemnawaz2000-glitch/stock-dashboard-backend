from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from ml_utils import load_model, score_signal, generate_chart
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.get("/signals")
def get_signals():
    try:
        with open("data/signals.json", "r") as f:
            signals = json.load(f)
        for signal in signals:
            reason = signal.get("reason", "")
            signal["confidence_score"] = "N/A"
            chart_path = f"charts/{signal['ticker']}_chart.html"
            if not os.path.exists(chart_path):
                try:
                    generate_chart(signal["ticker"], save_html=True)
                except:
                    continue
            signal["chart_url"] = f"charts/{signal['ticker']}_chart.html"
        return signals
    except FileNotFoundError:
        return {"message": "No signals yet."}

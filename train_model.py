import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Replace this with real labeled trade data in the future
data = [
    {"rsi": 30, "macd": 1, "macd_signal": 0.5, "ema5": 12, "ema20": 10, "volume": 500000, "success": 1},
    {"rsi": 75, "macd": -1, "macd_signal": -1.2, "ema5": 9, "ema20": 11, "volume": 300000, "success": 0},
    {"rsi": 45, "macd": 0.3, "macd_signal": 0.2, "ema5": 10.5, "ema20": 10, "volume": 600000, "success": 1},
    {"rsi": 25, "macd": 1.1, "macd_signal": 0.9, "ema5": 8, "ema20": 6, "volume": 800000, "success": 1},
    {"rsi": 60, "macd": -0.5, "macd_signal": -0.3, "ema5": 13, "ema20": 13.5, "volume": 200000, "success": 0},
]

df = pd.DataFrame(data)
X = df.drop(columns=["success"])
y = df["success"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "ml_stock_model.pkl")
print("✅ Model trained with feature names and saved.")

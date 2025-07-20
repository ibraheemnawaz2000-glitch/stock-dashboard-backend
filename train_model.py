import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Sample data — replace this with real indicators later
X = [
    [1, 0, 1],  # e.g. RSI oversold + breakout
    [0, 1, 0],  # e.g. EMA crossover
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0]
]
y = [1, 1, 0, 1, 0]  # 1 = successful, 0 = not successful

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "ml_stock_model.pkl")
print("✅ Model trained and saved as ml_stock_model.pkl")

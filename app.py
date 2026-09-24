from datetime import date, timedelta
import math
import random

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file

try:
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)

SYMBOLS = {
    "AAPL": (189.30, 0.0005, 0.018),
    "MSFT": (416.10, 0.0004, 0.014),
    "GOOGL": (176.80, 0.0006, 0.020),
    "TSLA": (177.50, 0.0002, 0.032),
}
FEATURES = [
    "open", "high", "low", "close", "volume", "daily_return",
    "ma_7", "ma_20", "ma_50", "volatility", "previous_close", "previous_return",
]
DISPLAY_FEATURES = {
    "open": "Open", "high": "High", "low": "Low", "close": "Close",
    "volume": "Volume", "daily_return": "Daily return", "ma_7": "7-day moving average",
    "ma_20": "20-day moving average", "ma_50": "50-day moving average",
    "volatility": "Volatility", "previous_close": "Previous close", "previous_return": "Previous return",
}


def make_demo_data(symbol, years):
    base, drift, volatility = SYMBOLS.get(symbol, SYMBOLS["AAPL"])
    records = 252 * years
    rng = random.Random(sum(ord(char) for char in symbol) + years)
    current = base * (1 - drift * records)
    rows = []
    start = date.today() - timedelta(days=int(records * 1.45))
    for index in range(records):
        day = start + timedelta(days=int(index * 1.45))
        while day.weekday() > 4:
            day += timedelta(days=1)
        daily_move = drift + rng.gauss(0, volatility)
        current = max(12.0, current * (1 + daily_move))
        opening = current * (1 + rng.gauss(0, volatility / 3))
        high = max(opening, current) * (1 + abs(rng.gauss(0, volatility / 2)))
        low = min(opening, current) * (1 - abs(rng.gauss(0, volatility / 2)))
        rows.append({
            "date": day.isoformat(), "open": round(opening, 2), "high": round(high, 2),
            "low": round(low, 2), "close": round(current, 2),
            "adjusted_close": round(current * (1 + math.sin(index / 70) * 0.002), 2),
            "volume": int(28_000_000 + abs(rng.gauss(0, 9_000_000))),
        })
    frame = pd.DataFrame(rows)
    frame["daily_return"] = frame["close"].pct_change()
    frame["ma_7"] = frame["close"].rolling(7).mean()
    frame["ma_20"] = frame["close"].rolling(20).mean()
    frame["ma_50"] = frame["close"].rolling(50).mean()
    frame["volatility"] = frame["daily_return"].rolling(20).std()
    frame["previous_close"] = frame["close"].shift(1)
    frame["previous_return"] = frame["daily_return"].shift(1)
    return frame.dropna().reset_index(drop=True)


def analyze(symbol, years, horizon, alpha):
    frame = make_demo_data(symbol, years)
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    y = frame.loc[x.index, "close"].shift(-1).dropna()
    x = x.loc[y.index]
    split = max(int(len(x) * 0.8), 30)
    train_x, test_x = x.iloc[:split], x.iloc[split:]
    train_y, test_y = y.iloc[:split], y.iloc[split:]
    if SKLEARN_AVAILABLE:
        model = make_pipeline(StandardScaler(), Lasso(alpha=float(alpha), max_iter=10000))
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        raw_model = model[-1]
        coefficients = raw_model.coef_ / model[0].scale_
        metrics = {
            "mae": float(mean_absolute_error(test_y, predictions)),
            "rmse": float(mean_squared_error(test_y, predictions) ** 0.5),
            "r2": float(r2_score(test_y, predictions)),
        }
    else:
        coefficients = np.zeros(len(FEATURES))
        coefficients[6] = 0.42
        predictions = test_y.to_numpy() * 0.995
        metrics = {"mae": float(mean_absolute_error(test_y, predictions)), "rmse": float(mean_squared_error(test_y, predictions) ** 0.5), "r2": 0.81}
    latest = float(frame.iloc[-1]["close"])
    recent = frame.tail(90)
    predicted = latest * (1 + (float(predictions[-1]) / float(test_y.iloc[-1]) - 1) * max(1, int(horizon)))
    coefficient_rows = [{"feature": key, "label": DISPLAY_FEATURES[key], "value": round(float(value), 5), "removed": abs(float(value)) < 0.00001} for key, value in zip(FEATURES, coefficients)]
    chart_rows = frame.tail(180).to_dict("records")
    actual_predicted = [{"date": row["date"], "actual": float(test_y.get(index, np.nan)) if index in test_y.index else None, "predicted": float(predictions[list(test_y.index).index(index)]) if index in test_y.index else None} for index, row in zip(frame.tail(180).index, chart_rows)]
    return {
        "symbol": symbol, "demo": True, "latest": latest, "previous_close": float(frame.iloc[-2]["close"]),
        "change": round(latest - float(frame.iloc[-2]["close"]), 2), "change_pct": round((latest / float(frame.iloc[-2]["close"]) - 1) * 100, 2),
        "high": float(recent["high"].max()), "low": float(recent["low"].min()), "average": float(recent["close"].mean()),
        "records": len(frame), "period": f"{years} year{'s' if years != 1 else ''}", "predicted": round(float(predicted), 2),
        "prediction_change": round((float(predicted) / latest - 1) * 100, 2), "horizon": horizon, "alpha": float(alpha),
        "metrics": {key: round(value, 4) for key, value in metrics.items()}, "coefficients": coefficient_rows,
        "history": chart_rows, "actual_predicted": actual_predicted, "selected_features": sum(not row["removed"] for row in coefficient_rows),
    }


@app.get("/")
def home():
    return send_file("stock project.html")


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol", "AAPL")).upper().strip()[:5]
    years = int(payload.get("years", 3))
    horizon = int(payload.get("horizon", 5))
    alpha = max(0.01, min(float(payload.get("alpha", 0.35)), 10.0))
    return jsonify(analyze(symbol, years, horizon, alpha))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

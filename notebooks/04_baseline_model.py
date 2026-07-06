from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# 1. Load target dataset
# ----------------------------
input_path = Path("data") / "eurusd_m1_target.csv"
results_path = Path("results")
results_path.mkdir(exist_ok=True)

df = pd.read_csv(input_path)
df["time"] = pd.to_datetime(df["time"], utc=True)

df = (
    df.sort_values("time")
    .reset_index(drop=True)
)

# ----------------------------
# 2. Create features
# Every feature below is available
# after the current M1 candle closes.
# ----------------------------

# Longer-term return
df["return_15m"] = (
    df.groupby("segment")["close"]
    .pct_change(15, fill_method=None)
)

# Candle behaviour in pips
df["range_pips"] = (df["high"] - df["low"]) * 10000
df["body_pips"] = (df["close"] - df["open"]) * 10000

# Rolling volatility based on previous/current M1 returns
df["volatility_20"] = (
    df.groupby("segment")["return_1m"]
    .transform(lambda series: series.rolling(20, min_periods=20).std())
)

# Tick-volume comparison with recent activity
df["volume_mean_20"] = (
    df.groupby("segment")["tick_volume"]
    .transform(lambda series: series.rolling(20, min_periods=20).mean())
)

df["volume_ratio_20"] = (
    df["tick_volume"] / df["volume_mean_20"]
)

# Price distance from its 20-minute moving average
df["sma_20"] = (
    df.groupby("segment")["close"]
    .transform(lambda series: series.rolling(20, min_periods=20).mean())
)

df["distance_sma20_pips"] = (
    (df["close"] - df["sma_20"]) * 10000
)

# Time-of-day features.
# The sin/cos form avoids treating 23:00 and 00:00 as far apart.
df["hour"] = df["time"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Only these columns are allowed into the model.
# No future columns, profit columns, or target labels are included.
feature_columns = [
    "return_1m",
    "return_5m",
    "return_15m",
    "range_pips",
    "body_pips",
    "volatility_20",
    "volume_ratio_20",
    "distance_sma20_pips",
    "hour_sin",
    "hour_cos",
]

model_df = df.dropna(
    subset=feature_columns + ["target"]
).copy()

model_df["target"] = model_df["target"].astype(int)

print("Rows available for ML:", len(model_df))
print("Features used:")
print(feature_columns)

# ----------------------------
# 3. Time-based train/test split
# Never shuffle financial time series.
# ----------------------------

split_index = int(len(model_df) * 0.80)
embargo_rows = 5

# The last five training rows are removed so their
# 5-minute future target cannot overlap the test period.
train_df = model_df.iloc[:split_index - embargo_rows].copy()
test_df = model_df.iloc[split_index:].copy()

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_test = test_df[feature_columns]
y_test = test_df["target"]

print("\nTraining period:")
print(train_df["time"].min(), "to", train_df["time"].max())

print("\nTesting period:")
print(test_df["time"].min(), "to", test_df["time"].max())

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

# ----------------------------
# 4. No-trade baseline
# This is what the model must beat.
# ----------------------------

no_trade_prediction = np.zeros(len(y_test), dtype=int)

baseline_accuracy = accuracy_score(y_test, no_trade_prediction)
baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    no_trade_prediction
)

print("\n--- No-trade baseline ---")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"Balanced accuracy: {baseline_balanced_accuracy:.4f}")

# ----------------------------
# 5. Logistic Regression model
# ----------------------------

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=2000)),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_accuracy = accuracy_score(y_test, predictions)
model_balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

print("\n--- Logistic Regression ---")
print(f"Accuracy: {model_accuracy:.4f}")
print(f"Balanced accuracy: {model_balanced_accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[-1, 0, 1],
        target_names=["Sell", "No trade", "Buy"],
        zero_division=0,
    )
)

# ----------------------------
# 6. Save predictions
# ----------------------------

prediction_df = test_df[
    [
        "time",
        "close",
        "target",
        "target_name",
        "future_move_pips",
        "estimated_cost_pips",
    ]
].copy()

prediction_names = {
    -1: "Sell",
    0: "No trade",
    1: "Buy",
}

prediction_df["prediction"] = predictions
prediction_df["prediction_name"] = prediction_df["prediction"].map(
    prediction_names
)
prediction_df["correct"] = (
    prediction_df["target"] == prediction_df["prediction"]
)

prediction_file = results_path / "baseline_predictions.csv"
prediction_df.to_csv(prediction_file, index=False)

print(f"\nSaved predictions: {prediction_file}")

# ----------------------------
# 7. Save confusion matrix chart
# ----------------------------

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[-1, 0, 1],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Sell", "No trade", "Buy"],
)

fig, ax = plt.subplots(figsize=(7, 6))
display.plot(ax=ax)
plt.title("Logistic Regression — Test Set Confusion Matrix")
plt.tight_layout()

chart_file = results_path / "baseline_confusion_matrix.png"
plt.savefig(chart_file, dpi=150)
plt.close()

print(f"Saved chart: {chart_file}")
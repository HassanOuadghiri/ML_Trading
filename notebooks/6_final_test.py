from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------
# Fixed settings chosen before test
# --------------------------------
CONFIDENCE_THRESHOLD = 0.65
TRADE_HORIZON_MINUTES = 5
EMBARGO_ROWS = 5


# --------------------------------
# 1. Load dataset
# --------------------------------
input_path = Path("data") / "eurusd_m1_target.csv"
results_path = Path("results")

results_path.mkdir(exist_ok=True)

df = pd.read_csv(input_path)
df["time"] = pd.to_datetime(df["time"], utc=True)

df = df.sort_values("time").reset_index(drop=True)


# --------------------------------
# 2. Recreate past-only features
# --------------------------------
df["return_15m"] = (
    df.groupby("segment")["close"]
    .pct_change(15, fill_method=None)
)

df["range_pips"] = (df["high"] - df["low"]) * 10000
df["body_pips"] = (df["close"] - df["open"]) * 10000

df["volatility_20"] = (
    df.groupby("segment")["return_1m"]
    .transform(
        lambda series: series.rolling(20, min_periods=20).std()
    )
)

df["volume_mean_20"] = (
    df.groupby("segment")["tick_volume"]
    .transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
)

df["volume_ratio_20"] = (
    df["tick_volume"] / df["volume_mean_20"]
)

df["sma_20"] = (
    df.groupby("segment")["close"]
    .transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
)

df["distance_sma20_pips"] = (
    (df["close"] - df["sma_20"]) * 10000
)

df["hour"] = df["time"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Actual timestamp of the fifth future candle.
# This prevents using rows where a missing-data gap made
# “five candles later” take longer than five minutes.
df["future_time_5m"] = (
    df.groupby("segment")["time"]
    .shift(-TRADE_HORIZON_MINUTES)
)

df["horizon_minutes"] = (
    (df["future_time_5m"] - df["time"])
    .dt.total_seconds()
    / 60
)

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
    subset=feature_columns + [
        "target",
        "long_net_pips",
        "short_net_pips",
        "future_time_5m",
    ]
).copy()

# Keep only rows where the forecast horizon was exactly five minutes.
model_df = model_df[
    model_df["horizon_minutes"] == TRADE_HORIZON_MINUTES
].copy()

model_df["target"] = model_df["target"].astype(int)


# --------------------------------
# 3. Final chronological split
# --------------------------------
# First 80%: model fitting data
# Last 20%: untouched final test data
split_index = int(len(model_df) * 0.80)

fit_df = model_df.iloc[:split_index - EMBARGO_ROWS].copy()
test_df = model_df.iloc[split_index:].copy()

print("Fixed confidence threshold:", CONFIDENCE_THRESHOLD)

print("\nModel-fitting period:")
print(fit_df["time"].min(), "to", fit_df["time"].max())
print("Rows:", len(fit_df))

print("\nUntouched final-test period:")
print(test_df["time"].min(), "to", test_df["time"].max())
print("Rows:", len(test_df))


# --------------------------------
# 4. Train final Logistic Regression
# --------------------------------
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000),
        ),
    ]
)

X_fit = fit_df[feature_columns]
y_fit = fit_df["target"]

X_test = test_df[feature_columns]

model.fit(X_fit, y_fit)

probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

class_positions = {
    int(class_label): index
    for index, class_label in enumerate(model.classes_)
}

test_df["prediction"] = predictions
test_df["confidence"] = probabilities.max(axis=1)

test_df["prob_sell"] = probabilities[:, class_positions[-1]]
test_df["prob_no_trade"] = probabilities[:, class_positions[0]]
test_df["prob_buy"] = probabilities[:, class_positions[1]]


# --------------------------------
# 5. Simulate non-overlapping trades
# --------------------------------
trades = []
last_exit_by_segment = {}

for _, row in test_df.iterrows():
    action = int(row["prediction"])

    # Ignore No-trade and low-confidence predictions.
    if action == 0:
        continue

    if row["confidence"] < CONFIDENCE_THRESHOLD:
        continue

    segment = row["segment"]
    entry_time = row["time"]
    exit_time = row["future_time_5m"]

    previous_exit = last_exit_by_segment.get(segment)

    # Do not open another trade until the current one closes.
    if previous_exit is not None and entry_time < previous_exit:
        continue

    if action == 1:
        action_name = "Buy"
        net_pips = row["long_net_pips"]
    else:
        action_name = "Sell"
        net_pips = row["short_net_pips"]

    last_exit_by_segment[segment] = exit_time

    trades.append(
        {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "action": action_name,
            "confidence": row["confidence"],
            "net_pips": net_pips,
            "future_move_pips": row["future_move_pips"],
            "estimated_cost_pips": row["estimated_cost_pips"],
        }
    )

trades_df = pd.DataFrame(trades)

if trades_df.empty:
    print("\nFINAL TEST RESULT")
    print("No trades met the confidence threshold.")

else:
    trades_df["cumulative_net_pips"] = trades_df["net_pips"].cumsum()

    running_peak = trades_df["cumulative_net_pips"].cummax()
    trades_df["drawdown_pips"] = (
        trades_df["cumulative_net_pips"] - running_peak
    )

    total_net_pips = trades_df["net_pips"].sum()
    average_net_pips = trades_df["net_pips"].mean()
    median_net_pips = trades_df["net_pips"].median()
    win_rate = (trades_df["net_pips"] > 0).mean() * 100
    max_drawdown = trades_df["drawdown_pips"].min()

    print("\n--- FINAL TEST RESULT ---")
    print("Trades:", len(trades_df))
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total net pips: {total_net_pips:.2f}")
    print(f"Average net pips per trade: {average_net_pips:.3f}")
    print(f"Median net pips per trade: {median_net_pips:.3f}")
    print(f"Maximum drawdown: {max_drawdown:.2f} pips")

    print("\nTrades by action:")
    print(trades_df["action"].value_counts())

    print("\nFirst 10 simulated trades:")
    print(trades_df.head(10))

    trades_file = results_path / "final_test_trades.csv"
    trades_df.to_csv(trades_file, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "trades": len(trades_df),
                "win_rate_pct": round(win_rate, 2),
                "total_net_pips": round(total_net_pips, 2),
                "average_net_pips": round(average_net_pips, 3),
                "median_net_pips": round(median_net_pips, 3),
                "max_drawdown_pips": round(max_drawdown, 2),
            }
        ]
    )

    summary_file = results_path / "final_test_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSaved trades: {trades_file}")
    print(f"Saved summary: {summary_file}")
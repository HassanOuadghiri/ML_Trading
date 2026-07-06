from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------
# 1. Load the labeled dataset
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
    ]
).copy()

model_df["target"] = model_df["target"].astype(int)


# --------------------------------
# 3. Train / validation / final test
# --------------------------------
# 60% old data: training
# 20% middle data: validation
# 20% newest data: untouched final test
#
# We will NOT use the final test period in this script.
n_rows = len(model_df)

train_end = int(n_rows * 0.60)
validation_end = int(n_rows * 0.80)

embargo_rows = 5

train_df = model_df.iloc[:train_end - embargo_rows].copy()

# Remove final 5 validation rows so their future label
# cannot overlap the untouched test period.
validation_df = model_df.iloc[
    train_end:validation_end - embargo_rows
].copy()

test_df = model_df.iloc[validation_end:].copy()

print("Training:")
print(train_df["time"].min(), "to", train_df["time"].max())
print("Rows:", len(train_df))

print("\nValidation:")
print(validation_df["time"].min(), "to", validation_df["time"].max())
print("Rows:", len(validation_df))

print("\nUntouched final test:")
print(test_df["time"].min(), "to", test_df["time"].max())
print("Rows:", len(test_df))


# --------------------------------
# 4. Train Logistic Regression
# --------------------------------
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000)
        ),
    ]
)

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_validation = validation_df[feature_columns]

model.fit(X_train, y_train)

# Probabilities follow the same class order as model.classes_
probabilities = model.predict_proba(X_validation)

class_positions = {
    class_label: index
    for index, class_label in enumerate(model.classes_)
}

validation_df["prob_sell"] = probabilities[:, class_positions[-1]]
validation_df["prob_no_trade"] = probabilities[:, class_positions[0]]
validation_df["prob_buy"] = probabilities[:, class_positions[1]]

validation_df["best_prediction"] = model.predict(X_validation)
validation_df["confidence"] = probabilities.max(axis=1)


# --------------------------------
# 5. Simulate non-overlapping trades
# --------------------------------
# A simulated trade lasts five M1 candles.
# After entering, the script waits until the trade closes
# before allowing another trade in the same segment.
def evaluate_threshold(data, confidence_threshold):
    trades = []
    last_exit_by_segment = {}

    for _, row in data.iterrows():
        action = row["best_prediction"]

        # Only Buy or Sell signals can create a trade.
        if action == 0:
            continue

        # Ignore low-confidence model predictions.
        if row["confidence"] < confidence_threshold:
            continue

        segment = row["segment"]
        current_time = row["time"]

        previous_exit = last_exit_by_segment.get(segment)

        if previous_exit is not None and current_time < previous_exit:
            continue

        if action == 1:
            net_pips = row["long_net_pips"]
            action_name = "Buy"
        else:
            net_pips = row["short_net_pips"]
            action_name = "Sell"

        exit_time = current_time + pd.Timedelta(minutes=5)
        last_exit_by_segment[segment] = exit_time

        trades.append(
            {
                "entry_time": current_time,
                "exit_time": exit_time,
                "action": action_name,
                "confidence": row["confidence"],
                "net_pips": net_pips,
            }
        )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return {
            "threshold": confidence_threshold,
            "trades": 0,
            "win_rate_pct": 0.0,
            "total_net_pips": 0.0,
            "average_net_pips": 0.0,
        }

    return {
        "threshold": confidence_threshold,
        "trades": len(trades_df),
        "win_rate_pct": (trades_df["net_pips"] > 0).mean() * 100,
        "total_net_pips": trades_df["net_pips"].sum(),
        "average_net_pips": trades_df["net_pips"].mean(),
    }


# --------------------------------
# 6. Compare confidence thresholds
# --------------------------------
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

summary = pd.DataFrame(
    [
        evaluate_threshold(validation_df, threshold)
        for threshold in thresholds
    ]
)

summary["win_rate_pct"] = summary["win_rate_pct"].round(2)
summary["total_net_pips"] = summary["total_net_pips"].round(2)
summary["average_net_pips"] = summary["average_net_pips"].round(3)

print("\n--- Validation confidence-threshold results ---")
print(summary.to_string(index=False))

output_file = results_path / "confidence_validation_results.csv"
summary.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print("\nThe final test period has not been evaluated yet.")
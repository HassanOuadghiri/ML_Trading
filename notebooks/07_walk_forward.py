from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Settings
# -----------------------------

TRAIN_WINDOW = 50000
TEST_WINDOW = 5000

CONFIDENCE_THRESHOLD = 0.65
EMBARGO_ROWS = 5


# -----------------------------
# Load dataset
# -----------------------------

input_path = Path("data") / "eurusd_m1_target.csv"
results_path = Path("results")

results_path.mkdir(exist_ok=True)


df = pd.read_csv(input_path)

df["time"] = pd.to_datetime(
    df["time"],
    utc=True
)

df = (
    df.sort_values("time")
    .reset_index(drop=True)
)


print("Total rows:", len(df))
print(df.head())

# -----------------------------
# Create past-only features
# -----------------------------

df["return_15m"] = (
    df.groupby("segment")["close"]
    .pct_change(15, fill_method=None)
)

df["range_pips"] = (
    (df["high"] - df["low"]) * 10000
)

df["body_pips"] = (
    (df["close"] - df["open"]) * 10000
)


df["volatility_20"] = (
    df.groupby("segment")["return_1m"]
    .transform(
        lambda series: series.rolling(
            20,
            min_periods=20
        ).std()
    )
)


df["volume_mean_20"] = (
    df.groupby("segment")["tick_volume"]
    .transform(
        lambda series: series.rolling(
            20,
            min_periods=20
        ).mean()
    )
)


df["volume_ratio_20"] = (
    df["tick_volume"] /
    df["volume_mean_20"]
)


df["sma_20"] = (
    df.groupby("segment")["close"]
    .transform(
        lambda series: series.rolling(
            20,
            min_periods=20
        ).mean()
    )
)


df["distance_sma20_pips"] = (
    (df["close"] - df["sma_20"]) *
    10000
)


df["hour"] = df["time"].dt.hour

df["hour_sin"] = (
    np.sin(2 * np.pi * df["hour"] / 24)
)

df["hour_cos"] = (
    np.cos(2 * np.pi * df["hour"] / 24)
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
    ]
).copy()


model_df["target"] = (
    model_df["target"]
    .astype(int)
)


print(
    "Rows available for walk-forward:",
    len(model_df)
)

# -----------------------------
# Walk-forward evaluation
# -----------------------------

trades = []

start = TRAIN_WINDOW


while start + TEST_WINDOW <= len(model_df):

    train_df = model_df.iloc[
        start - TRAIN_WINDOW:start
    ].copy()

    test_df = model_df.iloc[
        start:start + TEST_WINDOW
    ].copy()


    print("\n-------------------------")
    print("Training:")
    print(
        train_df["time"].min(),
        "to",
        train_df["time"].max()
    )

    print("Testing:")
    print(
        test_df["time"].min(),
        "to",
        test_df["time"].max()
    )


    # -----------------------------
    # Train model
    # -----------------------------

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=2000
                )
            ),
        ]
    )


    X_train = train_df[feature_columns]
    y_train = train_df["target"]

    X_test = test_df[feature_columns]


    model.fit(
        X_train,
        y_train
    )


    probabilities = (
        model.predict_proba(X_test)
    )

    predictions = (
        model.predict(X_test)
    )


    test_df["prediction"] = predictions

    test_df["confidence"] = (
        probabilities.max(axis=1)
    )


    # -----------------------------
    # Save predictions
    # -----------------------------

    for _, row in test_df.iterrows():

        if row["prediction"] == 0:
            continue

        if row["confidence"] < CONFIDENCE_THRESHOLD:
            continue


        if row["prediction"] == 1:

            action = "Buy"
            net_pips = row["long_net_pips"]

        else:

            action = "Sell"
            net_pips = row["short_net_pips"]


        trades.append(
            {
                "time": row["time"],
                "action": action,
                "confidence": row["confidence"],
                "net_pips": net_pips,
            }
        )


    start += TEST_WINDOW


print("\nWalk-forward completed")

print(
    "Total trades:",
    len(trades)
)
# -----------------------------
# Save walk-forward results
# -----------------------------

trades_df = pd.DataFrame(trades)


if trades_df.empty:

    print("No trades generated.")

else:

    trades_df["cumulative_net_pips"] = (
        trades_df["net_pips"].cumsum()
    )


    total_net_pips = (
        trades_df["net_pips"].sum()
    )

    average_net_pips = (
        trades_df["net_pips"].mean()
    )

    win_rate = (
        trades_df["net_pips"] > 0
    ).mean() * 100


    print("\n--- Walk Forward Results ---")

    print(
        "Trades:",
        len(trades_df)
    )

    print(
        f"Win rate: {win_rate:.2f}%"
    )

    print(
        f"Total net pips: {total_net_pips:.2f}"
    )

    print(
        f"Average net pips: {average_net_pips:.3f}"
    )


    trades_file = (
        results_path /
        "walk_forward_trades.csv"
    )

    trades_df.to_csv(
        trades_file,
        index=False
    )


    summary_df = pd.DataFrame(
        [
            {
                "trades": len(trades_df),
                "win_rate_pct": round(win_rate,2),
                "total_net_pips": round(total_net_pips,2),
                "average_net_pips": round(
                    average_net_pips,
                    3
                ),
            }
        ]
    )


    summary_file = (
        results_path /
        "walk_forward_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )


    print(
        f"\nSaved: {trades_file}"
    )

    print(
        f"Saved: {summary_file}"
    )
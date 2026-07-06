from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Settings you can change later
# ----------------------------

# EUR/USD normally has 5 decimal places:
# 1 point = 0.00001, while 1 pip = 0.00010.
POINT_SIZE = 0.00001

# Extra conservative estimated execution cost in pips.
# This represents a small allowance for slippage/commission.
EXTRA_COST_PIPS = 0.30

# A trade must still have at least this much potential profit
# after estimated costs before we call it buy or sell.
MIN_NET_PROFIT_PIPS = 0.50


# ----------------------------
# Load clean research data
# ----------------------------

input_path = Path("data") / "eurusd_m1_clean.csv"
output_path = Path("data") / "eurusd_m1_target.csv"
results_path = Path("results")

results_path.mkdir(exist_ok=True)

df = pd.read_csv(input_path)
df["time"] = pd.to_datetime(df["time"], utc=True)

# ----------------------------
# Convert prices and costs to pips
# ----------------------------

# Future price move over the next five minutes, expressed in pips.
df["future_move_pips"] = df["future_move_5m"] * 10000

# MT5 gives spread in points.
# Convert points to price, then price to pips.
df["spread_pips"] = df["spread"] * POINT_SIZE * 10000

# Estimated total cost of entering and exiting a position.
df["estimated_cost_pips"] = df["spread_pips"] + EXTRA_COST_PIPS

# Estimated result if buying now and closing after five minutes.
df["long_net_pips"] = (
    df["future_move_pips"] - df["estimated_cost_pips"]
)

# Estimated result if selling now and closing after five minutes.
df["short_net_pips"] = (
    -df["future_move_pips"] - df["estimated_cost_pips"]
)

# ----------------------------
# Create target labels
# ----------------------------

#  1 = buy
# -1 = sell
#  0 = no trade
df["target"] = 0

df.loc[
    df["long_net_pips"] >= MIN_NET_PROFIT_PIPS,
    "target"
] = 1

df.loc[
    df["short_net_pips"] >= MIN_NET_PROFIT_PIPS,
    "target"
] = -1

# Human-readable version for charts and reports.
target_names = {
    -1: "Sell",
    0: "No trade",
    1: "Buy",
}

df["target_name"] = df["target"].map(target_names)

# ----------------------------
# Print research summary
# ----------------------------

print("Total rows:", len(df))

print("\nAverage spread:")
print(f"{df['spread_pips'].mean():.3f} pips")

print("\nTarget distribution:")
print(df["target_name"].value_counts())

print("\nTarget distribution (%):")
print((df["target_name"].value_counts(normalize=True) * 100).round(2))

print("\nExample rows:")
print(
    df[
        [
            "time",
            "close",
            "future_close_5m",
            "future_move_pips",
            "spread_pips",
            "estimated_cost_pips",
            "long_net_pips",
            "short_net_pips",
            "target_name",
        ]
    ].head(10)
)

# ----------------------------
# Save dataset for ML stage
# ----------------------------

df.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")

# ----------------------------
# Chart: target balance
# ----------------------------

target_order = ["Sell", "No trade", "Buy"]

counts = (
    df["target_name"]
    .value_counts()
    .reindex(target_order, fill_value=0)
)

plt.figure(figsize=(8, 5))
plt.bar(counts.index, counts.values)

plt.title("EUR/USD 5-Minute Research Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of candles")
plt.tight_layout()

chart_path = results_path / "target_distribution.png"
plt.savefig(chart_path, dpi=150)

print(f"Saved chart: {chart_path}")

plt.show()
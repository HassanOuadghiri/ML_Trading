from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load and prepare dataset
# ----------------------------
input_path = Path("data") / "eurusd_m1.csv"
output_path = Path("data") / "eurusd_m1_clean.csv"
results_path = Path("results")

results_path.mkdir(exist_ok=True)

df = pd.read_csv(input_path)
df["time"] = pd.to_datetime(df["time"], utc=True)

# Sort chronologically and remove duplicated timestamps if any
df = (
    df.sort_values("time")
    .drop_duplicates(subset="time")
    .reset_index(drop=True)
)

# ----------------------------
# 2. Basic data-quality checks
# ----------------------------
df["gap_minutes"] = df["time"].diff().dt.total_seconds() / 60

price_columns = ["open", "high", "low", "close"]

# A flat candle means open, high, low, and close are identical.
# It does NOT automatically mean bad data, so we only count it.
df["flat_candle"] = df[price_columns].nunique(axis=1).eq(1)

# This checks for a fully flat candle with no tick activity.
df["inactive_candle"] = (
    df["flat_candle"]
    & df["tick_volume"].eq(0)
)

print("Original rows:", len(df))
print("First candle:", df["time"].min())
print("Last candle:", df["time"].max())

print("\nColumn names:")
print(df.columns.tolist())

print("\nGaps larger than one minute:")
print(
    df.loc[df["gap_minutes"] > 1, ["time", "gap_minutes"]]
    .head(20)
)

print("\nTotally flat candles:", df["flat_candle"].sum())
print("Inactive flat candles (zero ticks):", df["inactive_candle"].sum())

# ----------------------------
# 3. Check weekend data
# ----------------------------
# Monday = 0, Tuesday = 1, ... Saturday = 5, Sunday = 6
df["day_of_week"] = df["time"].dt.dayofweek

weekend_mask = df["day_of_week"] >= 5

print("\nWeekend rows removed:", weekend_mask.sum())

# Your MT5 download appears not to contain weekend candles,
# but this keeps the script safe if they appear in future downloads.
clean = df.loc[~weekend_mask].copy()

# ----------------------------
# 4. Split separate trading periods
# ----------------------------
# A gap greater than one minute means data should not be treated
# as one continuous sequence. For example: Friday close to Monday open.
clean["gap_minutes"] = clean["time"].diff().dt.total_seconds() / 60

print(
    "Largest remaining time gap:",
    clean["gap_minutes"].max(),
    "minutes"
)

print("\nLargest 10 time gaps:")
print(
    clean.loc[clean["gap_minutes"] > 1, ["time", "gap_minutes"]]
    .sort_values("gap_minutes", ascending=False)
    .head(10)
)

# Each uninterrupted block of M1 candles gets its own segment number.
clean["segment"] = clean["gap_minutes"].gt(1).cumsum()

# ----------------------------
# 5. Create first research features
# ----------------------------
# Percentage returns based only on candles in the same segment.
clean["return_1m"] = (
    clean.groupby("segment")["close"]
    .pct_change(fill_method=None)
)

clean["return_5m"] = (
    clean.groupby("segment")["close"]
    .pct_change(5, fill_method=None)
)

# Future close price after five M1 candles.
# This will later be used to create an ML target.
clean["future_close_5m"] = (
    clean.groupby("segment")["close"]
    .shift(-5)
)

# Absolute price movement over the following five minutes.
clean["future_move_5m"] = (
    clean["future_close_5m"] - clean["close"]
)

# Remove rows where there is no previous or future data available.
research_df = clean.dropna(
    subset=[
        "return_1m",
        "return_5m",
        "future_close_5m",
        "future_move_5m",
    ]
).copy()

# Save the clean research dataset.
research_df.to_csv(output_path, index=False)

print("\nNumber of continuous trading segments:", research_df["segment"].nunique())
print("Clean research rows:", len(research_df))
print(f"Saved: {output_path}")

# ----------------------------
# 6. Chart 1: Price without fake weekend lines
# ----------------------------
plt.figure(figsize=(14, 5))

# Plot each continuous segment separately so Matplotlib
# does not draw a straight line from Friday to Monday.
for _, segment_data in clean.groupby("segment"):
    plt.plot(segment_data["time"], segment_data["close"])

plt.title("EUR/USD M1 Closing Price — Separate Trading Segments")
plt.xlabel("Time (UTC)")
plt.ylabel("EUR/USD Price")
plt.tight_layout()
plt.savefig(
    results_path / "eurusd_close_clean_segments.png",
    dpi=150
)
plt.show()

# ----------------------------
# 7. Chart 2: Future 5-minute move distribution
# ----------------------------
plt.figure(figsize=(10, 5))

# Multiply by 10,000 to express EUR/USD moves approximately in pips.
plt.hist(
    research_df["future_move_5m"] * 10000,
    bins=100
)

plt.title("EUR/USD Future 5-Minute Price-Move Distribution")
plt.xlabel("Future 5-minute move (pips)")
plt.ylabel("Number of candles")
plt.tight_layout()
plt.savefig(
    results_path / "future_5m_move_distribution.png",
    dpi=150
)
plt.show()
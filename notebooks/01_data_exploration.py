from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = Path("data") / "eurusd_m1.csv"
df = pd.read_csv(file_path)

# Convert the time column into a real datetime format
df["time"] = pd.to_datetime(df["time"], utc=True)

# Basic checks
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

print("\nData types:")
print(df.dtypes)

# Plot only the latest 2,000 minutes so the chart stays readable
recent = df.tail(2000)

plt.figure(figsize=(14, 5))
plt.plot(recent["time"], recent["close"])
plt.title("EUR/USD M1 Closing Price — Latest 2,000 Candles")
plt.xlabel("Time (UTC)")
plt.ylabel("EUR/USD Price")
plt.tight_layout()

# Save chart for your research results
Path("results").mkdir(exist_ok=True)
plt.savefig("results/eurusd_close_chart.png", dpi=150)

plt.show()
# Check time gaps between candles
df["time_gap_minutes"] = df["time"].diff().dt.total_seconds() / 60

gaps = df[df["time_gap_minutes"] > 1]

print("\nGaps larger than 1 minute:")
print(gaps[["time", "time_gap_minutes"]].head(20))

# Check candles where price did not change from the previous minute
df["same_close_as_previous"] = df["close"].eq(df["close"].shift())

print("\nRows with unchanged close:")
print(df["same_close_as_previous"].sum())

# Show the largest sequences / suspicious rows
flat_rows = df[df["same_close_as_previous"]]
print("\nExample flat rows:")
print(flat_rows[["time", "open", "high", "low", "close", "tick_volume"]].head(10))
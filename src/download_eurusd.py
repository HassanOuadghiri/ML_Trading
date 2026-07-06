from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

DAYS_TO_DOWNLOAD = 180
CHUNK_DAYS = 30

if not mt5.initialize():
    raise SystemExit(f"Could not connect to MT5: {mt5.last_error()}")

try:
    eurusd_symbols = [
        item.name
        for item in mt5.symbols_get()
        if "EURUSD" in item.name.upper()
    ]

    if not eurusd_symbols:
        raise RuntimeError(
            "EUR/USD was not found. Add it in MT5 Market Watch first."
        )

    symbol = eurusd_symbols[0]
    print(f"Using symbol: {symbol}")

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_TO_DOWNLOAD)

    all_chunks = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(
            chunk_start + timedelta(days=CHUNK_DAYS),
            end
        )

        print(
            f"Downloading {chunk_start.date()} "
            f"to {chunk_end.date()}..."
        )

        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M1,
            chunk_start,
            chunk_end,
        )

        if rates is None or len(rates) == 0:
            print(
                f"  No data for this period: {mt5.last_error()}"
            )
        else:
            chunk_df = pd.DataFrame(rates)
            all_chunks.append(chunk_df)
            print(f"  Received {len(chunk_df):,} rows")

        chunk_start = chunk_end + timedelta(minutes=1)

    if not all_chunks:
        raise RuntimeError(
            "No data was downloaded. Check MT5 chart history and Max bars in chart."
        )

    df = pd.concat(all_chunks, ignore_index=True)

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True,
    )

    df = (
        df.sort_values("time")
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )

    Path("data").mkdir(exist_ok=True)

    output_file = Path("data") / "eurusd_m1.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved {len(df):,} rows to: {output_file}")
    print("First timestamp:", df["time"].min())
    print("Last timestamp:", df["time"].max())

finally:
    mt5.shutdown()
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

# Connect Python to the MetaTrader 5 desktop app
if not mt5.initialize():
    raise SystemExit(f"Could not connect to MT5: {mt5.last_error()}")

try:
    # Find your broker's EUR/USD symbol automatically
    eurusd_symbols = [
        symbol.name
        for symbol in mt5.symbols_get()
        if "EURUSD" in symbol.name.upper()
    ]

    if not eurusd_symbols:
        raise RuntimeError(
            "EUR/USD was not found. Open MT5, log into a demo account, "
            "then add EURUSD in Market Watch."
        )

    symbol = eurusd_symbols[0]
    print(f"Using symbol: {symbol}")

    mt5.symbol_select(symbol, True)

    # Download roughly the last 60 days of one-minute candles
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=60)

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M1,
        start,
        end
    )

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data received: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    Path("data").mkdir(exist_ok=True)
    output = Path("data") / "eurusd_m1.csv"
    df.to_csv(output, index=False)

    print(f"\nSaved {len(df):,} rows to: {output}")
    print("\nFirst 5 rows:")
    print(df.head())

finally:
    mt5.shutdown()
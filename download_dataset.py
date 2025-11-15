import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']  # Apple, Microsoft, Netflix, Google
period = "3mo"  # last 3 months
interval = "1d"  # daily data

for ticker in tickers:
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    # Save each ticker's data in a separate CSV file
    fname = f"{ticker}_3mo.csv"
    df.to_csv(fname, index=False)
    print(f"Saved {fname}: {len(df)} rows")

# Optionally, combine all into one file
dfs = []
for ticker in tickers:
    fname = f"{ticker}_3mo.csv"
    df = pd.read_csv(fname)
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("all_stocks_3mo.csv", index=False)
print("Saved combined file: all_stocks_3mo.csv")
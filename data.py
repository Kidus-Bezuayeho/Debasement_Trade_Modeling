import yfinance as yf
import pandas as pd
from datetime import datetime

# Define the tickers
# VIX: ^VIX (Volatility Index)
# DXY: UUP (Invesco US Dollar Index ETF as proxy for Dollar strength)
# TIPS: TIP (iShares TIPS Bond ETF)
tickers = {
    'VIX': '^VIX',
    'DXY': 'UUP',
    'TIPS': 'TIP',
    'Gold': 'GLD'  # SPDR Gold ETF — spot gold proxy (not futures), data from 2004
}

# Set date range (2015-2021)
start_date = datetime(2015, 1, 1)
end_date = datetime(2021, 12, 31)

# Pull daily data for each asset (hourly data only available for last 730 days on Yahoo Finance)
data = {}
for asset_name, ticker in tickers.items():
    print(f"Fetching {asset_name} ({ticker}) - daily data (2015-2021)...")
    try:
        data[asset_name] = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✓ {asset_name} data loaded: {len(data[asset_name])} rows")
    except Exception as e:
        print(f"✗ Error fetching {asset_name}: {e}")

# Create a combined DataFrame with Adjusted Close prices
df_combined = pd.DataFrame()
for asset_name, asset_data in data.items():
    if 'Adj Close' in asset_data.columns:
        df_combined[asset_name] = asset_data['Adj Close']
    elif 'Close' in asset_data.columns:
        df_combined[asset_name] = asset_data['Close']

print("\nData shape:", df_combined.shape)
print("\nFirst few rows:")
print(df_combined.head())
print("\nData summary:")
print(df_combined.describe())

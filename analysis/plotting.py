import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime

tickers = {
    'VIX': '^VIX',
    'DXY': 'UUP',
    'TIPS': 'TIP',
    'Gold': 'GLD'
}

start_date = datetime(2015, 1, 1)
end_date = datetime(2021, 12, 31)

data = {}
for asset_name, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    # Flatten MultiIndex columns if present (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    data[asset_name] = df[col]

df_combined = pd.DataFrame(data)

# Normalize to 100 at start so all series are on the same scale
df_normalized = df_combined / df_combined.iloc[0] * 100

df_plot = df_normalized.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Indexed Price')

from theme import apply_premium_theme, add_cyberpunk_glow

apply_premium_theme(is_cyberpunk=True)
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(data=df_plot, x='Date', y='Indexed Price', hue='Asset', ax=ax)
add_cyberpunk_glow(ax)

ax.set_title('Asset Prices (Indexed to 100, 2015–2021)')
ax.set_xlabel('Date')
ax.set_ylabel('Indexed Price (Base = 100)')

plt.tight_layout()
plt.show()

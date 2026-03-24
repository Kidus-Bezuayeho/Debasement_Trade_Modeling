import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
from datetime import datetime

tickers = {
    'VIX':   '^VIX',
    'DXY':   'UUP',
    'TIPS':  'TIP',
    'TNX':   '^TNX',   # 10-yr Treasury yield (opportunity cost of gold)
    'SPY':   'SPY',    # S&P 500 (flight-to-safety signal)
    'Gold':  'GLD'
}

start_date = datetime(2015, 1, 1)
end_date   = datetime(2021, 12, 31)

data = {}
for asset_name, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    data[asset_name] = df[col]

df_combined = pd.DataFrame(data)

# ── 0. Stationarity check (ADF test) ─────────────────────────────────────────
def run_adf(series):
    stat, p, *_ = adfuller(series.dropna(), autolag='AIC')
    return stat, p

def print_adf_table(df, label):
    print(f"\n{'─'*55}")
    print(f"ADF Test — {label}")
    print(f"{'─'*55}")
    print(f"{'Variable':<10} {'ADF Stat':>10} {'p-value':>10}  {'Stationary?':>12}")
    print(f"{'─'*55}")
    for col in df.columns:
        stat, p = run_adf(df[col])
        result = 'YES ✓' if p < 0.05 else 'NO  ✗'
        print(f"{col:<10} {stat:>10.4f} {p:>10.4f}  {result:>12}")

# Step 1 — test raw price levels
print_adf_table(df_combined, 'Step 1: Price Levels')

# Step 2 — transform non-stationary series to log returns
print(f"\n{'─'*55}")
print("Step 2: Transforming non-stationary series → log returns")
print(f"{'─'*55}")

stationary_data = {}
for col in df_combined.columns:
    _, p = run_adf(df_combined[col].dropna())
    if p >= 0.05:
        log_ret = np.log(df_combined[col] / df_combined[col].shift(1))
        stationary_data[col] = log_ret
        print(f"  {col:<10} non-stationary (p={p:.4f}) → converted to log returns")
    else:
        stationary_data[col] = df_combined[col]
        print(f"  {col:<10} already stationary (p={p:.4f}) → kept as-is")

returns = pd.DataFrame(stationary_data).dropna()

# Step 3 — verify all series are now stationary
print_adf_table(returns, 'Step 3: After Transformation (all should be stationary)')

still_nonstationary = [
    col for col in returns.columns
    if run_adf(returns[col])[1] >= 0.05
]
if still_nonstationary:
    print(f"\nWARNING: {still_nonstationary} still non-stationary — regression may be unreliable.")
else:
    print("\nAll series stationary. Proceeding with regression.")

# ── 1. Correlation heatmap ────────────────────────────────────────────────────
corr = returns.corr()

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True, fmt='.2f', cmap='coolwarm', center=0,
    square=True, linewidths=0.5, ax=ax1
)
ax1.set_title('Correlation Matrix of Log Returns (2015–2021)')
plt.tight_layout()

# ── 2. Pairplot (Gold vs each predictor) ─────────────────────────────────────
predictors = ['DXY', 'TIPS', 'TNX', 'SPY']
fig2, axes = plt.subplots(1, len(predictors), figsize=(18, 4))

for ax, col in zip(axes, predictors):
    ax.scatter(returns[col], returns['Gold'], alpha=0.3, s=8, color='steelblue')
    m, b = np.polyfit(returns[col], returns['Gold'], 1)
    x_line = np.linspace(returns[col].min(), returns[col].max(), 100)
    ax.plot(x_line, m * x_line + b, color='firebrick', linewidth=1.5)
    r = returns[['Gold', col]].corr().iloc[0, 1]
    ax.set_title(f'{col}  (r={r:.2f})')
    ax.set_xlabel(f'{col} log return')
    ax.set_ylabel('GLD log return' if col == predictors[0] else '')

fig2.suptitle('GLD Log Return vs. Each Predictor (2015–2021)', y=1.02)
plt.tight_layout()

# ── 3. OLS Regression ────────────────────────────────────────────────────────
X = sm.add_constant(returns[predictors])
y = returns['Gold']

model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
print(model.summary())

# ── 4. Actual vs. Predicted price plot ───────────────────────────────────────
actual_price    = np.exp(returns['Gold'].cumsum()) * 100
predicted_price = np.exp(model.fittedvalues.cumsum()) * 100

df_plot = pd.DataFrame({
    'Date':          returns.index,
    'Actual GLD':    actual_price.values,
    'Predicted GLD': predicted_price.values
}).melt(id_vars='Date', var_name='Series', value_name='Indexed Price')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.theme import apply_premium_theme, add_cyberpunk_glow

apply_premium_theme(is_cyberpunk=True)
fig3, ax3 = plt.subplots(figsize=(12, 6))

sns.lineplot(data=df_plot, x='Date', y='Indexed Price', hue='Series', ax=ax3)
add_cyberpunk_glow(ax3)

ax3.set_title('GLD: Actual vs. Predicted (OLS on Log Returns, 2015–2021)')
ax3.text(0.01, 0.97, f'R² = {model.rsquared:.4f}', transform=ax3.transAxes,
         verticalalignment='top', fontsize=10, color='gray')
ax3.set_xlabel('Date')
ax3.set_ylabel('Indexed Price (Base = 100)')

plt.tight_layout()
plt.show()

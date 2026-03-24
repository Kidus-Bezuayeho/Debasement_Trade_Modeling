import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ── Data ──────────────────────────────────────────────────────────────────────
tickers = {
    'VIX':  '^VIX',
    'DXY':  'UUP',
    'TIPS': 'TIP',
    'TNX':  '^TNX',
    'SPY':  'SPY',
    'Gold': 'GLD'
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
returns = np.log(df_combined / df_combined.shift(1)).dropna()

# ── Feature Engineering ───────────────────────────────────────────────────────
# XGBoost can exploit non-linearities that OLS misses:
#   - Lagged GLD returns (momentum / mean-reversion)
#   - Rolling volatility (captures sudden vol spikes)
#   - VIX re-included — XGBoost handles noisy features better than OLS

features = pd.DataFrame(index=returns.index)

# Contemporaneous macro returns (same-day signals)
for col in ['VIX', 'DXY', 'TIPS', 'TNX', 'SPY']:
    features[col] = returns[col]

# Lagged GLD returns — momentum & mean-reversion signals
for lag in [1, 2, 3, 5]:
    features[f'Gold_lag{lag}'] = returns['Gold'].shift(lag)

# Lagged predictor returns (t-1) — yesterday's macro moves
for col in ['VIX', 'DXY', 'TNX', 'SPY']:
    features[f'{col}_lag1'] = returns[col].shift(1)

# Rolling volatility of GLD (5-day and 20-day)
features['Gold_vol5']  = returns['Gold'].rolling(5).std()
features['Gold_vol20'] = returns['Gold'].rolling(20).std()

# Rolling mean return of GLD (5-day) — short-term trend
features['Gold_ma5'] = returns['Gold'].rolling(5).mean()

target = returns['Gold']

df_model = features.join(target.rename('target')).dropna()
X = df_model.drop(columns='target')
y = df_model['target']

# ── Time-Series Train / Test Split (no shuffling — preserves temporal order) ──
split = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Train: {X_train.index[0].date()} → {X_train.index[-1].date()}  ({len(X_train)} days)")
print(f"Test:  {X_test.index[0].date()} → {X_test.index[-1].date()}  ({len(X_test)} days)")

# ── XGBoost with TimeSeriesSplit cross-validation ─────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
cv_r2 = []

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,          # shallow trees — avoids overfitting on noisy daily returns
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    xgb.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    pred = xgb.predict(X_train.iloc[val_idx])
    r2 = r2_score(y_train.iloc[val_idx], pred)
    cv_r2.append(r2)
    print(f"  CV Fold {fold+1}  R² = {r2:.4f}")

print(f"\nCV Mean R²: {np.mean(cv_r2):.4f}  ±  {np.std(cv_r2):.4f}")

# ── Final fit on full train set, evaluate on held-out test ───────────────────
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

test_r2   = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest R²:   {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.6f}")

# ── Plot 1: Feature Importance ────────────────────────────────────────────────
importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values()

sns.set_theme(style='darkgrid')
fig1, ax1 = plt.subplots(figsize=(8, 6))
importance.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('XGBoost Feature Importance (Gain)')
ax1.set_xlabel('Importance Score')
plt.tight_layout()

# ── Plot 2: Actual vs Predicted (indexed price, test period only) ─────────────
actual_price    = np.exp(y_test.cumsum()) * 100
predicted_price = np.exp(pd.Series(y_pred, index=y_test.index).cumsum()) * 100

df_plot = pd.DataFrame({
    'Date':          y_test.index,
    'Actual GLD':    actual_price.values,
    'Predicted GLD': predicted_price.values
}).melt(id_vars='Date', var_name='Series', value_name='Indexed Price')

fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_plot, x='Date', y='Indexed Price', hue='Series', ax=ax2)
ax2.set_title('XGBoost — GLD Actual vs. Predicted (Test Period)')
ax2.text(0.01, 0.97, f'R² = {test_r2:.4f}  |  RMSE = {test_rmse:.6f}',
         transform=ax2.transAxes, verticalalignment='top', fontsize=10, color='gray')
ax2.set_xlabel('Date')
ax2.set_ylabel('Indexed Price (Base = 100)')

plt.tight_layout()
plt.show()

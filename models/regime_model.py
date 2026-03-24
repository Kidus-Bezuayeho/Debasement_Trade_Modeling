import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
from datetime import datetime

# ── Event dates to exclude from the post-break model ─────────────────────────
# These are dates where gold moved for idiosyncratic/event-driven reasons
# that shouldn't be baked into the structural baseline.
# Add/remove dates as needed.
EVENT_DATES = [
    '2022-02-24',  # Russia invades Ukraine
    '2022-03-16',  # FOMC liftoff (first hike)
    '2022-06-15',  # 75bp surprise hike
    '2023-03-10',  # SVB collapse
    '2023-03-19',  # Credit Suisse rescue
    '2024-04-13',  # Iran attacks Israel
]

# ── Date ranges ───────────────────────────────────────────────────────────────
PRE_START  = datetime(2015, 1, 1)
PRE_END    = datetime(2021, 12, 31)
POST_START = datetime(2022, 1, 1)
POST_END   = datetime(2024, 12, 31)
FULL_START = PRE_START

# ── Fetch data ────────────────────────────────────────────────────────────────
tickers = {
    'Gold': 'GLD',
    'TIPS': 'TIP',
    'DXY':  'UUP',
    'VIX':  '^VIX',
    'MOVE': '^MOVE',
    'ZQ':   'ZQ=F',   # 30-day fed funds futures — daily Δ = rate expectations surprise
}

print("Fetching data...")
data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=FULL_START, end=POST_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    data[name] = df[col]
    print(f"  {name} ({ticker}): {len(data[name])} rows")

df_raw = pd.DataFrame(data).dropna(how='all')

# ── Differencing ──────────────────────────────────────────────────────────────
# Price-based series → log returns (first diff of log price)
# Level/index series → first difference
# Fed funds futures → rate surprise = -Δ(ZQ price), since ZQ = 100 - rate

df_diff = pd.DataFrame(index=df_raw.index)

for col in ['Gold', 'TIPS', 'DXY']:
    df_diff[col] = np.log(df_raw[col] / df_raw[col].shift(1))

for col in ['VIX', 'MOVE']:
    df_diff[col] = df_raw[col].diff()

# Rate surprise: a rise in ZQ price = dovish surprise → negate to get hawkish surprise
df_diff['FFSurprise'] = -df_raw['ZQ'].diff()

df_diff = df_diff.dropna()

# ── ADF stationarity check ────────────────────────────────────────────────────
print(f"\n{'─'*58}")
print("ADF Test — Differenced Series (all should be stationary)")
print(f"{'─'*58}")
print(f"{'Variable':<12} {'ADF Stat':>10} {'p-value':>10}  {'Stationary?':>12}")
print(f"{'─'*58}")
for col in df_diff.columns:
    stat, p, *_ = adfuller(df_diff[col].dropna(), autolag='AIC')
    status = 'YES ✓' if p < 0.05 else 'NO  ✗'
    print(f"{col:<12} {stat:>10.4f} {p:>10.4f}  {status:>12}")

# ── Split into pre-break and post-break periods ───────────────────────────────
predictors = ['TIPS', 'DXY', 'VIX', 'MOVE', 'FFSurprise']

pre  = df_diff.loc[PRE_START:PRE_END, ['Gold'] + predictors].dropna()
post = df_diff.loc[POST_START:POST_END, ['Gold'] + predictors].dropna()

# Remove event dates from post-break model
event_idx = pd.to_datetime(EVENT_DATES)
post_clean = post[~post.index.isin(event_idx)]
excluded   = len(post) - len(post_clean)
print(f"\nPost-break: {len(post)} days total, {excluded} event dates excluded → {len(post_clean)} used for model")

# ── Model 1: Pre-break OLS (2015–2021) ───────────────────────────────────────
X_pre = sm.add_constant(pre[predictors])
y_pre = pre['Gold']
model_pre = sm.OLS(y_pre, X_pre).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

print(f"\n{'═'*58}")
print("MODEL 1 — Pre-Break OLS (2015–2021) with HAC Std Errors")
print(f"{'═'*58}")
print(model_pre.summary())

# ── Model 2: Post-break OLS (2022–2024, excl. event dates) ───────────────────
X_post = sm.add_constant(post_clean[predictors])
y_post = post_clean['Gold']
model_post = sm.OLS(y_post, X_post).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

print(f"\n{'═'*58}")
print("MODEL 2 — Post-Break OLS (2022–2024, excl. events) with HAC Std Errors")
print(f"{'═'*58}")
print(model_post.summary())

# ── Residuals from post-break model (expected baseline for Gold) ──────────────
# Apply post-break model to the full post-break period (including event dates)
# Residual > 0 → Gold outperformed macro expectations
# Residual < 0 → Gold underperformed macro expectations
X_post_full = sm.add_constant(post[predictors], has_constant='add')
residuals   = post['Gold'] - model_post.predict(X_post_full)

print(f"\nPost-Break Residuals Summary:")
print(residuals.describe())

# ── Coefficient comparison table ──────────────────────────────────────────────
print(f"\n{'─'*58}")
print("Coefficient Comparison: Pre vs Post Break")
print(f"{'─'*58}")
print(f"{'Variable':<14} {'Pre coef':>10} {'Pre p':>8}  {'Post coef':>10} {'Post p':>8}")
print(f"{'─'*58}")
for var in ['const'] + predictors:
    c1, p1 = model_pre.params.get(var, np.nan), model_pre.pvalues.get(var, np.nan)
    c2, p2 = model_post.params.get(var, np.nan), model_post.pvalues.get(var, np.nan)
    print(f"{var:<14} {c1:>10.4f} {p1:>8.4f}  {c2:>10.4f} {p2:>8.4f}")

# ── Helper: render a text block as a PDF page ─────────────────────────────────
def text_page(pdf, title, body):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.text(0.01, 0.97, body, transform=ax.transAxes,
            fontsize=7.5, family='monospace', verticalalignment='top',
            wrap=True)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# ── Build coefficient comparison text ─────────────────────────────────────────
coef_lines = [
    f"{'Variable':<14} {'Pre coef':>10} {'Pre p':>8}  {'Post coef':>10} {'Post p':>8}",
    '─' * 58,
]
for var in ['const'] + predictors:
    c1 = model_pre.params.get(var, np.nan)
    p1 = model_pre.pvalues.get(var, np.nan)
    c2 = model_post.params.get(var, np.nan)
    p2 = model_post.pvalues.get(var, np.nan)
    coef_lines.append(f"{var:<14} {c1:>10.4f} {p1:>8.4f}  {c2:>10.4f} {p2:>8.4f}")

residuals_summary = residuals.describe()
resid_lines = [
    f"  count : {residuals_summary['count']:.0f}",
    f"  mean  : {residuals_summary['mean']:.6f}",
    f"  std   : {residuals_summary['std']:.6f}",
    f"  min   : {residuals_summary['min']:.6f}",
    f"  25%   : {residuals_summary['25%']:.6f}",
    f"  50%   : {residuals_summary['50%']:.6f}",
    f"  75%   : {residuals_summary['75%']:.6f}",
    f"  max   : {residuals_summary['max']:.6f}",
    f"  cumul : {residuals.sum():.6f}",
]

# ── Plots & PDF export ────────────────────────────────────────────────────────
sns.set_theme(style='darkgrid')
out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'regime_model_results.pdf')

with PdfPages(out_path) as pdf:
    # Page 1: Pre-break model summary
    text_page(pdf, 'Model 1 — Pre-Break OLS (2015–2021) with HAC Std Errors',
              str(model_pre.summary()))

    # Page 2: Post-break model summary
    text_page(pdf, 'Model 2 — Post-Break OLS (2022–2024, excl. events) with HAC Std Errors',
              str(model_post.summary()))

    # Page 3: Coefficient comparison + residuals summary
    text_page(pdf, 'Coefficient Comparison & Residuals Summary',
              'COEFFICIENT COMPARISON: Pre vs Post Break\n' +
              '\n'.join(coef_lines) +
              '\n\nPOST-BREAK RESIDUALS SUMMARY\n' +
              '\n'.join(resid_lines) +
              '\n\nEVENT DATES EXCLUDED FROM POST-BREAK MODEL\n' +
              '\n'.join(f'  {d}' for d in EVENT_DATES))

    # Page 4: Charts
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(residuals.index, residuals, 0,
                     where=(residuals >= 0), color='steelblue', alpha=0.6, label='Gold above baseline')
    ax1.fill_between(residuals.index, residuals, 0,
                     where=(residuals < 0),  color='firebrick', alpha=0.6, label='Gold below baseline')
    for d in event_idx:
        ax1.axvline(pd.Timestamp(d), color='orange', linewidth=1, linestyle='--', alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title('Post-Break Model Residuals — Gold vs. Macro Baseline (2022–2024)')
    ax1.set_ylabel('Residual (log return)')
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    residuals.cumsum().plot(ax=ax2, color='steelblue')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Cumulative Residuals')
    ax2.set_ylabel('Cumulative log return')
    ax2.set_xlabel('')

    ax3 = fig.add_subplot(gs[1, 1])
    coef_df = pd.DataFrame({
        'Pre-Break':  model_pre.params[predictors],
        'Post-Break': model_post.params[predictors],
    })
    coef_df.plot(kind='bar', ax=ax3, color=['steelblue', 'firebrick'], width=0.6)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('Coefficient Shift: Pre vs Post Break')
    ax3.set_ylabel('Coefficient')
    ax3.tick_params(axis='x', rotation=30)
    ax3.legend(fontsize=9)

    plt.suptitle(f'Regime Model  |  Pre R²={model_pre.rsquared:.3f}  Post R²={model_post.rsquared:.3f}',
                 fontsize=13)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\nResults saved to {out_path}")

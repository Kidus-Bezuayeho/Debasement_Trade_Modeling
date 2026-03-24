import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import yfinance as yf
from datetime import datetime

# ── Event dates (same as regime_model.py) ────────────────────────────────────
EVENT_DATES = [
    '2022-03-16',  # FOMC liftoff — first hike in tightening cycle
    '2022-06-15',  # FOMC — 75bp hike (jumbo move)
    '2023-03-10',  # SVB failure — Fed Bank Term Funding Program / discount window
    '2023-03-19',  # Credit Suisse rescue — swap lines / global central-bank coordination
    '2023-12-13',  # FOMC — dovish pivot / cut guidance for 2024 (positive risk-asset shock)
    '2024-02-02',  # Presidential candidate: would not reappoint Powell
    '2024-08-08',  # Presidential remarks — should have say in Fed (independence debate)
    '2024-09-18',  # FOMC — first rate cut of cycle (50bp; large dovish repricing)
    '2024-11-07',  # FOMC — follow-on cut (continued easing path)
    '2025-04-17',  # Escalated removal rhetoric vs Fed chair (independence risk)
    '2025-04-22',  # Walk-back: no intention to fire chair (positive tail-risk reduction)
    '2025-05-04',  # Interview: won't remove chair before term ends (independence stabilizing)
    '2025-06-12',  # Rates pressure; won't fire chair but may "force something" (headline)
]

# ── Date ranges (same as regime_model.py) ────────────────────────────────────
PRE_START  = datetime(2015, 1, 1)
PRE_END    = datetime(2021, 12, 31)
POST_START = datetime(2022, 1, 1)
POST_END   = datetime(2025, 12, 31)
FULL_START = PRE_START

# ── Fetch data ────────────────────────────────────────────────────────────────
tickers = {
    'MOVE': '^MOVE',
    'VIX':  '^VIX',
    'TLT':  'TLT',
    'IEF':  'IEF',
    'GLD':  'GLD',
    'IAU':  'IAU',
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
df_diff = pd.DataFrame(index=df_raw.index)

for col in ['TLT', 'IEF', 'GLD', 'IAU']:
    df_diff[col] = np.log(df_raw[col] / df_raw[col].shift(1))

for col in ['MOVE', 'VIX']:
    df_diff[col] = df_raw[col].diff()

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

event_idx = pd.to_datetime(EVENT_DATES)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — MOVE vs VIX: Isolating Bond-Specific Stress
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*58}")
print("ANALYSIS 1 — MOVE vs VIX: Bond-Specific Stress")
print(f"{'═'*58}")

# MOVE/VIX ratio (raw levels)
ratio = df_raw['MOVE'] / df_raw['VIX']

# OLS: ΔMOVE ~ const + ΔVIX
mv_data = df_diff[['MOVE', 'VIX']].dropna()
X_mv = sm.add_constant(mv_data['VIX'])
y_mv = mv_data['MOVE']
model_mv = sm.OLS(y_mv, X_mv).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
print(model_mv.summary())

bond_stress_residuals = model_mv.resid

print(f"\nBond-Specific Stress Residuals Summary:")
print(bond_stress_residuals.describe())

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Treasury Selloff at Fed-Focused Event Dates
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*58}")
print("ANALYSIS 2 — Treasury Selloff at Fed-Focused Event Dates")
print(f"{'═'*58}")

WINDOWS = [0, 1, 3, 5, 10]
assets   = ['TLT', 'IEF', 'GLD']

# Build per-event cumulative returns
per_event = {}
for ed in EVENT_DATES:
    ed_ts = pd.Timestamp(ed)
    # Find the nearest available trading day on or after event date
    future_dates = df_diff.index[df_diff.index >= ed_ts]
    if len(future_dates) == 0:
        continue
    start_pos = df_diff.index.get_loc(future_dates[0])
    row = {}
    for w in WINDOWS:
        end_pos = min(start_pos + w, len(df_diff) - 1)
        window_slice = df_diff.iloc[start_pos:end_pos + 1]
        for asset in assets:
            cum_ret = window_slice[asset].sum()
            row[f'{asset}_d{w}'] = round(float(cum_ret), 6)
    per_event[ed] = row

# ── Estimation window: pre-break daily sigma per asset ────────────────────────
estimation_window = df_diff.loc[PRE_START:PRE_END]
sigma_est = {asset: float(estimation_window[asset].std()) for asset in assets}
print(f"\nEstimation-window daily sigma: " +
      "  ".join(f"{a}={sigma_est[a]:.5f}" for a in assets))

# ── Bootstrap p-value (two-sided, H0: mean = 0) ───────────────────────────────
_rng = np.random.default_rng(42)

def bootstrap_pvalue(vals, n_bootstrap=10_000):
    """Two-sided bootstrap p-value for H0: mean = 0."""
    arr = np.asarray(vals, dtype=float)
    observed = np.abs(arr.mean())
    centered = arr - arr.mean()          # enforce null
    boot_means = np.abs(
        _rng.choice(centered, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
    )
    return float((boot_means >= observed).mean())

# ── Adjusted Patell / BMP test ────────────────────────────────────────────────
def adjusted_patell_test(vals, sigma_daily, n_days):
    """
    Boehmer, Musumeci & Poulsen (1991) standardised cross-sectional test.

    Each event return is first standardised by the pre-event estimation-window
    volatility, then the cross-sectional standard deviation of those SAR values
    is used as the denominator — this corrects for the inflated variance and
    cross-sectional correlation that arise when events are clustered.

        SAR_e  = cumret_e / (sigma_daily * sqrt(n_days))
        Z_BMP  = mean(SAR) / (std(SAR) / sqrt(N))  ~  t(N-1)
    """
    arr = np.asarray(vals, dtype=float)
    N = len(arr)
    if N < 3 or sigma_daily <= 0:
        return np.nan, np.nan
    sar = arr / (sigma_daily * np.sqrt(n_days))
    bmp_stat = sar.mean() / (sar.std(ddof=1) / np.sqrt(N))
    p_val = 2.0 * stats.t.sf(np.abs(bmp_stat), df=N - 1)
    return float(bmp_stat), float(p_val)

# ── Aggregate across events ───────────────────────────────────────────────────
HDR = f"{'Window':<6} {'Asset':<6} {'Mean':>9} {'p(t)':>9} {'p(boot)':>9} {'p(BMP)':>9} {'BMPstat':>9}  n"
print(f"\n{'─'*len(HDR)}")
print(HDR)
print(f"{'─'*len(HDR)}")

aggregate = {}
for w in WINDOWS:
    agg_row = {}
    n_days = w + 1          # window spans days 0 … w inclusive
    for asset in assets:
        key = f'{asset}_d{w}'
        vals = [per_event[ed][key] for ed in per_event if key in per_event[ed]]
        if len(vals) < 2:
            continue
        vals_arr = np.array(vals)
        t_stat, p_ttest = stats.ttest_1samp(vals_arr, 0)
        p_boot           = bootstrap_pvalue(vals_arr)
        bmp_stat, p_bmp  = adjusted_patell_test(vals_arr, sigma_est[asset], n_days)
        agg_row[asset] = {
            'mean':     round(float(vals_arr.mean()), 6),
            'median':   round(float(np.median(vals_arr)), 6),
            't_stat':   round(float(t_stat), 6),
            'p_ttest':  round(float(p_ttest), 6),
            'p_boot':   round(float(p_boot), 6),
            'bmp_stat': round(float(bmp_stat) if not np.isnan(bmp_stat) else float('nan'), 6),
            'p_bmp':    round(float(p_bmp)    if not np.isnan(p_bmp)    else float('nan'), 6),
            'n':        len(vals),
        }
        print(f"d{w:<5} {asset:<6} {agg_row[asset]['mean']:>9.4f} "
              f"{agg_row[asset]['p_ttest']:>9.4f} "
              f"{agg_row[asset]['p_boot']:>9.4f} "
              f"{agg_row[asset]['p_bmp']:>9.4f} "
              f"{agg_row[asset]['bmp_stat']:>9.4f}  "
              f"{agg_row[asset]['n']}")
    aggregate[f'd{w}'] = agg_row

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Weekly Rotation Regression
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*58}")
print("ANALYSIS 3 — Weekly Bond→Gold Rotation Regression")
print(f"{'═'*58}")

# Resample to weekly sums
weekly = df_diff[['TLT', 'IEF', 'GLD', 'IAU']].resample('W').sum()
weekly['BondReturn'] = (weekly['TLT'] + weekly['IEF']) / 2
weekly['GoldReturn'] = (weekly['GLD'] + weekly['IAU']) / 2
weekly = weekly.dropna(subset=['BondReturn', 'GoldReturn'])

# ADF on weekly series
print(f"\n{'─'*40}")
print("ADF — Weekly BondReturn and GoldReturn")
print(f"{'─'*40}")
for col in ['BondReturn', 'GoldReturn']:
    stat, p, *_ = adfuller(weekly[col].dropna(), autolag='AIC')
    status = 'YES ✓' if p < 0.05 else 'NO  ✗'
    print(f"  {col:<14} ADF={stat:.4f}  p={p:.4f}  {status}")

# Split pre/post
weekly_pre  = weekly.loc[PRE_START:PRE_END]
weekly_post = weekly.loc[POST_START:POST_END]

def rotation_model(df, label):
    X = sm.add_constant(df['BondReturn'])
    y = df['GoldReturn']
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1}), label

model_rot_pre,  _ = rotation_model(weekly_pre,  'Pre-Break Weekly Rotation 2015-2021')
model_rot_post, _ = rotation_model(weekly_post, 'Post-Break Weekly Rotation 2022-2025')

print(f"\n{'═'*58}")
print("Weekly Rotation — Pre-Break (2015–2021)")
print(f"{'═'*58}")
print(model_rot_pre.summary())

print(f"\n{'═'*58}")
print("Weekly Rotation — Post-Break (2022–2025)")
print(f"{'═'*58}")
print(model_rot_post.summary())

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

# ── Build text summaries ──────────────────────────────────────────────────────
ratio_summary = ratio.describe()
ratio_lines = (
    f"  count : {ratio_summary['count']:.0f}\n"
    f"  mean  : {ratio_summary['mean']:.4f}\n"
    f"  std   : {ratio_summary['std']:.4f}\n"
    f"  min   : {ratio_summary['min']:.4f}\n"
    f"  25%   : {ratio_summary['25%']:.4f}\n"
    f"  50%   : {ratio_summary['50%']:.4f}\n"
    f"  75%   : {ratio_summary['75%']:.4f}\n"
    f"  max   : {ratio_summary['max']:.4f}"
)

_shdr = f"{'Win':<5} {'Asset':<5} {'Mean':>9} {'p(t)':>9} {'p(boot)':>9} {'p(BMP)':>9} {'BMPstat':>9}"
selloff_lines = [
    'AGGREGATE EVENT-STUDY RESULTS',
    '(p(t)=standard t-test, p(boot)=bootstrapped [H0: mean=0, two-sided, B=10,000],',
    ' p(BMP)=Adjusted Patell/BMP test correcting for cross-sectional correlation)',
    '',
    _shdr,
    '─' * len(_shdr),
]
for w in WINDOWS:
    wk = f'd{w}'
    for asset in assets:
        a = aggregate[wk].get(asset, {})
        selloff_lines.append(
            f"d{w:<4} {asset:<5} "
            f"{a.get('mean',    float('nan')):>9.4f} "
            f"{a.get('p_ttest', float('nan')):>9.4f} "
            f"{a.get('p_boot',  float('nan')):>9.4f} "
            f"{a.get('p_bmp',   float('nan')):>9.4f} "
            f"{a.get('bmp_stat',float('nan')):>9.4f}"
        )
    selloff_lines.append('')
selloff_lines += ['', 'Per-event detail:',
                  f"  {'Event':<12} {'TLT d0':>8} {'TLT d5':>8} {'TLT d10':>8}  {'IEF d0':>8} {'IEF d5':>8}  {'GLD d0':>8} {'GLD d5':>8}"]
for ed, row in per_event.items():
    selloff_lines.append(
        f"  {ed:<12} {row.get('TLT_d0', float('nan')):>8.4f} {row.get('TLT_d5', float('nan')):>8.4f} "
        f"{row.get('TLT_d10', float('nan')):>8.4f}  {row.get('IEF_d0', float('nan')):>8.4f} "
        f"{row.get('IEF_d5', float('nan')):>8.4f}  {row.get('GLD_d0', float('nan')):>8.4f} "
        f"{row.get('GLD_d5', float('nan')):>8.4f}"
    )

rot_cmp_lines = [
    f"{'Variable':<14} {'Pre coef':>10} {'Pre p':>8}  {'Post coef':>10} {'Post p':>8}",
    '─' * 58,
]
for var in ['const', 'BondReturn']:
    rot_cmp_lines.append(
        f"{var:<14} {model_rot_pre.params.get(var, float('nan')):>10.4f} "
        f"{model_rot_pre.pvalues.get(var, float('nan')):>8.4f}  "
        f"{model_rot_post.params.get(var, float('nan')):>10.4f} "
        f"{model_rot_post.pvalues.get(var, float('nan')):>8.4f}"
    )

# ── Plots & PDF export ────────────────────────────────────────────────────────
import sys
from theme import apply_premium_theme
apply_premium_theme(is_cyberpunk=False)
out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'bond_stress_test_results.pdf')

with PdfPages(out_path) as pdf:
    # Page 1: MOVE/VIX OLS summary + ratio stats
    text_page(pdf, 'Analysis 1 — MOVE vs VIX: Bond-Specific Stress',
              'OLS: ΔMOVE ~ const + ΔVIX (HAC errors)\n\n' +
              str(model_mv.summary()) +
              '\n\nMOVE/VIX RATIO SUMMARY (raw levels)\n' + ratio_lines)

    # Page 2: Treasury selloff event study
    text_page(pdf, 'Analysis 2 — Treasury Selloff at Fed-Focused Event Dates',
              '\n'.join(selloff_lines))

    # Page 3: Rotation regression summaries
    text_page(pdf, 'Analysis 3 — Weekly Rotation: Pre-Break (2015–2021)',
              str(model_rot_pre.summary()))

    text_page(pdf, 'Analysis 3 — Weekly Rotation: Post-Break (2022–2025)',
              str(model_rot_post.summary()) +
              '\n\nCOEFFICIENT COMPARISON: Pre vs Post\n' +
              '\n'.join(rot_cmp_lines))

    # Page 5: Charts
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ratio.plot(ax=ax1, color='steelblue', linewidth=0.8)
    for d in event_idx:
        ax1.axvline(pd.Timestamp(d), color='orange', linewidth=1, linestyle='--', alpha=0.8)
    ax1.set_title('MOVE / VIX Ratio — Bond Stress Relative to Equity Fear')
    ax1.set_ylabel('MOVE / VIX')
    ax1.set_xlabel('')

    ax2 = fig.add_subplot(gs[1])
    window_labels = [f'd{w}' for w in WINDOWS]
    tlt_means = [aggregate[w]['TLT']['mean'] for w in window_labels]
    ief_means = [aggregate[w]['IEF']['mean'] for w in window_labels]
    gld_means = [aggregate[w]['GLD']['mean'] for w in window_labels]
    x = np.arange(len(WINDOWS))
    width = 0.25
    ax2.bar(x - width, tlt_means, width, label='TLT', color='firebrick', alpha=0.8)
    ax2.bar(x,         ief_means, width, label='IEF', color='darkorange', alpha=0.8)
    ax2.bar(x + width, gld_means, width, label='GLD', color='steelblue', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'+{w}d' for w in WINDOWS])
    ax2.set_title(
        f'Mean Cumulative Return at Fed-Focused Events (across {len(EVENT_DATES)} events)'
    )
    ax2.set_ylabel('Mean cumulative log return')
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(weekly_pre['BondReturn'],  weekly_pre['GoldReturn'],
                color='steelblue', alpha=0.4, s=15, label='Pre-break (2015–2021)')
    ax3.scatter(weekly_post['BondReturn'], weekly_post['GoldReturn'],
                color='firebrick', alpha=0.4, s=15, label='Post-break (2022–2025)')
    x_range = np.linspace(weekly['BondReturn'].min(), weekly['BondReturn'].max(), 100)
    for mdl, color in [(model_rot_pre, 'steelblue'), (model_rot_post, 'firebrick')]:
        y_fit = mdl.params['const'] + mdl.params['BondReturn'] * x_range
        ax3.plot(x_range, y_fit, color=color, linewidth=2)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=0.5)
    ax3.set_title(f'Weekly Bond→Gold Rotation  |  Pre β={model_rot_pre.params["BondReturn"]:.3f}  Post β={model_rot_post.params["BondReturn"]:.3f}')
    ax3.set_xlabel('BondReturn (weekly log return, TLT+IEF avg)')
    ax3.set_ylabel('GoldReturn (weekly log return, GLD+IAU avg)')
    ax3.legend(fontsize=9)

    plt.suptitle('Bond Stress & Capital Rotation Analysis', fontsize=13)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\nResults saved to {out_path}")

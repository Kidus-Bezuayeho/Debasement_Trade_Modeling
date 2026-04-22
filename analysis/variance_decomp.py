import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import yfinance as yf
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply_premium_theme

# ── Date ranges ───────────────────────────────────────────────────────────────
PERIODS = [
    ('Pre-2023',  datetime(2015, 1,  1), datetime(2022, 12, 31)),
    ('2024–Now',  datetime(2024, 1,  1), datetime(2025, 12, 31)),
]
PERIOD_LABELS = [p[0] for p in PERIODS]
FULL_START = PERIODS[0][1]
FULL_END   = PERIODS[-1][2]

# ── Cholesky ordering ─────────────────────────────────────────────────────────
COL_ORDER     = ['FFSurprise', 'DXY', 'TIPS', 'MOVE', 'VIX', 'Gold']
FEVD_HORIZONS = [1, 5, 10, 20]

EF_DRIVERS = ['DXY', 'TIPS', 'MOVE', 'VIX']   # event-free: no FFSurprise
EF_TARGET  = 'Gold'

# ── Fetch data ────────────────────────────────────────────────────────────────
tickers = {
    'Gold': 'GLD',
    'TIPS': 'TIP',
    'DXY':  'UUP',
    'VIX':  '^VIX',
    'MOVE': '^MOVE',
    'ZQ':   'ZQ=F',
}

print("Fetching data...")
data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=FULL_START, end=FULL_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    data[name] = df[col]
    print(f"  {name} ({ticker}): {len(data[name])} rows")

df_raw  = pd.DataFrame(data).dropna(how='all')

# ── Differencing ──────────────────────────────────────────────────────────────
df_diff = pd.DataFrame(index=df_raw.index)
for col in ['Gold', 'TIPS', 'DXY']:
    df_diff[col] = np.log(df_raw[col] / df_raw[col].shift(1))
for col in ['VIX', 'MOVE']:
    df_diff[col] = df_raw[col].diff()
df_diff['FFSurprise'] = -df_raw['ZQ'].diff()
df_diff = df_diff.dropna()

# ── ADF stationarity check ────────────────────────────────────────────────────
print(f"\n{'─'*58}")
print("ADF Test — Differenced Series (all should be stationary)")
print(f"{'─'*58}")
print(f"{'Variable':<14} {'ADF Stat':>10} {'p-value':>10}  {'Stationary?':>12}")
print(f"{'─'*58}")
for col in COL_ORDER:
    stat, p, *_ = adfuller(df_diff[col].dropna(), autolag='AIC')
    status = 'YES ✓' if p < 0.05 else 'NO  ✗'
    print(f"{col:<14} {stat:>10.4f} {p:>10.4f}  {status:>12}")

# ── Helper: fit VAR, select lag, compute FEVD ─────────────────────────────────
gold_idx = COL_ORDER.index('Gold')

def fit_and_decompose(df_period, label, maxlags=15):
    print(f"\n{'═'*62}")
    print(f"VAR — {label}  (N={len(df_period)})")
    print(f"{'═'*62}")
    var_model = VAR(df_period)
    lag_sel   = var_model.select_order(maxlags=maxlags)
    optimal_lag = max(lag_sel.aic, 1)
    print(f"Lag order (AIC={lag_sel.aic}, BIC={lag_sel.bic}, HQIC={lag_sel.hqic})")
    print(f"  → Using AIC lag: {optimal_lag}")
    result = var_model.fit(optimal_lag)
    stable  = result.is_stable()
    max_mod = max(abs(result.roots))
    print(f"  Stability: {'STABLE ✓' if stable else 'UNSTABLE ✗'}  (max root={max_mod:.4f})")
    fevd = result.fevd(max(FEVD_HORIZONS))
    hdr  = f"{'Horizon':<8} " + " ".join(f"{c:>10}" for c in COL_ORDER)
    print(f"\nFEVD — Gold equation\n{'─'*len(hdr)}\n{hdr}\n{'─'*len(hdr)}")
    for h in FEVD_HORIZONS:
        row = fevd.decomp[gold_idx, h - 1, :]
        print(f"{h:<8} " + " ".join(f"{v:>10.4f}" for v in row) +
              f"   [Σ={row.sum():.4f}]")
    return result, fevd, optimal_lag, lag_sel

def ols_variance_decomp(df_period, drivers, target):
    sub = df_period[drivers + [target]].dropna()
    X, y = sub[drivers].values, sub[target].values
    ols_res = sm.OLS(y, sm.add_constant(X)).fit()
    betas = ols_res.params[1:]
    var_y = np.var(y, ddof=1)
    cov_xy = np.array([np.cov(X[:, j], y, ddof=1)[0, 1] for j in range(X.shape[1])])
    pratt = (betas * cov_xy) / var_y
    shares = {d: float(pratt[j]) for j, d in enumerate(drivers)}
    shares['Unexplained'] = float(1.0 - ols_res.rsquared)
    return shares, ols_res.rsquared

# ── Fit all three periods ─────────────────────────────────────────────────────
period_slices = [
    df_diff.loc[start:end, COL_ORDER].dropna()
    for _, start, end in PERIODS
]
fits = [
    fit_and_decompose(sl, label)
    for sl, (label, _, _) in zip(period_slices, PERIODS)
]
results  = [f[0] for f in fits]
fevds    = [f[1] for f in fits]
lags     = [f[2] for f in fits]
lag_sels = [f[3] for f in fits]

# ── Event-free OLS decomposition ──────────────────────────────────────────────
ef_period_slices = [
    df_diff.loc[start:end, EF_DRIVERS + [EF_TARGET]].dropna()
    for _, start, end in PERIODS
]
ef_decomps = [ols_variance_decomp(sl, EF_DRIVERS, EF_TARGET) for sl in ef_period_slices]

# ── Build text helpers ────────────────────────────────────────────────────────
def fevd_table_text(fevd_obj, label, lag):
    lines = [
        f"FEVD — Gold Equation  [{label},  VAR({lag})]",
        "",
        f"Cholesky ordering: {' → '.join(COL_ORDER)}",
        "",
        f"{'Horizon':<8} " + " ".join(f"{c:>11}" for c in COL_ORDER) + "   Sum",
        "─" * 80,
    ]
    for h in FEVD_HORIZONS:
        row = fevd_obj.decomp[gold_idx, h - 1, :]
        lines.append(
            f"{h:<8} " + " ".join(f"{v:>11.4f}" for v in row) +
            f"   {row.sum():.4f}"
        )
    return "\n".join(lines)

def delta_table_text(fevd_a, fevd_b, label_a, label_b):
    lines = [
        f"FEVD Δ ({label_b} − {label_a}) — Gold equation",
        f"(positive = driver explains MORE of gold's variance in {label_b})",
        "",
        f"{'Horizon':<8} " + " ".join(f"{c:>11}" for c in COL_ORDER),
        "─" * 80,
    ]
    for h in FEVD_HORIZONS:
        delta = fevd_b.decomp[gold_idx, h-1, :] - fevd_a.decomp[gold_idx, h-1, :]
        lines.append(f"{h:<8} " + " ".join(f"{v:>+11.4f}" for v in delta))
    return "\n".join(lines)

def diag_text(label, result, lag_sel_obj, lag):
    stable = result.is_stable()
    roots  = sorted(abs(result.roots), reverse=True)
    lines = [
        f"VAR DIAGNOSTICS — {label}",
        f"  Observations : {int(result.nobs)}",
        f"  Lag selected : {lag}  (AIC={lag_sel_obj.aic}, BIC={lag_sel_obj.bic}, HQIC={lag_sel_obj.hqic})",
        f"  Stability    : {'STABLE ✓' if stable else 'UNSTABLE ✗'}",
        f"  Max root mod : {roots[0]:.5f}",
        "",
        "  Root moduli (top 10):",
    ]
    for r in roots[:10]:
        lines.append(f"    {r:.5f}")
    return "\n".join(lines)

# ── Plots & PDF ───────────────────────────────────────────────────────────────
apply_premium_theme(is_cyberpunk=False)
out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'variance_decomp_results.pdf')

def text_page(pdf, title, body):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.text(0.01, 0.97, body, transform=ax.transAxes,
            fontsize=7.5, family='monospace', verticalalignment='top')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

with PdfPages(out_path) as pdf:

    # Page 1: VAR diagnostics — all three periods
    diag_body = "\n\n".join(
        diag_text(label, results[i], lag_sels[i], lags[i])
        for i, (label, _, _) in enumerate(PERIODS)
    )
    text_page(pdf, 'VAR Diagnostics — All Three Periods', diag_body)

    # Pages 2–4: FEVD tables per period
    for i, (label, _, _) in enumerate(PERIODS):
        text_page(pdf, f'FEVD — {label}',
                  fevd_table_text(fevds[i], label, lags[i]))

    # Page 5: Delta table
    delta_body = delta_table_text(fevds[0], fevds[1], PERIOD_LABELS[0], PERIOD_LABELS[1])
    text_page(pdf, 'FEVD Regime Shift — Delta Table', delta_body)

    # Page 6: Grouped horizontal bar chart — % explanatory power, 3 periods
    PERIOD_COLORS = ['#457b9d', '#e9c46a', '#e63946']  # blue, yellow, red
    driver_labels  = COL_ORDER[::-1]
    driver_indices = [COL_ORDER.index(d) for d in driver_labels]
    n_drivers = len(driver_labels)
    n_periods = len(PERIODS)
    bar_w   = 0.35
    offsets = np.linspace(-(n_periods - 1) / 2, (n_periods - 1) / 2, n_periods) * bar_w

    fig, axes = plt.subplots(1, 4, figsize=(22, 7), sharey=True)
    fig.suptitle('FEVD — Gold Equation: Explanatory Power by Driver (All Periods)',
                 fontsize=13, y=1.02)

    driver_xlabels = COL_ORDER  # left-to-right order on x-axis

    for ax, h in zip(axes, FEVD_HORIZONS):
        x = np.arange(n_drivers)
        for pi, ((label, _, _), clr, offset) in enumerate(zip(PERIODS, PERIOD_COLORS, offsets)):
            row  = fevds[pi].decomp[gold_idx, h - 1, :] * 100
            vals = [row[COL_ORDER.index(d)] for d in driver_xlabels]
            bars = ax.bar(x + offset, vals, width=bar_w,
                          color=clr, label=label, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=6.5, color='white')

        ax.set_xticks(x)
        ax.set_xticklabels(driver_xlabels, fontsize=9)
        ax.set_ylabel('% of Gold variance explained', fontsize=8)
        ax.set_title(f'Horizon = {h} day{"s" if h > 1 else ""}', fontsize=10)
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.tick_params(axis='y', labelsize=7)
        if h == FEVD_HORIZONS[0]:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Page 7: Regime shift horizontal bars — 2024-Now minus Pre-2023
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    fig.suptitle(f'FEVD Regime Shift — Gold Equation: {PERIOD_LABELS[1]} − {PERIOD_LABELS[0]}',
                 fontsize=13, y=1.02)

    for ax, h in zip(axes, FEVD_HORIZONS):
        pre_row  = fevds[0].decomp[gold_idx, h - 1, :]
        post_row = fevds[1].decomp[gold_idx, h - 1, :]
        deltas   = [post_row[i] - pre_row[i] for i in driver_indices]
        bar_colors = ['#e63946' if d < 0 else '#2a9d8f' for d in deltas]
        bars = ax.barh(driver_labels, deltas, color=bar_colors,
                       edgecolor='white', linewidth=0.4, height=0.6)
        for bar, val in zip(bars, deltas):
            ha     = 'left'  if val >= 0 else 'right'
            offset = 0.002   if val >= 0 else -0.002
            ax.text(bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:+.3f}', va='center', ha=ha, fontsize=7.5)
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
        x_lim = max(abs(d) for d in deltas) * 1.6 or 0.05
        ax.set_xlim(-x_lim, x_lim)
        ax.set_title(f'Horizon = {h} day{"s" if h > 1 else ""}', fontsize=10)
        ax.set_xlabel('Δ Variance share', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             'Green = driver explains MORE of gold\'s variance in 2024–Now   '
             'Red = driver explains LESS',
             ha='center', fontsize=8, style='italic')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Page 8: Event-free OLS Pratt variance decomposition
    EF_BAR_LABELS = EF_DRIVERS + ['Unexplained']
    EF_COLORS = {
        'DXY':         '#457b9d',
        'TIPS':        '#2a9d8f',
        'MOVE':        '#e9c46a',
        'VIX':         '#f4a261',
        'Unexplained': '#555555',
    }
    PERIOD_HATCHES = ['', '///']

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle('Gold Variance Attribution — Event-Free (OLS Pratt Measure)',
                 fontsize=12, y=1.02)

    bar_w   = 0.32
    x       = np.arange(len(EF_BAR_LABELS))
    offsets = [-(bar_w / 2), bar_w / 2]

    for pi, ((label, _, _), hatch, offset) in enumerate(zip(PERIODS, PERIOD_HATCHES, offsets)):
        shares_dict, r2 = ef_decomps[pi]
        vals       = [shares_dict.get(d, 0.0) * 100 for d in EF_BAR_LABELS]
        bar_colors = [EF_COLORS[d] for d in EF_BAR_LABELS]
        bars = ax.bar(x + offset, vals, width=bar_w,
                      color=bar_colors, hatch=hatch,
                      edgecolor='white', linewidth=0.5,
                      alpha=0.88, label=f'{label}  (R²={r2:.2f})')
        for bar, val in zip(bars, vals):
            if abs(val) >= 0.3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=7.5, color='white')

    all_vals = [ef_decomps[pi][0].get(d, 0.0) * 100
                for pi in range(len(PERIODS)) for d in EF_BAR_LABELS]
    ax.set_ylim(0, max(all_vals) * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(EF_BAR_LABELS, fontsize=10)
    ax.set_ylabel('% of Gold variance', fontsize=10)
    ax.set_xlabel('Driver', fontsize=10)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=9, loc='upper left')
    ax.tick_params(axis='y', labelsize=8)
    fig.text(0.5, -0.03,
             "Pratt's measure: β_i · Cov(X_i, Gold) / Var(Gold) · 100.  "
             "Bars sum to 100%.  No horizon assumed — full-period realized variance.",
             ha='center', fontsize=7.5, style='italic', color='#aaaaaa')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\nResults saved to {out_path}")

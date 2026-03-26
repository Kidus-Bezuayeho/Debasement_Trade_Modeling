import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import yfinance as yf
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply_premium_theme

# ── Date ranges ───────────────────────────────────────────────────────────────
PRE_START  = datetime(2015, 1, 1)
PRE_END    = datetime(2021, 12, 31)
POST_START = datetime(2022, 1, 1)
POST_END   = datetime(2025, 12, 31)
FULL_START = PRE_START

# ── Cholesky ordering: most exogenous → most endogenous ───────────────────────
# Rationale: Fed policy shocks drive dollar & rates first; rate moves feed into
# inflation expectations (TIPS) and bond vol (MOVE); equity fear (VIX) reacts
# to all; gold is the final absorber.
COL_ORDER = ['FFSurprise', 'DXY', 'TIPS', 'MOVE', 'VIX', 'Gold']

FEVD_HORIZONS = [1, 5, 10, 20]

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
    df = yf.download(ticker, start=FULL_START, end=POST_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    data[name] = df[col]
    print(f"  {name} ({ticker}): {len(data[name])} rows")

df_raw = pd.DataFrame(data).dropna(how='all')

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

# ── Split periods ─────────────────────────────────────────────────────────────
pre  = df_diff.loc[PRE_START:PRE_END,  COL_ORDER].dropna()
post = df_diff.loc[POST_START:POST_END, COL_ORDER].dropna()

# ── Helper: fit VAR, select lag, compute FEVD ─────────────────────────────────
def fit_and_decompose(df_period, label, maxlags=15):
    print(f"\n{'═'*62}")
    print(f"VAR — {label}  (N={len(df_period)})")
    print(f"{'═'*62}")

    var_model = VAR(df_period)

    # Lag order selection
    lag_sel = var_model.select_order(maxlags=maxlags)
    optimal_lag = max(lag_sel.aic, 1)   # floor at 1
    print(f"\nLag order selection (AIC={lag_sel.aic}, BIC={lag_sel.bic}, HQIC={lag_sel.hqic})")
    print(f"  → Using AIC-selected lag: {optimal_lag}")

    # Fit
    result = var_model.fit(optimal_lag)
    stable  = result.is_stable()
    max_mod = max(abs(result.roots))
    print(f"  Stability: {'STABLE ✓' if stable else 'UNSTABLE ✗'}  (max root modulus={max_mod:.4f})")

    # FEVD
    fevd = result.fevd(max(FEVD_HORIZONS))
    gold_idx = COL_ORDER.index('Gold')

    # Print FEVD table for Gold equation
    hdr = f"{'Horizon':<8} " + " ".join(f"{c:>10}" for c in COL_ORDER)
    print(f"\nFEVD — Gold equation")
    print(f"{'─'*len(hdr)}")
    print(hdr)
    print(f"{'─'*len(hdr)}")
    for h in FEVD_HORIZONS:
        row = fevd.decomp[gold_idx, h - 1, :]
        row_sum = row.sum()
        print(f"{h:<8} " + " ".join(f"{v:>10.4f}" for v in row) +
              f"   [Σ={row_sum:.4f}]")

    return result, fevd, optimal_lag

result_pre,  fevd_pre,  lag_pre  = fit_and_decompose(pre,  'Pre-Break (2015–2021)')
result_post, fevd_post, lag_post = fit_and_decompose(post, 'Post-Break (2022–2025)')

# ── Delta table ───────────────────────────────────────────────────────────────
gold_idx = COL_ORDER.index('Gold')
print(f"\n{'─'*62}")
print("FEVD Δ (Post − Pre) — Gold equation")
print(f"{'─'*62}")
hdr = f"{'Horizon':<8} " + " ".join(f"{c:>10}" for c in COL_ORDER)
print(hdr)
print(f"{'─'*62}")
for h in FEVD_HORIZONS:
    row_pre  = fevd_pre.decomp[gold_idx, h - 1, :]
    row_post = fevd_post.decomp[gold_idx, h - 1, :]
    delta    = row_post - row_pre
    print(f"{h:<8} " + " ".join(f"{v:>+10.4f}" for v in delta))

# ── Build text for PDF pages ──────────────────────────────────────────────────
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

def delta_table_text():
    lines = [
        "FEVD Δ (Post−Pre) — Gold equation",
        "(positive = driver explains MORE of gold's variance post-2022)",
        "",
        f"{'Horizon':<8} " + " ".join(f"{c:>11}" for c in COL_ORDER),
        "─" * 80,
    ]
    for h in FEVD_HORIZONS:
        row_pre  = fevd_pre.decomp[gold_idx, h - 1, :]
        row_post = fevd_post.decomp[gold_idx, h - 1, :]
        delta    = row_post - row_pre
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

# Re-run lag selection objects for diagnostics text (needed for text page)
lag_sel_pre  = VAR(pre).select_order(maxlags=15)
lag_sel_post = VAR(post).select_order(maxlags=15)

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

    # Page 1: VAR diagnostics
    text_page(pdf, 'VAR Diagnostics — Pre-Break & Post-Break',
              diag_text('Pre-Break (2015–2021)',  result_pre,  lag_sel_pre,  lag_pre) +
              '\n\n' +
              diag_text('Post-Break (2022–2025)', result_post, lag_sel_post, lag_post))

    # Page 2: FEVD pre-break
    text_page(pdf, 'Forecast Error Variance Decomposition — Pre-Break (2015–2021)',
              fevd_table_text(fevd_pre,  'Pre-Break 2015–2021',  lag_pre))

    # Page 3: FEVD post-break + delta
    text_page(pdf, 'FEVD — Post-Break (2022–2025) & Regime Shift',
              fevd_table_text(fevd_post, 'Post-Break 2022–2025', lag_post) +
              '\n\n' + delta_table_text())

    # Page 4: Stacked bar charts (pre vs post, horizons 1/5/10/20)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FEVD — Gold Equation: Pre vs Post Break', fontsize=13)
    colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261', '#6d6875']

    for ax, h in zip(axes.flat, FEVD_HORIZONS):
        pre_row  = fevd_pre.decomp[gold_idx, h - 1, :]
        post_row = fevd_post.decomp[gold_idx, h - 1, :]
        x      = np.arange(2)
        bottom = np.zeros(2)
        for i, (col, clr) in enumerate(zip(COL_ORDER, colors)):
            vals = [pre_row[i], post_row[i]]
            ax.bar(x, vals, bottom=bottom, color=clr, label=col, width=0.5)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(['Pre-Break\n2015–2021', 'Post-Break\n2022–2025'])
        ax.set_title(f'Horizon = {h} day{"s" if h > 1 else ""}')
        ax.set_ylabel('Fraction of variance')
        ax.set_ylim(0, 1.05)
        if h == FEVD_HORIZONS[0]:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Page 5: Horizontal bar chart — regime shift (post − pre) per horizon
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    fig.suptitle('FEVD Regime Shift — Gold Equation: Post-Break minus Pre-Break',
                 fontsize=13, y=1.02)

    driver_labels = COL_ORDER[::-1]          # reverse so top bar = first driver
    driver_indices = [COL_ORDER.index(d) for d in driver_labels]

    for ax, h in zip(axes, FEVD_HORIZONS):
        pre_row  = fevd_pre.decomp[gold_idx, h - 1, :]
        post_row = fevd_post.decomp[gold_idx, h - 1, :]
        deltas   = [post_row[i] - pre_row[i] for i in driver_indices]

        bar_colors = ['#e63946' if d < 0 else '#2a9d8f' for d in deltas]
        bars = ax.barh(driver_labels, deltas, color=bar_colors, edgecolor='white',
                       linewidth=0.5, height=0.6)

        # Value labels
        for bar, val in zip(bars, deltas):
            x_pos = bar.get_width()
            ha = 'left' if val >= 0 else 'right'
            offset = 0.002 if val >= 0 else -0.002
            ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                    f'{val:+.3f}', va='center', ha=ha, fontsize=7.5)

        ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f'Horizon = {h} day{"s" if h > 1 else ""}', fontsize=10)
        ax.set_xlabel('Δ Variance share', fontsize=8)
        x_lim = max(abs(d) for d in deltas) * 1.5
        ax.set_xlim(-x_lim, x_lim)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=7)

    fig.text(0.5, -0.02,
             'Green = driver explains MORE of gold\'s variance post-2022   '
             'Red = driver explains LESS',
             ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\nResults saved to {out_path}")

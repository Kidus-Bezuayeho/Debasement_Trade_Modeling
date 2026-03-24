# Debasement Trade Modeling

Empirical research code that downloads market data from Yahoo Finance (`yfinance`) and studies how **gold** (primarily via GLD) co-moves with macro and risk proxies: volatility (VIX, MOVE), the dollar (UUP), inflation-linked bonds (TIPS), rate expectations, Treasuries, and broad equities. The focus is exploratory modeling and regime-style comparison—not a production trading system.

## Repository layout

- **`data/`** — Scripts that fetch and inspect raw series (e.g. combined adjusted closes for a fixed date window).
- **`models/`** — Regression, regime-break OLS, and XGBoost experiments on gold and related factors.
- **`analysis/`** — Additional studies (bond stress framing) and simple visualization utilities.
- **`outputs/`** — PDF reports produced by the regime and bond-stress pipelines. The directory must exist before those scripts run (the code does not create it).

## Requirements

- Python 3.x  
- Dependencies (install with pip, for example):

```bash
pip install yfinance pandas numpy matplotlib seaborn statsmodels scipy xgboost scikit-learn
```

## How to run

From the repository root (`Debasement_Trade_Modeling`), run any script directly:

```bash
python data/data.py
python models/regression.py
python models/regime_model.py
python models/xgboost_model.py
python analysis/plotting.py
python analysis/bond_stress_test.py
```

Most scripts open interactive figure windows via `matplotlib` (`plt.show()`). The two PDF exporters write without blocking on a display:

| Script | Output |
|--------|--------|
| `models/regime_model.py` | `outputs/regime_model_results.pdf` |
| `analysis/bond_stress_test.py` | `outputs/bond_stress_test_results.pdf` |

## What each script does (short)

- **`data/data.py`** — Pulls daily prices for VIX, UUP, TIP, GLD (2015–2021) and prints sample stats.
- **`models/regression.py`** — ADF stationarity checks, log-return transforms where needed, OLS relating gold returns to macro factors (2015–2021 sample).
- **`models/regime_model.py`** — Pre-break OLS (2015–2021) vs post-break OLS (2022–2024) with selected event days excluded; exports a multi-page PDF under `outputs/`.
- **`models/xgboost_model.py`** — Feature engineering and time-series cross-validation with `XGBRegressor` for GLD.
- **`analysis/plotting.py`** — Indexed (base 100) line chart of selected assets over 2015–2021.
- **`analysis/bond_stress_test.py`** — MOVE, duration ETFs, and gold in a stress-style setup; exports a PDF under `outputs/`.

## Data and network

All data is pulled live from Yahoo Finance. You need an internet connection. Symbol availability and `yfinance` behavior can change over time.

## Disclaimer

This repository is for **research and education** only. It is not investment advice.

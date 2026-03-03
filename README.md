# Kalman-EM Time Series Forecaster

> A production-ready time series forecasting framework based on the **Kalman filter** with parameters learned by the **Expectation-Maximisation (EM) algorithm**, packaged as an interactive **Streamlit application**.

---

## Table of contents

1. [Overview](#overview)
2. [Mathematical background](#mathematical-background)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Streamlit application](#streamlit-application)
6. [Command-line examples](#command-line-examples)
7. [Python API](#python-api)
8. [Datasets](#datasets)
9. [Results](#results)
10. [Documentation](#documentation)
11. [Project structure](#project-structure)

---

## Overview

This project implements a **Linear Gaussian State-Space model** whose parameters are estimated from data via the **EM algorithm** (Expectation-Maximisation). The framework handles:

- **Univariate and multivariate** time series
- **Missing observations** (NaN-aware Kalman filter)
- **Strongly seasonal series** via an optional **STL pre-decomposition** pipeline
- **Honest backtesting** with one-step-ahead predictions
- **Multi-step forecasting** with calibrated uncertainty intervals

Three real-world domains are demonstrated out-of-the-box:

| Domain | Dataset | Best MAPE |
|--------|---------|-----------|
| Finance | Apple stock prices (AAPL, 2022–2024) | ~0.8 % |
| Energy | French electricity consumption (RTE, 2020–2024) | ~1.9 % |
| Transport | SNCF TGV monthly traffic (2018–2025) | ~1.0 % |

---

## Mathematical background

### State-space model

```
x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)   (state equation)
y_t = H x_t    + v_t,   v_t ~ N(0, R)   (observation equation)
x_0 ~ N(μ₀, Σ₀)
```

| Symbol | Shape | Role |
|--------|-------|------|
| `x_t` | `(d,)` | Latent state — e.g. level + trend |
| `y_t` | `(m,)` | Observed time series |
| `F` | `(d, d)` | State transition matrix |
| `H` | `(m, d)` | Observation matrix |
| `Q` | `(d, d)` | Process noise covariance |
| `R` | `(m, m)` | Observation noise covariance |

### EM algorithm

All six parameters `{F, H, Q, R, μ₀, Σ₀}` are learned end-to-end from data.

**E-step** — run the Kalman filter (forward) and RTS smoother (backward) to compute sufficient statistics:

```
E[x_t | Y],   E[x_t xₜᵀ | Y],   E[x_t x_{t-1}ᵀ | Y]
```

**M-step** — closed-form updates:

```
F  ← (Σ_t E[x_t x_{t-1}ᵀ]) (Σ_t E[x_{t-1} x_{t-1}ᵀ])⁻¹
Q  ← (1/T) Σ_t (E[x_t xₜᵀ] − F E[x_t x_{t-1}ᵀ]ᵀ)
H  ← (Σ_t y_t E[xₜᵀ]) (Σ_t E[x_t xₜᵀ])⁻¹
R  ← (1/T) Σ_t (y_t yₜᵀ − H y_t E[xₜᵀ]ᵀ)
```

### STL + Kalman-EM pipeline (for seasonal series)

When the series exhibits strong seasonality, a pre-decomposition step is applied before fitting:

```
Raw series
    │
    ▼
STL decomposition  (Seasonal-Trend via Loess)
    ├── trend      →  linear extrapolation
    ├── seasonal   →  day-of-year / month profile
    └── residual   →  fed to Kalman-EM
                           │
                           ▼
                      Kalman-EM fit
                           │
                           ▼
                   Recomposition: ŷ = ŷ_residual + seasonal + trend
```

This reduces residual variance to ~16 % of total variance and cuts MAPE by 60 %.

---

## Architecture

```
kalman_em/
├── kalman_filter.py   Core filter + RTS smoother (NaN-aware, Joseph form)
├── em.py              EM algorithm — E-step / M-step, spectral clipping
├── model.py           KalmanEM public API  (fit / filter / smooth / forecast)
└── __init__.py

examples/
├── stock_prediction.py     AAPL price smoothing + 30-day forecast
├── evaluate.py             One-step-ahead backtest with metrics
├── energy_prediction.py    French electricity: STL + Kalman-EM
└── transport_prediction.py SNCF TGV traffic: STL + Kalman-EM

data/
├── AAPL_prices.csv          752 trading days (2022–2024)
├── france_conso_elec.csv    1 827 daily observations (2020–2024)
├── sncf_tgv_mensuel.csv     96 monthly observations (2018–2025)
└── README.md                Data provenance and column descriptions

docs/
├── kalman_em_theory.pdf     Full theoretical derivations (Kalman + EM)
└── numerical_issues.md      Numerical pitfalls and best practices

app.py                       Streamlit Forecaster application
requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd kalman_time_series

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Matrix operations, linear algebra |
| `scipy` | ≥ 1.10 | Eigenvalue computation, numerical utilities |
| `pandas` | ≥ 2.0 | Data loading and date handling |
| `matplotlib` | ≥ 3.7 | Visualisation |
| `statsmodels` | ≥ 0.14 | STL decomposition |
| `streamlit` | ≥ 1.32 | Interactive web application |

---

## Streamlit application

The **Forecaster** app provides a no-code interface to the entire pipeline.

### Launch

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Workflow

```
1. Upload any CSV  →  date column + numeric value column
2. Configure preprocessing (STL period auto-detected)
3. Set model hyperparameters  →  Run Analysis
4. Explore the 4 result tabs
```

### Sidebar controls

| Control | Description |
|---------|-------------|
| **Upload CSV** | Any CSV with a date column and a numeric value column |
| **Date column** | Automatically guessed; can be overridden |
| **Value column** | All numeric columns are listed |
| **Remove seasonality (STL)** | Toggle STL pre-decomposition |
| **STL period** | Auto-detected from data frequency (daily→365, monthly→12, …) |
| **Latent dimension d** | Size of hidden state vector (1–6; default 2 = level + trend) |
| **Max EM iterations** | 20–500 (default 200) |
| **Test set ratio** | Fraction held out for backtesting (default 20 %) |
| **Convergence tolerance** | EM stopping criterion (default 1e-5) |
| **Run Analysis** | Triggers the full pipeline |

### Result tabs

#### 📊 Raw Data
Plot of the uploaded series and a data preview table.

#### 🔧 Preprocessing
If STL is enabled: four-panel decomposition (observed / trend / seasonal / residual) and the **residual variance ratio** — a measure of how much structure STL captured.

#### 📈 Backtest
One-step-ahead prediction results on the held-out test set.

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE (%)** | Mean Absolute Percentage Error |
| **Coverage ±2σ (%)** | Fraction of actuals inside the ±2σ prediction band (target ≈ 95 %) |

Three-panel figure:
1. Actual vs predicted with ±2σ confidence band
2. Absolute error per time step
3. Normalised residuals (should follow N(0,1) for a well-fitted model)

#### 🔬 Model Parameters
Learned state-space matrices displayed as annotated heatmaps:

| Parameter | Meaning |
|-----------|---------|
| **F** | State transition — encodes the learned dynamics |
| **H** | Observation — maps latent state to observed series |
| **Q** | Process noise — uncertainty in state evolution |
| **R** | Observation noise — measurement uncertainty |
| **Σ₀** | Initial state uncertainty |
| **ρ(F)** | Spectral radius — must be < 1 for stable dynamics |

EM convergence curve (log-likelihood vs iteration) is also shown.

### Tips for best results

- **Finance / smooth signals**: disable STL, use `d=2`.
- **Daily data with annual cycle** (energy, weather): enable STL with `period=365`.
- **Monthly data** (transport, sales): enable STL with `period=12`.
- **Coverage ≈ 95 %** is the calibration target for ±2σ intervals. If coverage is lower, try increasing `d`.
- If EM does not converge, reduce the tolerance or increase iterations.

---

## Command-line examples

### Stock prediction

```bash
python examples/stock_prediction.py \
    --csv data/AAPL_prices.csv \
    --latent 2 \
    --days 30 \
    --iters 200
```

### Backtest / validation

```bash
python examples/evaluate.py \
    --csv data/AAPL_prices.csv \
    --test_ratio 0.2 \
    --latent 2
```

### Electricity consumption (STL + Kalman-EM)

```bash
python examples/energy_prediction.py \
    --csv data/france_conso_elec.csv \
    --days 30 \
    --test_ratio 0.15
```

### TGV traffic

```bash
# Scheduled trains volume
python examples/transport_prediction.py \
    --csv data/sncf_tgv_mensuel.csv \
    --target trains_prevus \
    --days 12

# Punctuality rate
python examples/transport_prediction.py \
    --csv data/sncf_tgv_mensuel.csv \
    --target ponctualite_pct \
    --days 12
```

---

## Python API

```python
import numpy as np
from kalman_em import KalmanEM

# Y: (T, m) array — observations (NaN allowed for missing values)
Y = prices.reshape(-1, 1)

# --- Fit ---
model = KalmanEM(
    d=2,           # latent state dimension
    n_iter=200,    # max EM iterations
    tol=1e-5,      # convergence tolerance
    diagonal_R=True,
    verbose=True,
)
model.fit(Y, standardise=True)

# --- Learned parameters ---
print(model.params_)    # dict: F, H, Q, R, mu0, Sigma0
print(model.log_liks_)  # log-likelihood curve

# --- Smoothing (uses full series) ---
smooth_mean, smooth_var = model.smooth(Y)    # (T, m), (T, m)

# --- One-step-ahead backtest ---
y_pred, y_var = model.predict_one_step(Y_test, Y_context=Y_train)

# --- Multi-step forecast ---
fore_mean, fore_var = model.forecast(Y, n_steps=30)  # (30, m), (30, m)
```

---

## Datasets

See [`data/README.md`](data/README.md) for full provenance, API sources, licences, and column descriptions.

| File | Period | Frequency | Rows | Source |
|------|--------|-----------|------|--------|
| `AAPL_prices.csv` | 2022–2024 | Daily | 752 | Yahoo Finance |
| `france_conso_elec.csv` | 2020–2024 | Daily | 1 827 | RTE / ODRE (Etalab 2.0) |
| `sncf_tgv_mensuel.csv` | 2018–2025 | Monthly | 96 | SNCF Voyageurs (Etalab 2.0) |

---

## Results

### AAPL stock — backtest (last 20 % of data)

| MAE (USD) | RMSE (USD) | MAPE (%) | Coverage ±2σ (%) |
|-----------|------------|----------|------------------|
| ~1.6 | ~2.1 | ~0.8 | ~96 |

### French electricity — STL + Kalman-EM

| Setup | MAPE (%) |
|-------|----------|
| Raw Kalman-EM (d=2) | ~5.0 |
| **STL + Kalman-EM** | **~1.9** |

### SNCF TGV monthly traffic

| Target | MAPE (%) | Coverage ±2σ (%) |
|--------|----------|------------------|
| `trains_prevus` | ~2.5 | ~95 |
| `ponctualite_pct` | ~1.0 | ~97 |

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/kalman_em_theory.pdf`](docs/kalman_em_theory.pdf) | Full theoretical derivations: Kalman filter recursion, RTS smoother, EM E-step / M-step proofs, convergence theorem, prediction intervals |
| [`docs/numerical_issues.md`](docs/numerical_issues.md) | Numerical pitfalls encountered and fixes applied (spectral clipping, Joseph form, STL pipeline, pandas indexing) |
| [`data/README.md`](data/README.md) | Dataset provenance, column descriptions, API sources, licences |

---

## Project structure

```
kalman_time_series/
│
├── app.py                        Streamlit Forecaster application
├── requirements.txt              Python dependencies
├── README.md                     This file
│
├── kalman_em/                    Core library
│   ├── __init__.py
│   ├── kalman_filter.py          Kalman filter + RTS smoother
│   ├── em.py                     EM algorithm (E-step / M-step)
│   └── model.py                  KalmanEM public class
│
├── examples/                     Ready-to-run scripts
│   ├── stock_prediction.py       AAPL smoothing + forecast
│   ├── evaluate.py               One-step-ahead backtest
│   ├── energy_prediction.py      French electricity (STL + Kalman-EM)
│   └── transport_prediction.py   SNCF TGV traffic (STL + Kalman-EM)
│
├── data/                         Local CSV datasets
│   ├── AAPL_prices.csv
│   ├── france_conso_elec.csv
│   ├── sncf_tgv_mensuel.csv
│   └── README.md                 Data provenance
│
└── docs/                         Documentation
    ├── kalman_em_theory.pdf       Theoretical foundations
    └── numerical_issues.md        Numerical best practices
```

---

## Licence

Data files are distributed under their original licences (see [`data/README.md`](data/README.md)).
Source code is provided for educational and research purposes.

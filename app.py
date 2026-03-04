"""
Forecaster — Streamlit application for time series prediction using Kalman-EM.

Pipeline
--------
  Upload CSV → select columns → optional STL decomposition
  → Kalman-EM fit → one-step-ahead backtest → results display

Usage
-----
  streamlit run app.py
"""

import io
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from kalman_em import KalmanEM

# ---------------------------------------------------------------------------
# Optional STL import
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except ImportError:
    _HAS_STL = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecaster — Kalman-EM",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Auto-configuration — BIC-based model selection
# ---------------------------------------------------------------------------

def _n_free_params(d: int) -> int:
    """Free parameters for KalmanEM with m=1 observation dim."""
    # F(d²) + H(d) + Q(d(d+1)/2) + R(1, diagonal) + mu0(d) + Sigma0(d(d+1)/2)
    return d * d + d + d * (d + 1) // 2 + 1 + d + d * (d + 1) // 2


def _downsample(arr: np.ndarray, max_pts: int) -> np.ndarray:
    """Return a regularly-strided subsample of arr keeping at most max_pts points."""
    if len(arr) <= max_pts:
        return arr
    stride = max(1, len(arr) // max_pts)
    return arr[::stride]


def _auto_configure(dates, values, stl_period: int) -> dict:
    """
    Quick BIC-based selection of latent dim d, STL usage, and n_iter.
    Subsamples to MAX_AUTOCONF_PTS before fitting to stay fast on large datasets.
    """
    MAX_AUTOCONF_PTS = 500
    n = len(values)
    Y = values.copy().astype(float)
    seasonality_strength = 0.0
    use_stl = False

    # --- Seasonality strength (on full series — cheap) ---
    if stl_period > 1 and _HAS_STL and n >= 2 * stl_period + 1:
        try:
            trend_s, seas_s, resid_s = stl_decompose(dates, values, stl_period)
            detrended = values - trend_s.values
            var_dt = np.var(detrended)
            var_res = np.var(resid_s.values)
            seasonality_strength = float(max(0.0, 1.0 - var_res / (var_dt + 1e-10)))
            use_stl = seasonality_strength > 0.3
            if use_stl:
                Y = resid_s.values.astype(float)
        except Exception:
            pass

    # --- Subsample for BIC estimation ---
    Y = _downsample(Y, MAX_AUTOCONF_PTS)

    # --- Standardise ---
    mu_y, std_y = float(np.nanmean(Y)), float(np.nanstd(Y))
    Y_std = ((Y - mu_y) / std_y).reshape(-1, 1) if std_y > 0 else Y.reshape(-1, 1)
    T = len(Y_std)

    # --- BIC for d = 1..4 ---
    QUICK_ITERS = 20
    bic_scores: dict[int, float] = {}
    lls_by_d:   dict[int, list]  = {}

    for d in [1, 2, 3, 4]:
        try:
            m = KalmanEM(d=d, n_iter=QUICK_ITERS, tol=1e-4,
                         diagonal_R=True, diagonal_Q=False,
                         n_restarts=1, verbose=False)
            m.fit(Y_std, standardise=False)
            ll = m.log_liks_[-1] if m.log_liks_ else -np.inf
            bic = -2.0 * ll + np.log(T) * _n_free_params(d)
            bic_scores[d] = float(bic)
            lls_by_d[d]   = m.log_liks_
        except Exception:
            bic_scores[d] = float("inf")

    best_d = min(bic_scores, key=bic_scores.get)

    # --- Suggest n_iter ---
    converged_at = len(lls_by_d.get(best_d, []))
    if converged_at < QUICK_ITERS:
        raw = max(50, converged_at * 3)
    else:
        raw = 200
    suggested_n_iter = int(round(raw / 10) * 10)
    suggested_n_iter = max(20, min(500, suggested_n_iter))

    return {
        "use_stl":              use_stl,
        "stl_period":           stl_period,
        "d":                    best_d,
        "n_iter":               suggested_n_iter,
        "bic_scores":           bic_scores,
        "seasonality_strength": seasonality_strength,
    }


# ---------------------------------------------------------------------------
# Dataset presets — recommended settings per known dataset file
# ---------------------------------------------------------------------------

DATASET_PRESETS: dict[str, dict] = {
    "airline_passengers.csv": {
        "use_stl": True, "stl_period": 12, "d": 2, "n_iter": 100, "n_forecast": 24,
    },
    "sunspots_monthly.csv": {
        "use_stl": False, "stl_period": 132, "d": 4, "n_iter": 200, "n_forecast": 60,
    },
    "france_conso_elec.csv": {
        "use_stl": True, "stl_period": 365, "d": 2, "n_iter": 200, "n_forecast": 30,
    },
    "paris_temperature_daily.csv": {
        "use_stl": True, "stl_period": 365, "d": 2, "n_iter": 200, "n_forecast": 30,
    },
    "sncf_tgv_mensuel.csv": {
        "use_stl": True, "stl_period": 12, "d": 2, "n_iter": 200, "n_forecast": 12,
    },
    "aapl_prices.csv": {
        "use_stl": False, "stl_period": 12, "d": 2, "n_iter": 200, "n_forecast": 30,
    },
    "etth1.csv": {
        "use_stl": True, "stl_period": 24, "d": 2, "n_iter": 200, "n_forecast": 168,
    },
}

_SIDEBAR_DEFAULTS = {
    "sb_use_stl": False, "sb_stl_period": 12, "sb_d": 2,
    "sb_n_iter": 200, "sb_n_forecast": 30,
}


def _apply_preset(fname: str) -> bool:
    """Inject preset sidebar values into session_state. Returns True if preset found."""
    preset = DATASET_PRESETS.get(fname.lower(), {})
    base = {"use_stl": False, "stl_period": 12, "d": 2, "n_iter": 200, "n_forecast": 30}
    merged = {**base, **preset}
    st.session_state["sb_use_stl"]    = merged["use_stl"]
    st.session_state["sb_stl_period"] = merged["stl_period"]
    st.session_state["sb_d"]          = merged["d"]
    st.session_state["sb_n_iter"]     = merged["n_iter"]
    st.session_state["sb_n_forecast"] = merged["n_forecast"]
    return bool(preset)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def load_series(df: pd.DataFrame, date_col: str, val_col: str):
    """Return (dates as np.datetime64 array, values as float array)."""
    sub = df[[date_col, val_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col])
    sub = sub.sort_values(date_col).reset_index(drop=True)
    dates  = sub[date_col].values          # np.datetime64
    values = sub[val_col].values.astype(float)
    return dates, values


def infer_stl_period(dates) -> int:
    """Guess a sensible STL period from the median gap between dates."""
    if len(dates) < 3:
        return 1
    ts = pd.to_datetime(dates)
    gaps = (ts[1:] - ts[:-1]).days
    med = float(np.median(gaps))
    if med <= 2:
        return 365     # daily → annual seasonality
    if med <= 10:
        return 52      # weekly
    if med <= 45:
        return 12      # monthly
    if med <= 120:
        return 4       # quarterly
    return 1

# ---------------------------------------------------------------------------
# STL helpers (mirrors energy_prediction.py)
# ---------------------------------------------------------------------------

def stl_decompose(dates, values, period: int):
    """Return (trend, seasonal, resid) as pd.Series aligned on dates."""
    s = pd.Series(values, index=pd.to_datetime(dates))
    res = STL(s, period=period, robust=True).fit()
    return res.trend, res.seasonal, res.resid


def project_seasonal(seasonal_hist, n_fore: int, period: int) -> np.ndarray:
    """Tile the last complete seasonal cycle forward for n_fore steps."""
    cycle = np.array(seasonal_hist[-period:])   # shape (period,)
    return np.array([cycle[i % period] for i in range(n_fore)])

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_std, alpha=2.0) -> dict:
    err  = y_true - y_pred
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else np.nan
    lo   = y_pred - alpha * y_std
    hi   = y_pred + alpha * y_std
    cov  = float(np.mean((y_true >= lo) & (y_true <= hi)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, f"Coverage ±{alpha}σ (%)": cov}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(dates, values, cfg: dict, callback=None) -> dict:
    """
    Full Kalman-EM pipeline.

    Parameters
    ----------
    dates  : (T,) np.datetime64 array
    values : (T,) float array
    cfg    : dict with keys:
               use_stl, stl_period,
               d, n_iter, test_ratio, tol

    Returns
    -------
    dict with all results needed for display.
    """
    T = len(values)
    n_test  = max(1, int(T * cfg["test_ratio"]))
    n_train = T - n_test

    dates_train  = dates[:n_train]
    dates_test   = dates[n_train:]
    values_train = values[:n_train]
    values_test  = values[n_train:]

    stl_info = None

    # ---- STL branch --------------------------------------------------------
    if cfg["use_stl"] and _HAS_STL and cfg["stl_period"] > 1:
        period = cfg["stl_period"]

        # Decompose entire series so we have test STL components for recomposition
        trend_all, seas_all, resid_all = stl_decompose(dates, values, period)

        trend_train  = trend_all.values[:n_train]
        seas_train   = seas_all.values[:n_train]
        resid_train  = resid_all.values[:n_train]
        trend_test   = trend_all.values[n_train:]
        seas_test    = seas_all.values[n_train:]
        resid_test   = resid_all.values[n_train:]

        stl_info = {
            "trend":    trend_all.values,
            "seasonal": seas_all.values,
            "resid":    resid_all.values,
            "period":   period,
        }

        Y_train = resid_train.reshape(-1, 1)
        Y_test  = resid_test.reshape(-1, 1)

        model = KalmanEM(d=cfg["d"], n_iter=cfg["n_iter"], tol=cfg["tol"],
                         diagonal_R=True, diagonal_Q=False,
                         n_restarts=cfg["n_restarts"], verbose=False)
        model.fit(Y_train, standardise=True, callback=callback)

        # One-step-ahead on test residuals
        pred_raw, var_raw = model.predict_one_step(Y_test, Y_context=Y_train)
        resid_pred = pred_raw[:, 0]
        resid_std  = np.sqrt(np.abs(var_raw[:, 0]))

        # Recompose
        y_pred = resid_pred + seas_test + trend_test
        y_std  = resid_std   # uncertainty from residual model

    # ---- Direct Kalman branch ----------------------------------------------
    else:
        Y_train = values_train.reshape(-1, 1)
        Y_test  = values_test.reshape(-1, 1)

        model = KalmanEM(d=cfg["d"], n_iter=cfg["n_iter"], tol=cfg["tol"],
                         diagonal_R=True, diagonal_Q=False,
                         n_restarts=cfg["n_restarts"], verbose=False)
        model.fit(Y_train, standardise=True, callback=callback)

        pred_raw, var_raw = model.predict_one_step(Y_test, Y_context=Y_train)
        y_pred = pred_raw[:, 0]
        y_std  = np.sqrt(np.abs(var_raw[:, 0]))

    metrics = compute_metrics(values_test, y_pred, y_std)

    return {
        "model":        model,
        "dates_train":  dates_train,
        "dates_test":   dates_test,
        "values_train": values_train,
        "values_test":  values_test,
        "y_pred":       y_pred,
        "y_std":        y_std,
        "metrics":      metrics,
        "stl":          stl_info,
        "log_liks":     model.log_liks_,
        "params":       model.params_,
        "n_restarts":   cfg["n_restarts"],
    }

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_DATE_FMT = mdates.DateFormatter("%b %Y")


def _fmt_date_axis(ax, dates):
    span_days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
    if span_days > 500:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif span_days > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(_DATE_FMT)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def fig_raw(dates, values, val_col: str):
    MAX_PLOT = 3000
    d = _downsample(np.asarray(dates),  MAX_PLOT)
    v = _downsample(np.asarray(values), MAX_PLOT)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(pd.to_datetime(d), v, color="steelblue", lw=1)
    ax.set_title(f"Raw time series — {val_col}")
    ax.set_ylabel(val_col)
    _fmt_date_axis(ax, d)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_stl(dates, values, trend, seasonal, resid, val_col: str):
    MAX_PLOT = 3000
    dt = pd.to_datetime(_downsample(np.asarray(dates), MAX_PLOT))
    panels = [
        (_downsample(np.asarray(values),   MAX_PLOT), "Observed",  "steelblue"),
        (_downsample(np.asarray(trend),    MAX_PLOT), "Trend",     "darkorange"),
        (_downsample(np.asarray(seasonal), MAX_PLOT), "Seasonal",  "green"),
        (_downsample(np.asarray(resid),    MAX_PLOT), "Residual",  "purple"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5]})
    for ax, (data, label, color) in zip(axes, panels):
        ax.plot(dt, data, color=color, lw=1)
        ax.set_ylabel(label)
        _fmt_date_axis(ax, dt)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"STL decomposition — {val_col}")
    fig.tight_layout()
    return fig


def fig_backtest(dates_train, values_train, dates_test, values_test,
                 y_pred, y_std, metrics: dict):
    MAX_PLOT = 3000
    # Downsample train for display; keep test at full resolution (important for validation)
    dtr_d = _downsample(np.asarray(dates_train),  MAX_PLOT)
    vtr_d = _downsample(np.asarray(values_train), MAX_PLOT)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    dt_train = pd.to_datetime(dtr_d)
    dt_test  = pd.to_datetime(dates_test)

    # Panel 1 — predictions vs actual
    ax = axes[0]
    ax.plot(dt_train, vtr_d, color="steelblue", lw=1,
            label="Train", alpha=0.6)
    ax.plot(dt_test, values_test, color="steelblue", lw=1.8,
            label="Test (actual)", zorder=4)
    ax.plot(dt_test, y_pred, color="crimson", lw=1.8, linestyle="--",
            label="Kalman 1-step", zorder=5)
    ax.fill_between(dt_test, y_pred - 2*y_std, y_pred + 2*y_std,
                    color="crimson", alpha=0.15, label="±2σ")
    ax.axvline(dt_test[0], color="gray", linestyle=":", lw=1.2)
    ax.set_ylabel("Value")
    ax.set_title("One-step-ahead backtest")
    ax.legend(loc="upper left", fontsize=9)
    _fmt_date_axis(ax, np.concatenate([dates_train, dates_test]))
    ax.grid(True, alpha=0.3)

    metric_str = "  |  ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
    ax.text(0.01, 0.04, metric_str, transform=ax.transAxes,
            fontsize=8.5, color="darkred",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    # Panel 2 — absolute error
    ax2 = axes[1]
    abs_err = np.abs(values_test - y_pred)
    ax2.bar(dt_test, abs_err, color="orange", alpha=0.7, width=1.5)
    ax2.axhline(abs_err.mean(), color="red", lw=1.5, linestyle="--",
                label=f"MAE = {abs_err.mean():.3f}")
    ax2.set_ylabel("|Error|")
    ax2.set_title("Absolute prediction error")
    ax2.legend(fontsize=9)
    _fmt_date_axis(ax2, dates_test)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — normalised residuals
    ax3 = axes[2]
    residuals = (values_test - y_pred) / np.where(y_std > 0, y_std, 1e-9)
    ax3.plot(dt_test, residuals, color="purple", lw=0.9, alpha=0.9)
    ax3.axhline( 2, color="red", lw=1, linestyle="--", alpha=0.6)
    ax3.axhline(-2, color="red", lw=1, linestyle="--", alpha=0.6, label="±2σ")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("Normalised residual")
    ax3.set_title("Normalised residuals  (≈ N(0,1) if model is well-fitted)")
    ax3.legend(fontsize=9)
    _fmt_date_axis(ax3, dates_test)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _future_dates(dates, n: int) -> np.ndarray:
    """Generate n future dates with the same median frequency as the series."""
    last = pd.Timestamp(dates[-1])
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    step = max(1, int(np.median(gaps)))
    return np.array(
        [last + pd.Timedelta(days=i * step) for i in range(1, n + 1)],
        dtype="datetime64[ns]",
    )


def fig_reconstruction_forecast(
    dates_train, values_train,
    mu_train, std_train,
    dates_test, values_test,
    mu_fore_test, std_fore_test,
    dates_future, y_future, std_future,
    val_col: str,
):
    """
    Validation-first layout:
      [0, T_train]  : observed train (line) + Kalman smooth (blue)
      [T_train, T]  : observed test (dots) + multi-step forecast (red) — visual validation
      [T, T+n_fore] : pure future forecast (red dashed) — no ground truth
    """
    C_OBS   = "#2d2d2d"
    C_RECON = "#2196F3"
    C_FORE  = "#E53935"

    MAX_PLOT = 2000   # max points per series for readable rendering

    fig, ax = plt.subplots(figsize=(13, 5))

    # Downsample for display only (model data unchanged)
    dtr  = _downsample(np.asarray(dates_train),  MAX_PLOT)
    vtr  = _downsample(np.asarray(values_train), MAX_PLOT)
    mtr  = _downsample(np.asarray(mu_train),     MAX_PLOT)
    str_ = _downsample(np.asarray(std_train),    MAX_PLOT)
    dte  = _downsample(np.asarray(dates_test),   MAX_PLOT // 4)
    vte  = _downsample(np.asarray(values_test),  MAX_PLOT // 4)
    mft  = _downsample(np.asarray(mu_fore_test), MAX_PLOT // 4)
    sft  = _downsample(np.asarray(std_fore_test),MAX_PLOT // 4)

    dt_train  = pd.to_datetime(dtr)
    dt_test   = pd.to_datetime(dte)
    dt_future = pd.to_datetime(dates_future)

    # Observed — train as line, test as scatter dots (ground truth for validation)
    ax.plot(dt_train, vtr, color=C_OBS, lw=1.2, alpha=0.7,
            label="Observed (train)", zorder=1)
    ax.scatter(dt_test, vte, color=C_OBS, s=18, alpha=0.85, zorder=4,
               label="Observed (test — ground truth)")

    # Kalman smooth on train
    ax.fill_between(dt_train,
                    mtr - 2 * str_,
                    mtr + 2 * str_,
                    color=C_RECON, alpha=0.15, label="±2σ reconstruction", zorder=2)
    ax.plot(dt_train, mtr, color=C_RECON, lw=1.8,
            label="Reconstruction (smooth)", zorder=3)

    # Multi-step forecast over test period (validation zone)
    ax.fill_between(dt_test,
                    mft - 2 * sft,
                    mft + 2 * sft,
                    color=C_FORE, alpha=0.18, label="±2σ forecast", zorder=2)
    ax.plot(dt_test, mft, color=C_FORE, lw=2, linestyle="--",
            label="Forecast (validation)", zorder=3)

    # Pure future forecast (already short — no downsampling needed)
    ax.fill_between(dt_future,
                    y_future - 2 * std_future,
                    y_future + 2 * std_future,
                    color=C_FORE, alpha=0.10, zorder=2)
    ax.plot(dt_future, y_future, color=C_FORE, lw=2, linestyle="--",
            label="Forecast (future)", zorder=3)

    # Vertical separators
    ax.axvline(dt_test[0],   color="#888",  linestyle=":", lw=1.2, label="train | test")
    ax.axvline(dt_future[0], color=C_FORE,  linestyle=":", lw=1.2, label="test | future")

    # Anchor y-axis on observed data range only
    values_all = np.concatenate([values_train, values_test])
    y_lo = float(np.nanmin(values_all))
    y_hi = float(np.nanmax(values_all))
    margin = 0.12 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - margin, y_hi + margin)

    ax.set_ylabel(val_col)
    ax.set_title(f"Kalman-EM — Reconstruction & Forecast — {val_col}")
    ax.legend(loc="upper left", fontsize=9)
    all_dates = np.concatenate([dates_train, dates_test, dates_future])
    _fmt_date_axis(ax, all_dates)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def fig_loglik(log_liks):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(log_liks, color="steelblue", lw=1.5, marker="o", markersize=2)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("EM convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_matrix(M: np.ndarray, title: str):
    """Heatmap of a 2-D matrix."""
    fig, ax = plt.subplots(figsize=(max(2.5, M.shape[1] * 0.8 + 1),
                                    max(2.0, M.shape[0] * 0.8 + 0.8)))
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(M[i, j]) > 0.6 * np.abs(M).max() else "black")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels([f"j={k}" for k in range(M.shape[1])], fontsize=8)
    ax.set_yticklabels([f"i={k}" for k in range(M.shape[0])], fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parameter display
# ---------------------------------------------------------------------------

def show_params(params: dict, log_liks: list, n_restarts: int = 1):
    st.subheader("EM Convergence")
    st.pyplot(fig_loglik(log_liks), use_container_width=False)

    restart_note = f" (best of {n_restarts} restarts)" if n_restarts > 1 else ""
    st.markdown(f"**Converged in {len(log_liks)} iterations{restart_note}** "
                f"— final log-likelihood: {log_liks[-1]:.4f}")
    st.divider()

    # Scalar spectral radius of F
    F = params["F"]
    rho = float(np.max(np.abs(np.linalg.eigvals(F))))
    st.metric("Spectral radius ρ(F)", f"{rho:.5f}",
              help="Must be < 1 for stable dynamics")

    matrices = [
        ("F — Transition matrix",     params["F"]),
        ("H — Observation matrix",    params["H"]),
        ("Q — Process noise cov",     params["Q"]),
        ("R — Observation noise cov", params["R"]),
        ("Σ₀ — Initial state cov",    params["Sigma0"]),
    ]

    cols = st.columns(2)
    for idx, (label, M) in enumerate(matrices):
        with cols[idx % 2]:
            st.subheader(label)
            if M.ndim == 1:
                M = M.reshape(1, -1)
            st.pyplot(fig_matrix(M, label), use_container_width=True)
            with st.expander("Raw values"):
                st.dataframe(pd.DataFrame(
                    M,
                    index=[f"row {i}" for i in range(M.shape[0])],
                    columns=[f"col {j}" for j in range(M.shape[1])],
                ))

    st.subheader("μ₀ — Initial state mean")
    st.dataframe(pd.DataFrame(params["mu0"].reshape(1, -1),
                               columns=[f"x{i}" for i in range(len(params["mu0"]))]),
                 use_container_width=False)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> dict | None:
    """Render sidebar and return configuration dict, or None if not ready."""
    st.sidebar.title("Forecaster")
    st.sidebar.caption("Kalman-EM · time series prediction")
    st.sidebar.divider()

    # Initialise session-state defaults (first run only)
    for k, v in _SIDEBAR_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --- Upload ---
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.sidebar.info("Upload a CSV to get started.")
        return None

    # Detect file change → apply preset
    if st.session_state.get("_uploaded_name") != uploaded.name:
        st.session_state["_uploaded_name"] = uploaded.name
        preset_applied = _apply_preset(uploaded.name)
        st.session_state["_preset_applied"] = preset_applied

    file_bytes = uploaded.read()
    df = parse_csv(file_bytes)

    st.sidebar.success(f"{len(df):,} rows · {df.shape[1]} columns")
    if st.session_state.get("_preset_applied"):
        st.sidebar.info(f"⚡ Preset applied for **{uploaded.name}**")

    # --- Column selection ---
    st.sidebar.subheader("Column selection")
    all_cols = list(df.columns)

    # Guess date column
    date_guess = next(
        (c for c in all_cols if "date" in c.lower() or "time" in c.lower()), all_cols[0]
    )
    date_col = st.sidebar.selectbox("Date column", all_cols,
                                    index=all_cols.index(date_guess))

    num_cols = [c for c in all_cols if c != date_col and
                pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.sidebar.error("No numeric column found (excluding date column).")
        return None
    val_col = st.sidebar.selectbox("Value column", num_cols)

    # Parse series for period inference
    try:
        dates, values = load_series(df, date_col, val_col)
    except Exception as e:
        st.sidebar.error(f"Cannot parse series: {e}")
        return None

    suggested_period = infer_stl_period(dates)

    # --- Auto-configure: BIC-based model selection (runs once per series) ---
    autoconf_key = f"{val_col}_{len(values)}_{suggested_period}"
    if st.session_state.get("_autoconf_key") != autoconf_key:
        with st.spinner("🔍 Analysing dataset…"):
            acfg = _auto_configure(dates, values, suggested_period)
        st.session_state["_autoconf"]        = acfg
        st.session_state["_autoconf_key"]    = autoconf_key
        st.session_state["sb_use_stl"]       = acfg["use_stl"]
        st.session_state["sb_stl_period"]    = acfg["stl_period"]
        st.session_state["sb_d"]             = acfg["d"]
        st.session_state["sb_n_iter"]        = acfg["n_iter"]
        st.session_state["_preset_applied"]  = False

    acfg = st.session_state.get("_autoconf", {})
    if acfg:
        bic    = acfg.get("bic_scores", {})
        best_d = acfg.get("d", "?")
        s_str  = acfg.get("seasonality_strength", 0.0)
        with st.sidebar.expander("🔍 Auto-configuration", expanded=False):
            st.caption(f"Seasonality strength: **{s_str:.0%}**")
            st.caption(f"STL: {'✅ enabled' if acfg.get('use_stl') else '❌ disabled'} · "
                       f"Recommended d: **{best_d}**")
            if bic:
                bic_df = pd.DataFrame({
                    "d":    list(bic.keys()),
                    "BIC":  [f"{v:.1f}" if not np.isinf(v) else "—" for v in bic.values()],
                    "":     ["✓" if d == best_d else "" for d in bic.keys()],
                })
                st.dataframe(bic_df, use_container_width=True, hide_index=True)

    # --- Preprocessing ---
    st.sidebar.subheader("Preprocessing")
    stl_available = _HAS_STL
    use_stl = st.sidebar.toggle(
        "Remove seasonality / trend (STL)",
        key="sb_use_stl",
        disabled=not stl_available,
        help="Seasonal-Trend decomposition via Loess (statsmodels required)."
             if stl_available else "statsmodels not installed.",
    )

    stl_period = 1
    if use_stl:
        stl_period = st.sidebar.number_input(
            "STL period",
            min_value=2, max_value=10_000,
            key="sb_stl_period",
            help="Number of time steps per seasonal cycle "
                 f"(auto-detected: {suggested_period})",
        )

    # --- Model hyperparameters ---
    st.sidebar.subheader("Model hyperparameters")
    d = st.sidebar.slider("Latent dimension d", 1, 6, key="sb_d",
                          help="Dimension of the hidden state vector")
    n_iter = st.sidebar.slider("Max EM iterations", 20, 500, step=10, key="sb_n_iter")
    test_ratio = st.sidebar.slider("Test set ratio", 0.05, 0.40, 0.20, step=0.05,
                                   format="%.2f")
    tol = st.sidebar.select_slider(
        "Convergence tolerance",
        options=[1e-3, 1e-4, 1e-5, 1e-6],
        value=1e-5,
        format_func=lambda x: f"{x:.0e}",
        help="Relative criterion: ΔLL / (1 + |LL|) < tol",
    )
    n_restarts = st.sidebar.slider(
        "Random restarts",
        min_value=1, max_value=5, value=1,
        help="Run EM multiple times with different initialisations and keep the best result.",
    )

    st.sidebar.subheader("Forecast")
    n_forecast = st.sidebar.slider(
        "Forecast horizon",
        min_value=5, max_value=365, step=5, key="sb_n_forecast",
        help="Number of steps to project beyond the last observation.",
    )

    st.sidebar.divider()
    run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    return {
        "df":          df,
        "file_bytes":  file_bytes,
        "date_col":    date_col,
        "val_col":     val_col,
        "dates":       dates,
        "values":      values,
        "use_stl":     use_stl,
        "stl_period":  int(stl_period),
        "d":           d,
        "n_iter":      n_iter,
        "test_ratio":  test_ratio,
        "tol":         tol,
        "n_restarts":  n_restarts,
        "n_forecast":  n_forecast,
        "run":         run_btn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = sidebar()

    # Landing page when no file uploaded
    if cfg is None:
        st.title("Forecaster — Kalman-EM")
        st.markdown("""
Welcome to **Forecaster**, a time series prediction app powered by the
**Kalman filter with EM parameter estimation**.

### How it works
1. **Upload** a CSV file with a date column and a numeric value column.
2. **Configure** optional STL pre-processing (removes trend and seasonality)
   and model hyperparameters in the sidebar.
3. Click **Run Analysis** — the app will:
   - Train a Kalman-EM model on the training portion of your data.
   - Run an honest **one-step-ahead backtest** on the held-out test set.
   - Display **model parameters** (F, H, Q, R matrices) and **performance metrics**.

### Tips
- Enable *Remove seasonality / trend* for series with strong seasonal patterns
  (electricity consumption, monthly traffic, etc.).
- For stock prices or smooth signals, the raw Kalman-EM usually works well.
- The spectral radius ρ(F) < 1 confirms the model produces stable dynamics.
        """)
        return

    # ---- Data preview (always visible once a file is loaded) ----
    tab_raw, tab_pre, tab_bt, tab_fore, tab_params = st.tabs([
        "📊 Raw Data",
        "🔧 Preprocessing",
        "📈 Backtest",
        "🔮 Reconstruction & Forecast",
        "🔬 Model Parameters",
    ])

    with tab_raw:
        st.subheader(f"Series: {cfg['val_col']}  vs  {cfg['date_col']}")
        st.pyplot(fig_raw(cfg["dates"], cfg["values"], cfg["val_col"]),
                  use_container_width=True)
        st.caption(f"{len(cfg['values']):,} observations")
        with st.expander("Data preview (first 50 rows)"):
            st.dataframe(cfg["df"].head(50), use_container_width=True)

    # ---- Clear stale results when series changes ----
    series_key = f"{cfg['val_col']}_{len(cfg['values'])}"
    if st.session_state.get("_series_key") != series_key:
        st.session_state.pop("results", None)
    st.session_state["_series_key"] = series_key

    # ---- Trigger pipeline ----
    if cfg["run"]:
        conv_slot = st.empty()

        def _em_callback(i, n_iter, log_liks):
            # Update every 5 iterations to avoid flooding re-renders
            if i % 5 != 0 and i < n_iter - 1:
                return
            with conv_slot.container():
                st.progress(
                    min((i + 1) / n_iter, 1.0),
                    text=f"EM iteration {i + 1}/{n_iter} · log-lik = {log_liks[-1]:.4f}",
                )
                if len(log_liks) > 1:
                    st.line_chart(
                        {"Log-likelihood": log_liks},
                        height=160,
                        use_container_width=True,
                    )

        try:
            results = run_pipeline(cfg["dates"], cfg["values"], cfg,
                                   callback=_em_callback)
            conv_slot.empty()
            st.session_state["results"] = results
            st.session_state["val_col"] = cfg["val_col"]
        except Exception as e:
            conv_slot.empty()
            st.error(f"Pipeline failed: {e}")
            st.session_state.pop("results", None)

    results = st.session_state.get("results")

    # ---- Preprocessing tab ----
    with tab_pre:
        if results is None:
            st.info("Click **Run Analysis** in the sidebar to start.")
        elif results["stl"] is None:
            st.success("No pre-processing applied — Kalman-EM trained on raw series.")
        else:
            stl = results["stl"]
            # Use the dates/values from the pipeline run — never cfg (stale mismatch)
            dates_all  = np.concatenate([results["dates_train"], results["dates_test"]])
            values_all = np.concatenate([results["values_train"], results["values_test"]])
            st.subheader("STL decomposition")
            st.pyplot(
                fig_stl(dates_all, values_all,
                        stl["trend"], stl["seasonal"], stl["resid"],
                        cfg["val_col"]),
                use_container_width=True,
            )
            var_ratio = float(np.var(stl["resid"]) / np.var(values_all) * 100)
            st.metric(
                "Residual variance / Total variance",
                f"{var_ratio:.1f} %",
                help="Lower → STL captured more structure; Kalman trains on a simpler signal.",
            )

    # ---- Backtest tab ----
    with tab_bt:
        if results is None:
            st.info("Click **Run Analysis** in the sidebar to start.")
        else:
            m = results["metrics"]
            cols = st.columns(len(m))
            for col, (k, v) in zip(cols, m.items()):
                col.metric(k, f"{v:.3f}")

            st.pyplot(
                fig_backtest(
                    results["dates_train"], results["values_train"],
                    results["dates_test"],  results["values_test"],
                    results["y_pred"],      results["y_std"],
                    m,
                ),
                use_container_width=True,
            )

    # ---- Reconstruction & Forecast tab ----
    with tab_fore:
        if results is None:
            st.info("Click **Run Analysis** in the sidebar to start.")
        else:
            model        = results["model"]
            stl          = results["stl"]
            dates_train  = results["dates_train"]
            dates_test   = results["dates_test"]
            values_train = results["values_train"]
            values_test  = results["values_test"]
            dates_all    = np.concatenate([dates_train, dates_test])
            values_all   = np.concatenate([values_train, values_test])
            n_train      = len(values_train)
            n_fore       = cfg["n_forecast"]

            n_test = len(values_test)

            # --- Kalman smooth on TRAIN only ---
            Y_train_kal = (stl["resid"][:n_train].reshape(-1, 1)
                           if stl is not None
                           else values_train.reshape(-1, 1))
            mu_s, var_s = model.smooth(Y_train_kal)
            if stl is not None:
                mu_train_plot = mu_s[:, 0] + stl["seasonal"][:n_train] + stl["trend"][:n_train]
            else:
                mu_train_plot = mu_s[:, 0]
            std_train_plot = np.sqrt(np.abs(var_s[:, 0]))

            # --- Multi-step forecast from end of TRAIN → covers test + future ---
            # This enables visual validation: forecast vs actual test dots
            n_total_fore = n_test + n_fore
            y_fore_all_raw, y_fore_all_var = model.forecast(Y_train_kal, n_steps=n_total_fore)

            # Test portion of forecast (validation zone)
            mu_fore_test_resid  = y_fore_all_raw[:n_test, 0]
            std_fore_test       = np.sqrt(np.abs(y_fore_all_var[:n_test, 0]))
            if stl is not None:
                mu_fore_test = (mu_fore_test_resid
                                + stl["seasonal"][n_train:]
                                + stl["trend"][n_train:])
            else:
                mu_fore_test = mu_fore_test_resid

            # Future portion of forecast (no ground truth)
            dates_future        = _future_dates(dates_all, n_fore)
            mu_future_resid     = y_fore_all_raw[n_test:, 0]
            std_future          = np.sqrt(np.abs(y_fore_all_var[n_test:, 0]))
            if stl is not None:
                seas_future  = project_seasonal(stl["seasonal"], n_fore, stl["period"])
                trend        = stl["trend"]
                slope        = (trend[-1] - trend[max(0, len(trend) - 10)]) / min(10, len(trend) - 1)
                trend_future = trend[-1] + slope * np.arange(1, n_fore + 1)
                mu_future    = mu_future_resid + seas_future + trend_future
            else:
                mu_future = mu_future_resid

            st.pyplot(
                fig_reconstruction_forecast(
                    dates_train, values_train,
                    mu_train_plot, std_train_plot,
                    dates_test, values_test,
                    mu_fore_test, std_fore_test,
                    dates_future, mu_future, std_future,
                    cfg["val_col"],
                ),
                use_container_width=True,
            )

            # Table of forecast values (future only) + download
            fore_df = pd.DataFrame({
                "Date":      pd.to_datetime(dates_future).strftime("%Y-%m-%d"),
                "Forecast":  np.round(mu_future, 4),
                "Lower_2σ":  np.round(mu_future - 2 * std_future, 4),
                "Upper_2σ":  np.round(mu_future + 2 * std_future, 4),
            })
            col_tbl, col_dl = st.columns([3, 1])
            with col_tbl:
                with st.expander(f"Forecast values (next {n_fore} steps beyond test)"):
                    st.dataframe(fore_df, use_container_width=True)
            with col_dl:
                st.download_button(
                    label="⬇ Download forecast CSV",
                    data=fore_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"forecast_{cfg['val_col']}_{n_fore}steps.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ---- Model parameters tab ----
    with tab_params:
        if results is None:
            st.info("Click **Run Analysis** in the sidebar to start.")
        else:
            show_params(results["params"], results["log_liks"],
                        results.get("n_restarts", 1))


if __name__ == "__main__":
    main()

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
    Universal auto-configuration for any dataset.

    Decisions taken automatically:
      1. Log transform  — if series is strictly positive and max/min > 10 or CV > 0.5
      2. STL            — if STL reduces residual variance by > 10 % (low bar, intentionally)
      3. Latent dim d   — BIC over d = 1..4 on the final work signal
      4. n_iter         — extrapolated from quick-run convergence speed

    All decisions are made on the (possibly log-transformed, possibly STL-residual) signal
    to ensure the Kalman receives the most stationary signal possible.
    """
    MAX_AUTOCONF_PTS = 500
    QUICK_ITERS      = 20
    n = len(values)
    raw = values.astype(float)

    # ── 1. Log transform detection ────────────────────────────────────────────
    all_pos = bool(np.all(raw > 0))
    if all_pos:
        cv    = float(np.std(raw) / (np.mean(raw) + 1e-10))
        ratio = float(np.max(raw) / (np.min(raw) + 1e-10))
        log_transform = (cv > 0.5) or (ratio > 5)
    else:
        log_transform = False

    work = np.log(raw) if log_transform else raw.copy()

    # ── 2. STL: try it, keep it if variance reduction ≥ 10 % ─────────────────
    use_stl              = False
    seasonality_strength = 0.0
    var_reduction        = 0.0
    Y                    = work.copy()

    if stl_period > 1 and _HAS_STL and n >= 2 * stl_period + 1:
        try:
            trend_s, seas_s, resid_s = stl_decompose(dates, work, stl_period)
            var_work  = float(np.var(work))
            var_resid = float(np.var(resid_s.values))
            var_reduction = float(max(0.0, 1.0 - var_resid / (var_work + 1e-10)))

            # Seasonality strength (relative to detrended signal)
            detrended = work - trend_s.values
            var_dt    = float(np.var(detrended))
            seasonality_strength = float(max(0.0, 1.0 - var_resid / (var_dt + 1e-10)))

            # Low threshold: keep STL if it removes at least 10 % of variance
            use_stl = var_reduction >= 0.10
            if use_stl:
                Y = resid_s.values.astype(float)
        except Exception:
            pass

    # ── 3. BIC over d = 1..4 on the final (subsampled) signal ───────────────
    Y = _downsample(Y, MAX_AUTOCONF_PTS)
    mu_y, std_y = float(np.nanmean(Y)), float(np.nanstd(Y))
    Y_std = ((Y - mu_y) / std_y).reshape(-1, 1) if std_y > 0 else Y.reshape(-1, 1)
    T     = len(Y_std)

    bic_scores: dict[int, float] = {}
    lls_by_d:   dict[int, list]  = {}

    for d in [1, 2, 3, 4]:
        try:
            m = KalmanEM(d=d, n_iter=QUICK_ITERS, tol=1e-4,
                         diagonal_R=True, diagonal_Q=False,
                         n_restarts=1, verbose=False)
            m.fit(Y_std, standardise=False)
            ll  = m.log_liks_[-1] if m.log_liks_ else -np.inf
            bic = -2.0 * ll + np.log(T) * _n_free_params(d)
            bic_scores[d] = float(bic)
            lls_by_d[d]   = list(m.log_liks_) if m.log_liks_ else []
        except Exception:
            bic_scores[d] = float("inf")

    best_d = min(bic_scores, key=lambda k: bic_scores[k])

    # ── 4. Suggest n_iter ─────────────────────────────────────────────────────
    converged_at = len(lls_by_d.get(best_d, []))
    raw_iter     = max(50, converged_at * 3) if converged_at < QUICK_ITERS else 200
    suggested_n_iter = max(20, min(500, int(round(raw_iter / 10) * 10)))

    return {
        "log_transform":        log_transform,
        "use_stl":              use_stl,
        "stl_period":           stl_period,
        "var_reduction":        var_reduction,
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
        "use_stl": True, "stl_period": 12, "log_transform": True, "d": 2, "n_iter": 100, "n_forecast": 24,
    },
    "co2_mauna_loa.csv": {
        "use_stl": True, "stl_period": 12, "log_transform": False, "d": 2, "n_iter": 200, "n_forecast": 24,
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
        "use_stl": False, "stl_period": 12, "log_transform": True, "d": 2, "n_iter": 200, "n_forecast": 30,
    },
    "btc_usd_daily.csv": {
        "use_stl": False, "stl_period": 1, "log_transform": True, "d": 3, "n_iter": 200, "n_forecast": 30,
    },

    "airline_passengers.csv": {
        "use_stl": True, "stl_period": 12, "log_transform": True, "d": 2, "n_iter": 100, "n_forecast": 24,
    },
    "etth1.csv": {
        "use_stl": True, "stl_period": 168, "d": 2, "n_iter": 200, "n_forecast": 168,
    },
}

_SIDEBAR_DEFAULTS = {
    "sb_use_stl": False, "sb_stl_period": 12, "sb_d": 2,
    "sb_n_iter": 200, "sb_n_forecast": 30,
}


def _apply_preset(fname: str) -> bool:
    """Inject preset sidebar values into session_state. Returns True if preset found."""
    preset = DATASET_PRESETS.get(fname.lower(), {})
    base = {"use_stl": False, "stl_period": 12, "log_transform": False,
            "d": 2, "n_iter": 200, "n_forecast": 30}
    merged = {**base, **preset}
    st.session_state["sb_use_stl"]       = merged["use_stl"]
    st.session_state["sb_stl_period"]    = merged["stl_period"]
    st.session_state["sb_log_transform"] = merged["log_transform"]
    st.session_state["sb_d"]             = merged["d"]
    st.session_state["sb_n_iter"]        = merged["n_iter"]
    st.session_state["sb_n_forecast"]    = merged["n_forecast"]
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
    ts  = pd.to_datetime(dates)
    # Use seconds for precision — handles sub-daily frequencies
    med_sec = float(np.median((ts[1:] - ts[:-1]).total_seconds()))
    if med_sec < 120:          # minute-level
        return 60              # hourly cycle
    if med_sec < 7200:         # hourly (< 2 h)
        return 24              # daily cycle
    if med_sec < 172800:       # daily (< 2 days)
        return 365             # annual cycle
    if med_sec < 864000:       # weekly (< 10 days)
        return 52
    if med_sec < 3456000:      # monthly (< 40 days)
        return 12
    if med_sec < 10368000:     # quarterly (< 120 days)
        return 4
    return 1

# ---------------------------------------------------------------------------
# STL helpers (mirrors energy_prediction.py)
# ---------------------------------------------------------------------------

def stl_decompose(dates, values, period: int):
    """Return (trend, seasonal, resid) as pd.Series aligned on dates."""
    s = pd.Series(values, index=pd.to_datetime(dates))
    res = STL(s, period=period, robust=True).fit()
    return res.trend, res.seasonal, res.resid


def project_seasonal(seasonal_hist, n_fore: int, period: int, n_cycles: int = 5) -> np.ndarray:
    """
    Tile the average of the last n_cycles seasonal cycles forward for n_fore steps.
    Falls back to fewer cycles if the history is too short.
    """
    hist = np.array(seasonal_hist)
    avail = len(hist) // period          # number of complete cycles available
    k = max(1, min(n_cycles, avail))     # how many cycles we can actually use
    cycles = hist[-(k * period):]        # shape (k*period,)
    cycle  = cycles.reshape(k, period).mean(axis=0)   # average across cycles
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
               use_stl, stl_period, log_transform,
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

    # ---- Log transform (applied before everything, reversed at the end) ------
    log_transform = cfg.get("log_transform", False) and np.all(values > 0)
    if log_transform:
        work_values  = np.log(values)
        work_train   = np.log(values_train)
        work_test    = np.log(values_test)
    else:
        work_values  = values
        work_train   = values_train
        work_test    = values_test

    stl_info = None

    # ---- STL branch --------------------------------------------------------
    if cfg["use_stl"] and _HAS_STL and cfg["stl_period"] > 1:
        period = cfg["stl_period"]

        # Decompose entire (possibly log-transformed) series
        trend_all, seas_all, resid_all = stl_decompose(dates, work_values, period)

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

        pred_raw, var_raw = model.predict_one_step(Y_test, Y_context=Y_train)
        resid_pred = pred_raw[:, 0]
        resid_std  = np.sqrt(np.abs(var_raw[:, 0]))

        # Recompose in work space, then back-transform
        y_work_pred = resid_pred + seas_test + trend_test
        y_pred = np.exp(y_work_pred) if log_transform else y_work_pred
        y_std  = y_pred * resid_std if log_transform else resid_std

    # ---- Direct Kalman branch ----------------------------------------------
    else:
        Y_train = work_train.reshape(-1, 1)
        Y_test  = work_test.reshape(-1, 1)

        model = KalmanEM(d=cfg["d"], n_iter=cfg["n_iter"], tol=cfg["tol"],
                         diagonal_R=True, diagonal_Q=False,
                         n_restarts=cfg["n_restarts"], verbose=False)
        model.fit(Y_train, standardise=True, callback=callback)

        pred_raw, var_raw = model.predict_one_step(Y_test, Y_context=Y_train)
        y_work_pred = pred_raw[:, 0]
        y_work_std  = np.sqrt(np.abs(var_raw[:, 0]))

        # Back-transform to original space
        y_pred = np.exp(y_work_pred) if log_transform else y_work_pred
        y_std  = y_pred * y_work_std if log_transform else y_work_std

    metrics = compute_metrics(values_test, y_pred, y_std)

    return {
        "model":          model,
        "dates_train":    dates_train,
        "dates_test":     dates_test,
        "values_train":   values_train,
        "values_test":    values_test,
        "work_train":     work_train,   # log-space or original, used for forecast tab
        "log_transform":  log_transform,
        "y_pred":         y_pred,
        "y_std":          y_std,
        "metrics":        metrics,
        "stl":            stl_info,
        "log_liks":       model.log_liks_,
        "params":         model.params_,
        "n_restarts":     cfg["n_restarts"],
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


# ---------------------------------------------------------------------------
# Shared visual theme
# ---------------------------------------------------------------------------

_P = {
    "obs":   "#1E2A3A",   # deep navy    — observed data
    "recon": "#2E86DE",   # vivid blue   — Kalman reconstruction
    "fore":  "#E84855",   # warm red     — forecast
    "trend": "#F0A500",   # amber        — STL trend
    "seas":  "#3BB273",   # emerald      — STL seasonal
    "resid": "#7B5EA7",   # muted purple — residuals
    "grid":  "#EBEBEB",   # very light gray grid
}


def _style_ax(ax, *, yonly=False):
    """Clean, minimal axes style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_P["grid"])
    ax.spines["bottom"].set_color(_P["grid"])
    ax.tick_params(colors="#666666", labelsize=10)
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    ax.title.set_color("#111111")
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("semibold")
    axis = "y" if yonly else "both"
    ax.grid(True, color=_P["grid"], linewidth=0.9, linestyle="-", axis="y")
    ax.set_axisbelow(True)


def fig_raw(dates, values, val_col: str):
    MAX_PLOT = 3000
    d = _downsample(np.asarray(dates),  MAX_PLOT)
    v = _downsample(np.asarray(values), MAX_PLOT)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("white")
    ax.plot(pd.to_datetime(d), v, color=_P["obs"], lw=1.4, alpha=0.9)
    ax.set_title(f"Raw time series — {val_col}")
    ax.set_ylabel(val_col, fontsize=10)
    _fmt_date_axis(ax, d)
    _style_ax(ax)
    fig.tight_layout(pad=1.5)
    return fig


def fig_stl(dates, values, trend, seasonal, resid, val_col: str):
    MAX_PLOT = 3000
    dt = pd.to_datetime(_downsample(np.asarray(dates), MAX_PLOT))
    panels = [
        (_downsample(np.asarray(values),   MAX_PLOT), "Observed", _P["obs"]),
        (_downsample(np.asarray(trend),    MAX_PLOT), "Trend",    _P["trend"]),
        (_downsample(np.asarray(seasonal), MAX_PLOT), "Seasonal", _P["seas"]),
        (_downsample(np.asarray(resid),    MAX_PLOT), "Residual", _P["resid"]),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5]})
    fig.patch.set_facecolor("white")
    for ax, (data, label, color) in zip(axes, panels):
        ax.plot(dt, data, color=color, lw=1.4)
        ax.set_ylabel(label, fontsize=10)
        _style_ax(ax)
    axes[0].set_title(f"STL decomposition — {val_col}")
    _fmt_date_axis(axes[-1], dt)
    fig.tight_layout(pad=1.5, h_pad=0.6)
    return fig


def fig_backtest(dates_train, values_train, dates_test, values_test,
                 y_pred, y_std, metrics: dict):
    MAX_PLOT = 3000
    dtr_d = _downsample(np.asarray(dates_train),  MAX_PLOT)
    vtr_d = _downsample(np.asarray(values_train), MAX_PLOT)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.patch.set_facecolor("white")
    dt_train = pd.to_datetime(dtr_d)
    dt_test  = pd.to_datetime(dates_test)

    # Panel 1 — predictions vs actual
    ax = axes[0]
    ax.plot(dt_train, vtr_d, color=_P["obs"], lw=1.2, alpha=0.35, label="Train")
    ax.plot(dt_test, values_test, color=_P["obs"], lw=2,
            label="Test (actual)", zorder=4)
    ax.fill_between(dt_test, y_pred - 2*y_std, y_pred + 2*y_std,
                    color=_P["fore"], alpha=0.12, zorder=2, label="±2σ")
    ax.plot(dt_test, y_pred, color=_P["fore"], lw=2, linestyle="--",
            label="Kalman 1-step", zorder=5)
    ax.axvline(dt_test[0], color="#CCCCCC", linestyle="--", lw=1, zorder=1)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title("One-step-ahead backtest")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="#DDDDDD")
    _fmt_date_axis(ax, np.concatenate([dates_train, dates_test]))
    _style_ax(ax)
    metric_str = "   ".join(f"{k}  {v:.3f}" for k, v in metrics.items())
    ax.text(0.99, 0.97, metric_str, transform=ax.transAxes,
            fontsize=9, color="#444444", ha="right", va="top",
            bbox=dict(facecolor="#F7F7F7", alpha=0.95, edgecolor="#DDDDDD",
                      boxstyle="round,pad=0.45"))

    # Panel 2 — absolute error (area)
    ax2 = axes[1]
    abs_err = np.abs(values_test - y_pred)
    ax2.fill_between(dt_test, 0, abs_err, color=_P["fore"], alpha=0.20)
    ax2.plot(dt_test, abs_err, color=_P["fore"], lw=1.2)
    ax2.axhline(abs_err.mean(), color=_P["fore"], lw=1.5, linestyle="--",
                label=f"MAE = {abs_err.mean():.3f}")
    ax2.set_ylabel("|Error|", fontsize=10)
    ax2.set_title("Absolute error")
    ax2.legend(fontsize=9, framealpha=0.95, edgecolor="#DDDDDD")
    _fmt_date_axis(ax2, dates_test)
    _style_ax(ax2)

    # Panel 3 — normalised residuals
    ax3 = axes[2]
    residuals = (values_test - y_pred) / np.where(y_std > 0, y_std, 1e-9)
    ax3.fill_between(dt_test, residuals, 0,
                     where=np.abs(residuals) > 2,
                     color=_P["fore"], alpha=0.25, label="Outside ±2σ")
    ax3.plot(dt_test, residuals, color=_P["resid"], lw=1.2)
    ax3.axhline( 2, color="#BBBBBB", lw=1, linestyle="--")
    ax3.axhline(-2, color="#BBBBBB", lw=1, linestyle="--", label="±2σ bounds")
    ax3.axhline(0,  color="#999999", lw=0.8)
    ax3.set_ylabel("Norm. residual", fontsize=10)
    ax3.set_title("Normalised residuals  (≈ N(0,1) if well-fitted)")
    ax3.legend(fontsize=9, framealpha=0.95, edgecolor="#DDDDDD")
    _fmt_date_axis(ax3, dates_test)
    _style_ax(ax3)

    fig.tight_layout(pad=1.5, h_pad=1.2)
    return fig


def _future_dates(dates, n: int) -> np.ndarray:
    """Generate n future dates with the same median frequency as the series."""
    ts   = pd.to_datetime(dates)
    last = ts[-1]
    # Use timedelta directly (handles sub-daily frequencies correctly)
    step = pd.to_timedelta(np.median((ts[1:] - ts[:-1]).total_seconds()), unit="s")
    return np.array(
        [last + step * i for i in range(1, n + 1)],
        dtype="datetime64[ns]",
    )


def fig_reconstruction_forecast(
    dates_train, values_train,
    mu_train, std_train,
    dates_test, values_test,
    mu_fore_test, std_fore_test,
    dates_future, y_future, std_future,
    val_col: str,
    reliable_horizon: int = 0,
):
    """
    Validation-first layout:
      [0, T_train]  : observed train (line) + Kalman smooth (blue)
      [T_train, T]  : observed test (dots) + multi-step forecast (red) — visual validation
      [T, T+n_fore] : pure future forecast (red dashed) — no ground truth

    reliable_horizon : if > 0 and < n_test, draw a shaded unreliable zone beyond that point.
    """
    C_OBS   = _P["obs"]
    C_RECON = _P["recon"]
    C_FORE  = _P["fore"]

    MAX_PLOT = 2000

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

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

    # Observed train (line) + test (scatter)
    ax.plot(dt_train, vtr, color=C_OBS, lw=1.3, alpha=0.45,
            label="Observed (train)", zorder=1)
    ax.scatter(dt_test, vte, color=C_OBS, s=14, alpha=0.7, zorder=4,
               label="Observed (test)")

    # Kalman reconstruction on train
    ax.fill_between(dt_train, mtr - 2*str_, mtr + 2*str_,
                    color=C_RECON, alpha=0.10, zorder=2)
    ax.plot(dt_train, mtr, color=C_RECON, lw=2,
            label="Reconstruction", zorder=3)

    # Multi-step forecast over test period
    ax.fill_between(dt_test, mft - 2*sft, mft + 2*sft,
                    color=C_FORE, alpha=0.12, zorder=2)
    ax.plot(dt_test, mft, color=C_FORE, lw=2, linestyle="--",
            label="Forecast (validation)", zorder=3)

    # Unreliable zone
    if reliable_horizon > 0 and reliable_horizon < len(dates_test):
        dt_horizon = pd.to_datetime(dates_test[reliable_horizon])
        ax.axvspan(dt_horizon, dt_future[-1],
                   color="#BBBBBB", alpha=0.13, zorder=0,
                   label=f"Beyond reliable horizon (~{reliable_horizon} steps)")
        ax.axvline(dt_horizon, color="#AAAAAA", linestyle="--", lw=1, zorder=5)

    # Pure future forecast
    ax.fill_between(dt_future, y_future - 2*std_future, y_future + 2*std_future,
                    color=C_FORE, alpha=0.08, zorder=2)
    ax.plot(dt_future, y_future, color=C_FORE, lw=2, linestyle="--",
            label="Forecast (future)", zorder=3)

    # Zone separators
    ax.axvline(dt_test[0],   color="#CCCCCC", linestyle="--", lw=1, zorder=1)
    ax.axvline(dt_future[0], color="#CCCCCC", linestyle="--", lw=1, zorder=1)

    # Zone labels at top
    ylims = ax.get_ylim()
    y_label = ylims[1] - 0.02 * (ylims[1] - ylims[0])
    for x_mid, label in [
        (dt_train[len(dt_train)//2],  "Train"),
        (dt_test[len(dt_test)//2],    "Validation"),
        (dt_future[len(dt_future)//2],"Forecast"),
    ]:
        ax.text(x_mid, y_label, label, ha="center", va="top",
                fontsize=9, color="#888888", style="italic")

    # y-axis anchor on observed data
    values_all = np.concatenate([values_train, values_test])
    y_lo = float(np.nanmin(values_all))
    y_hi = float(np.nanmax(values_all))
    margin = 0.15 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - margin, y_hi + margin)

    ax.set_ylabel(val_col, fontsize=11)
    ax.set_title(f"Kalman-EM — Reconstruction & Forecast — {val_col}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95,
              edgecolor="#DDDDDD", ncol=2)
    all_dates = np.concatenate([dates_train, dates_test, dates_future])
    _fmt_date_axis(ax, all_dates)
    _style_ax(ax)
    fig.tight_layout(pad=1.5)
    return fig


def fig_loglik(log_liks):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("white")
    ax.fill_between(range(len(log_liks)), log_liks, min(log_liks),
                    color=_P["recon"], alpha=0.10)
    ax.plot(log_liks, color=_P["recon"], lw=2)
    ax.set_xlabel("EM iteration", fontsize=10)
    ax.set_ylabel("Log-likelihood", fontsize=10)
    ax.set_title("EM convergence")
    _style_ax(ax)
    fig.tight_layout(pad=1.5)
    return fig


def fig_matrix(M: np.ndarray, title: str):
    """Heatmap of a 2-D matrix."""
    fig, ax = plt.subplots(figsize=(max(3.0, M.shape[1] * 1.1 + 1),
                                    max(2.5, M.shape[0] * 1.1 + 0.8)))
    fig.patch.set_facecolor("white")
    vmax = float(np.abs(M).max()) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors="#555555")
    ax.set_title(title, fontsize=11, fontweight="semibold", color="#111111")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "#222222")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels([f"j={k}" for k in range(M.shape[1])], fontsize=9)
    ax.set_yticklabels([f"i={k}" for k in range(M.shape[0])], fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Forecast quality report
# ---------------------------------------------------------------------------

_GRADE_COLOR = {"Excellent": "🟢", "Good": "🟡", "Poor": "🔴"}


def _grade(value: float, thresholds: tuple, labels=("Excellent", "Good", "Poor")) -> str:
    """Return a grade label given ascending thresholds (low-is-better)."""
    if value <= thresholds[0]:
        return labels[0]
    if value <= thresholds[1]:
        return labels[1]
    return labels[2]


def _grade_coverage(cov: float) -> str:
    """Coverage should be ~95 %; penalise both too low and too high."""
    if 88 <= cov <= 100:
        return "Excellent"
    if 75 <= cov < 88:
        return "Good"
    return "Poor"


def show_forecast_quality(results: dict, val_col: str):
    """
    Guided forecast quality report written for non-specialists.
    Plain-language narrative + simple cards + actionable advice.
    Technical details hidden in an expert expander.
    """
    metrics      = results["metrics"]
    log_liks     = results["log_liks"]
    params       = results["params"]
    y_pred       = results["y_pred"]
    y_std        = results["y_std"]
    values_test  = results["values_test"]
    values_train = results["values_train"]
    stl          = results["stl"]

    # ── Derived quantities ───────────────────────────────────────────────────
    mae      = float(metrics.get("MAE",  np.nan))
    rmse     = float(metrics.get("RMSE", np.nan))
    mape     = float(metrics.get("MAPE (%)", np.nan))
    coverage = float(metrics.get(next(k for k in metrics if "Coverage" in k), 0.0))

    series_mean  = float(np.mean(np.abs(values_test)))   # scale reference
    norm_resid   = (values_test - y_pred) / np.where(y_std > 0, y_std, 1e-9)
    nr_std       = float(norm_resid.std())
    nr_mean      = float(norm_resid.mean())

    F   = params["F"]
    rho = float(np.max(np.abs(np.linalg.eigvals(F))))

    if len(log_liks) >= 10:
        ll_delta  = abs(log_liks[-1] - log_liks[-10]) / (abs(log_liks[-1]) + 1e-10)
        converged = ll_delta < 5e-3
    else:
        converged = True

    # Effective forecast horizon (variance saturation)
    _Y_kal = (stl["resid"][:len(values_train)] if stl else values_train).reshape(-1, 1)
    var_steps = results["model"].forecast(_Y_kal, n_steps=min(200, len(values_test)))[1][:, 0]
    growth    = np.diff(var_steps)
    flat_idx  = int(np.argmax(growth < 1e-4 * growth[0])) if np.any(growth < 1e-4 * growth[0]) else len(growth)
    horizon_steps = max(1, flat_idx)

    # ── Grades ───────────────────────────────────────────────────────────────
    g_mape     = _grade(mape, (5, 15)) if not np.isnan(mape) else "Good"
    g_coverage = _grade_coverage(coverage)
    g_nr       = _grade(abs(nr_std - 1.0), (0.25, 0.6))
    g_rho      = "Excellent" if 0.7 <= rho < 1.0 else ("Good" if rho < 0.7 else "Poor")
    g_converge = "Excellent" if converged else "Poor"

    overall_grades = [g_mape, g_coverage, g_nr, g_rho, g_converge]
    n_poor = overall_grades.count("Poor")
    n_good = overall_grades.count("Good")
    if n_poor >= 2:
        overall = "Poor"
    elif n_poor == 1 or n_good >= 2:
        overall = "Good"
    else:
        overall = "Excellent"

    # ── Plain-language building blocks ───────────────────────────────────────

    # Accuracy sentence
    if not np.isnan(mape):
        if mape < 5:
            acc_sentence = (f"The model is **very accurate**: on average, its predictions are off by "
                            f"only **{mape:.1f} %** of the true value "
                            f"({mae:.2f} {val_col} units per step).")
        elif mape < 15:
            acc_sentence = (f"The model shows **acceptable accuracy**: predictions deviate by "
                            f"**{mape:.1f} %** on average "
                            f"({mae:.2f} {val_col} units per step).")
        else:
            acc_sentence = (f"The model's accuracy is **limited**: average deviation is "
                            f"**{mape:.1f} %** ({mae:.2f} {val_col} units per step). "
                            "See recommendations below.")
    else:
        acc_sentence = (f"Average prediction error: **{mae:.2f} {val_col} units** per step "
                        "(percentage error not available — the series contains zeros).")

    # Consistency sentence (RMSE vs MAE)
    rmse_ratio = rmse / mae if mae > 0 else 1.0
    if rmse_ratio < 1.5:
        consist_sentence = "Errors are **consistent** — there are no catastrophic outlier predictions."
    elif rmse_ratio < 2.5:
        consist_sentence = ("Errors are **mostly consistent** but a few predictions were significantly "
                            "off — check the absolute error panel in the chart above.")
    else:
        consist_sentence = ("There are **occasional large errors** (some predictions are far from "
                            "the truth) — the model struggles with certain periods.")

    # Confidence sentence
    if g_coverage == "Excellent":
        conf_sentence = (f"The uncertainty bands are **well-calibrated**: {coverage:.0f} % of true "
                         "values fall inside the predicted range (ideal ≈ 95 %).")
    elif coverage < 88:
        conf_sentence = (f"The model is **slightly over-confident**: only {coverage:.0f} % of true "
                         "values fall inside its predicted range (expected ≈ 95 %). "
                         "Take the confidence bands as a lower bound on actual uncertainty.")
    else:
        conf_sentence = (f"The uncertainty bands are **a little wide** ({coverage:.0f} % coverage). "
                         "The model is cautious — predictions are conservative.")

    # Horizon sentence
    if horizon_steps >= 50:
        hor_sentence = (f"The model maintains **meaningful predictive power for at least "
                        f"{horizon_steps} steps** ahead before uncertainty dominates.")
    elif horizon_steps >= 10:
        hor_sentence = (f"The model is reliable for roughly **{horizon_steps} steps ahead**. "
                        "Beyond that, forecasts converge to the series average.")
    else:
        hor_sentence = (f"The model has a **short reliable horizon (~{horizon_steps} steps)**. "
                        "Long-range forecasts will quickly revert to the series mean.")

    # ── What went well / What to improve ─────────────────────────────────────
    positives, improvements = [], []

    if g_mape == "Excellent":
        positives.append(f"Prediction accuracy is excellent (average error < 5 %)")
    if g_coverage == "Excellent":
        positives.append("Confidence intervals are well-calibrated (~95 % coverage)")
    if g_rho in ("Excellent", "Good") and rho < 1.0:
        positives.append("The model learned stable, non-diverging dynamics from the data")
    if g_converge == "Excellent":
        positives.append("Model training completed successfully")
    if rmse_ratio < 1.5:
        positives.append("Errors are uniform — no catastrophic prediction failures")
    if horizon_steps >= 30:
        positives.append(f"Good forecast horizon — reliable for {horizon_steps}+ steps ahead")

    if not converged:
        improvements.append(
            "**Training did not fully complete** — increase *EM iterations* in the sidebar "
            "(try 300–500). Parameters may be sub-optimal.")
    if rho >= 1.0:
        improvements.append(
            "**The model dynamics are unstable** (may diverge at long horizons). "
            "Try reducing the *latent dimension d* or running more iterations.")
    if g_mape == "Poor" and not np.isnan(mape):
        improvements.append(
            f"**High prediction error ({mape:.1f} %)** — consider: "
            "① enabling *Remove seasonality/trend* if the series has cycles; "
            "② increasing *latent dimension d*; "
            "③ running more EM iterations.")
    if stl is None and not np.isnan(mape) and mape > 10:
        improvements.append(
            "**Seasonality/trend removal is off** — if the series has regular cycles "
            "(daily, weekly, annual), enabling STL decomposition often reduces error significantly.")
    if coverage < 80:
        improvements.append(
            f"**Confidence bands are too narrow** ({coverage:.0f} % coverage, expected 95 %) — "
            "the model underestimates its own uncertainty. Try more EM iterations or a larger d.")
    if nr_std > 1.5:
        improvements.append(
            "**Prediction intervals are under-sized** — true values frequently fall outside "
            "the shaded band. Increase iterations or d to better estimate the noise level.")
    if horizon_steps < 10:
        improvements.append(
            f"**Short forecast horizon (~{horizon_steps} steps)** — the model forgets the "
            "past quickly. For longer horizons, try a larger *latent dimension d* or enable STL.")

    # Horizon warning flag (used in render + improvements)
    n_test       = len(values_test)
    horizon_warn = horizon_steps < n_test // 2

    if horizon_warn:
        improvements.append(
            f"**The test set is much longer than the reliable forecast horizon** "
            f"({n_test} test steps vs ~{horizon_steps} reliable steps). "
            f"The multi-step forecast in the Reconstruction & Forecast tab will diverge "
            f"from actual observations after ~{horizon_steps} steps — this is expected and "
            f"does **not** affect the backtest score above. "
            f"To reduce this effect: shorten the test ratio, or use the model only for "
            f"short-horizon forecasting (≤ {horizon_steps} steps)."
        )

    # ── Render ───────────────────────────────────────────────────────────────
    st.subheader("One-step-ahead Backtest Quality")

    st.info(
        "ℹ️ **What this report measures:** the model's ability to predict **one step at a time**, "
        "each time using the true previous value as input (one-step-ahead backtest). "
        "This is the most honest measure of model fit, but it is **not the same** as the "
        "multi-step forecast shown in the *Reconstruction & Forecast* tab, which predicts "
        "many steps ahead without feedback from true values."
    )

    _VERDICT_ICON = {"Excellent": "✅", "Good": "⚠️", "Poor": "❌"}
    _VERDICT_DESC = {
        "Excellent": "The one-step-ahead predictions are reliable and well-calibrated.",
        "Good":      "The one-step-ahead predictions are usable but have room for improvement.",
        "Poor":      "The one-step-ahead prediction quality is limited — see recommendations below.",
    }
    verdict_color = {"Excellent": "success", "Good": "warning", "Poor": "error"}
    getattr(st, verdict_color[overall])(
        f"{_VERDICT_ICON[overall]} **Overall verdict: {overall}** — "
        f"{_VERDICT_DESC[overall]}"
    )

    # ── Narrative summary ────────────────────────────────────────────────────
    st.markdown("#### In plain words")
    st.markdown(
        f"{acc_sentence}  \n"
        f"{consist_sentence}  \n"
        f"{conf_sentence}  \n"
        f"{hor_sentence}"
    )

    # ── 4 simple metric cards ────────────────────────────────────────────────
    st.markdown("#### Key indicators")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label="Average error",
        value=f"{mape:.1f} %" if not np.isnan(mape) else f"{mae:.2f}",
        help=f"On average, each prediction is off by {mape:.1f} % of the true value "
             f"({mae:.2f} {val_col} units). Lower is better.",
    )
    c2.metric(
        label="Worst-case error",
        value=f"{rmse:.2f} {val_col}",
        help="Root Mean Squared Error. Penalises large occasional mistakes more than "
             "the average error. If much larger than the average error, some periods "
             "are poorly predicted.",
    )
    c3.metric(
        label="Confidence band accuracy",
        value=f"{coverage:.0f} %",
        help="What fraction of true values fall inside the shaded ±2σ band. "
             "Ideal value is ~95 %. "
             "Too low → the band is too narrow (model is over-confident). "
             "Too high → the band is too wide (model is over-cautious).",
    )
    c4.metric(
        label="Reliable forecast horizon",
        value=f"{horizon_steps} steps",
        help="How many steps ahead the model can predict before uncertainty "
             "overwhelms the signal. Beyond this point, forecasts converge "
             "to the historical average.",
    )

    # ── What went well / To improve ─────────────────────────────────────────
    col_pos, col_imp = st.columns(2)
    with col_pos:
        st.markdown("#### What went well")
        if positives:
            for p in positives:
                st.markdown(f"✅ {p}")
        else:
            st.markdown("_(nothing stands out positively)_")

    with col_imp:
        st.markdown("#### What to improve")
        if improvements:
            for imp in improvements:
                st.warning(imp)
        else:
            st.success("No significant issues detected.")

    # ── Expert details (hidden by default) ───────────────────────────────────
    with st.expander("🔬 Technical details (for experts)", expanded=False):
        st.markdown(f"""
| Diagnostic | Value | Grade | Technical interpretation |
|---|---|---|---|
| MAPE | {mape:.1f} % | {_GRADE_COLOR[g_mape]} {g_mape} | Mean absolute percentage error on the test set |
| Coverage ±2σ | {coverage:.1f} % | {_GRADE_COLOR[g_coverage]} {g_coverage} | Fraction of test points inside the 95 % prediction interval |
| RMSE / MAE | {rmse_ratio:.2f} | {"🟢" if rmse_ratio < 1.5 else "🟡" if rmse_ratio < 2.5 else "🔴"} | Ratio close to 1 → uniform errors; large ratio → outlier predictions |
| Spectral radius ρ(F) | {rho:.4f} | {_GRADE_COLOR[g_rho]} {g_rho} | Largest eigenvalue of F. Must be < 1 (stability). High → long memory. Low → fast mean-reversion. |
| EM convergence | {"Yes" if converged else "No"} | {_GRADE_COLOR[g_converge]} {g_converge} | Relative LL improvement over last 10 iterations < 0.5 % |
| Norm. residuals σ | {nr_std:.3f} (μ = {nr_mean:.3f}) | {_GRADE_COLOR[g_nr]} {g_nr} | Std of (y − ŷ)/σ_pred. Should be ≈ 1.0. > 1.5 → R under-estimated. < 0.6 → R over-estimated. |
| Effective horizon | {horizon_steps} steps | — | Steps until forecast variance growth < 10⁻⁴ of initial growth rate |
""")


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

    # --- Auto-configure: runs once per series, decides everything ---------------
    autoconf_key = f"{val_col}_{len(values)}_{suggested_period}"
    if st.session_state.get("_autoconf_key") != autoconf_key:
        with st.spinner("🔍 Analysing dataset…"):
            acfg = _auto_configure(dates, values, suggested_period)
        st.session_state["_autoconf"]           = acfg
        st.session_state["_autoconf_key"]       = autoconf_key
        st.session_state["sb_log_transform"]    = acfg["log_transform"]
        st.session_state["sb_use_stl"]          = acfg["use_stl"]
        st.session_state["sb_stl_period"]       = acfg["stl_period"]
        st.session_state["sb_d"]                = acfg["d"]
        st.session_state["sb_n_iter"]           = acfg["n_iter"]
        st.session_state["_preset_applied"]     = False

    acfg = st.session_state.get("_autoconf", {})
    if acfg:
        bic      = acfg.get("bic_scores", {})
        best_d   = acfg.get("d", "?")
        s_str    = acfg.get("seasonality_strength", 0.0)
        var_red  = acfg.get("var_reduction", 0.0)
        log_rec  = acfg.get("log_transform", False)
        with st.sidebar.expander("🔍 Auto-configuration", expanded=False):
            st.caption(
                f"{'🔢 Log transform: **on**' if log_rec else '🔢 Log transform: off'}  \n"
                f"{'📉 STL: **enabled**' if acfg.get('use_stl') else '📉 STL: disabled'}"
                f"{'  (variance reduction: ' + f'{var_red:.0%})' if acfg.get('use_stl') else ''}  \n"
                f"📐 Best latent dim: **d = {best_d}**  \n"
                f"🌊 Seasonality strength: **{s_str:.0%}**"
            )
            if bic:
                bic_df = pd.DataFrame({
                    "d":    list(bic.keys()),
                    "BIC":  [f"{v:.1f}" if not np.isinf(v) else "—" for v in bic.values()],
                    "":     ["✓" if d == best_d else "" for d in bic.keys()],
                })
                st.dataframe(bic_df, use_container_width=True, hide_index=True)

    # --- Preprocessing ---
    st.sidebar.subheader("Preprocessing")

    all_positive = bool(np.all(values > 0))
    log_transform = st.sidebar.toggle(
        "Log transform",
        key="sb_log_transform",
        value=st.session_state.get("sb_log_transform", False),
        disabled=not all_positive,
        help="Apply log(y) before modelling. Recommended for financial prices, "
             "passenger counts, and any series where variance grows with level. "
             "Forecasts are automatically back-transformed to the original scale."
             if all_positive else "Disabled: series contains non-positive values.",
    )

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
        "stl_period":    int(stl_period),
        "log_transform": log_transform,
        "d":             d,
        "n_iter":        n_iter,
        "test_ratio":    test_ratio,
        "tol":           tol,
        "n_restarts":    n_restarts,
        "n_forecast":    n_forecast,
        "run":           run_btn,
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

            st.divider()
            show_forecast_quality(results, cfg["val_col"])

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

            log_transform = results.get("log_transform", False)
            work_train    = results.get("work_train", values_train)

            # --- Kalman smooth on TRAIN only ---
            Y_train_kal = (stl["resid"][:n_train].reshape(-1, 1)
                           if stl is not None
                           else work_train.reshape(-1, 1))
            mu_s, var_s = model.smooth(Y_train_kal)
            if stl is not None:
                mu_train_work = mu_s[:, 0] + stl["seasonal"][:n_train] + stl["trend"][:n_train]
            else:
                mu_train_work = mu_s[:, 0]
            std_train_plot = np.sqrt(np.abs(var_s[:, 0]))
            mu_train_plot  = np.exp(mu_train_work) if log_transform else mu_train_work

            # --- Multi-step forecast from end of TRAIN → covers test + future ---
            n_total_fore = n_test + n_fore
            y_fore_all_raw, y_fore_all_var = model.forecast(Y_train_kal, n_steps=n_total_fore)

            # Test portion of forecast (in work space, then back-transform)
            mu_fore_test_resid = y_fore_all_raw[:n_test, 0]
            std_fore_test      = np.sqrt(np.abs(y_fore_all_var[:n_test, 0]))
            if stl is not None:
                mu_fore_test_work = (mu_fore_test_resid
                                     + stl["seasonal"][n_train:]
                                     + stl["trend"][n_train:])
            else:
                mu_fore_test_work = mu_fore_test_resid
            mu_fore_test = np.exp(mu_fore_test_work) if log_transform else mu_fore_test_work
            if log_transform:
                std_fore_test = mu_fore_test * std_fore_test

            # Future portion of forecast
            dates_future    = _future_dates(dates_all, n_fore)
            mu_future_resid = y_fore_all_raw[n_test:, 0]
            std_future      = np.sqrt(np.abs(y_fore_all_var[n_test:, 0]))
            if stl is not None:
                seas_future  = project_seasonal(stl["seasonal"], n_fore, stl["period"])
                trend        = stl["trend"]
                slope        = (trend[-1] - trend[max(0, len(trend) - 10)]) / min(10, len(trend) - 1)
                trend_future = trend[-1] + slope * np.arange(1, n_fore + 1)
                mu_future_work = mu_future_resid + seas_future + trend_future
            else:
                mu_future_work = mu_future_resid
            mu_future = np.exp(mu_future_work) if log_transform else mu_future_work
            if log_transform:
                std_future = mu_future * std_future

            # --- Effective forecast horizon ---
            var_fore_growth = np.diff(y_fore_all_var[:min(200, n_test + n_fore), 0])
            if len(var_fore_growth) > 0 and np.any(var_fore_growth < 1e-4 * var_fore_growth[0]):
                _horizon = max(1, int(np.argmax(var_fore_growth < 1e-4 * var_fore_growth[0])))
            else:
                _horizon = len(var_fore_growth)

            # --- Horizon warning ---
            if _horizon < n_test // 2:
                st.warning(
                    f"⚠️ **Model reliability warning** — this model can reliably forecast only "
                    f"~**{_horizon} steps** ahead (based on its learned dynamics). "
                    f"The validation zone (test set) spans **{n_test} steps**, which is much longer — "
                    f"beyond ~{_horizon} steps the forecast reverts to the series mean and no longer tracks the actual signal. "
                    f"The divergence in the validation zone is **expected**, not a bug.\n\n"
                    f"*Your **Future forecast** setting ({n_fore} steps) is a separate control — "
                    f"it sets how many steps are projected beyond the end of your data.*"
                )

            st.pyplot(
                fig_reconstruction_forecast(
                    dates_train, values_train,
                    mu_train_plot, std_train_plot,
                    dates_test, values_test,
                    mu_fore_test, std_fore_test,
                    dates_future, mu_future, std_future,
                    cfg["val_col"],
                    reliable_horizon=_horizon,
                ),
                use_container_width=True,
            )

            # --- Short-horizon forecast accuracy (within the reliable zone) ---
            n_eval = min(_horizon, n_test)
            if n_eval >= 5:
                err_h = values_test[:n_eval] - mu_fore_test[:n_eval]
                mae_h  = float(np.mean(np.abs(err_h)))
                mask_h = values_test[:n_eval] != 0
                mape_h = float(np.mean(np.abs(err_h[mask_h] / values_test[:n_eval][mask_h])) * 100) if mask_h.any() else np.nan
                fa, fb = st.columns(2)
                fa.metric(
                    f"Multi-step MAE (first {n_eval} steps)",
                    f"{mae_h:.3f}",
                    help=f"Average error of the multi-step forecast over the first {n_eval} steps "
                         f"(reliable horizon). Compare with the backtest MAE to see how much "
                         f"accuracy is lost going from one-step to multi-step forecasting.",
                )
                if not np.isnan(mape_h):
                    fb.metric(
                        f"Multi-step MAPE (first {n_eval} steps)",
                        f"{mape_h:.1f} %",
                        help="Percentage error of the multi-step forecast within the reliable horizon.",
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

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


def project_seasonal(dates_hist, seasonal_hist, dates_fore) -> np.ndarray:
    """Project historical seasonal pattern to forecast dates by day-of-period."""
    hist_ts  = pd.to_datetime(dates_hist)
    fore_ts  = pd.to_datetime(dates_fore)
    # Use day-of-year for daily; month for monthly; generically modulo period index
    doy_hist = hist_ts.dayofyear
    doy_fore = fore_ts.dayofyear
    profile  = pd.Series(seasonal_hist, index=doy_hist).groupby(level=0).mean()
    return np.array([profile.get(d, float(profile.mean())) for d in doy_fore])

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

def run_pipeline(dates, values, cfg: dict) -> dict:
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
        }

        Y_train = resid_train.reshape(-1, 1)
        Y_test  = resid_test.reshape(-1, 1)

        model = KalmanEM(d=cfg["d"], n_iter=cfg["n_iter"], tol=cfg["tol"],
                         diagonal_R=True, diagonal_Q=False, verbose=False)
        model.fit(Y_train, standardise=True)

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
                         diagonal_R=True, diagonal_Q=False, verbose=False)
        model.fit(Y_train, standardise=True)

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
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(pd.to_datetime(dates), values, color="steelblue", lw=1)
    ax.set_title(f"Raw time series — {val_col}")
    ax.set_ylabel(val_col)
    _fmt_date_axis(ax, dates)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_stl(dates, values, trend, seasonal, resid, val_col: str):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5]})
    dt = pd.to_datetime(dates)
    panels = [
        (values,   "Observed",   "steelblue"),
        (trend,    "Trend",      "darkorange"),
        (seasonal, "Seasonal",   "green"),
        (resid,    "Residual",   "purple"),
    ]
    for ax, (data, label, color) in zip(axes, panels):
        ax.plot(dt, data, color=color, lw=1)
        ax.set_ylabel(label)
        _fmt_date_axis(ax, dates)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"STL decomposition — {val_col}")
    fig.tight_layout()
    return fig


def fig_backtest(dates_train, values_train, dates_test, values_test,
                 y_pred, y_std, metrics: dict):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    dt_train = pd.to_datetime(dates_train)
    dt_test  = pd.to_datetime(dates_test)

    # Panel 1 — predictions vs actual
    ax = axes[0]
    ax.plot(dt_train, values_train, color="steelblue", lw=1,
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

def show_params(params: dict, log_liks: list):
    st.subheader("EM Convergence")
    st.pyplot(fig_loglik(log_liks), use_container_width=False)

    st.markdown(f"**Converged in {len(log_liks)} iterations** "
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

    # --- Upload ---
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.sidebar.info("Upload a CSV to get started.")
        return None

    file_bytes = uploaded.read()
    df = parse_csv(file_bytes)

    st.sidebar.success(f"{len(df):,} rows · {df.shape[1]} columns")

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

    # --- Preprocessing ---
    st.sidebar.subheader("Preprocessing")
    stl_available = _HAS_STL
    use_stl = st.sidebar.toggle(
        "Remove seasonality / trend (STL)",
        value=False,
        disabled=not stl_available,
        help="Seasonal-Trend decomposition via Loess (statsmodels required)."
             if stl_available else "statsmodels not installed.",
    )

    suggested_period = infer_stl_period(dates)
    stl_period = 1
    if use_stl:
        stl_period = st.sidebar.number_input(
            "STL period",
            min_value=2, max_value=10_000,
            value=suggested_period,
            help="Number of time steps per seasonal cycle "
                 f"(auto-detected: {suggested_period})",
        )

    # --- Model hyperparameters ---
    st.sidebar.subheader("Model hyperparameters")
    d = st.sidebar.slider("Latent dimension d", 1, 6, 2,
                          help="Dimension of the hidden state vector")
    n_iter = st.sidebar.slider("Max EM iterations", 20, 500, 200, step=10)
    test_ratio = st.sidebar.slider("Test set ratio", 0.05, 0.40, 0.20, step=0.05,
                                   format="%.2f")
    tol = st.sidebar.select_slider(
        "Convergence tolerance",
        options=[1e-3, 1e-4, 1e-5, 1e-6],
        value=1e-5,
        format_func=lambda x: f"{x:.0e}",
    )

    st.sidebar.divider()
    run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    return {
        "df":         df,
        "file_bytes": file_bytes,
        "date_col":   date_col,
        "val_col":    val_col,
        "dates":      dates,
        "values":     values,
        "use_stl":    use_stl,
        "stl_period": int(stl_period),
        "d":          d,
        "n_iter":     n_iter,
        "test_ratio": test_ratio,
        "tol":        tol,
        "run":        run_btn,
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
    tab_raw, tab_pre, tab_bt, tab_params = st.tabs([
        "📊 Raw Data",
        "🔧 Preprocessing",
        "📈 Backtest",
        "🔬 Model Parameters",
    ])

    with tab_raw:
        st.subheader(f"Series: {cfg['val_col']}  vs  {cfg['date_col']}")
        st.pyplot(fig_raw(cfg["dates"], cfg["values"], cfg["val_col"]),
                  use_container_width=True)
        st.caption(f"{len(cfg['values']):,} observations")
        with st.expander("Data preview (first 50 rows)"):
            st.dataframe(cfg["df"].head(50), use_container_width=True)

    # ---- Trigger pipeline ----
    if cfg["run"]:
        with st.spinner("Running Kalman-EM pipeline…"):
            try:
                results = run_pipeline(cfg["dates"], cfg["values"], cfg)
                st.session_state["results"] = results
                st.session_state["val_col"] = cfg["val_col"]
            except Exception as e:
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
            st.subheader("STL decomposition")
            st.pyplot(
                fig_stl(cfg["dates"], cfg["values"],
                        stl["trend"], stl["seasonal"], stl["resid"],
                        cfg["val_col"]),
                use_container_width=True,
            )
            var_ratio = float(np.var(stl["resid"]) / np.var(cfg["values"]) * 100)
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

    # ---- Model parameters tab ----
    with tab_params:
        if results is None:
            st.info("Click **Run Analysis** in the sidebar to start.")
        else:
            show_params(results["params"], results["log_liks"])


if __name__ == "__main__":
    main()

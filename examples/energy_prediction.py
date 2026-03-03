"""
French electricity consumption prediction with the Kalman-EM filter.

Data source
-----------
  RTE (Réseau de Transport d'Électricité) via Open Data Réseaux Energies (ODRE)
  URL    : https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def
  Licence: Etalab 2.0 (free reuse with attribution)

Modelling strategy
------------------
Electricity consumption exhibits two strong seasonalities (annual, weekly).
A plain Kalman filter (d=2) converges poorly on such seasonal data.

Chosen approach: STL decomposition + Kalman-EM on the residual
  1. STL (Seasonal-Trend decomposition using Loess) separates:
       conso(t) = trend(t) + seasonal(t) + residual(t)
  2. The Kalman-EM filter is trained on the residual (near-stationary series).
  3. The final forecast recombines: residual_forecast + projected_seasonal + trend.

This typically reduces MAPE from ~5% to ~2%.

Usage
-----
    python examples/energy_prediction.py [--csv data/france_conso_elec.csv] [--days 30]
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import STL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kalman_em import KalmanEM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Electricity consumption prediction — Kalman-EM")
    p.add_argument("--csv",     default="data/france_conso_elec.csv",
                   help="Local CSV (columns: Date, Conso_MW)")
    p.add_argument("--latent",  type=int, default=2,
                   help="Latent state dimension")
    p.add_argument("--days",    type=int, default=30,
                   help="Forecast horizon (days)")
    p.add_argument("--iters",   type=int, default=200,
                   help="Max EM iterations")
    p.add_argument("--test_ratio", type=float, default=0.15,
                   help="Fraction of the series reserved for testing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------

def stl_decompose(dates, values, period=365):
    """
    STL decomposition (Seasonal-Trend via Loess).

    Returns trend, seasonal, residual over the full series.
    period=365 captures annual seasonality.
    """
    s = pd.Series(values, index=pd.to_datetime(dates))
    stl = STL(s, period=period, robust=True)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid


def project_seasonal(dates_hist, seasonal_hist, dates_fore):
    """
    Projects the seasonal profile onto future dates
    by matching the day of the year.
    """
    df_hist = pd.DataFrame({
        "doy": pd.to_datetime(dates_hist).dayofyear,
        "s":   seasonal_hist,
    })
    profile = df_hist.groupby("doy")["s"].mean()

    fore_doy = pd.to_datetime(dates_fore).dayofyear
    return np.array([profile.get(d, profile.mean()) for d in fore_doy])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df["Date"].values, df.iloc[:, 1].values.astype(float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred, y_std):
    err  = y_true - y_pred
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / y_true)) * 100
    lo, hi = y_pred - 2*y_std, y_pred + 2*y_std
    cov  = np.mean((y_true >= lo) & (y_true <= hi)) * 100
    return {"MAE (MW)": mae, "RMSE (MW)": rmse,
            "MAPE (%)": mape, "Coverage ±2σ (%)": cov}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_full(dates, obs, smooth_mean, smooth_std,
              dates_fore, fore_mean, fore_std, title):
    """Full view: history + smoothing + forecast."""
    fig, ax = plt.subplots(figsize=(15, 5))

    # Historical data
    ax.plot(dates, obs / 1e3, color="steelblue", lw=0.8,
            alpha=0.7, label="Actual consumption")

    # Smoother
    ax.plot(dates, smooth_mean[:, 0] / 1e3, color="orange",
            lw=1.4, label="Kalman smoother")
    ax.fill_between(dates,
                    (smooth_mean[:, 0] - 2*smooth_std[:, 0]) / 1e3,
                    (smooth_mean[:, 0] + 2*smooth_std[:, 0]) / 1e3,
                    color="orange", alpha=0.15)

    # Forecast
    ax.plot(dates_fore, fore_mean[:, 0] / 1e3,
            color="red", lw=2, linestyle="--",
            label=f"Forecast {len(dates_fore)} days")
    ax.fill_between(dates_fore,
                    (fore_mean[:, 0] - 2*fore_std[:, 0]) / 1e3,
                    (fore_mean[:, 0] + 2*fore_std[:, 0]) / 1e3,
                    color="red", alpha=0.15, label="±2σ forecast")

    ax.axvline(dates[-1], color="gray", lw=1, linestyle=":")
    ax.set_ylabel("Consumption (GW)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("energy_forecast.png", dpi=150)
    print("Full plot → energy_forecast.png")
    plt.show()


def plot_backtest(dates_train, train_obs,
                  dates_test, test_obs, y_pred, y_std, metrics_dict):
    """Backtest view: end of history + predictions vs actuals."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # Panel 1: predictions vs actuals
    ax = axes[0]
    n_ctx = min(90, len(dates_train))
    ax.plot(dates_train[-n_ctx:], train_obs[-n_ctx:] / 1e3,
            color="steelblue", lw=1, alpha=0.6, label="Train (last 90 days)")
    ax.plot(dates_test, test_obs / 1e3, color="steelblue", lw=1.5,
            label="Test (actual)")
    ax.plot(dates_test, y_pred / 1e3, color="red", lw=1.5,
            linestyle="--", label="Kalman 1-step")
    ax.fill_between(dates_test,
                    (y_pred - 2*y_std) / 1e3,
                    (y_pred + 2*y_std) / 1e3,
                    color="red", alpha=0.15, label="±2σ")
    ax.axvline(dates_test[0], color="gray", lw=1, linestyle=":")
    ax.set_ylabel("Consumption (GW)")
    ax.set_title("Kalman-EM Backtest — French electricity consumption (one-step-ahead)")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    m_str = "  ".join(f"{k}: {v:.2f}" for k, v in metrics_dict.items())
    ax.text(0.01, 0.04, m_str, transform=ax.transAxes, fontsize=9,
            color="darkred", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # Panel 2: absolute error
    ax2 = axes[1]
    abs_err = np.abs(test_obs - y_pred) / 1e3
    ax2.bar(dates_test, abs_err, color="orange", alpha=0.7, width=1.2)
    ax2.axhline(abs_err.mean(), color="red", lw=1.2, linestyle="--",
                label=f"MAE = {abs_err.mean():.3f} GW")
    ax2.set_ylabel("|Error| (GW)")
    ax2.set_title("Daily absolute error")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(True, alpha=0.3)

    # Panel 3: normalised residuals
    ax3 = axes[2]
    res = (test_obs - y_pred) / y_std
    ax3.plot(dates_test, res, color="purple", lw=0.9, alpha=0.9)
    ax3.axhline( 2, color="red", lw=1, linestyle="--", alpha=0.6, label="±2σ")
    ax3.axhline(-2, color="red", lw=1, linestyle="--", alpha=0.6)
    ax3.axhline( 0, color="black", lw=0.8)
    ax3.set_ylabel("Normalised residual")
    ax3.set_title("Normalised residuals (target ≈ N(0,1))")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("energy_backtest.png", dpi=150)
    print("Backtest → energy_backtest.png")
    plt.show()


def plot_seasonal(dates, obs):
    """Empirical seasonal decomposition."""
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Conso_GW": obs / 1e3})
    df["Month"]   = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.dayofweek   # 0=Mon, 6=Sun

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Monthly profile
    monthly = df.groupby("Month")["Conso_GW"].agg(["mean", "std"])
    axes[0].bar(monthly.index, monthly["mean"], yerr=monthly["std"],
                color="steelblue", alpha=0.8, capsize=4)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"])
    axes[0].set_ylabel("Mean consumption (GW)")
    axes[0].set_title("Annual seasonality")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Weekly profile
    weekly = df.groupby("Weekday")["Conso_GW"].agg(["mean", "std"])
    axes[1].bar(weekly.index, weekly["mean"], yerr=weekly["std"],
                color="orange", alpha=0.8, capsize=4)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    axes[1].set_ylabel("Mean consumption (GW)")
    axes[1].set_title("Weekly seasonality")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("French electricity consumption 2020–2024 (source: RTE/ODRE)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("energy_seasonality.png", dpi=150)
    print("Seasonality → energy_seasonality.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load data
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)

    print(f"Loading: {csv_path}")
    dates, conso = load_csv(csv_path)
    T = len(conso)
    print(f"  {T} days  ({pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()})")
    print(f"  Consumption min={conso.min():.0f} MW, max={conso.max():.0f} MW, "
          f"mean={conso.mean():.0f} MW")

    print("\n[Data source]")
    print("  RTE (Réseau de Transport d'Électricité)")
    print("  Open Data Réseaux Energies (ODRE)")
    print("  https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def")
    print("  Licence: Etalab 2.0\n")

    # 2. Seasonality plot
    plot_seasonal(dates, conso)

    # 3. STL decomposition on the full series
    print("STL decomposition (annual seasonality, period=365)...")
    trend, seasonal, resid = stl_decompose(dates, conso, period=365)
    print(f"  Residual variance / total variance = "
          f"{np.var(resid) / np.var(conso):.1%}  "
          f"(target: < 20%)")

    # 4. Train / test split (on the residual)
    n_test  = max(1, int(T * args.test_ratio))
    n_train = T - n_test
    dates_train, dates_test = dates[:n_train], dates[n_train:]
    resid_train = resid[:n_train].values.reshape(-1, 1)
    resid_test  = resid[n_train:].values.reshape(-1, 1)
    print(f"Train: {n_train} days | Test: {n_test} days")

    # 5. Kalman-EM trained on the STL residual
    model = KalmanEM(d=args.latent, n_iter=args.iters, tol=1e-5,
                     diagonal_R=True, diagonal_Q=False, verbose=True)
    model.fit(resid_train, standardise=True)

    # 6. Smooth residual → reconstruct
    smooth_resid, smooth_var = model.smooth(resid_train)
    smooth_std = np.sqrt(np.abs(smooth_var))
    # Recombine with trend + seasonal
    trend_train    = trend[:n_train].values
    seasonal_train = seasonal[:n_train].values
    smooth_full    = smooth_resid[:, 0] + seasonal_train + trend_train
    smooth_full_std = smooth_std[:, 0]

    # 7. Out-of-sample forecast on the residual
    fore_resid, fore_var = model.forecast(resid_train, n_steps=args.days)
    fore_std_resid = np.sqrt(np.abs(fore_var[:, 0]))
    last = pd.Timestamp(dates_train[-1])
    fore_dates = pd.DatetimeIndex([last + pd.Timedelta(days=i+1) for i in range(args.days)])
    # Add projected seasonal component
    fore_seasonal = project_seasonal(dates_train, seasonal_train, fore_dates)
    fore_trend_ext = trend.iloc[-1] + np.arange(1, args.days + 1) * (trend.iloc[-1] - trend.iloc[-2])
    fore_full = fore_resid[:, 0] + fore_seasonal + fore_trend_ext

    # Wrap for plot_full (expected format)
    smooth_mean_wrap = smooth_full.reshape(-1, 1)
    smooth_std_wrap  = smooth_full_std.reshape(-1, 1)
    fore_mean_wrap   = fore_full.reshape(-1, 1)
    fore_std_wrap    = fore_std_resid.reshape(-1, 1)

    # 8. Full plot
    plot_full(dates_train, conso[:n_train],
              smooth_mean_wrap, smooth_std_wrap,
              fore_dates, fore_mean_wrap, fore_std_wrap,
              title="STL + Kalman-EM — French electricity consumption (RTE/ODRE)")

    # 9. One-step-ahead backtest on test residuals
    resid_pred_raw, resid_var_raw = model.predict_one_step(resid_test, Y_context=resid_train)
    resid_pred = resid_pred_raw[:, 0]
    resid_std  = np.sqrt(np.abs(resid_var_raw[:, 0]))
    # Recombine with seasonal + trend over the test period
    seasonal_test = seasonal[n_train:].values
    trend_test    = trend[n_train:].values
    y_pred = resid_pred + seasonal_test + trend_test
    y_std  = resid_std   # uncertainty is on the residual

    m = metrics(conso[n_train:], y_pred, y_std)
    print("\n" + "=" * 55)
    print("  TEST METRICS  (STL + Kalman-EM, one-step)")
    print("=" * 55)
    for k, v in m.items():
        print(f"  {k:<28}: {v:.2f}")
    print("=" * 55)

    # 10. First predictions table
    print(f"\n{'Date':<14}{'Actual (MW)':>12}{'Predicted (MW)':>15}{'Error':>10}{'±2σ':>10}")
    print("-" * 60)
    for i in range(min(10, n_test)):
        err = conso[n_train + i] - y_pred[i]
        print(f"{str(pd.Timestamp(dates_test[i]).date()):<14}"
              f"{conso[n_train+i]:>12.0f}{y_pred[i]:>15.0f}"
              f"{err:>10.0f}{2*y_std[i]:>10.0f}")

    # 11. Backtest plot
    plot_backtest(dates_train, conso[:n_train],
                  dates_test, conso[n_train:], y_pred, y_std, m)


if __name__ == "__main__":
    main()

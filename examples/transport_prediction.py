"""
French TGV traffic prediction with the Kalman-EM filter.

Data source
-----------
  SNCF Voyageurs — Monthly TGV punctuality (all routes, national aggregate)
  Portal : https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst
  API    : https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/
           regularite-mensuelle-tgv-aqst/records
  Licence: Licence Ouverte Etalab 2.0
  Period : 2018-01 → 2025-12  (96 months)

Available columns in the CSV
-----------------------------
  trains_prevus    : total number of scheduled TGV trains per month (national)
  annulations      : number of cancellations
  retards_15       : number of trains arriving with > 15 min delay
  ponctualite_pct  : punctuality rate = (scheduled - late) / scheduled × 100

Two Kalman-EM applications are demonstrated:
  1. Monthly train volume (trains_prevus) — strong seasonality
  2. Punctuality rate (ponctualite_pct) — bounded series [0,100], less seasonal

Usage
-----
    python examples/transport_prediction.py [--csv data/sncf_tgv_mensuel.csv] [--days 12]
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
    p = argparse.ArgumentParser(description="SNCF TGV traffic — Kalman-EM")
    p.add_argument("--csv",        default="data/sncf_tgv_mensuel.csv")
    p.add_argument("--target",     default="trains_prevus",
                   choices=["trains_prevus", "ponctualite_pct", "annulations"],
                   help="Target variable to predict")
    p.add_argument("--days",       type=int, default=12,
                   help="Forecast horizon in months")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--latent",     type=int, default=2)
    p.add_argument("--iters",      type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_std):
    err  = y_true - y_pred
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / np.where(y_true == 0, 1, y_true))) * 100
    cov  = np.mean((y_true >= y_pred - 2*y_std) & (y_true <= y_pred + 2*y_std)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "Coverage ±2σ (%)": cov}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(dates, values, target_label, title):
    """Overview of the raw series with event annotations."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(dates, values, color="steelblue", lw=1.5, marker="o",
            markersize=3, label=target_label)

    # COVID annotation
    covid_start = pd.Timestamp("2020-03-01")
    covid_end   = pd.Timestamp("2021-06-01")
    if dates[0] < covid_start < dates[-1]:
        ax.axvspan(covid_start, min(covid_end, dates[-1]),
                   color="red", alpha=0.08, label="COVID period")

    # Dec 2019 strike
    strike = pd.Timestamp("2019-12-01")
    if dates[0] < strike < dates[-1]:
        ax.axvline(strike, color="orange", lw=1.5, linestyle="--", label="Dec 2019 strike")

    ax.set_title(title)
    ax.set_ylabel(target_label)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("transport_overview.png", dpi=150)
    print("Overview → transport_overview.png")
    plt.show()


def plot_results(dates_train, train_obs,
                 smooth_mean, smooth_std,
                 dates_fore, fore_mean, fore_std,
                 dates_test, test_obs,
                 y_pred_test, y_std_test,
                 metrics_dict, target_label):

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # --- Panel 1: history + smoothing + forecast ---
    ax = axes[0]
    ax.plot(dates_train, train_obs, color="steelblue", lw=1,
            alpha=0.7, label="Training data")
    ax.plot(dates_test, test_obs, color="steelblue", lw=2,
            label="Test data (actual)", zorder=4)
    ax.plot(dates_train, smooth_mean[:, 0], color="orange", lw=1.5,
            label="Kalman smoother")
    ax.fill_between(dates_train,
                    smooth_mean[:, 0] - 2*smooth_std[:, 0],
                    smooth_mean[:, 0] + 2*smooth_std[:, 0],
                    color="orange", alpha=0.15)
    ax.plot(dates_fore, fore_mean[:, 0], color="red", lw=2,
            linestyle="--", label=f"Forecast {len(dates_fore)} months")
    ax.fill_between(dates_fore,
                    fore_mean[:, 0] - 2*fore_std[:, 0],
                    fore_mean[:, 0] + 2*fore_std[:, 0],
                    color="red", alpha=0.15, label="±2σ")
    ax.axvline(dates_train[-1], color="gray", lw=1, linestyle=":")

    # COVID annotation
    covid_start = pd.Timestamp("2020-03-01")
    covid_end   = pd.Timestamp("2021-06-01")
    if dates_train[0] < covid_start < dates_train[-1]:
        ax.axvspan(covid_start, min(covid_end, pd.Timestamp(dates_train[-1])),
                   color="red", alpha=0.06)
        ax.text(covid_start, ax.get_ylim()[0], " COVID", fontsize=8,
                color="red", va="bottom")

    ax.set_ylabel(target_label)
    ax.set_title(f"STL + Kalman-EM — SNCF TGV traffic ({target_label})")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(True, alpha=0.3)

    m_str = "  ".join(f"{k}: {v:.2f}" for k, v in metrics_dict.items())
    ax.text(0.01, 0.05, m_str, transform=ax.transAxes, fontsize=9,
            color="darkred", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    # --- Panel 2: absolute error on test ---
    ax2 = axes[1]
    abs_err = np.abs(test_obs - y_pred_test)
    ax2.bar(dates_test, abs_err, color="orange", alpha=0.75, width=20)
    ax2.axhline(abs_err.mean(), color="red", lw=1.2, linestyle="--",
                label=f"MAE = {abs_err.mean():.1f}")
    ax2.set_ylabel(f"|Error| ({target_label})")
    ax2.set_title("Absolute error (one-step-ahead, test period)")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: normalised residuals ---
    ax3 = axes[2]
    res_norm = (test_obs - y_pred_test) / np.where(y_std_test == 0, 1, y_std_test)
    ax3.bar(dates_test, res_norm, color="purple", alpha=0.7, width=20)
    ax3.axhline( 2, color="red", lw=1, linestyle="--", alpha=0.6, label="±2σ")
    ax3.axhline(-2, color="red", lw=1, linestyle="--", alpha=0.6)
    ax3.axhline( 0, color="black", lw=0.8)
    ax3.set_ylabel("Normalised residual")
    ax3.set_title("Normalised residuals (target ≈ N(0,1))")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("transport_backtest.png", dpi=150)
    print("Backtest → transport_backtest.png")
    plt.show()


def plot_monthly_profile(dates, values, target_label):
    """Monthly seasonal profile (January–December)."""
    df = pd.DataFrame({"m": pd.to_datetime(dates).month, "v": values})
    by_month = df.groupby("m")["v"].agg(["mean", "std"])
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_month.index, by_month["mean"], yerr=by_month["std"],
           color="steelblue", alpha=0.8, capsize=5, width=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylabel(target_label)
    ax.set_title(f"Monthly seasonal profile — {target_label} (SNCF TGV 2018–2025)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("transport_seasonality.png", dpi=150)
    print("Seasonal profile → transport_seasonality.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)

    # 1. Load data
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    dates  = df["Date"].values
    values = df[args.target].values.astype(float)
    T = len(values)

    target_labels = {
        "trains_prevus":   "Scheduled TGV trains / month",
        "ponctualite_pct": "Punctuality rate (%)",
        "annulations":     "Cancelled trains / month",
    }
    target_label = target_labels[args.target]

    print(f"\n{'='*60}")
    print(f"  Target: {target_label}")
    print(f"{'='*60}")
    print(f"\n[Data source]")
    print(f"  SNCF Voyageurs — Monthly TGV punctuality")
    print(f"  https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst")
    print(f"  Licence: Etalab 2.0\n")
    print(f"  {T} months  ({pd.Timestamp(dates[0]).strftime('%B %Y')} → "
          f"{pd.Timestamp(dates[-1]).strftime('%B %Y')})")
    print(f"  min={values.min():.1f}  max={values.max():.1f}  "
          f"mean={values.mean():.1f}  σ={values.std():.1f}\n")

    # 2. Overview + seasonal profile
    plot_overview(pd.to_datetime(dates), values, target_label,
                  f"SNCF TGV — {target_label} (2018–2025)")
    plot_monthly_profile(dates, values, target_label)

    # 3. STL decomposition (period=12 for monthly data)
    print("STL decomposition (period=12 — annual monthly seasonality)...")
    s = pd.Series(values, index=pd.to_datetime(dates))
    stl_res = STL(s, period=12, robust=True).fit()
    trend, seasonal, resid = stl_res.trend, stl_res.seasonal, stl_res.resid
    print(f"  Residual variance / total variance = "
          f"{float(np.var(resid)) / float(np.var(values)):.1%}")

    # 4. Train / test split
    n_test  = max(2, int(T * args.test_ratio))
    n_train = T - n_test
    dates_train, dates_test = dates[:n_train], dates[n_train:]
    resid_train = resid.values[:n_train].reshape(-1, 1)
    resid_test  = resid.values[n_train:].reshape(-1, 1)
    trend_train    = trend.values[:n_train]
    seasonal_train = seasonal.values[:n_train]
    print(f"  Train: {n_train} months | Test: {n_test} months\n")

    # 5. Kalman-EM on the residual
    model = KalmanEM(d=args.latent, n_iter=args.iters, tol=1e-5,
                     diagonal_R=True, diagonal_Q=False, verbose=True)
    model.fit(resid_train, standardise=True)

    # 6. Smoothing
    smooth_resid, smooth_var = model.smooth(resid_train)
    smooth_std = np.sqrt(np.abs(smooth_var))
    smooth_full = smooth_resid[:, 0] + seasonal_train + trend_train
    smooth_full_std = smooth_std[:, 0]

    # 7. Forecast n months ahead
    fore_resid, fore_var = model.forecast(resid_train, n_steps=args.days)
    fore_std_r = np.sqrt(np.abs(fore_var[:, 0]))
    last_date = pd.Timestamp(dates_train[-1])
    fore_dates = pd.DatetimeIndex(
        [last_date + pd.DateOffset(months=i+1) for i in range(args.days)]
    )
    # Seasonal projection: profile of corresponding months
    df_hist = pd.DataFrame({"m": pd.to_datetime(dates_train).month,
                             "s": seasonal_train})
    s_profile = df_hist.groupby("m")["s"].mean()
    fore_seasonal = np.array([s_profile.get(d.month, 0) for d in fore_dates])
    # Linear trend extension
    slope = float(trend.values[-1] - trend.values[-2])
    fore_trend = float(trend.values[-1]) + np.arange(1, args.days+1) * slope
    fore_full = fore_resid[:, 0] + fore_seasonal + fore_trend

    # 8. One-step-ahead backtest on test set
    resid_pred_r, resid_var_r = model.predict_one_step(resid_test, Y_context=resid_train)
    y_pred = resid_pred_r[:, 0] + seasonal.values[n_train:] + trend.values[n_train:]
    y_std  = np.sqrt(np.abs(resid_var_r[:, 0]))

    m = compute_metrics(values[n_train:], y_pred, y_std)

    print("\n" + "=" * 55)
    print("  TEST METRICS  (STL + Kalman-EM, one-step)")
    print("=" * 55)
    for k, v in m.items():
        print(f"  {k:<28}: {v:.2f}")
    print("=" * 55)

    # 9. Forecast table
    print(f"\n{'Month':<12}{'Forecast':>12}{'Lower 95%':>12}{'Upper 95%':>12}")
    print("-" * 55)
    for dt, fm, fs in zip(fore_dates, fore_full, fore_std_r):
        print(f"{dt.strftime('%Y-%m'):<12}{fm:>12.0f}{fm - 2*fs:>12.0f}{fm + 2*fs:>12.0f}")

    # 10. Full plot
    plot_results(
        pd.to_datetime(dates_train), values[:n_train],
        smooth_full.reshape(-1, 1), smooth_full_std.reshape(-1, 1),
        fore_dates, fore_full.reshape(-1, 1), fore_std_r.reshape(-1, 1),
        pd.to_datetime(dates_test), values[n_train:],
        y_pred, y_std,
        m, target_label,
    )


if __name__ == "__main__":
    main()

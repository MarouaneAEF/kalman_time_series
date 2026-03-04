"""
Solar cycle forecasting with the EM-Kalman filter — backtest vs real data.

Strategy
--------
  - Train on the first N_train months
  - Test on the remaining months
  - One-step-ahead predictions: at each t, predict y_t from history y_{1:t-1}
  - Metrics: MAE, RMSE, MAPE, confidence interval coverage

Usage
-----
    python examples/sunspot_prediction.py [--test_ratio 0.15] [--latent 4] [--iters 200]
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kalman_em import KalmanEM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Backtesting Kalman-EM on sunspots")
    p.add_argument("--csv",        default="data/sunspots_monthly.csv")
    p.add_argument("--test_ratio", type=float, default=0.15,
                   help="Fraction of series reserved for testing (default 0.15)")
    p.add_argument("--latent",     type=int,   default=4,
                   help="Latent state dimension (4 = cycle + harmonics)")
    p.add_argument("--iters",      type=int,   default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_std, alpha=2.0):
    err  = y_true - y_pred
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    # avoid division by zero for months with 0 sunspots
    nonzero = y_true > 0
    mape = np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100
    lo   = y_pred - alpha * y_std
    hi   = y_pred + alpha * y_std
    cov  = np.mean((y_true >= lo) & (y_true <= hi)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape,
            f"Coverage ±{alpha}σ (%)": cov}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_backtest(dates_train, obs_train,
                  dates_test, obs_test, y_pred, y_std,
                  metrics, title):

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # ---- Panel 1: historical series + test predictions ----
    ax = axes[0]
    ax.plot(dates_train, obs_train, color="steelblue", lw=0.7,
            label="Train", alpha=0.6)
    ax.plot(dates_test, obs_test, color="steelblue", lw=1.5,
            label="Test (actual)", zorder=4)
    ax.plot(dates_test, y_pred, color="red", lw=1.5, linestyle="--",
            label="Kalman 1-step", zorder=5)
    ax.fill_between(dates_test, y_pred - 2*y_std, y_pred + 2*y_std,
                    color="red", alpha=0.15, label="±2σ")
    ax.axvline(dates_test[0], color="gray", linestyle=":", lw=1.2)
    ax.set_ylabel("Monthly sunspot number")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(True, alpha=0.3)

    metric_str = "  ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
    ax.text(0.01, 0.04, metric_str, transform=ax.transAxes,
            fontsize=9, color="darkred",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # ---- Panel 2: absolute error ----
    ax2 = axes[1]
    abs_err = np.abs(obs_test - y_pred)
    ax2.bar(dates_test, abs_err, color="orange", alpha=0.7, width=20)
    ax2.axhline(abs_err.mean(), color="red", lw=1.2, linestyle="--",
                label=f"MAE = {abs_err.mean():.2f}")
    ax2.set_ylabel("|Error|")
    ax2.set_title("Absolute prediction error (one-step-ahead)")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: normalised residuals ----
    ax3 = axes[2]
    residuals = (obs_test - y_pred) / y_std
    ax3.plot(dates_test, residuals, color="purple", lw=0.8, alpha=0.9)
    ax3.axhline( 2, color="red", lw=1, linestyle="--", alpha=0.6)
    ax3.axhline(-2, color="red", lw=1, linestyle="--", alpha=0.6, label="±2σ")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("Normalised residual")
    ax3.set_title("Normalised residuals  (should be ≈ N(0,1))")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "sunspot_backtest.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
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
    df = df[["Date", "Sunspots"]].dropna().sort_values("Date").reset_index(drop=True)
    dates    = df["Date"].values
    sunspots = df["Sunspots"].values.astype(float)
    T        = len(sunspots)

    # 2. Train / test split
    n_test  = max(1, int(T * args.test_ratio))
    n_train = T - n_test
    print(f"Train: {n_train} months | Test: {n_test} months")

    dates_train = dates[:n_train]
    obs_train   = sunspots[:n_train]
    dates_test  = dates[n_train:]
    obs_test    = sunspots[n_train:]

    Y_train = obs_train.reshape(-1, 1)
    Y_test  = obs_test.reshape(-1, 1)

    # 3. Train on train set
    model = KalmanEM(d=args.latent, n_iter=args.iters, tol=1e-5,
                     diagonal_R=True, diagonal_Q=False, verbose=True)
    model.fit(Y_train, standardise=True)

    # 4. One-step-ahead predictions on test set
    y_pred_raw, y_var_raw = model.predict_one_step(Y_test, Y_context=Y_train)
    y_pred = y_pred_raw[:, 0]
    y_std  = np.sqrt(np.abs(y_var_raw[:, 0]))

    # 5. Metrics
    metrics = compute_metrics(obs_test, y_pred, y_std, alpha=2.0)

    print("\n" + "=" * 55)
    print("  TEST SET METRICS (one-step-ahead)")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v:.4f}")
    print("=" * 55)

    # 6. Table of first 10 predictions vs actuals
    print(f"\n{'Date':<14}{'Actual':>10}{'Predicted':>10}{'Error':>10}{'±2σ':>10}")
    print("-" * 55)
    for i in range(min(10, n_test)):
        err = obs_test[i] - y_pred[i]
        print(f"{str(pd.Timestamp(dates_test[i]).date()):<14}"
              f"{obs_test[i]:>10.1f}{y_pred[i]:>10.1f}"
              f"{err:>10.1f}{2*y_std[i]:>10.1f}")
    if n_test > 10:
        print(f"  ... ({n_test - 10} more months)")

    # 7. Plot
    plot_backtest(
        dates_train, obs_train,
        dates_test, obs_test, y_pred, y_std,
        metrics,
        title=f"Kalman-EM Backtest — Sunspots  (test = last {n_test} months)"
    )


if __name__ == "__main__":
    main()

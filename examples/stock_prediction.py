"""
Stock price prediction with the EM-Kalman filter.

Usage
-----
    python examples/stock_prediction.py [--csv data/AAPL_prices.csv] [--days 30]

The script:
1. Reads a local CSV file (columns: Date, Close).
2. Trains a KalmanEM model via the EM algorithm.
3. Smooths the historical series (RTS smoother).
4. Forecasts the next `--days` trading days.
5. Prints the forecast table and saves the plot.
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
    p = argparse.ArgumentParser(description="EM-Kalman stock predictor")
    p.add_argument("--csv",    default="data/AAPL_prices.csv",
                   help="Path to local CSV (columns: Date, Close)")
    p.add_argument("--latent", type=int, default=2,
                   help="Latent state dimension (2 = level + trend)")
    p.add_argument("--days",   type=int, default=30,
                   help="Forecast horizon (trading days)")
    p.add_argument("--iters",  type=int, default=200,
                   help="Max number of EM iterations")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    return df["Date"].values, df["Close"].values.astype(float)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(dates_hist, prices_hist,
                 smoothed_mean, smoothed_std,
                 dates_fore, fore_mean, fore_std,
                 title):

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_zoom = axes

    # --- Main panel ---
    ax.plot(dates_hist, prices_hist, color="steelblue", lw=1.2,
            label="Closing price", zorder=3)
    ax.plot(dates_hist, smoothed_mean[:, 0], color="orange", lw=1.5,
            label="Kalman smoother", zorder=4)
    ax.fill_between(dates_hist,
                    smoothed_mean[:, 0] - 2 * smoothed_std[:, 0],
                    smoothed_mean[:, 0] + 2 * smoothed_std[:, 0],
                    color="orange", alpha=0.2, label="±2σ smoother")

    ax.plot(dates_fore, fore_mean[:, 0], color="red", lw=1.8,
            linestyle="--", label=f"Forecast {len(dates_fore)} days", zorder=5)
    ax.fill_between(dates_fore,
                    fore_mean[:, 0] - 2 * fore_std[:, 0],
                    fore_mean[:, 0] + 2 * fore_std[:, 0],
                    color="red", alpha=0.15, label="±2σ forecast")

    ax.axvline(dates_hist[-1], color="gray", linestyle=":", lw=1.2)
    ax.set_ylabel("Price (USD)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(True, alpha=0.3)

    # --- Zoom: end of history + forecast ---
    n = min(60, len(dates_hist))
    zoom_dates  = list(dates_hist[-n:]) + list(dates_fore)
    zoom_prices = list(prices_hist[-n:]) + [np.nan] * len(dates_fore)

    ax_zoom.plot(zoom_dates, zoom_prices, color="steelblue", lw=1.2)
    ax_zoom.plot(dates_fore, fore_mean[:, 0], color="red", lw=1.8, linestyle="--")
    ax_zoom.fill_between(dates_fore,
                         fore_mean[:, 0] - 2 * fore_std[:, 0],
                         fore_mean[:, 0] + 2 * fore_std[:, 0],
                         color="red", alpha=0.2)
    ax_zoom.axvline(dates_hist[-1], color="gray", linestyle=":", lw=1.2)
    ax_zoom.set_ylabel("Zoom")
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax_zoom.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "kalman_forecast.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load CSV
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)

    print(f"Loading: {csv_path}")
    dates, prices = load_csv(csv_path)
    print(f"  {len(prices)} days  ({pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()})")

    Y = prices.reshape(-1, 1)

    # 2. EM-Kalman training
    model = KalmanEM(
        d=args.latent,
        n_iter=args.iters,
        tol=1e-5,
        diagonal_R=True,
        diagonal_Q=False,
        verbose=True,
    )
    model.fit(Y, standardise=True)

    # 3. Smoothing
    smooth_mean, smooth_var = model.smooth(Y)
    smooth_std = np.sqrt(np.abs(smooth_var))

    # 4. Forecast
    fore_mean, fore_var = model.forecast(Y, n_steps=args.days)
    fore_std = np.sqrt(np.abs(fore_var))

    # Forecast dates (trading days only)
    last = pd.Timestamp(dates[-1])
    fore_dates = []
    d = last
    while len(fore_dates) < args.days:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            fore_dates.append(d)

    # 5. Forecast table
    print(f"\n{'Date':<14}{'Forecast':>12}{'Lower 95%':>12}{'Upper 95%':>12}")
    print("-" * 57)
    for dt, m, s in zip(fore_dates, fore_mean[:, 0], fore_std[:, 0]):
        print(f"{str(dt.date()):<14}{m:>12.2f}{m - 2*s:>12.2f}{m + 2*s:>12.2f}")

    # 6. Plot
    ticker_name = os.path.splitext(os.path.basename(args.csv))[0]
    plot_results(
        dates, prices,
        smooth_mean, smooth_std,
        fore_dates, fore_mean, fore_std,
        title=f"Kalman-EM filter — {ticker_name}",
    )


if __name__ == "__main__":
    main()

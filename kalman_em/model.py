"""
High-level KalmanEM model: fit, filter, smooth, forecast.
"""

import numpy as np
from .em import run_em
from .kalman_filter import kalman_filter, rts_smoother


class KalmanEM:
    """
    Linear Gaussian state-space model with EM-learned parameters.

    State space:
        x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = H x_t    + v_t,  v_t ~ N(0, R)

    Parameters
    ----------
    d : int
        Latent state dimension.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood.
    diagonal_R : bool
        Restrict R to be diagonal (recommended for independent observed series).
    diagonal_Q : bool
        Restrict Q to be diagonal.
    verbose : bool
    """

    def __init__(self, d=2, n_iter=200, tol=1e-4,
                 diagonal_R=True, diagonal_Q=False,
                 n_restarts=1, verbose=True):
        self.d = d
        self.n_iter = n_iter
        self.tol = tol
        self.diagonal_R = diagonal_R
        self.diagonal_Q = diagonal_Q
        self.n_restarts = n_restarts
        self.verbose = verbose

        self.params_ = None
        self.log_liks_ = None
        self._scaler = None   # (mean, std) for standardisation

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, Y, init_params=None, standardise=True):
        """
        Fit model parameters via EM.

        Parameters
        ----------
        Y : (T, m) array-like  — observations (NaN allowed for missing)
        init_params : dict, optional
        standardise : bool — z-score the data before fitting (recommended)

        Returns
        -------
        self
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]

        if standardise:
            mu = np.nanmean(Y, axis=0)
            std = np.nanstd(Y, axis=0)
            std[std == 0] = 1.0
            self._scaler = (mu, std)
            Y = (Y - mu) / std
        else:
            self._scaler = None

        if self.verbose:
            print(f"Fitting KalmanEM  (d={self.d}, T={Y.shape[0]}, m={Y.shape[1]})")

        self.params_, self.log_liks_ = run_em(
            Y,
            d=self.d,
            n_iter=self.n_iter,
            tol=self.tol,
            init_params=init_params,
            diagonal_Q=self.diagonal_Q,
            diagonal_R=self.diagonal_R,
            n_restarts=self.n_restarts,
            verbose=self.verbose,
        )
        return self

    # ------------------------------------------------------------------
    # Filter / Smooth
    # ------------------------------------------------------------------

    def _preprocess(self, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if self._scaler is not None:
            mu, std = self._scaler
            Y = (Y - mu) / std
        return Y

    def _postprocess(self, Y_hat):
        if self._scaler is not None:
            mu, std = self._scaler
            Y_hat = Y_hat * std + mu
        return Y_hat

    def filter(self, Y):
        """
        Run Kalman filter on Y.

        Returns
        -------
        mu_filt : (T, m)  filtered observation means  H x_{t|t}
        var_filt : (T, m)  filtered observation variances  diag(H P_{t|t} H^T + R)
        """
        p = self.params_
        Ys = self._preprocess(Y)
        mu_filt, Sigma_filt, _, _, _ = kalman_filter(
            Ys, p["F"], p["H"], p["Q"], p["R"], p["mu0"], p["Sigma0"]
        )
        H = p["H"]
        R = p["R"]
        obs_mean = (H @ mu_filt.T).T
        obs_var  = np.array([np.diag(H @ S @ H.T + R) for S in Sigma_filt])

        return self._postprocess(obs_mean), obs_var * (self._scaler[1] ** 2 if self._scaler else 1)

    def smooth(self, Y):
        """
        Run Kalman smoother on Y.

        Returns
        -------
        mu_smooth : (T, m)  smoothed observation means
        var_smooth : (T, m)  smoothed observation variances
        """
        p = self.params_
        Ys = self._preprocess(Y)
        mu_filt, Sigma_filt, mu_pred, Sigma_pred, _ = kalman_filter(
            Ys, p["F"], p["H"], p["Q"], p["R"], p["mu0"], p["Sigma0"]
        )
        mu_smooth, Sigma_smooth, _ = rts_smoother(
            mu_filt, Sigma_filt, mu_pred, Sigma_pred, p["F"]
        )
        H = p["H"]
        R = p["R"]
        obs_mean = (H @ mu_smooth.T).T
        obs_var  = np.array([np.diag(H @ S @ H.T + R) for S in Sigma_smooth])

        return self._postprocess(obs_mean), obs_var * (self._scaler[1] ** 2 if self._scaler else 1)

    # ------------------------------------------------------------------
    # One-step-ahead predictions (for backtesting)
    # ------------------------------------------------------------------

    def predict_one_step(self, Y, Y_context=None):
        """
        One-step-ahead predictions on Y: at each t, predict y_t given y_{1:t-1}.

        Parameters
        ----------
        Y         : (T, m)  observations to predict
        Y_context : (T0, m) optional history to warm-up the filter before Y

        Returns
        -------
        y_pred  : (T, m)  predicted means  H x_{t|t-1}
        y_var   : (T, m)  predicted variances  diag(H P_{t|t-1} H^T + R)
        """
        p = self.params_
        F, H, Q, R = p["F"], p["H"], p["Q"], p["R"]

        # Warm-up from context if provided
        if Y_context is not None:
            Ycs = self._preprocess(Y_context)
            mu_filt_ctx, Sigma_filt_ctx, _, _, _ = kalman_filter(
                Ycs, F, H, Q, R, p["mu0"], p["Sigma0"]
            )
            mu0_use    = mu_filt_ctx[-1]
            Sigma0_use = Sigma_filt_ctx[-1]
        else:
            mu0_use, Sigma0_use = p["mu0"], p["Sigma0"]

        Ys = self._preprocess(Y)
        _, _, mu_pred, Sigma_pred, _ = kalman_filter(
            Ys, F, H, Q, R, mu0_use, Sigma0_use
        )

        y_pred = (H @ mu_pred.T).T                                   # (T, m)
        y_var  = np.array([np.diag(H @ S @ H.T + R) for S in Sigma_pred])

        scale2 = (self._scaler[1] ** 2) if self._scaler else 1.0
        return self._postprocess(y_pred), y_var * scale2

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(self, Y, n_steps=10):
        """
        One-step-ahead filtered estimates on Y, then project n_steps ahead.

        Returns
        -------
        y_forecast : (n_steps, m)  predicted observation means
        y_var      : (n_steps, m)  predicted observation variances
        """
        p = self.params_
        Ys = self._preprocess(Y)
        mu_filt, Sigma_filt, _, _, _ = kalman_filter(
            Ys, p["F"], p["H"], p["Q"], p["R"], p["mu0"], p["Sigma0"]
        )
        F, H, Q, R = p["F"], p["H"], p["Q"], p["R"]

        # Start from the last filtered state
        mu   = mu_filt[-1]
        Sigma = Sigma_filt[-1]

        y_means = []
        y_vars  = []
        for _ in range(n_steps):
            mu    = F @ mu
            Sigma = F @ Sigma @ F.T + Q
            y_means.append(H @ mu)
            y_vars.append(np.diag(H @ Sigma @ H.T + R))

        y_means = np.array(y_means)   # (n_steps, m)
        y_vars  = np.array(y_vars)    # (n_steps, m)

        return self._postprocess(y_means), y_vars * (self._scaler[1] ** 2 if self._scaler else 1)

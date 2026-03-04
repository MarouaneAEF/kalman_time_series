"""
Kalman Filter and RTS Smoother for linear Gaussian state-space models.

State space model:
    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
    y_t = H x_t    + v_t,   v_t ~ N(0, R)
    x_0 ~ N(mu0, Sigma0)
"""

import numpy as np


def _make_pd(A, eps=1e-8):
    """
    Enforce positive definiteness.

    Fast path: try Cholesky (O(d³/3)).  If it fails, fall back to eigen-
    projection which also clamps all eigenvalues to at least `eps`.
    """
    A = 0.5 * (A + A.T)
    try:
        np.linalg.cholesky(A)
        return A
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


def kalman_filter(Y, F, H, Q, R, mu0, Sigma0):
    """
    Forward Kalman filter pass.

    Parameters
    ----------
    Y : (T, m) array  — observations (NaN rows are treated as missing)
    F : (d, d)        — transition matrix
    H : (m, d)        — observation matrix
    Q : (d, d)        — process noise covariance
    R : (m, m)        — observation noise covariance
    mu0 : (d,)        — initial state mean
    Sigma0 : (d, d)   — initial state covariance

    Returns
    -------
    mu_filt    : (T, d)      filtered means  x_{t|t}
    Sigma_filt : (T, d, d)   filtered covariances  P_{t|t}
    mu_pred    : (T, d)      predicted means  x_{t|t-1}
    Sigma_pred : (T, d, d)   predicted covariances  P_{t|t-1}
    log_likelihood : float
    """
    T, m = Y.shape
    d = F.shape[0]

    mu_pred    = np.zeros((T, d))
    Sigma_pred = np.zeros((T, d, d))
    mu_filt    = np.zeros((T, d))
    Sigma_filt = np.zeros((T, d, d))
    log_likelihood = 0.0
    log2pi = np.log(2 * np.pi)

    mu    = mu0.copy()
    Sigma = _make_pd(Sigma0)

    for t in range(T):
        # --- Predict ---
        mu_p    = F @ mu
        Sigma_p = _make_pd(F @ Sigma @ F.T + Q)
        mu_pred[t]    = mu_p
        Sigma_pred[t] = Sigma_p

        # --- Update (handle missing observations) ---
        obs     = Y[t]
        missing = np.isnan(obs)
        if missing.all():
            mu_filt[t]    = mu_p
            Sigma_filt[t] = Sigma_p
            mu, Sigma = mu_p, Sigma_p
            continue

        obs_idx = ~missing
        obs_m   = int(obs_idx.sum())
        if missing.any():
            H_obs = H[obs_idx]
            R_obs = R[np.ix_(obs_idx, obs_idx)]
            y_obs = obs[obs_idx]
        else:
            H_obs, R_obs, y_obs = H, R, obs

        # Innovation covariance S = H P H^T + R
        S = _make_pd(H_obs @ Sigma_p @ H_obs.T + R_obs)

        # Cholesky of S — used for gain, log-det, and Mahalanobis distance
        L = np.linalg.cholesky(S)           # S = L L^T

        innovation = y_obs - H_obs @ mu_p

        # Kalman gain via solve:  K = Sigma_p H^T S^{-1}
        #   => S K^T = H Sigma_p  => K = solve(S, H Sigma_p)^T
        K = np.linalg.solve(S, H_obs @ Sigma_p).T

        mu = mu_p + K @ innovation

        # Joseph form for numerical stability: (I-KH) P (I-KH)^T + K R K^T
        IKH   = np.eye(d) - K @ H_obs
        Sigma = _make_pd(IKH @ Sigma_p @ IKH.T + K @ R_obs @ K.T)

        mu_filt[t]    = mu
        Sigma_filt[t] = Sigma

        # Log-likelihood contribution — all quantities from the Cholesky of S
        log_det = 2.0 * np.sum(np.log(np.diag(L)))   # log|S| — numerically stable
        v       = np.linalg.solve(L, innovation)       # L v = innov  (triangular)
        maha    = float(v @ v)                          # innov^T S^{-1} innov

        log_likelihood += -0.5 * (log_det + maha + obs_m * log2pi)

    return mu_filt, Sigma_filt, mu_pred, Sigma_pred, log_likelihood


def rts_smoother(mu_filt, Sigma_filt, mu_pred, Sigma_pred, F):
    """
    Rauch-Tung-Striebel (RTS) backward smoother.

    Returns
    -------
    mu_smooth    : (T, d)      smoothed means  x_{t|T}
    Sigma_smooth : (T, d, d)   smoothed covariances  P_{t|T}
    G            : (T-1, d, d) smoother gains  J_t  (t = 0 … T-2)
    """
    T, d = mu_filt.shape
    mu_smooth    = mu_filt.copy()
    Sigma_smooth = Sigma_filt.copy()
    G = np.zeros((T - 1, d, d))

    for t in range(T - 2, -1, -1):
        # Smoother gain: J = Sigma_filt[t] F^T Sigma_pred[t+1]^{-1}
        #   => Sigma_pred[t+1] J^T = F Sigma_filt[t]
        #   => J = solve(Sigma_pred[t+1], F @ Sigma_filt[t])^T
        J = np.linalg.solve(Sigma_pred[t + 1], F @ Sigma_filt[t]).T
        G[t] = J

        mu_smooth[t] = mu_filt[t] + J @ (mu_smooth[t + 1] - mu_pred[t + 1])
        diff = Sigma_smooth[t + 1] - Sigma_pred[t + 1]
        Sigma_smooth[t] = _make_pd(Sigma_filt[t] + J @ diff @ J.T)

    return mu_smooth, Sigma_smooth, G


def lag_one_covariance(Sigma_smooth, G):
    """
    Lag-one smoothed covariance  P_{t+1, t | T}  for t = 0 … T-2.

    Direct formula (Ghahramani & Hinton, 1996):
        P_{t+1, t | T} = Sigma_smooth[t+1] @ J_t^T

    Returns
    -------
    C : (T-1, d, d)  where C[t] = P_{t+1, t | T}
    """
    # C[t] = Sigma_smooth[t+1] @ G[t].T  for t = 0..T-2
    return np.einsum("tij,tkj->tik", Sigma_smooth[1:], G)
